# contextualization_manager.py
"""Runtime and dataset-map manager for train-time contextualization.

``ContextualizationManager`` connects the strategy registry, dataset adapters,
contextualized dataset loader, and ``DistilTrainer``. Static strategies build
their teacher prompt during dataset mapping. Dynamic strategies store serialized
row/reference metadata so the trainer can construct stage-1 and final teacher
prompts after generation.

Strategies produce chat messages; adapters merge those messages with the
original task shell when a dataset requires system-level formatting. The manager
also records concrete prompt/stage-1 variation ids and the original selection
specification for reproducibility and later analysis.
"""

from __future__ import annotations

import json
import re
from typing import Any

from dataset_adapters import get_dataset_adapter
from .contextualization_strategies import get_context_strategy


JsonLike = dict[str, Any] | list[Any] | str | int | float | bool | None


def make_jsonable(value: Any) -> JsonLike:
    """
    Recursively convert dataset objects into JSON-serializable values while
    preserving dict/list structure.

    Important: never fallback to str(value) for the whole raw example, because
    dynamic strategies need raw_example_json to decode back to a dict.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")

    if isinstance(value, dict):
        return {str(k): make_jsonable(v) for k, v in value.items()}

    # Hugging Face Dataset.map passes rows as LazyRow objects, not plain dicts.
    # LazyRow has keys()/__getitem__ but is not isinstance(value, dict).
    if hasattr(value, "keys") and hasattr(value, "__getitem__"):
        try:
            return {str(k): make_jsonable(value[k]) for k in value.keys()}
        except Exception:
            pass

    if isinstance(value, (list, tuple)):
        return [make_jsonable(v) for v in value]

    # numpy scalar, pandas scalar, etc.
    if hasattr(value, "item"):
        try:
            return make_jsonable(value.item())
        except Exception:
            pass

    # numpy array, torch tensor, etc.
    if hasattr(value, "tolist"):
        try:
            return make_jsonable(value.tolist())
        except Exception:
            pass

    # Last-resort fallback only for this leaf, not for the whole example.
    return str(value)


def json_dumps_safe(value: Any) -> str:
    """
    Serialize metadata so Hugging Face datasets can keep it as a plain column.

    The trainer later reconstructs raw_example_json/reference_json for dynamic
    strategies, so dict/list structure must be preserved.
    """
    return json.dumps(make_jsonable(value), ensure_ascii=False)


def json_loads_safe(value: Any) -> Any:
    """
    Inverse of json_dumps_safe, tolerant to already-decoded values.

    DistilTrainer calls this on example.get("raw_example_json") and
    example.get("reference_json"). Returning None for empty/missing values lets
    the trainer raise a clear dynamic-strategy error.
    """
    if value is None:
        return None

    if isinstance(value, (dict, list, int, float, bool)):
        return value

    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
        #value = value.decode("utf-8")

    if not isinstance(value, str):
        return value

    if value == "":
        return None

    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value
    

_CONTEXT_SPEC_RE = re.compile(r"^\s*([A-Za-z0-9_]+)(?:\(([^()]*)\))?\s*$")


def _parse_optional_positive_int(value: str, *, field_name: str) -> int | None:
    value = value.strip()

    if value == "":
        return None

    if value == "*":
        raise ValueError(
            f"{field_name}='*' is only supported by contextualized_train_loader.py, "
            "where random per-row selection is resolved before constructing "
            "ContextualizationManager."
        )

    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(
            f"Invalid {field_name}={value!r}. Expected a positive integer."
        ) from exc

    if parsed < 1:
        raise ValueError(
            f"Invalid {field_name}={value!r}. Expected a positive integer."
        )

    return parsed


def parse_context_strategy_spec(spec: str) -> tuple[str, int | None, int | None]:
    """
    Parse a fixed context strategy spec.

    Supported:
        strategy
        strategy(prompt_variation)
        strategy(prompt_variation, stage1_variation)

    Examples:
        self_feedback_concrete       -> ("self_feedback_concrete", None, None)
        self_feedback_concrete(2)    -> ("self_feedback_concrete", 2, None)
        self_feedback_concrete(2,3)  -> ("self_feedback_concrete", 2, 3)

    Random wildcard '*' is intentionally not resolved here. The loader resolves
    it per row and passes concrete integers to the manager.
    """
    match = _CONTEXT_SPEC_RE.match(spec)
    if not match:
        raise ValueError(
            f"Invalid context_strategy spec {spec!r}. Expected 'strategy', "
            "'strategy(k)', or 'strategy(k,m)'."
        )

    strategy_name = match.group(1)
    args_text = match.group(2)

    if args_text is None:
        return strategy_name, None, None

    parts = [part.strip() for part in args_text.split(",")]
    if len(parts) > 2:
        raise ValueError(
            f"Invalid context_strategy spec {spec!r}. Expected at most two "
            "variation arguments: strategy(prompt_variation, stage1_variation)."
        )

    prompt_variation = _parse_optional_positive_int(
        parts[0],
        field_name="prompt_variation",
    )

    stage1_variation = None
    if len(parts) == 2:
        stage1_variation = _parse_optional_positive_int(
            parts[1],
            field_name="stage1_variation",
        )

    return strategy_name, prompt_variation, stage1_variation


class ContextualizationManager:
    """
    Build contextualized train examples and runtime teacher/feedback prompts.
    """

    def __init__(
        self,
        context_strategy: str = "dataset_default",
        *,
        prompt_variation: int | None = None,
        stage1_variation: int | None = None,
    ) -> None:
        """
        Keep the manager responsible for:
        - parsing fixed strategy specs such as self_feedback_concrete(1,2);
        - holding the selected concrete strategy name;
        - passing selected variations to the strategy.

        Random specs with '*' or strategy pools with '|' are intentionally resolved
        earlier by contextualized_train_loader.py, once per dataset row.
        """
        (
            parsed_strategy,
            parsed_prompt_variation,
            parsed_stage1_variation,
        ) = parse_context_strategy_spec(context_strategy)

        self.context_strategy_spec = context_strategy
        self.context_strategy = parsed_strategy

        self.prompt_variation = (
            prompt_variation
            if prompt_variation is not None
            else parsed_prompt_variation
        )
        self.stage1_variation = (
            stage1_variation
            if stage1_variation is not None
            else parsed_stage1_variation
        )

        self.strategy = get_context_strategy(self.context_strategy)

    def _copy_optional_fields(self, *, formatted: dict[str, Any], output: dict[str, Any]) -> None:
        """
        Preserve optional multimodal fields if a future dataset uses them.
        """
        for key in ("image", "images"):
            if key in formatted:
                output[key] = formatted[key]

    def _resolve_prompt_variation(
        self,
        *,
        dataset_name: str,
        prompt_variation: int | None = None,
    ) -> int:
        """
        Resolve the final teacher-prompt variation.

        None selects the default prompt variation (variation 1).
        """
        selected = (
            prompt_variation
            if prompt_variation is not None
            else self.prompt_variation
        )

        return self.strategy.resolve_prompt_variation(
            selected,
            dataset_name=dataset_name,
        )


    def _resolve_stage1_variation(
        self,
        *,
        dataset_name: str,
        stage1_variation: int | None = None,
    ) -> int | None:
        """
        Resolve the auxiliary feedback/rationale prompt variation.

        For non-two-stage strategies, there is no stage1 prompt, so this returns
        None unless the user explicitly asked for an unsupported nontrivial stage1
        variation.
        """
        selected = (
            stage1_variation
            if stage1_variation is not None
            else self.stage1_variation
        )

        if hasattr(self.strategy, "resolve_stage1_variation"):
            return self.strategy.resolve_stage1_variation(
                selected,
                dataset_name=dataset_name,
            )

        if selected is not None and selected != 1:
            raise ValueError(
                f"Strategy {self.context_strategy!r} does not support stage1 "
                f"variations, but got stage1_variation={selected}."
            )

        return None

    def _prepare_raw_example(
        self,
        *,
        dataset_name: str,
        raw_example: dict[str, Any],
        prompt_variation: int,
    ) -> dict[str, Any]:
        """Apply an optional strategy-level view transformation safely."""
        prepared = self.strategy.prepare_raw_example(
            dataset_name=dataset_name,
            raw_example=raw_example,
            prompt_variation=prompt_variation,
        )
        prepared = make_jsonable(prepared)
        if not isinstance(prepared, dict):
            raise TypeError(
                f"Strategy {self.context_strategy!r}.prepare_raw_example() must "
                f"return a dict, but got {type(prepared).__name__}."
            )
        return prepared

    def _base_output(
        self,
        *,
        dataset_name: str,
        raw_example: dict[str, Any],
        formatted: dict[str, Any],
        reference: Any,
        teacher_prompt: Any,
        prompt_variation: int | None,
        stage1_variation: int | None,
        selection_spec: str | None = None,
        base_prompt_variation: int | None = None,
        row_index: int | None = None,
        selection_seed: int | None = None,
        prompt_variation_random: bool = False,
        stage1_variation_random: bool = False,
    ) -> dict[str, Any]:
        """
        Common output schema consumed by DistilTrainer.
        """
        output = {
            "prompt": formatted["prompt"],
            "teacher_prompt": teacher_prompt,
            "target": formatted["target"],
            "source_dataset": dataset_name,

            # Concrete strategy actually used by this row.
            "context_strategy": self.context_strategy,

            # Fixed manager-level spec, usually equal to context_strategy unless the
            # manager was built from something like "self_feedback_concrete(1,2)".
            "context_strategy_spec": self.context_strategy_spec,

            # Original user selection string, e.g.
            # "gold_hint|self_feedback_concrete(*,*)". Useful for later analysis.
            "context_strategy_selection_spec": selection_spec or self.context_strategy_spec,

            # Concrete resolved variations used by this row.
            "context_prompt_variation": prompt_variation,
            "context_base_prompt_variation": (
                prompt_variation if base_prompt_variation is None else base_prompt_variation
            ),
            "context_stage1_variation": stage1_variation,

            # Selection provenance. These fields are inert for every existing
            # strategy; they are consumed only by explicitly opted-in epoch
            # resampling strategies. -1 keeps the Arrow integer columns typed.
            "context_row_index": -1 if row_index is None else int(row_index),
            "context_selection_seed": -1 if selection_seed is None else int(selection_seed),
            "context_prompt_variation_random": bool(prompt_variation_random),
            "context_stage1_variation_random": bool(stage1_variation_random),

            "raw_example_json": json_dumps_safe(raw_example),
            "reference_json": json_dumps_safe(reference),
        }
        self._copy_optional_fields(formatted=formatted, output=output)
        return output

    def format_train_example(
        self,
        *,
        dataset_name: str,
        raw_example: dict[str, Any],
        prompt_variation: int | None = None,
        stage1_variation: int | None = None,
        selection_spec: str | None = None,
        base_prompt_variation: int | None = None,
        row_index: int | None = None,
        selection_seed: int | None = None,
        prompt_variation_random: bool = False,
        stage1_variation_random: bool = False,
    ) -> dict[str, Any]:
        """
        Format one raw training example for DistilTrainer.

        For static/precomputable strategies, this creates the final
        teacher_prompt immediately.

        For dynamic strategies, this stores metadata and uses the dataset-default
        teacher prompt as a placeholder. ``DistilTrainer`` replaces it at runtime
        after generation.

        The row/seed and wildcard-provenance fields are bookkeeping only. They
        let an explicitly opted-in strategy choose a deterministic epoch-level
        variation without replaying the original strategy-selection parser.
        Existing callers can omit them and preserve the old behavior.
        """
        raw_example = make_jsonable(raw_example)
        if not isinstance(raw_example, dict):
            raise TypeError(
                f"ContextualizationManager.format_train_example expected raw_example "
                f"to normalize to dict, but got {type(raw_example).__name__}: "
                f"{str(raw_example)[:300]}"
            )
        adapter = get_dataset_adapter(dataset_name)
        self.strategy.validate_dataset(dataset_name)

        if self.context_strategy == "dataset_default":
            # Usually bypassed by contextualized_train_loader.py, but keeping
            # this exact behavior makes manager-level tests simple.
            formatted = adapter.format_train_example(raw_example)
            reference = adapter.get_reference_for_context(raw_example, formatted)
            return self._base_output(
                dataset_name=dataset_name,
                raw_example=raw_example,
                formatted=formatted,
                reference=reference,
                teacher_prompt=formatted["teacher_prompt"],
                prompt_variation=None,
                stage1_variation=None,
                selection_spec=selection_spec,
                base_prompt_variation=base_prompt_variation,
                row_index=row_index,
                selection_seed=selection_seed,
                prompt_variation_random=prompt_variation_random,
                stage1_variation_random=stage1_variation_random,
            )

        selected_prompt_variation = self._resolve_prompt_variation(
            dataset_name=dataset_name,
            prompt_variation=prompt_variation,
        )
        selected_stage1_variation = self._resolve_stage1_variation(
            dataset_name=dataset_name,
            stage1_variation=stage1_variation,
        )
        raw_example = self._prepare_raw_example(
            dataset_name=dataset_name,
            raw_example=raw_example,
            prompt_variation=selected_prompt_variation,
        )
        formatted = adapter.format_train_example(raw_example)
        reference = adapter.get_reference_for_context(raw_example, formatted)

        if not self.strategy.can_precompute:
            # Runtime prompt will be built in DistilTrainer. The placeholder is
            # intentionally valid so static code paths and debugging displays do
            # not crash before replacement.
            return self._base_output(
                dataset_name=dataset_name,
                raw_example=raw_example,
                formatted=formatted,
                reference=reference,
                teacher_prompt=formatted["teacher_prompt"],
                prompt_variation=selected_prompt_variation,
                stage1_variation=selected_stage1_variation,
                selection_spec=selection_spec,
                base_prompt_variation=base_prompt_variation,
                row_index=row_index,
                selection_seed=selection_seed,
                prompt_variation_random=prompt_variation_random,
                stage1_variation_random=stage1_variation_random,
            )

        strategy_messages = self.strategy.build_messages(
            dataset_name=dataset_name,
            adapter=adapter,
            raw_example=raw_example,
            reference=reference,
            prompt_variation=selected_prompt_variation,
        )
        teacher_prompt = adapter.contextualize_messages(raw_example, strategy_messages)

        return self._base_output(
            dataset_name=dataset_name,
            raw_example=raw_example,
            formatted=formatted,
            reference=reference,
            teacher_prompt=teacher_prompt,
            prompt_variation=selected_prompt_variation,
            stage1_variation=selected_stage1_variation,
            selection_spec=selection_spec,
            base_prompt_variation=base_prompt_variation,
            row_index=row_index,
            selection_seed=selection_seed,
            prompt_variation_random=prompt_variation_random,
            stage1_variation_random=stage1_variation_random,
        )

    def build_teacher_prompt_runtime(
        self,
        *,
        dataset_name: str,
        raw_example: dict[str, Any],
        reference: Any,
        student_response: str | None = None,
        feedback_text: str | None = None,
        sibling_text: str | None = None,
        prompt_variation: int | None = None,
    ) -> Any:
        """
        Build final teacher prompt at runtime for dynamic strategies.

        Also works for static strategies, which is useful for tests/parity
        checks.
        """
        adapter = get_dataset_adapter(dataset_name)
        self.strategy.validate_dataset(dataset_name)

        if self.context_strategy == "dataset_default":
            return adapter.format_train_example(raw_example)["teacher_prompt"]

        selected_prompt_variation = self._resolve_prompt_variation(
            dataset_name=dataset_name,
            prompt_variation=prompt_variation,
        )
        raw_example = self._prepare_raw_example(
            dataset_name=dataset_name,
            raw_example=raw_example,
            prompt_variation=selected_prompt_variation,
        )

        strategy_messages = self.strategy.build_messages(
            dataset_name=dataset_name,
            adapter=adapter,
            raw_example=raw_example,
            reference=reference,
            student_response=student_response,
            feedback_text=feedback_text,
            sibling_text=sibling_text,
            prompt_variation=selected_prompt_variation,
        )
        return adapter.contextualize_messages(raw_example, strategy_messages)

    def build_feedback_prompt_runtime(
        self,
        *,
        dataset_name: str,
        raw_example: dict[str, Any],
        reference: Any,
        student_response: str,
        stage1_variation: int | None = None,
    ) -> Any:
        """
        Build the auxiliary feedback/rationale prompt for strategies that need
        generated feedback_text before the final teacher prompt.
        """
        adapter = get_dataset_adapter(dataset_name)
        self.strategy.validate_dataset(dataset_name)

        if not self.strategy.requires_feedback_generation:
            raise ValueError(
                f"Strategy {self.context_strategy!r} does not require feedback generation."
            )

        if not hasattr(self.strategy, "build_feedback_messages"):
            raise AttributeError(
                f"Strategy {self.context_strategy!r} has requires_feedback_generation=True "
                "but does not define build_feedback_messages(...)."
            )
        
        selected_stage1_variation = self._resolve_stage1_variation(
            dataset_name=dataset_name,
            stage1_variation=stage1_variation,
        )

        feedback_messages = self.strategy.build_feedback_messages(
            dataset_name=dataset_name,
            adapter=adapter,
            raw_example=raw_example,
            reference=reference,
            student_response=student_response,
            stage1_variation=selected_stage1_variation,
        )
        return adapter.contextualize_messages(raw_example, feedback_messages)
