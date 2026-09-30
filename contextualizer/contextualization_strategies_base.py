# contextualization_strategies_base.py
"""Base interfaces and shared utilities for contextualization strategies.

``ContextStrategy`` defines one-stage/static teacher contextualization;
``TwoStageStrategy`` adds an auxiliary generated feedback/rationale stage.
The interfaces expose dataset restrictions, canonical train views, prompt and
stage-1 variations, sibling requirements, and strategy-specific auxiliary
generation-length policies.
"""

from __future__ import annotations

import hashlib
import random
from abc import ABC
from string import Template
from typing import Any

FEEDBACK_MAX_USE_DATASET_DEFAULT = "dataset_default"

# ---------------------------------------------------------------------------
# Per-epoch prompt-variation resampling policies
# ---------------------------------------------------------------------------
EPOCH_VARIATION_POLICY_SHUFFLED_CYCLE = "shuffled_cycle"
EPOCH_VARIATION_POLICY_RANDOM = "random"

EPOCH_VARIATION_POLICIES = (
    EPOCH_VARIATION_POLICY_SHUFFLED_CYCLE,
    EPOCH_VARIATION_POLICY_RANDOM,
)


def _stable_epoch_rng(*parts: object) -> random.Random:
    """Return a deterministic RNG for epoch-level contextualization decisions."""
    key = "::".join(str(part) for part in parts)
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))


# =============================================================================
# Generic rendering helpers
# =============================================================================


def _adapter_question_text(adapter, raw_example: dict) -> str:
    """
    Dataset-agnostic question extraction.

    Delegates to DatasetAdapter.get_question_text, which should reproduce the
    exact column recovery behavior for each registered dataset.
    """
    return adapter.get_question_text(raw_example)


def _adapter_compact_answer_text(adapter, raw_example: dict, reference: Any) -> str:
    """
    Compact final answer / target decision.

    Delegates to DatasetAdapter.get_compact_reference_text_for_context.
    """
    return adapter.get_compact_reference_text_for_context(raw_example, reference)


def _adapter_full_reference_text(adapter, raw_example: dict, reference: Any) -> str:
    """
    Full gold response / full reference solution.

    Delegates to DatasetAdapter.get_full_reference_text_for_context.
    """
    return adapter.get_full_reference_text_for_context(raw_example, reference)


def _adapter_full_reference_text_old(adapter, raw_example: dict, reference: Any) -> str:
    """
    Compatibility full-reference rendering used by strategies that intentionally
    retain JSON-style list formatting.
    """
    if hasattr(adapter, "get_full_reference_text_for_context_old"):
        return adapter.get_full_reference_text_for_context_old(raw_example, reference)

    return adapter.get_full_reference_text_for_context(raw_example, reference)


def _adapter_source_context_text(adapter, raw_example: dict) -> str:
    """Optional source document for source-grounded contextualization."""
    if hasattr(adapter, "get_source_context_text_for_context"):
        return adapter.get_source_context_text_for_context(raw_example)
    return ""


def _adapter_short_hint_text(adapter, raw_example: dict, reference: Any) -> str:
    """
    Short oracle hint.

    Uses DatasetAdapter.get_short_hint_text_for_context when available. This is
    important for ToolUse, where gold_hint should use the compact action/input
    hint, not just a JSON dump of golden_answer.
    """
    if hasattr(adapter, "get_short_hint_text_for_context"):
        return adapter.get_short_hint_text_for_context(raw_example, reference)
    return _adapter_compact_answer_text(adapter, raw_example, reference)





# =============================================================================
# Base strategy interface
# =============================================================================

class ContextStrategy(ABC):
    """
    Base class for train-time contextualization strategies.

    Attributes
    ----------
    name:
        CLI/config strategy name.

    requires_student_response:
        True when the final contextualized teacher prompt depends on a first
        model response. Example: tooluse_failure_mode, self_feedback.

    requires_feedback_generation:
        True when the strategy requires a separate generated feedback/rationale
        before building the final teacher prompt. Example: self_feedback,
        self_feedback_structured, self_feedback_concrete, rationalization.

    can_precompute:
        True when ContextualizationManager can build teacher_prompt during
        dataset.map. False when DistilTrainer must build teacher_prompt at
        runtime.

    supported_datasets:
        Optional set of dataset names. ToolUse-only strategies should declare
        {"tooluse"}.

    Prompt variation contract
    -------------------------
    - prompt_variation is 1-based.
    - prompt_variation=None means "use variation 1".
    - A strategy can define prompt_templates = [...]
      and use the default build_messages implementation.
    - A strategy can override build_messages when it needs custom logic.

    Per-epoch resampling contract
    -----------------------------
    Existing strategies keep one concrete prompt variation per mapped row for
    the whole run. A strategy may opt into epoch-level variation changes by
    setting ``resample_prompt_variation_per_epoch = True``. The trainer applies
    that behavior only to rows whose variation was selected through ``(*)``;
    explicit controls such as ``strategy(3)`` always stay fixed.

    Epoch 0 always uses the concrete variation stored by dataset.map. For the
    default ``shuffled_cycle`` policy, later epochs walk a deterministic per-row
    permutation, rotated so that the stored map-time variation is first. Thus a
    row sees every valid variation once before repeating any of them.

    ``selection_rng_namespace`` is intentionally separate from the strategy
    name. It lets an experimental derived strategy preserve the exact map-time
    RNG assignments of its control strategy while changing only what happens in
    later epochs. Existing strategies leave it as ``None``, so their historical
    RNG keys remain byte-for-byte unchanged.

    A strategy that opts into epoch resampling must make
    ``prepare_raw_example`` safe to apply repeatedly with different variations,
    because the trainer rebuilds the mapped row through ContextualizationManager.
    """

    name: str = "base"

    # Canonical adapter training view selected before dataset.map. Existing
    # strategies use the dataset default view unless they opt into another view.
    train_view: str = "default"

    # Raw source-only views have no gold completion and must reject dataset/SFT
    # supervision before training starts.
    supports_dataset_targets: bool = True

    requires_student_response: bool = False
    requires_feedback_generation: bool = False
    requires_sibling_selection: bool = False
    sibling_fallback_to_gold: bool = True
    can_precompute: bool = True
    supported_datasets: set[str] | None = None

    # Optional runtime optimizations for SDPO-like strategies.
    sibling_only_if_student_incorrect: bool = False
    feedback_only_without_correct_solution: bool = False

    # Optional simple-template variations.
    prompt_templates: list[str] = []

    # Existing behavior is fixed-per-row. Experimental strategies can opt in.
    resample_prompt_variation_per_epoch: bool = False
    epoch_variation_policy: str = EPOCH_VARIATION_POLICY_SHUFFLED_CYCLE

    # Optional RNG alias used only by the loader when constructing the row-level
    # selection seed. None preserves the historical strategy name/key exactly.
    selection_rng_namespace: str | None = None

    # Optional max-token override for generated stage-1 feedback/rationale/rewrite.
    #
    # None:
    #   preserve current behavior and use args.feedback_max_new_tokens.
    #
    # int:
    #   use this fixed max_new_tokens for this strategy.
    #
    # FEEDBACK_MAX_USE_DATASET_DEFAULT / "dataset_default":
    #   use adapter.default_max_new_tokens for the source dataset.
    feedback_max_new_tokens: int | str | None = None

    @property
    def num_prompt_variations(self) -> int:
        return max(1, len(self.prompt_templates))

    def get_valid_prompt_variations(self, *, dataset_name: str) -> list[int]:
        """Return every valid 1-based prompt variation for this dataset."""
        valid: list[int] = []
        for variation in range(1, self.num_prompt_variations + 1):
            try:
                self.validate_prompt_variation(variation, dataset_name=dataset_name)
            except ValueError:
                continue
            valid.append(variation)
        return valid

    def resolve_epoch_variation_policy(self) -> str:
        """Return the validated epoch-level variation policy."""
        policy = str(self.epoch_variation_policy)
        if policy not in EPOCH_VARIATION_POLICIES:
            raise ValueError(
                f"Strategy {self.name!r} declares epoch_variation_policy="
                f"{self.epoch_variation_policy!r}, which is not one of "
                f"{list(EPOCH_VARIATION_POLICIES)}."
            )
        return policy

    def resolve_epoch_prompt_variation(
        self,
        *,
        dataset_name: str,
        base_variation: int,
        row_index: int,
        seed: int,
        epoch_index: int,
    ) -> int:
        """Resolve the concrete prompt variation for one mapped row and epoch.

        This method does not decide *whether* the row came from wildcard syntax;
        the loader records that provenance and the trainer checks it before
        calling here. It only implements the opted-in strategy's epoch policy.
        """
        self.validate_dataset(dataset_name)
        self.validate_prompt_variation(base_variation, dataset_name=dataset_name)

        epoch_index = int(epoch_index)
        if epoch_index <= 0 or not self.resample_prompt_variation_per_epoch:
            return int(base_variation)

        valid = self.get_valid_prompt_variations(dataset_name=dataset_name)
        if not valid:
            raise ValueError(
                f"Strategy {self.name!r} has no valid prompt variations for "
                f"dataset {dataset_name!r}."
            )
        if base_variation not in valid:
            raise ValueError(
                f"Base prompt variation {base_variation} is not valid for strategy "
                f"{self.name!r} on dataset {dataset_name!r}. Valid: {valid}."
            )

        policy = self.resolve_epoch_variation_policy()

        if policy == EPOCH_VARIATION_POLICY_SHUFFLED_CYCLE:
            rng = _stable_epoch_rng(
                "context_epoch_variation",
                seed,
                dataset_name,
                row_index,
                self.name,
                policy,
            )
            order = list(valid)
            rng.shuffle(order)
            start = order.index(base_variation)
            order = order[start:] + order[:start]
            return int(order[epoch_index % len(order)])

        if policy == EPOCH_VARIATION_POLICY_RANDOM:
            rng = _stable_epoch_rng(
                "context_epoch_variation",
                seed,
                dataset_name,
                row_index,
                self.name,
                policy,
                epoch_index,
            )
            return int(valid[rng.randrange(len(valid))])

        # resolve_epoch_variation_policy already validates this; keep an explicit
        # defensive branch in case the method is overridden incorrectly.
        raise AssertionError(f"Unhandled epoch variation policy: {policy!r}")

    def validate_dataset(self, dataset_name: str) -> None:
        if self.supported_datasets is not None and dataset_name not in self.supported_datasets:
            raise ValueError(
                f"Strategy {self.name!r} does not support dataset {dataset_name!r}. "
                f"Supported datasets: {sorted(self.supported_datasets)}"
            )

    def validate_prompt_variation(
        self,
        prompt_variation: int | None,
        *,
        dataset_name: str | None = None,
    ) -> None:
        """
        Validate a 1-based prompt variation.

        Override this only for special cases, for example if a variation is
        only valid for ToolUse.
        """
        if prompt_variation is None:
            return

        if prompt_variation < 1 or prompt_variation > self.num_prompt_variations:
            raise ValueError(
                f"Strategy {self.name!r} has {self.num_prompt_variations} prompt "
                f"variation(s), but got variation {prompt_variation}."
            )

    def resolve_prompt_variation(
        self,
        prompt_variation: int | None,
        *,
        dataset_name: str | None = None,
    ) -> int:
        """
        Return a valid 1-based variation index.
        """
        if prompt_variation is None:
            prompt_variation = 1

        self.validate_prompt_variation(
            prompt_variation,
            dataset_name=dataset_name,
        )
        return prompt_variation

    def prepare_raw_example(
        self,
        *,
        dataset_name: str,
        raw_example: dict,
        prompt_variation: int | None = None,
    ) -> dict:
        """Return the row used to build both student and teacher prompts.

        Most strategies contextualize only the teacher and therefore return the
        row unchanged. Strategies that intentionally vary the task seen by both
        models may override this hook. Implementations must return a new mapping
        rather than mutate the dataset row in place.
        """
        self.validate_dataset(dataset_name)
        self.resolve_prompt_variation(
            prompt_variation,
            dataset_name=dataset_name,
        )
        return raw_example

    def get_prompt_template(
        self,
        prompt_variation: int,
        *,
        dataset_name: str | None = None,
    ) -> str:
        """
        Return the selected template.

        Default behavior uses self.prompt_templates.
        Override only if templates are dataset-specific.
        """
        if not self.prompt_templates:
            raise NotImplementedError(
                f"Strategy {self.name!r} does not define prompt_templates and "
                "does not override build_messages(...)."
            )

        return self.prompt_templates[prompt_variation - 1]

    def build_template_variables(
        self,
        *,
        dataset_name: str,
        adapter,
        raw_example: dict,
        reference: Any,
        student_response: str | None = None,
        feedback_text: str | None = None,
        sibling_text: str | None = None,
    ) -> dict[str, Any]:
        """
        Common variables available to prompt_templates.

        Individual strategies can override this to add strategy-specific fields.
        """
        return {
            "question": _adapter_question_text(adapter, raw_example),
            "compact_answer": _adapter_compact_answer_text(adapter, raw_example, reference),
            "full_reference": _adapter_full_reference_text(adapter, raw_example, reference),
            "short_hint": _adapter_short_hint_text(adapter, raw_example, reference),
            "source_context": _adapter_source_context_text(adapter, raw_example),
            "student_response": student_response or "",
            "feedback_text": feedback_text or "",
            "sibling_text": sibling_text or "",
        }

    def render_prompt_template(
        self,
        template: str,
        variables: dict[str, Any],
    ) -> str:
        """
        Render a template using $variable syntax.

        Using string.Template avoids accidental conflicts with JSON braces.
        """
        return Template(template).substitute(
            {key: str(value) for key, value in variables.items()}
        ).strip()

    def build_messages(
        self,
        *,
        dataset_name: str,
        adapter,
        raw_example: dict,
        reference: Any,
        student_response: str | None = None,
        feedback_text: str | None = None,
        sibling_text: str | None = None,
        prompt_variation: int | None = None,
    ) -> list[dict[str, str]] | str:
        """
        Default implementation for template-based strategies.
        """
        self.validate_dataset(dataset_name)

        variation = self.resolve_prompt_variation(
            prompt_variation,
            dataset_name=dataset_name,
        )

        template = self.get_prompt_template(
            variation,
            dataset_name=dataset_name,
        )

        variables = self.build_template_variables(
            dataset_name=dataset_name,
            adapter=adapter,
            raw_example=raw_example,
            reference=reference,
            student_response=student_response,
            feedback_text=feedback_text,
            sibling_text=sibling_text,
        )

        content = self.render_prompt_template(template, variables)

        return [{"role": "user", "content": content}]
    
    def resolve_feedback_max_new_tokens(
        self,
        *,
        dataset_name: str,
        adapter,
        configured_max_new_tokens: int,
    ) -> int:
        """
        Resolve the max_new_tokens used for stage-1 feedback/rationale/rewrite
        generation.

        Default behavior:
            feedback_max_new_tokens is None -> use args.feedback_max_new_tokens.

        Strategies that need longer auxiliary generations can override the class
        attribute with either:
            - an integer;
            - FEEDBACK_MAX_USE_DATASET_DEFAULT / "dataset_default".
        """
        value = self.feedback_max_new_tokens

        if value is None:
            resolved = configured_max_new_tokens

        elif isinstance(value, int):
            resolved = value

        elif value == FEEDBACK_MAX_USE_DATASET_DEFAULT:
            dataset_default = getattr(adapter, "default_max_new_tokens", None)
            resolved = configured_max_new_tokens if dataset_default is None else dataset_default

        else:
            raise ValueError(
                f"Invalid feedback_max_new_tokens={value!r} for strategy {self.name!r}. "
                f"Expected None, a positive integer, or {FEEDBACK_MAX_USE_DATASET_DEFAULT!r}."
            )

        resolved = int(resolved)
        if resolved < 1:
            raise ValueError(
                f"Resolved feedback max_new_tokens must be positive for strategy "
                f"{self.name!r} on dataset {dataset_name!r}, got {resolved}."
            )

        return resolved




class TwoStageStrategy(ContextStrategy):
    """
    Base class for generic two-stage strategies.

    Stage 1:
        Generate feedback/rationale from a stage1 prompt.

    Stage 2:
        Build the final teacher prompt from the original question plus the
        generated stage1 text, passed as feedback_text.
    """

    name: str = "two_stage_base"
    requires_student_response = False
    requires_feedback_generation = True
    can_precompute = False

    stage1_templates: list[str] = []

    # Fallback text used when feedback_text is unavailable.
    default_feedback_text: str = ""

    @property
    def num_stage1_variations(self) -> int:
        return max(1, len(self.stage1_templates))

    def validate_stage1_variation(
        self,
        stage1_variation: int | None,
        *,
        dataset_name: str | None = None,
    ) -> None:
        if stage1_variation is None:
            return

        if stage1_variation < 1 or stage1_variation > self.num_stage1_variations:
            raise ValueError(
                f"Strategy {self.name!r} has {self.num_stage1_variations} stage1 "
                f"variation(s), but got variation {stage1_variation}."
            )

    def resolve_stage1_variation(
        self,
        stage1_variation: int | None,
        *,
        dataset_name: str | None = None,
    ) -> int:
        if stage1_variation is None:
            stage1_variation = 1

        self.validate_stage1_variation(
            stage1_variation,
            dataset_name=dataset_name,
        )
        return stage1_variation

    def get_stage1_template(
        self,
        stage1_variation: int,
        *,
        dataset_name: str | None = None,
    ) -> str:
        if not self.stage1_templates:
            raise NotImplementedError(
                f"Strategy {self.name!r} does not define stage1_templates and "
                "does not override build_feedback_messages(...)."
            )

        return self.stage1_templates[stage1_variation - 1]

    def build_template_variables(
        self,
        *,
        dataset_name: str,
        adapter,
        raw_example: dict,
        reference: Any,
        student_response: str | None = None,
        feedback_text: str | None = None,
        sibling_text: str | None = None,
    ) -> dict[str, Any]:
        variables = super().build_template_variables(
            dataset_name=dataset_name,
            adapter=adapter,
            raw_example=raw_example,
            reference=reference,
            student_response=student_response,
            feedback_text=feedback_text,
            sibling_text=sibling_text,
        )

        if not variables["feedback_text"]:
            variables["feedback_text"] = self.default_feedback_text

        return variables

    def build_feedback_messages(
        self,
        *,
        dataset_name: str,
        adapter,
        raw_example: dict,
        reference: Any,
        student_response: str | None,
        stage1_variation: int | None = None,
    ) -> list[dict[str, str]]:
        self.validate_dataset(dataset_name)

        variation = self.resolve_stage1_variation(
            stage1_variation,
            dataset_name=dataset_name,
        )

        template = self.get_stage1_template(
            variation,
            dataset_name=dataset_name,
        )

        variables = self.build_template_variables(
            dataset_name=dataset_name,
            adapter=adapter,
            raw_example=raw_example,
            reference=reference,
            student_response=student_response,
        )

        content = self.render_prompt_template(template, variables)

        return [{"role": "user", "content": content}]
