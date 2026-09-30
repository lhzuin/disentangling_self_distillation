# contextualized_train_loader.py
"""Contextualized training-dataset loader and strategy-spec resolver.

The loader selects the canonical dataset training view, parses strategy/variation
expressions, performs deterministic per-row strategy and variation selection,
and delegates row formatting to ``ContextualizationManager``. Strategy pools
must share one canonical training view, and source-only views are rejected for
dataset-target supervision.
"""

from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass
from typing import Literal

from datasets import load_from_disk

from .contextualization_manager import ContextualizationManager
from .contextualization_strategies import get_context_strategy
from dataset_adapters import get_dataset_adapter


VariationChoice = int | Literal["random"] | None


_CONTEXT_CHOICE_RE = re.compile(r"^\s*([A-Za-z0-9_]+)(?:\(([^()]*)\))?\s*$")
_CONTEXT_POOL_NAME_RE = re.compile(r"(^|\|)(\s*)([A-Za-z0-9_]+)")


@dataclass(frozen=True)
class ContextRowSelection:
    """Concrete row-level selection plus provenance of wildcard choices."""

    strategy_name: str
    prompt_variation: int
    stage1_variation: int | None
    prompt_variation_random: bool
    stage1_variation_random: bool


def _stable_rng(*parts: object) -> random.Random:
    """
    Deterministic per-row RNG.

    This makes random strategy/variation assignment reproducible for the same:
      seed, dataset name, row index, and context_strategy string.
    """
    key = "::".join(str(part) for part in parts)
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))


def _selection_rng_spec(context_strategy: str) -> str:
    """Return the string used in the historical row-selection RNG key.

    Existing strategies are returned byte-for-byte unchanged. A derived
    experimental strategy may declare ``selection_rng_namespace`` to reuse its
    control strategy's RNG identity. Only the strategy-name token is replaced;
    whitespace, pool separators and variation syntax are preserved exactly.
    """

    def replace_name(match: re.Match[str]) -> str:
        prefix, whitespace, strategy_name = match.groups()
        strategy = get_context_strategy(strategy_name)
        namespace = getattr(strategy, "selection_rng_namespace", None)
        rng_name = strategy_name if namespace in (None, "") else str(namespace)
        return f"{prefix}{whitespace}{rng_name}"

    return _CONTEXT_POOL_NAME_RE.sub(replace_name, context_strategy)


def _parse_variation_choice(value: str, *, field_name: str) -> VariationChoice:
    """
    Parse one variation argument.

    Supported:
      ""  -> None, meaning default variation
      "*" -> "random", meaning random valid variation per row
      "2" -> 2, meaning fixed variation 2
    """
    value = value.strip()

    if value == "":
        return None

    if value == "*":
        return "random"

    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(
            f"Invalid {field_name}={value!r}. Expected a positive integer or '*'."
        ) from exc

    if parsed < 1:
        raise ValueError(
            f"Invalid {field_name}={value!r}. Expected a positive integer or '*'."
        )

    return parsed


def _parse_context_choice_spec(
    spec: str,
) -> tuple[str, VariationChoice, VariationChoice]:
    """
    Parse one strategy choice.

    Supported:
      strategy
      strategy(prompt_variation)
      strategy(prompt_variation, stage1_variation)

    Examples:
      gold_hint
      gold_hint(2)
      gold_hint(*)
      self_feedback_concrete(1,2)
      self_feedback_concrete(*,*)

    This function parses one choice only. Pools are handled by splitting the
    full context_strategy string on '|'.
    """
    match = _CONTEXT_CHOICE_RE.match(spec)
    if not match:
        raise ValueError(
            f"Invalid context strategy choice {spec!r}. Expected 'strategy', "
            "'strategy(k)', or 'strategy(k,m)'."
        )

    strategy_name = match.group(1)
    args_text = match.group(2)

    # Validate the strategy name early.
    get_context_strategy(strategy_name)

    if args_text is None:
        return strategy_name, None, None

    parts = [part.strip() for part in args_text.split(",")]
    if len(parts) > 2:
        raise ValueError(
            f"Invalid context strategy choice {spec!r}. Expected at most two "
            "variation arguments: strategy(prompt_variation, stage1_variation)."
        )

    prompt_choice = _parse_variation_choice(
        parts[0],
        field_name="prompt_variation",
    )

    stage1_choice: VariationChoice = None
    if len(parts) == 2:
        stage1_choice = _parse_variation_choice(
            parts[1],
            field_name="stage1_variation",
        )

    return strategy_name, prompt_choice, stage1_choice


def _strategies_in_spec(context_strategy: str, *, dataset_name: str):
    """Resolve every semantic strategy that may be selected from a pool."""
    choices = [part.strip() for part in context_strategy.split("|") if part.strip()]
    if not choices:
        raise ValueError(f"Empty context_strategy specification: {context_strategy!r}")

    strategies = []
    for choice in choices:
        strategy_name, _prompt_choice, _stage1_choice = _parse_context_choice_spec(choice)
        strategy = get_context_strategy(strategy_name)
        strategy.validate_dataset(dataset_name)
        strategies.append(strategy)
    return strategies


def resolve_context_strategy_train_view(
    context_strategy: str,
    *,
    dataset_name: str,
) -> str:
    """Return the single canonical train view required by a strategy pool.

    Pools are resolved per row, so all strategies in one pool must consume the
    same raw dataset. This prevents accidental row-index alignment across
    incompatible canonical views.
    """
    strategies = _strategies_in_spec(context_strategy, dataset_name=dataset_name)
    views = {str(getattr(strategy, "train_view", "default")) for strategy in strategies}
    if len(views) != 1:
        details = {strategy.name: getattr(strategy, "train_view", "default") for strategy in strategies}
        raise ValueError(
            "A context-strategy pool cannot mix different canonical train views. "
            f"Got: {details}"
        )
    return next(iter(views))


def validate_context_strategy_training_mode(
    context_strategy: str,
    *,
    dataset_name: str,
    optimal_policy_source: str | None,
) -> None:
    """Reject dataset/SFT supervision for strategies whose view has no target."""
    if optimal_policy_source != "dataset":
        return
    unsupported = [
        strategy.name
        for strategy in _strategies_in_spec(context_strategy, dataset_name=dataset_name)
        if not bool(getattr(strategy, "supports_dataset_targets", True))
    ]
    if unsupported:
        raise ValueError(
            f"Context strategy/strategies {unsupported} use a source-only training "
            "view with no gold completion. Use --optimal_policy_source teacher."
        )


def _valid_prompt_variations(strategy, *, dataset_name: str) -> list[int]:
    """Compatibility wrapper around the strategy-level variation API."""
    return strategy.get_valid_prompt_variations(dataset_name=dataset_name)


def _resolve_prompt_choice(
    *,
    strategy,
    dataset_name: str,
    choice: VariationChoice,
    rng: random.Random,
) -> int:
    """
    Resolve the teacher/final prompt variation.

    None resolves to the default variation (variation 1).
    """
    if choice == "random":
        valid = _valid_prompt_variations(strategy, dataset_name=dataset_name)
        if not valid:
            raise ValueError(
                f"No valid teacher prompt variations for strategy {strategy.name!r} "
                f"on dataset {dataset_name!r}."
            )
        return valid[rng.randrange(len(valid))]

    if choice is None:
        return strategy.resolve_prompt_variation(
            None,
            dataset_name=dataset_name,
        )

    strategy.validate_prompt_variation(
        choice,
        dataset_name=dataset_name,
    )
    return choice


def _valid_stage1_variations(strategy, *, dataset_name: str) -> list[int]:
    if not hasattr(strategy, "num_stage1_variations"):
        return []

    valid = []
    for variation in range(1, strategy.num_stage1_variations + 1):
        try:
            strategy.validate_stage1_variation(
                variation,
                dataset_name=dataset_name,
            )
            valid.append(variation)
        except ValueError:
            pass

    return valid


def _resolve_stage1_choice(
    *,
    strategy,
    dataset_name: str,
    choice: VariationChoice,
    rng: random.Random,
) -> int | None:
    """
    Resolve the stage1 feedback/rationale prompt variation.

    For strategies without a stage1 prompt, None and explicit 1 are accepted as
    no-ops. Nontrivial stage1 choices raise a clear error.
    """
    if not hasattr(strategy, "resolve_stage1_variation"):
        if choice in (None, 1):
            return None
        raise ValueError(
            f"Strategy {strategy.name!r} does not support stage1 variations, "
            f"but got {choice!r}."
        )

    if choice == "random":
        valid = _valid_stage1_variations(strategy, dataset_name=dataset_name)
        if not valid:
            raise ValueError(
                f"No valid stage1 variations for strategy {strategy.name!r} "
                f"on dataset {dataset_name!r}."
            )
        return valid[rng.randrange(len(valid))]

    if choice is None:
        return strategy.resolve_stage1_variation(
            None,
            dataset_name=dataset_name,
        )

    strategy.validate_stage1_variation(
        choice,
        dataset_name=dataset_name,
    )
    return choice


def _select_context_for_row(
    *,
    context_strategy: str,
    dataset_name: str,
    row_index: int,
    seed: int,
) -> ContextRowSelection:
    """Resolve one concrete row selection while preserving historical RNG order.

    RNG consumption remains exactly: strategy draw -> prompt draw -> stage1
    draw. Existing strategy specifications also keep the exact historical RNG
    key. Only strategies that explicitly define ``selection_rng_namespace`` can
    alias that key to a control strategy.
    
    Default selection:
      context_strategy='self_feedback_concrete'
      -> self_feedback_concrete, prompt variation 1, stage1 variation 1

    Random strategy behavior:
      context_strategy='gold_hint|self_feedback_concrete'
      -> random semantic strategy per row, default variation 1

    Random variation behavior:
      context_strategy='self_feedback_concrete(*,*)'
      -> fixed semantic strategy, random prompt + stage1 variation per row
    """
    choices = [part.strip() for part in context_strategy.split("|") if part.strip()]
    if not choices:
        raise ValueError(f"Empty context_strategy specification: {context_strategy!r}")

    rng = _stable_rng(
        "context_strategy_selection",
        seed,
        dataset_name,
        row_index,
        _selection_rng_spec(context_strategy),
    )

    selected_choice = choices[rng.randrange(len(choices))]
    strategy_name, prompt_choice, stage1_choice = _parse_context_choice_spec(selected_choice)
    strategy = get_context_strategy(strategy_name)
    strategy.validate_dataset(dataset_name)

    prompt_variation = _resolve_prompt_choice(
        strategy=strategy,
        dataset_name=dataset_name,
        choice=prompt_choice,
        rng=rng,
    )
    stage1_variation = _resolve_stage1_choice(
        strategy=strategy,
        dataset_name=dataset_name,
        choice=stage1_choice,
        rng=rng,
    )

    return ContextRowSelection(
        strategy_name=strategy_name,
        prompt_variation=prompt_variation,
        stage1_variation=stage1_variation,
        prompt_variation_random=prompt_choice == "random",
        stage1_variation_random=stage1_choice == "random",
    )


def _select_strategy_and_variations_for_row(
    *,
    context_strategy: str,
    dataset_name: str,
    row_index: int,
    seed: int,
) -> tuple[str, int, int | None]:
    """Compatibility API returning the historical three concrete values."""
    selection = _select_context_for_row(
        context_strategy=context_strategy,
        dataset_name=dataset_name,
        row_index=row_index,
        seed=seed,
    )
    return (
        selection.strategy_name,
        selection.prompt_variation,
        selection.stage1_variation,
    )


def _format_train_example_with_selected_context(
    *,
    raw_example,
    row_index: int,
    dataset_name: str,
    seed: int,
    context_strategy: str,
):
    """
    Resolve the row-level strategy/variation selection, then delegate actual
    formatting to ContextualizationManager.

    This keeps responsibilities separated:
      - loader: row-level selection and reproducibility;
      - manager: dataset adapter + strategy invocation + metadata schema;
      - strategy: prompt wording and variation validation.
    """
    selection = _select_context_for_row(
        context_strategy=context_strategy,
        dataset_name=dataset_name,
        row_index=row_index,
        seed=seed,
    )

    manager = ContextualizationManager(
        selection.strategy_name,
        prompt_variation=selection.prompt_variation,
        stage1_variation=selection.stage1_variation,
    )

    return manager.format_train_example(
        dataset_name=dataset_name,
        raw_example=raw_example,
        prompt_variation=selection.prompt_variation,
        stage1_variation=selection.stage1_variation,
        selection_spec=context_strategy,
        base_prompt_variation=selection.prompt_variation,
        row_index=row_index,
        selection_seed=seed,
        prompt_variation_random=selection.prompt_variation_random,
        stage1_variation_random=selection.stage1_variation_random,
    )


def load_contextualized_train_dataset(
    *,
    dataset_name: str,
    seed: int,
    context_strategy: str = "dataset_default",
    optimal_policy_source: str | None = None,
):
    """
    Load training data with optional contextualization strategy.

    dataset_default is intentionally delegated to adapter.load_train(seed)
    to use the adapter-defined dataset-default training path.

    Extended context_strategy syntax:
      strategy
          Use one strategy with its default variations.

      strategy(k)
          Use fixed teacher/final prompt variation k.

      strategy(k,m)
          Use fixed teacher/final prompt variation k and fixed stage1
          feedback/rationale variation m.

      strategy(*)
          Randomly select a valid teacher/final prompt variation per row.

      strategy(*,*)
          Randomly select valid teacher/final and stage1 variations per row.

      strategy_a|strategy_b
          Randomly select one semantic strategy per row.
    """
    validate_context_strategy_training_mode(
        context_strategy,
        dataset_name=dataset_name,
        optimal_policy_source=optimal_policy_source,
    )
    train_view = resolve_context_strategy_train_view(
        context_strategy,
        dataset_name=dataset_name,
    )

    if context_strategy == "dataset_default":
        adapter = get_dataset_adapter(dataset_name)
        return adapter.load_train(seed, train_view=train_view)

    adapter = get_dataset_adapter(dataset_name)
    raw_dataset = adapter.load_raw_train(train_view=train_view)

    formatted = raw_dataset.map(
        lambda ex, idx: _format_train_example_with_selected_context(
            raw_example=ex,
            row_index=idx,
            dataset_name=dataset_name,
            seed=seed,
            context_strategy=context_strategy,
        ),
        with_indices=True,
        remove_columns=raw_dataset.column_names,
        load_from_cache_file=False,
    )

    return formatted.shuffle(
        seed=seed,
        load_from_cache_file=False,
    ), None
