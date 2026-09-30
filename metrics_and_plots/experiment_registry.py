"""Central experiment metadata used by metrics and plotting scripts.

Keep identifiers (folder names) separate from presentation labels.  Analysis
code should group with canonical identifiers and apply aliases only while
rendering tables, titles, and legends.  This prevents a display-name change
from silently changing aggregation behavior.

This module is intentionally dependency-free so other metrics builders can
reuse it without importing matplotlib, pandas, or the training stack.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping


# Historical output-folder name -> canonical method identifier.
LEGACY_METHOD_NAMES: dict[str, str] = {
    "forward": "fwd_student_ema",
    "backward": "bwd_student_ema",
    "forward_offpolicy": "fwd_student_frozen",
    "backward_offpolicy": "bwd_student_frozen",
    "sdft24": "fwd_teacher_frozen",
    "sdft24_onpolicy": "fwd_teacher_ema",
    "forward_current_policy": "fwd_student_sync",
    "backward_current_policy": "bwd_student_sync",
    "label_sft": "ce_dataset_sft",
    "sdft24_ce": "ce_teacher_frozen",
    "bwd_teacher_frozen_offline": "bwd_teacher_frozen",
    "bwd_teacher_ema_online": "bwd_teacher_ema",
    "jsd_student_ema": "jsd_student_ema",
    "bwd_student_ema005": "bwd_student_ema005",
    "fwd_student_ema005": "fwd_student_ema005",
    "bwd_teacher_ema005": "bwd_teacher_ema005",
    "fwd_teacher_ema005": "fwd_teacher_ema005",
    "bwd_student_ema010": "bwd_student_ema010",
    "fwd_student_ema010": "fwd_student_ema010",
    "bwd_teacher_ema010": "bwd_teacher_ema010",
    "fwd_teacher_ema010": "fwd_teacher_ema010",
    "bwd_student_ema025": "bwd_student_ema025",
    "fwd_student_ema025": "fwd_student_ema025",
    "bwd_teacher_ema025": "bwd_teacher_ema025",
    "fwd_teacher_ema025": "fwd_teacher_ema025",
    "cpt": "cpt",
}


DEFAULT_EXPERIMENTS: tuple[str, ...] = (
    "ce_dataset_sft",
    "bwd_student_frozen",
    "fwd_student_frozen",
    "bwd_student_ema",
    "bwd_student_ema005",
    "fwd_student_ema005",
    "bwd_teacher_ema005",
    "fwd_teacher_ema005",
    "bwd_student_ema010",
    "fwd_student_ema010",
    "bwd_teacher_ema010",
    "fwd_teacher_ema010",
    "bwd_student_ema025",
    "fwd_student_ema025",
    "bwd_teacher_ema025",
    "fwd_teacher_ema025",
    "jsd_student_ema",
    "fwd_student_ema",
    "cpt",
)


# Edit these dictionaries to change plot text without changing folder parsing,
# joins, filters, or aggregation keys.  Unknown identifiers display unchanged.
METHOD_DISPLAY_ALIASES: dict[str, str] = {
    "bwd_student_ema025": "rev_student_ema025",
    "bwd_student_ema010": "rev_student_ema010",
    "bwd_student_ema005": "rev_student_ema005",
    "bwd_student_ema": "rev_student_ema",
    "bwd_student_frozen": "rev_student_frozen",
    "bwd_teacher_ema": "rev_teacher_ema",
    "bwd_teacher_ema005": "rev_teacher_ema005",
    "bwd_teacher_ema010": "rev_teacher_ema010",
    "bwd_teacher_ema025": "rev_teacher_ema025",
    "bwd_teacher_frozen": "rev_teacher_frozen",
}


# Canonical dataset identifier -> report-facing label.  Keep this mapping
# presentation-only: dataset discovery, metric column names, joins, filters,
# output directories, and CSV values must continue to use canonical names.
DATASET_DISPLAY_ALIASES: dict[str, str] = {
    "tooluse": "ToolUse",
    "science": "Chemistry L-3",
    "math_contradiction": "Math Contradiction",
    "spatial_contradiction2": "Spatial Contradiction",
}

STRATEGY_DISPLAY_ALIASES: dict[str, str] = {
    "rationalization_v2_student": "rationalization_student",
    "rationalization_v2_teacher": "rationalization_teacher",
    "rationalization_student": "rationalization_v1_student",
    "rationalization_teacher": "rationalization_v1_teacher",
    "goldresp_rewrite_example_v3_var_student": "goldresp_rewrite_ex_var_student",
    "goldresp_rewrite_example_v3_var_teacher": "goldresp_rewrite_ex_var_teacher",
    "goldresp_rewrite_example_v3_student": "goldresp_rewrite_ex_student",
    "goldresp_rewrite_example_v3_teacher": "goldresp_rewrite_ex_teacher",
    "goldresp_rewrite_example_v2_var_student": "goldresp_rewrite_ex_v2_var_student",
    "goldresp_rewrite_example_v2_var_teacher": "goldresp_rewrite_ex_v2_var_teacher",
    "goldresp_rewrite_example_v2_student": "goldresp_rewrite_ex_v2_student",
    "goldresp_rewrite_example_v2_teacher": "goldresp_rewrite_ex_v2_teacher",
    "goldresp_rewrite_example_var_student": "goldresp_rewrite_ex_v1_var_student",
    "goldresp_rewrite_example_var_teacher": "goldresp_rewrite_ex_v1_var_teacher",
    "goldresp_rewrite_example_student": "goldresp_rewrite_ex_v1_student",
    "goldresp_rewrite_example_teacher": "goldresp_rewrite_ex_v1_teacher",
    "gold_hint2": "gold_hint",
    "gold_hint": "gold_hint_v1",
    "goldansw_example_v2": "goldansw_example",
    "goldansw_example": "goldansw_example_v1",
    # Historical spelling: ``sfvar`` encodes the same standalone ``var``
    # metadata token used by the other strategies.  Normalize it before the
    # generic token pass so it is displayed consistently as ``(var)``.
    "sfvar_fb_student": "sf_var_fb_student",
}


# Ordered, display-only token abbreviations applied after the exact aliases
# above.  Exact aliases remain the right place for semantic/version renames;
# these rules only remove repetitive wording from plot legends.  Keeping this
# layer separate guarantees that filesystem discovery, filtering, joins, CSV
# identities, and aggregation keys continue to use canonical strategy names.
STRATEGY_DISPLAY_REPLACEMENTS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"example"), "ex"),
    (re.compile(r"guidance"), "guid"),
    (re.compile(r"(?:^|_)fb_student(?=_|$)"), "@S"),
    (re.compile(r"(?:^|_)fb_teacher(?=_|$)"), "@T"),
    (re.compile(r"_student(?=_|$)"), "@S"),
    (re.compile(r"_teacher(?=_|$)"), "@T"),
)

# ``var`` is experiment metadata rather than the main strategy name.  It is
# removed from its original token position and rendered once as a suffix.
# Token boundaries deliberately use underscores instead of ``\b`` because
# underscores count as word characters in regular expressions.
STRATEGY_VARIATION_TOKEN = re.compile(r"(?:^|_)var(?=_|$)")


def canonical_method_name(raw_name: str) -> str | None:
    """Return the canonical method identifier, or ``None`` if it is unknown."""

    if raw_name in LEGACY_METHOD_NAMES:
        return LEGACY_METHOD_NAMES[raw_name]
    if raw_name in known_canonical_methods():
        return raw_name
    return None


def known_canonical_methods() -> set[str]:
    """Return every centrally registered canonical method identifier."""

    return set(LEGACY_METHOD_NAMES.values()) | set(DEFAULT_EXPERIMENTS)


def known_raw_methods() -> set[str]:
    """Return accepted historical and canonical output-folder identifiers."""

    return set(LEGACY_METHOD_NAMES) | known_canonical_methods()


def candidate_raw_method_names(explicit_methods: Iterable[str] | None = None) -> list[str]:
    """Return parser candidates longest-first to avoid prefix collisions."""

    candidates = known_raw_methods()
    if explicit_methods:
        explicit = set(explicit_methods)
        candidates.update(explicit)
        for legacy, canonical in LEGACY_METHOD_NAMES.items():
            if canonical in explicit:
                candidates.add(legacy)
    return sorted(candidates, key=lambda value: (-len(value), value))


def method_display_name(
    method: object,
    aliases: Mapping[str, str] = METHOD_DISPLAY_ALIASES,
) -> str:
    """Return the configured presentation label for a method identifier."""

    text = str(method or "").strip()
    return aliases.get(text, text)


def dataset_display_name(
    dataset: object,
    aliases: Mapping[str, str] = DATASET_DISPLAY_ALIASES,
) -> str:
    """Return the report-facing label for a canonical dataset identifier."""

    text = str(dataset or "").strip()
    return aliases.get(text, text)


def strategy_display_name(
    strategy: object,
    aliases: Mapping[str, str] = STRATEGY_DISPLAY_ALIASES,
) -> str:
    """Return a concise presentation label for a strategy identifier.

    Exact aliases are resolved first so version-dependent names remain
    unambiguous.  The ordered regex rules then abbreviate only common display
    tokens.  Neither layer changes the identifier stored in metric rows.
    """

    text = str(strategy or "").strip()
    display = aliases.get(text, text)
    is_variation = bool(
        STRATEGY_VARIATION_TOKEN.search(text)
        or STRATEGY_VARIATION_TOKEN.search(display)
    )
    display = STRATEGY_VARIATION_TOKEN.sub("", display).strip("_")
    for pattern, replacement in STRATEGY_DISPLAY_REPLACEMENTS:
        display = pattern.sub(replacement, display)
    if is_variation:
        display = f"{display} (var)" if display else "(var)"
    return display

