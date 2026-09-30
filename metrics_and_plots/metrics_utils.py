"""Reusable, dependency-light helpers for experiment metrics builders.

The functions here are deliberately domain-neutral: safe JSON/number parsing,
weighted aggregation, version parsing, command-line metadata extraction, and
run hyperparameter resolution.  Plot orchestration and single-phase metric
definitions remain in their owning builder.
"""

from __future__ import annotations

import csv
import json
import math
import re
import shlex
import warnings
from collections.abc import Iterable, Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from statistics import mean
from typing import Any


CHECKPOINT_RE = re.compile(r"checkpoint-(\d+)$")
RUN_PREFIX_PATTERN = r"(?:v\d+|lr(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?)"
VERSION_RE = re.compile(
    rf"^(?P<prefix>{RUN_PREFIX_PATTERN})(?:_?s(?P<seed>\d+))?(?:_?(?P<suffix>.*))$",
    re.IGNORECASE,
)
LEARNING_RATE_PREFIX_RE = re.compile(
    r"^lr(?P<value>(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?)$",
    re.IGNORECASE,
)
EPOCH_SUFFIX_RE = re.compile(r"^(?P<base>.+?)_ep(?P<epoch>\d+(?:\.\d+)?)$")


def is_nan(value: object) -> bool:
    """Return whether *value* is a floating-point NaN."""

    return isinstance(value, float) and math.isnan(value)


def try_float(value: object) -> float | None:
    """Parse a finite float, returning ``None`` for empty/invalid values."""

    if value is None or value == "" or is_nan(value):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def avg_last_n(values: Iterable[object], n: int = 3) -> float | None:
    """Return the mean of the last *n* valid numeric values."""

    valid = [parsed for value in values if (parsed := try_float(value)) is not None]
    return float(mean(valid[-n:])) if valid else None


def format_number(value: object, digits: int = 3) -> str:
    """Format a number compactly with at most *digits* decimal places."""

    parsed = try_float(value)
    if parsed is None:
        return ""
    rounded = round(parsed, digits)
    if abs(rounded) < 0.5 * 10 ** (-digits):
        rounded = 0.0
    return f"{rounded:.{digits}f}".rstrip("0").rstrip(".")


def format_learning_rate(value: object) -> str:
    """Format a learning rate consistently for legends and CSVs."""

    parsed = try_float(value)
    if parsed is None:
        return ""
    mantissa, exponent = f"{parsed:.3E}".split("E")
    mantissa = mantissa.rstrip("0").rstrip(".")
    return f"{mantissa}e{int(exponent)}"


def format_epoch(value: object, *, prefix: str = "ep") -> str:
    """Format integral epochs without a trailing decimal (``ep2``)."""

    parsed = try_float(value)
    if parsed is None:
        return ""
    number = str(int(parsed)) if parsed.is_integer() else f"{parsed:g}"
    return f"{prefix}{number}"


def weighted_mean(values: Sequence[object], weights: Sequence[object]) -> float | None:
    """Return a weighted mean over positions with valid values and weights."""

    pairs: list[tuple[float, float]] = []
    for value, weight in zip(values, weights):
        parsed_value = try_float(value)
        parsed_weight = try_float(weight)
        if parsed_value is None or parsed_weight is None or parsed_weight <= 0:
            continue
        pairs.append((parsed_value, parsed_weight))
    denominator = sum(weight for _, weight in pairs)
    if not pairs or denominator <= 0:
        return None
    return sum(value * weight for value, weight in pairs) / denominator


def unweighted_mean(values: Iterable[object]) -> float | None:
    """Return the arithmetic mean of valid numeric values."""

    valid = [parsed for value in values if (parsed := try_float(value)) is not None]
    return float(mean(valid)) if valid else None


def first_nonempty(rows: Iterable[Mapping[str, Any]], key: str) -> Any:
    """Return the first non-empty value stored under *key*."""

    for row in rows:
        value = row.get(key)
        if value not in (None, "", []):
            return value
    return ""


def gm_of_one_plus(values: Iterable[object]) -> float | None:
    """Geometric mean of ``1 + x`` over valid values."""

    valid = [parsed for value in values if (parsed := try_float(value)) is not None]
    if not valid or any(value <= -1.0 for value in valid):
        return None
    return math.exp(sum(math.log1p(value) for value in valid) / len(valid))


def gm_positive_with_floor(values: Iterable[object], epsilon: float = 1e-3) -> float | None:
    """Geometric mean after flooring valid values at *epsilon*."""

    valid = [parsed for value in values if (parsed := try_float(value)) is not None]
    if not valid:
        return None
    floored = [max(epsilon, value) for value in valid]
    return math.exp(sum(math.log(value) for value in floored) / len(floored))


def relative_delta(value: object, baseline: object) -> float | None:
    """Return ``(value - baseline) / baseline`` when the baseline is nonzero."""

    parsed_value = try_float(value)
    parsed_baseline = try_float(baseline)
    if parsed_value is None or parsed_baseline in (None, 0.0):
        return None
    return (parsed_value - parsed_baseline) / parsed_baseline


def accuracy_delta(value: object, baseline: object) -> float | None:
    """Return the signed absolute accuracy change ``value - baseline``.

    Unlike :func:`relative_delta`, this remains in the same units as the
    supplied accuracies.  For accuracies stored in ``[0, 1]``, a result of
    ``0.05`` means a gain of five percentage points.
    """

    parsed_value = try_float(value)
    parsed_baseline = try_float(baseline)
    if parsed_value is None or parsed_baseline is None:
        return None
    return parsed_value - parsed_baseline


def normalized_progress(value: object, baseline: object, expert: object) -> float | None:
    """Return headroom-normalized progress from baseline toward expert."""

    parsed_value = try_float(value)
    parsed_baseline = try_float(baseline)
    parsed_expert = try_float(expert)
    if parsed_value is None or parsed_baseline is None or parsed_expert is None:
        return None
    denominator = parsed_expert - parsed_baseline
    if denominator == 0:
        return None
    return (parsed_value - parsed_baseline) / denominator


def bounded_normalized_progress(
    value: object,
    baseline: object,
    expert: object,
    floor: object = 0.0,
) -> float | None:
    """Return piecewise normalized progress clipped to ``[-1, 1]``.

    Improvements are normalized by the reference headroom
    ``expert - baseline``.  Degradations are normalized by the available
    downward range ``baseline - floor``.  Clipping gives the metric its
    explicit bounded contract even when an observed score exceeds the chosen
    reference or falls below the chosen floor.
    """

    parsed_value = try_float(value)
    parsed_baseline = try_float(baseline)
    parsed_expert = try_float(expert)
    parsed_floor = try_float(floor)
    if None in (parsed_value, parsed_baseline, parsed_expert, parsed_floor):
        return None

    assert parsed_value is not None
    assert parsed_baseline is not None
    assert parsed_expert is not None
    assert parsed_floor is not None

    if parsed_value >= parsed_baseline:
        denominator = parsed_expert - parsed_baseline
        if denominator <= 0:
            return None
        raw = (parsed_value - parsed_baseline) / denominator
    else:
        denominator = parsed_baseline - parsed_floor
        if denominator <= 0:
            return None
        raw = (parsed_value - parsed_baseline) / denominator
    return max(-1.0, min(1.0, raw))


def slugify(value: object) -> str:
    """Return a conservative filesystem-safe label."""

    text = re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower())
    return re.sub(r"_+", "_", text).strip("_") or "metric"


def readable_metric_name(metric: object) -> str:
    """Convert an internal snake/slash metric key into a plot label."""

    return str(metric or "").replace("_", " ")


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    """Write dictionaries to UTF-8 CSV, creating parent directories."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


@lru_cache(maxsize=None)
def load_json(path: Path) -> Any | None:
    """Load JSON once per path; malformed/missing files return ``None``."""

    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError, TypeError):
        return None


def split_version_name(version: str) -> tuple[str, str]:
    """Split a legacy or LR-labelled run name into prefix and suffix."""

    prefix, suffix, _ = split_seeded_version_name(version)
    return prefix, suffix


def split_seeded_version_name(version: str) -> tuple[str, str, int | None]:
    """Split a run name while also extracting an optional ``s<seed>`` token."""

    match = VERSION_RE.match(version)
    if not match:
        return version, "", None
    seed = int(match.group("seed")) if match.group("seed") else None
    return match.group("prefix"), (match.group("suffix") or "").strip("_"), seed


def learning_rate_from_run_prefix(run_prefix: str) -> float | None:
    """Read an explicit ``lr...`` prefix; legacy ``vN`` labels return ``None``."""

    match = LEARNING_RATE_PREFIX_RE.fullmatch(str(run_prefix).strip())
    if not match:
        return None
    value = float(match.group("value"))
    return value if math.isfinite(value) and value > 0 else None


def seedless_version_name(version_prefix: str, version_suffix: str) -> str:
    """Recombine version components after removing a seed segment."""

    return f"{version_prefix}{version_suffix}" if version_suffix else version_prefix


def split_epoch_suffix(name: str) -> tuple[str, float | None]:
    """Remove only a terminal ``_epN`` marker from an experiment folder name."""

    match = EPOCH_SUFFIX_RE.match(str(name).strip())
    if not match:
        return str(name).strip(), None
    return match.group("base"), try_float(match.group("epoch"))


def checkpoint_step(path: Path) -> int:
    """Return a checkpoint's integer step, or ``-1`` for a non-checkpoint."""

    match = CHECKPOINT_RE.match(path.name)
    return int(match.group(1)) if match else -1


def sorted_checkpoints(phase_dir: Path) -> list[Path]:
    """Return checkpoint directories sorted by numeric step."""

    if not phase_dir.is_dir():
        return []
    return sorted(
        (path for path in phase_dir.iterdir() if path.is_dir() and checkpoint_step(path) >= 0),
        key=checkpoint_step,
    )


def read_command_file(phase_dir: Path) -> str | None:
    """Read the first supported training-command metadata file."""

    for name in ("train_command.txt", "command.txt"):
        try:
            return (phase_dir / name).read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
    return None


def normalize_command_text(text: str) -> str:
    """Collapse shell line continuations/newlines before tokenization."""

    return text.replace("\\\n", " ").replace("\n", " ")


def extract_arg_from_command(command: str, arg_name: str) -> str | None:
    """Extract ``--name value`` or ``--name=value`` from a saved command."""

    if not command:
        return None
    normalized = normalize_command_text(command)
    try:
        tokens = shlex.split(normalized)
    except ValueError:
        tokens = normalized.split()
    flag = f"--{arg_name}"
    for index, token in enumerate(tokens):
        if token == flag and index + 1 < len(tokens):
            return tokens[index + 1]
        if token.startswith(f"{flag}="):
            return token.split("=", 1)[1]
    return None


def _consistent_numeric_value(
    values: Iterable[object],
    *,
    description: str,
    location: Path,
) -> float | None:
    parsed = [value for item in values if (value := try_float(item)) is not None]
    if not parsed:
        return None
    distinct = {f"{value:.12g}" for value in parsed}
    if len(distinct) > 1:
        warnings.warn(
            f"Multiple {description} values found under {location}: {parsed}. "
            f"Using {parsed[0]}.",
            stacklevel=2,
        )
    return parsed[0]


def numeric_arg_from_phase_commands(
    version_dir: Path,
    datasets: Iterable[str],
    arg_name: str,
) -> float | None:
    """Resolve a numeric CLI argument across saved per-phase commands."""

    values = []
    for dataset in datasets:
        command = read_command_file(version_dir / dataset)
        if command:
            values.append(extract_arg_from_command(command, arg_name))
    return _consistent_numeric_value(
        values,
        description=arg_name.replace("_", " "),
        location=version_dir,
    )


def experiment_config(version_dir: Path) -> Mapping[str, Any]:
    """Return a run's experiment config as a mapping."""

    data = load_json(version_dir / "experiment_config.json")
    return data if isinstance(data, Mapping) else {}


def numeric_experiment_parameter(
    version_dir: Path,
    parameter: str,
    datasets: Iterable[str],
    *,
    fallback: object = None,
) -> float | None:
    """Resolve a numeric run parameter using authoritative metadata first.

    Priority:
      1. ``experiment_config.json['common_args'][parameter]``;
      2. top-level ``experiment_config.json[parameter]``;
      3. saved per-phase command argument;
      4. explicit fallback (for legacy runs only).
    """

    config = experiment_config(version_dir)
    common_args = config.get("common_args")
    if isinstance(common_args, Mapping):
        value = try_float(common_args.get(parameter))
        if value is not None:
            return value
    value = try_float(config.get(parameter))
    if value is not None:
        return value
    command_value = numeric_arg_from_phase_commands(version_dir, datasets, parameter)
    if command_value is not None:
        return command_value
    return try_float(fallback)
