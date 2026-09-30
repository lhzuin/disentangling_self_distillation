#!/usr/bin/env python3
"""Build the three canonical raw-ish CSVs for single-phase experiments.

This is the deliberately small data-extraction counterpart of the repository's
plotting/aggregation stack.  It scans one single-phase outputs tree and writes
*only* these files into ``--output-dir``:

* ``context_single_phase_all_per_seed.csv``
* ``single_phase_training_dynamics_steps.csv``
* ``single_phase_training_dynamics_checkpoints.csv``

Design principles
-----------------
1. One outputs root, all runs.  There are intentionally no experiment/seed/LR/
   epoch/context filters.
2. Single phase only.  Runs whose authoritative ``experiment_config.json``
   declares more than one phase are ignored.
3. Preserve information instead of summarizing it.  The steps table contains
   one row per deduplicated *actual* training step, not temporal bins.
4. Reuse persisted analysis artefacts when possible.  In particular,
   ``training_eval_curve.json`` is the preferred checkpoint-accuracy source so
   the extractor still works after Trainer checkpoints have been cleaned up.
5. Missing historical fields remain empty.  No interpolation or fabrication is
   performed.
6. Model-specific base accuracies and lm-eval baselines come from
   ``model_registry.py``.  ``--model`` is a fallback/consistency check when a
   run does not record ``initial_model``.

The per-seed table intentionally keeps the familiar structure of
``context_single_phase_all_per_seed.csv`` while simplifying the learning
contract: only ``final_accuracy`` and ``accuracy_delta`` are supported (no
normalized/bounded-normalized learning).  The historical ``learning_metric``
/``learning`` columns are retained for compatibility and can select either of
those two quantities.

LM-eval columns keep the existing delta columns and add final checkpoint values
and raw stderr.  Final values/deltas use percent/percentage-point units, while
stderr keeps lm-eval's native 0--1 scale.  Final values are read directly when
present and otherwise reconstructed from the model-registry baseline + delta.

Training-step metrics are exported with their original training-log key names
where possible (for example ``completions/mean_length``).  The only deliberate
rename is the per-step ``learning_rate`` -> ``step_learning_rate`` to avoid a
collision with the run-level learning rate column.
"""

from __future__ import annotations

import argparse
import ast
import csv
import math
import os
import re
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for candidate in (SCRIPT_DIR, PROJECT_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

try:
    from metrics_utils import (  # type: ignore
        learning_rate_from_run_prefix,
        load_json,
        numeric_experiment_parameter,
        split_seeded_version_name,
        split_epoch_suffix,
        try_float,
    )
    from experiment_registry import (  # type: ignore
        candidate_raw_method_names,
        canonical_method_name,
    )
    from model_registry import (  # type: ignore
        DEFAULT_MODEL_KEY,
        resolve_model_spec,
    )
    from dataset_adapters import get_evaluation_dataset_names  # type: ignore
except ImportError as exc:  # pragma: no cover - repository installation error
    raise SystemExit(
        "Could not import project modules. Put this file in metrics_and_plots/ "
        "and run it from the repository's Python environment."
    ) from exc


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PER_SEED_CSV = "context_single_phase_all_per_seed.csv"
STEPS_CSV = "single_phase_training_dynamics_steps.csv"
CHECKPOINTS_CSV = "single_phase_training_dynamics_checkpoints.csv"

CHECKPOINT_RE = re.compile(r"^checkpoint-(?P<step>\d+)$")
TRAIN_PROGRESS_RE = re.compile(r"(?P<step>\d+)\s*/\s*(?P<total>\d+)")

DEFAULT_WORKERS = min(8, max(1, os.cpu_count() or 1))
PRIOR_TASK_KEYS = (
    "hellaswag",
    "mmlu",
    "truthfulqa_mc2",
    "winogrande",
    "ifeval",
)
LMEVAL_TASK_KEYS = (*PRIOR_TASK_KEYS, "humaneval")

# Known evidence that a dict literal is a genuine per-step Trainer record.
RAW_LOG_STEP_MARKERS = {
    "loss",
    "grad_norm",
    "kl_approx",
    "entropy",
    "teacher_entropy",
    "completions/mean_length",
    "ce_loss",
}

# Top-level fields that describe where a record lives rather than a training
# metric.  They are exported explicitly as coordinates/provenance columns.
STEP_COORDINATE_KEYS = {
    "phase_step_index",
    "reported_step",
    "reported_total_steps",
    "phase_total_steps",
    "phase_fraction",
    "phase_offset",
    "global_step",
    "global_fraction",
    "line_number",
    "source_log",
    "phase",
}

PER_SEED_BASE_FIELDS = [
    "protocol",
    "new_name",
    "context_suffix",
    "version_prefix",
    "version_suffix",
    "run_version",
    "training_entropy_first3_mean",
    "training_entropy_last3_mean",
    "training_entropy_delta",
    "training_response_length_first3_mean",
    "training_response_length_last3_mean",
    "training_response_length_delta",
    "LR",
    "num_train_epochs",
    "learning_metric",
    "learning",
    "lmeval_aggregation",
    "lmeval_component_count",
    "trained_dataset",
    "final_accuracy",
    "accuracy_delta",
]

STEP_BASE_FIELDS = [
    "run_id",
    "experiment",
    "context_suffix",
    "version",
    "version_prefix",
    "version_suffix",
    "seed",
    "trained_dataset",
    "learning_rate",
    "num_train_epochs",
    "step_data_source",
    "reported_step",
    "reported_total_steps",
    "phase_step_index",
    "phase_fraction",
    "global_step",
    "global_fraction",
    "epoch",
    "source_log",
    "line_number",
]

CHECKPOINT_BASE_FIELDS = [
    "run_id",
    "experiment",
    "context_suffix",
    "version",
    "seed",
    "trained_dataset",
    "learning_rate",
    "num_train_epochs",
    "checkpoint_step",
    "training_fraction",
    "is_baseline",
    "accuracy",
    "baseline_accuracy",
    "accuracy_delta_from_baseline",
    "accuracy_delta_from_previous",
    "response_length_main_mean",
    "response_length_main_median",
    "response_length_main_unit",
    "response_char_mean",
    "response_whitespace_tokens_mean",
    "response_hf_tokens_mean",
    "response_length_delta_from_previous",
    "eval_num_responses",
    "checkpoint_dir",
]


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def safe_json(path: Path) -> Mapping[str, Any]:
    value = load_json(path)
    return value if isinstance(value, Mapping) else {}


def safe_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None or value == "":
        return None
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return None


def finite_mean(values: Iterable[Any]) -> float | None:
    parsed = [x for value in values if (x := try_float(value)) is not None]
    return float(statistics.fmean(parsed)) if parsed else None


def relative_delta(value: Any, baseline: Any) -> float | None:
    current = try_float(value)
    base = try_float(baseline)
    if current is None or base in (None, 0.0):
        return None
    return (current - base) / base


def absolute_delta(value: Any, baseline: Any) -> float | None:
    current = try_float(value)
    base = try_float(baseline)
    if current is None or base is None:
        return None
    return current - base


def format_learning_rate(value: Any) -> str:
    parsed = try_float(value)
    if parsed is None:
        return ""
    mantissa, exponent = f"{parsed:.12E}".split("E")
    mantissa = mantissa.rstrip("0").rstrip(".")
    return f"{mantissa}e{int(exponent)}"


def format_number(value: Any, significant_digits: int) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        if value == 0:
            return "0"
        return f"{value:.{significant_digits}g}"
    return value


def write_csv_rows(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    preferred_fields: Sequence[str],
    significant_digits: int,
) -> list[str]:
    all_keys: set[str] = set()
    for row in rows:
        all_keys.update(str(key) for key in row)
    fields = [field for field in preferred_fields if field in all_keys]
    fields.extend(sorted(all_keys - set(fields)))

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: format_number(row.get(field), significant_digits)
                    for field in fields
                }
            )
    return fields


def checkpoint_step_from_path(path: str | Path | None) -> int | None:
    if not path:
        return None
    match = CHECKPOINT_RE.match(Path(str(path)).name)
    return int(match.group("step")) if match else None


# ---------------------------------------------------------------------------
# Run discovery and metadata
# ---------------------------------------------------------------------------


def phase_like_directory(path: Path) -> bool:
    if not path.is_dir() or CHECKPOINT_RE.match(path.name):
        return False
    if (path / "train_command.txt").exists() or (path / "command.txt").exists():
        return True
    try:
        if any(path.glob("*.log")):
            return True
        return any(
            child.is_dir() and CHECKPOINT_RE.match(child.name)
            for child in path.iterdir()
        )
    except OSError:
        return False


def version_parts(version: str) -> tuple[str, str, int | None]:
    return split_seeded_version_name(version)


def infer_phase_from_version_suffix(version_suffix: str) -> str | None:
    suffix = str(version_suffix or "").strip("_")
    if not suffix.endswith("_only"):
        return None
    phase = suffix[: -len("_only")].strip("_")
    return phase or None


def resolve_single_phase(
    version_dir: Path,
    config: Mapping[str, Any],
) -> tuple[str | None, str, list[str]]:
    """Resolve exactly one training phase and reject sequential runs."""
    warnings: list[str] = []
    sequence = config.get("phase_sequence")
    if isinstance(sequence, list):
        cleaned = [str(x).strip() for x in sequence if str(x).strip()]
        if len(cleaned) > 1:
            return None, "experiment_config.sequential", warnings
        if len(cleaned) == 1:
            configured = cleaned[0]
            _, suffix, _ = version_parts(version_dir.name)
            inferred = infer_phase_from_version_suffix(suffix)
            if inferred is not None and inferred != configured:
                warnings.append(
                    f"{version_dir}: config phase {configured!r} disagrees with "
                    f"version suffix {inferred!r}; using config"
                )
            return configured, "experiment_config.phase_sequence", warnings

    phase_dirs = [
        child.name
        for child in version_dir.iterdir()
        if phase_like_directory(child)
    ]
    if len(phase_dirs) == 1:
        return phase_dirs[0], "single_existing_phase_dir", warnings
    if len(phase_dirs) > 1:
        return None, "multiple_phase_dirs", warnings

    _, suffix, _ = version_parts(version_dir.name)
    inferred = infer_phase_from_version_suffix(suffix)
    if inferred is not None:
        return inferred, "version_suffix", warnings
    return None, "unresolved", warnings


def strip_folder_metadata(folder_name: str, phase: str) -> tuple[str, float | None]:
    """Strip only phase/epoch metadata, preserving method/context identity."""
    text, epoch_hint = split_epoch_suffix(folder_name)
    phase_suffix = f"_{phase}_only"
    if text.endswith(phase_suffix):
        text = text[: -len(phase_suffix)].strip("_")
    text, second_epoch_hint = split_epoch_suffix(text)
    if epoch_hint is None:
        epoch_hint = second_epoch_hint
    if text.endswith(phase_suffix):
        text = text[: -len(phase_suffix)].strip("_")
    return text, epoch_hint


def parse_method_and_context(
    folder_name: str,
    phase: str,
    config: Mapping[str, Any],
) -> tuple[str, str, float | None]:
    """Parse registered method prefixes; retain unknown folders losslessly."""
    cleaned, epoch_hint = strip_folder_metadata(folder_name, phase)
    for raw_base in candidate_raw_method_names(None):
        if cleaned == raw_base:
            method = canonical_method_name(raw_base) or raw_base
            strategy = str(config.get("context_strategy") or "").strip()
            return method, strategy or "context", epoch_hint
        prefix = raw_base + "_"
        if cleaned.startswith(prefix):
            suffix = cleaned[len(prefix) :].strip("_")
            if suffix:
                method = canonical_method_name(raw_base) or raw_base
                return method, suffix, epoch_hint

    # Unknown experimental method: never guess a split at the last underscore.
    strategy = str(config.get("context_strategy") or "").strip()
    return cleaned, strategy or "context", epoch_hint


@dataclass(frozen=True)
class RunDescriptor:
    outputs_root: Path
    experiment_folder: str
    version_dir: Path
    run_id: str
    experiment: str
    context_suffix: str
    phase: str
    phase_source: str
    version: str
    version_prefix: str
    version_suffix: str
    seed: int | None
    learning_rate: float | None
    num_train_epochs: float | None
    config: Mapping[str, Any]


def discover_runs(args: argparse.Namespace) -> tuple[list[RunDescriptor], list[str]]:
    root = args.outputs_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Outputs root does not exist: {root}")

    runs: list[RunDescriptor] = []
    warnings: list[str] = []
    for exp_dir in sorted(root.iterdir(), key=lambda p: p.name):
        if not exp_dir.is_dir():
            continue
        for version_dir in sorted(exp_dir.iterdir(), key=lambda p: p.name):
            if not version_dir.is_dir():
                continue
            config = safe_json(version_dir / args.experiment_config_name)
            try:
                has_phase_dir = any(
                    phase_like_directory(child)
                    for child in version_dir.iterdir()
                    if child.is_dir()
                )
            except OSError:
                has_phase_dir = False
            if not config and not has_phase_dir:
                continue

            phase, phase_source, run_warnings = resolve_single_phase(version_dir, config)
            warnings.extend(run_warnings)
            if phase is None:
                continue

            method, context_suffix, epoch_hint = parse_method_and_context(
                exp_dir.name, phase, config
            )
            version_prefix, version_suffix, seed_from_name = version_parts(version_dir.name)
            raw_seed = config.get("seed")
            try:
                seed = int(raw_seed) if raw_seed not in (None, "") else seed_from_name
            except (TypeError, ValueError):
                seed = seed_from_name

            recorded_lr = numeric_experiment_parameter(
                version_dir,
                "learning_rate",
                [phase],
            )
            named_lr = learning_rate_from_run_prefix(version_prefix)
            lr = recorded_lr if recorded_lr is not None else named_lr
            if (
                recorded_lr is not None
                and named_lr is not None
                and not math.isclose(recorded_lr, named_lr, rel_tol=1e-12, abs_tol=0.0)
            ):
                warnings.append(
                    f"{version_dir}: run label encodes learning rate {named_lr:g}, "
                    f"but recorded metadata specifies {recorded_lr:g}; using metadata"
                )
            epochs = numeric_experiment_parameter(
                version_dir,
                "num_train_epochs",
                [phase],
                fallback=epoch_hint,
            )
            run_id = str(version_dir.resolve().relative_to(root))
            runs.append(
                RunDescriptor(
                    outputs_root=root,
                    experiment_folder=exp_dir.name,
                    version_dir=version_dir.resolve(),
                    run_id=run_id,
                    experiment=method,
                    context_suffix=context_suffix,
                    phase=phase,
                    phase_source=phase_source,
                    version=version_dir.name,
                    version_prefix=version_prefix,
                    version_suffix=version_suffix,
                    seed=seed,
                    learning_rate=lr,
                    num_train_epochs=epochs,
                    config=config,
                )
            )

    runs.sort(
        key=lambda r: (
            r.experiment,
            r.context_suffix,
            r.phase,
            r.version,
            str(r.version_dir),
        )
    )
    return runs, warnings


# ---------------------------------------------------------------------------
# Model-aware baselines
# ---------------------------------------------------------------------------


def resolve_run_model_spec(
    run: RunDescriptor,
    cli_model: str | None,
) -> tuple[Any, list[str]]:
    warnings: list[str] = []
    recorded = str(run.config.get("initial_model") or "").strip()
    selected = recorded or (str(cli_model).strip() if cli_model else DEFAULT_MODEL_KEY)
    spec = resolve_model_spec(selected)

    if recorded and cli_model:
        try:
            cli_spec = resolve_model_spec(str(cli_model))
            if getattr(cli_spec, "key", None) != getattr(spec, "key", None):
                warnings.append(
                    f"{run.run_id}: --model resolves to {getattr(cli_spec, 'key', cli_model)!r} "
                    f"but experiment_config initial_model resolves to "
                    f"{getattr(spec, 'key', recorded)!r}; using run metadata"
                )
        except Exception as exc:  # defensive only
            warnings.append(f"{run.run_id}: could not validate --model: {exc!r}")
    return spec, warnings


# ---------------------------------------------------------------------------
# Per-step training records
# ---------------------------------------------------------------------------


def flatten_numeric_dict(obj: Mapping[str, Any], prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in obj.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            out.update(flatten_numeric_dict(value, full_key))
            continue
        parsed = try_float(value)
        if parsed is not None:
            out[full_key] = float(parsed)
    return out


def record_numeric_metrics(record: Mapping[str, Any]) -> dict[str, float]:
    numeric: dict[str, float] = {}
    nested = record.get("numeric_metrics")
    if isinstance(nested, Mapping):
        for key, value in nested.items():
            parsed = try_float(value)
            if parsed is not None:
                numeric[str(key)] = float(parsed)

    # Older/alternate analyzers expose numeric fields at top level.
    for key, value in record.items():
        if key in STEP_COORDINATE_KEYS or key == "numeric_metrics" or key == "metrics":
            continue
        parsed = try_float(value)
        if parsed is not None:
            numeric.setdefault(str(key), float(parsed))

    if "entropy_delta_teacher_minus_student" not in numeric:
        student = numeric.get("entropy")
        teacher = numeric.get("teacher_entropy")
        if student is not None and teacher is not None:
            numeric["entropy_delta_teacher_minus_student"] = teacher - student
    return numeric


def extract_first_dict_literal(line: str) -> Mapping[str, Any] | None:
    start = line.find("{")
    end = line.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        value = ast.literal_eval(line[start : end + 1])
    except Exception:
        return None
    return value if isinstance(value, Mapping) else None


def raw_log_records(run: RunDescriptor, log_glob: str) -> list[dict[str, Any]]:
    """Conservative raw-log fallback compatible with the current analyzers."""
    phase_dir = run.version_dir / run.phase
    if not phase_dir.is_dir():
        return []

    rows: list[dict[str, Any]] = []
    for log_path in sorted(phase_dir.glob(log_glob), key=lambda p: p.name):
        last_step: int | None = None
        last_total: int | None = None
        try:
            handle = log_path.open("r", encoding="utf-8", errors="replace")
        except OSError:
            continue
        with handle:
            for line_number, line in enumerate(handle, start=1):
                matches = TRAIN_PROGRESS_RE.findall(line)
                if matches:
                    step_s, total_s = matches[-1]
                    try:
                        last_step, last_total = int(step_s), int(total_s)
                    except ValueError:
                        pass
                if "{" not in line or "}" not in line:
                    continue
                obj = extract_first_dict_literal(line)
                if obj is None:
                    continue
                numeric = flatten_numeric_dict(obj)
                if not numeric or not RAW_LOG_STEP_MARKERS.intersection(numeric):
                    continue
                row: dict[str, Any] = {
                    "phase": run.phase,
                    "source_log": str(log_path),
                    "line_number": line_number,
                    "reported_step": last_step,
                    "reported_total_steps": last_total,
                    "numeric_metrics": numeric,
                }
                for key, value in numeric.items():
                    row[key] = value
                rows.append(row)

    totals = [
        int(row["reported_total_steps"])
        for row in rows
        if isinstance(row.get("reported_total_steps"), int)
        and int(row["reported_total_steps"]) > 0
    ]
    valid_steps = [
        int(row["reported_step"])
        for row in rows
        if isinstance(row.get("reported_step"), int)
        and int(row["reported_step"]) > 0
    ]
    total = max(totals) if totals else (max(valid_steps) if valid_steps else None)
    for index, row in enumerate(rows, start=1):
        step = row.get("reported_step")
        if not isinstance(step, int) or step <= 0:
            step = index
            row["reported_step"] = step
        if total:
            row["phase_total_steps"] = total
            row["phase_fraction"] = min(1.0, max(0.0, step / total))
    return rows


def load_training_records(
    run: RunDescriptor,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], str, list[str]]:
    warnings: list[str] = []
    path = run.version_dir / args.training_log_stats_name
    data = safe_json(path)
    if data:
        rows = data.get("combined_training_log_series")
        if isinstance(rows, list):
            selected = [
                dict(row)
                for row in rows
                if isinstance(row, Mapping)
                and str(row.get("phase") or "") == run.phase
            ]
            return selected, "training_log_stats_json", warnings
        warnings.append(f"{path}: missing combined_training_log_series")

    if args.raw_log_fallback:
        rows = raw_log_records(run, args.log_glob)
        if rows:
            return rows, "raw_log_fallback", warnings
    return [], "missing", warnings


def genuine_step_record(record: Mapping[str, Any]) -> bool:
    numeric = record_numeric_metrics(record)
    if not numeric:
        return False
    step = record.get("reported_step")
    return (
        bool(RAW_LOG_STEP_MARKERS.intersection(numeric))
        and isinstance(step, int)
        and step > 0
    )


def normalize_and_deduplicate_records(
    records: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    candidates = [dict(row) for row in records if genuine_step_record(row)]
    if not candidates:
        return [], 0

    totals = [
        value
        for row in candidates
        if (value := try_float(row.get("phase_total_steps") or row.get("reported_total_steps")))
        is not None
    ]
    max_step = max(int(row["reported_step"]) for row in candidates)
    total = max(int(max(totals)) if totals else max_step, max_step, 1)

    by_step: dict[int, dict[str, Any]] = {}
    duplicates = 0
    for row in candidates:
        step = int(row["reported_step"])
        numeric = record_numeric_metrics(row)
        row["_numeric_metrics"] = numeric
        fraction = try_float(row.get("phase_fraction"))
        row["_fraction"] = min(
            1.0,
            max(0.0, float(fraction if fraction is not None else step / total)),
        )
        row["_metric_count"] = len(numeric)
        previous = by_step.get(step)
        if previous is None:
            by_step[step] = row
            continue
        duplicates += 1
        # Resumed logs can repeat steps. Prefer the richer record; on ties,
        # prefer the later encountered record (usually the resumed/latest one).
        if row["_metric_count"] >= previous.get("_metric_count", 0):
            by_step[step] = row

    return [by_step[step] for step in sorted(by_step)], duplicates


def metric_output_column(metric_key: str) -> str | None:
    if metric_key == "epoch":
        return None  # already an explicit coordinate column
    if metric_key == "learning_rate":
        return "step_learning_rate"
    return metric_key


def build_step_rows(
    run: RunDescriptor,
    records: Sequence[Mapping[str, Any]],
    step_source: str,
) -> tuple[list[dict[str, Any]], set[str]]:
    rows: list[dict[str, Any]] = []
    metric_columns: set[str] = set()
    for record in records:
        numeric = record.get("_numeric_metrics", {})
        if not isinstance(numeric, Mapping):
            numeric = record_numeric_metrics(record)
        epoch = try_float(record.get("epoch"))
        if epoch is None:
            epoch = try_float(numeric.get("epoch"))
        row: dict[str, Any] = {
            "run_id": run.run_id,
            "experiment": run.experiment,
            "context_suffix": run.context_suffix,
            "version": run.version,
            "version_prefix": run.version_prefix,
            "version_suffix": run.version_suffix,
            "seed": run.seed,
            "trained_dataset": run.phase,
            "learning_rate": run.learning_rate,
            "num_train_epochs": run.num_train_epochs,
            "step_data_source": step_source,
            "reported_step": record.get("reported_step"),
            "reported_total_steps": record.get("reported_total_steps")
            or record.get("phase_total_steps"),
            "phase_step_index": record.get("phase_step_index"),
            "phase_fraction": record.get("phase_fraction")
            if record.get("phase_fraction") is not None
            else record.get("_fraction"),
            "global_step": record.get("global_step"),
            "global_fraction": record.get("global_fraction"),
            "epoch": epoch,
            "source_log": str(record.get("source_log") or ""),
            "line_number": record.get("line_number"),
        }
        for metric_key, value in numeric.items():
            column = metric_output_column(str(metric_key))
            if column is None:
                continue
            row[column] = value
            metric_columns.add(column)
        rows.append(row)
    return rows, metric_columns


def endpoint_summary(
    records: Sequence[Mapping[str, Any]],
    metric_key: str,
    window: int = 3,
) -> tuple[float | None, float | None, float | None]:
    values: list[float] = []
    for record in records:
        numeric = record.get("_numeric_metrics")
        if not isinstance(numeric, Mapping):
            numeric = record_numeric_metrics(record)
        value = try_float(numeric.get(metric_key))
        if value is not None:
            values.append(value)
    if not values:
        return None, None, None
    first = float(statistics.fmean(values[:window]))
    last = float(statistics.fmean(values[-window:]))
    return first, last, last - first


# ---------------------------------------------------------------------------
# Checkpoint accuracy and response length
# ---------------------------------------------------------------------------


def load_accuracy_points(
    run: RunDescriptor,
    args: argparse.Namespace,
    model_spec: Any,
) -> tuple[dict[int, dict[str, Any]], bool]:
    """Load target-task checkpoint accuracy, preferring persisted curve JSON."""
    points: dict[int, dict[str, Any]] = {}
    path = run.version_dir / args.training_eval_curve_name
    curve = safe_json(path)
    if curve:
        baseline_obj = curve.get("baseline", {})
        baseline = (
            baseline_obj.get(run.phase)
            if isinstance(baseline_obj, Mapping)
            else None
        )
        if isinstance(baseline, Mapping):
            value = try_float(baseline.get("accuracy"))
            if value is not None:
                points[0] = {
                    "checkpoint_step": 0,
                    "accuracy": value,
                    "is_baseline": True,
                    "checkpoint_dir": "",
                }

        phases = curve.get("phases", {})
        phase_rows = phases.get(run.phase, []) if isinstance(phases, Mapping) else []
        if isinstance(phase_rows, list):
            for item in phase_rows:
                if not isinstance(item, Mapping):
                    continue
                step = try_float(item.get("phase_step") or item.get("global_step"))
                value = try_float(item.get(f"{run.phase}_eval_accuracy"))
                if step is None or value is None:
                    continue
                step_i = int(step)
                points[step_i] = {
                    "checkpoint_step": step_i,
                    "accuracy": value,
                    "is_baseline": False,
                    "checkpoint_dir": str(item.get("checkpoint_dir") or ""),
                }

        # Combined is the most compact persistent fallback and survives cleanup.
        if not any(step > 0 for step in points):
            combined = curve.get("combined", {})
            rows = combined.get(run.phase, []) if isinstance(combined, Mapping) else []
            if isinstance(rows, list):
                for item in rows:
                    if not isinstance(item, (list, tuple)) or len(item) < 2:
                        continue
                    step = try_float(item[0])
                    value = try_float(item[1])
                    if step is None or value is None:
                        continue
                    step_i = int(step)
                    points[step_i] = {
                        "checkpoint_step": step_i,
                        "accuracy": value,
                        "is_baseline": step_i == 0,
                        "checkpoint_dir": "",
                    }
        if points:
            return points, True

    # Checkpoints may still exist on non-cleaned runs.
    phase_dir = run.version_dir / run.phase
    if phase_dir.is_dir():
        checkpoints = sorted(
            (p for p in phase_dir.glob("checkpoint-*") if p.is_dir()),
            key=lambda p: checkpoint_step_from_path(p) or -1,
        )
        for checkpoint in checkpoints:
            step = checkpoint_step_from_path(checkpoint)
            if step is None:
                continue
            result = safe_json(checkpoint / f"eval_{run.phase}_results.json")
            accuracy = try_float(result.get("accuracy"))
            if accuracy is not None:
                points[step] = {
                    "checkpoint_step": step,
                    "accuracy": accuracy,
                    "is_baseline": False,
                    "checkpoint_dir": str(checkpoint),
                }

    # Add the registry baseline when the persisted curve did not contain step 0.
    baseline = try_float(getattr(model_spec, "dataset_base_accuracy", {}).get(run.phase))
    if baseline is not None and 0 not in points:
        points[0] = {
            "checkpoint_step": 0,
            "accuracy": baseline,
            "is_baseline": True,
            "checkpoint_dir": "",
        }
    return points, False


def extract_length_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    main_field = str(
        summary.get("length_field_used_for_main_mean")
        or summary.get("length_field")
        or ""
    )
    main_mean = try_float(summary.get("main_mean_length"))
    main_median = None
    if main_field and isinstance(summary.get(main_field), Mapping):
        main_median = try_float(summary[main_field].get("median"))

    def mean_for(field: str) -> float | None:
        obj = summary.get(field)
        return try_float(obj.get("mean")) if isinstance(obj, Mapping) else None

    return {
        "response_length_main_mean": main_mean,
        "response_length_main_median": main_median,
        "response_length_main_unit": main_field,
        "response_char_mean": mean_for("char_length"),
        "response_whitespace_tokens_mean": mean_for("whitespace_token_length"),
        "response_hf_tokens_mean": mean_for("hf_token_length"),
        "eval_num_responses": try_float(
            summary.get("num_responses_with_text") or summary.get("num_records")
        ),
    }


def load_length_points(
    run: RunDescriptor,
    args: argparse.Namespace,
) -> tuple[dict[int, dict[str, Any]], bool]:
    points: dict[int, dict[str, Any]] = {}
    data = safe_json(run.version_dir / args.response_length_name)
    if not data:
        return points, False

    phases = data.get("phases", {})
    phase_obj = phases.get(run.phase) if isinstance(phases, Mapping) else None
    checkpoints = phase_obj.get("checkpoints", []) if isinstance(phase_obj, Mapping) else []
    if isinstance(checkpoints, list):
        for checkpoint in checkpoints:
            if not isinstance(checkpoint, Mapping):
                continue
            step = try_float(checkpoint.get("phase_step") or checkpoint.get("global_step"))
            eval_lengths = checkpoint.get("eval_response_lengths")
            summary = eval_lengths.get(run.phase) if isinstance(eval_lengths, Mapping) else None
            if step is None or not isinstance(summary, Mapping):
                continue
            step_i = int(step)
            points[step_i] = {
                "checkpoint_step": step_i,
                "checkpoint_dir": str(checkpoint.get("checkpoint_dir") or ""),
                **extract_length_summary(summary),
            }

    if not points:
        combined = data.get("combined_checkpoint_series", {})
        rows = combined.get(run.phase, []) if isinstance(combined, Mapping) else []
        if isinstance(rows, list):
            for item in rows:
                if not isinstance(item, Mapping):
                    continue
                step = try_float(item.get("phase_step") or item.get("global_step"))
                if step is None:
                    continue
                step_i = int(step)
                points[step_i] = {
                    "checkpoint_step": step_i,
                    "checkpoint_dir": str(item.get("checkpoint_dir") or ""),
                    "response_length_main_mean": try_float(item.get("main_mean_length")),
                    "response_length_main_median": None,
                    "response_length_main_unit": str(item.get("length_field") or ""),
                    "response_char_mean": None,
                    "response_whitespace_tokens_mean": None,
                    "response_hf_tokens_mean": None,
                    "eval_num_responses": try_float(
                        item.get("num_responses_with_text") or item.get("num_records")
                    ),
                }
    return points, True


def training_total_steps(records: Sequence[Mapping[str, Any]]) -> int | None:
    totals = [
        value
        for row in records
        if (value := try_float(row.get("phase_total_steps") or row.get("reported_total_steps")))
        is not None
    ]
    if totals:
        return int(max(totals))
    steps = [
        int(row["reported_step"])
        for row in records
        if isinstance(row.get("reported_step"), int)
    ]
    return max(steps) if steps else None


def merge_checkpoint_rows(
    run: RunDescriptor,
    accuracy_points: Mapping[int, Mapping[str, Any]],
    length_points: Mapping[int, Mapping[str, Any]],
    total_steps: int | None,
) -> list[dict[str, Any]]:
    steps = sorted(set(accuracy_points) | set(length_points))
    if not steps:
        return []
    nonzero = [step for step in steps if step > 0]
    denominator = total_steps or (max(nonzero) if nonzero else 1)
    baseline_accuracy = try_float(accuracy_points.get(0, {}).get("accuracy"))

    rows: list[dict[str, Any]] = []
    previous_accuracy: float | None = None
    previous_length: float | None = None
    for step in steps:
        merged = dict(accuracy_points.get(step, {}))
        merged.update(
            {
                key: value
                for key, value in length_points.get(step, {}).items()
                if value not in (None, "")
            }
        )
        accuracy = try_float(merged.get("accuracy"))
        response_length = try_float(merged.get("response_whitespace_tokens_mean"))
        if response_length is None:
            response_length = try_float(merged.get("response_length_main_mean"))

        row = {
            "run_id": run.run_id,
            "experiment": run.experiment,
            "context_suffix": run.context_suffix,
            "version": run.version,
            "seed": run.seed,
            "trained_dataset": run.phase,
            "learning_rate": run.learning_rate,
            "num_train_epochs": run.num_train_epochs,
            "checkpoint_step": step,
            "training_fraction": 0.0
            if step == 0
            else min(1.0, max(0.0, step / max(1, denominator))),
            "is_baseline": bool(merged.get("is_baseline") or step == 0),
            "accuracy": accuracy,
            "baseline_accuracy": baseline_accuracy,
            "accuracy_delta_from_baseline": (
                accuracy - baseline_accuracy
                if accuracy is not None and baseline_accuracy is not None
                else None
            ),
            "accuracy_delta_from_previous": (
                accuracy - previous_accuracy
                if accuracy is not None and previous_accuracy is not None
                else None
            ),
            "response_length_main_mean": try_float(merged.get("response_length_main_mean")),
            "response_length_main_median": try_float(merged.get("response_length_main_median")),
            "response_length_main_unit": str(merged.get("response_length_main_unit") or ""),
            "response_char_mean": try_float(merged.get("response_char_mean")),
            "response_whitespace_tokens_mean": try_float(
                merged.get("response_whitespace_tokens_mean")
            ),
            "response_hf_tokens_mean": try_float(merged.get("response_hf_tokens_mean")),
            "response_length_delta_from_previous": (
                response_length - previous_length
                if response_length is not None and previous_length is not None
                else None
            ),
            "eval_num_responses": try_float(merged.get("eval_num_responses")),
            "checkpoint_dir": str(merged.get("checkpoint_dir") or ""),
        }
        rows.append(row)
        if accuracy is not None:
            previous_accuracy = accuracy
        if response_length is not None:
            previous_length = response_length
    return rows


# ---------------------------------------------------------------------------
# Final per-dataset accuracies
# ---------------------------------------------------------------------------


def accuracy_series_from_curve(
    run: RunDescriptor,
    dataset: str,
    args: argparse.Namespace,
) -> list[tuple[int, float]]:
    curve = safe_json(run.version_dir / args.training_eval_curve_name)
    if curve:
        combined = curve.get("combined", {})
        rows = combined.get(dataset, []) if isinstance(combined, Mapping) else []
        points: list[tuple[int, float]] = []
        if isinstance(rows, list):
            for item in rows:
                if not isinstance(item, (list, tuple)) or len(item) < 2:
                    continue
                step = try_float(item[0])
                value = try_float(item[1])
                if step is not None and value is not None and int(step) > 0:
                    points.append((int(step), value))
        if points:
            return sorted(points)

    phase_dir = run.version_dir / run.phase
    if not phase_dir.is_dir():
        return []
    points = []
    for checkpoint in sorted(
        (p for p in phase_dir.glob("checkpoint-*") if p.is_dir()),
        key=lambda p: checkpoint_step_from_path(p) or -1,
    ):
        step = checkpoint_step_from_path(checkpoint)
        if step is None:
            continue
        result = safe_json(checkpoint / f"eval_{dataset}_results.json")
        value = try_float(result.get("accuracy"))
        if value is not None:
            points.append((step, value))
    return points


def final_accuracy_for_dataset(
    run: RunDescriptor,
    dataset: str,
    args: argparse.Namespace,
) -> float | None:
    points = accuracy_series_from_curve(run, dataset, args)
    return finite_mean(value for _, value in points[-3:]) if points else None


# ---------------------------------------------------------------------------
# LM-eval
# ---------------------------------------------------------------------------


def find_lmeval_summary_path(run: RunDescriptor, name: str) -> Path | None:
    direct = run.version_dir / name
    if direct.exists():
        return direct
    matches = list(run.version_dir.rglob(name))
    if not matches:
        return None

    def key(path: Path) -> tuple[int, float]:
        step = -1
        for parent in path.parents:
            parsed = checkpoint_step_from_path(parent)
            if parsed is not None:
                step = parsed
                break
        try:
            mtime = path.stat().st_mtime
        except OSError:
            mtime = 0.0
        return step, mtime

    return sorted(matches, key=key)[-1]


def _percent_value(block: Mapping[str, Any]) -> float | None:
    value = try_float(block.get("value_percent"))
    if value is not None:
        return value
    raw = try_float(block.get("value"))
    return raw * 100.0 if raw is not None else None


def _raw_stderr(block: Mapping[str, Any]) -> float | None:
    raw = try_float(block.get("stderr"))
    if raw is not None:
        return raw
    percent = try_float(block.get("stderr_percent"))
    return percent / 100.0 if percent is not None else None


def load_lmeval_features(
    run: RunDescriptor,
    model_spec: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    path = find_lmeval_summary_path(run, args.lm_eval_summary_name)
    out: dict[str, Any] = {
        "lmeval": None,
        "lmeval_final": None,
        "lmeval_aggregation": "mean_delta",
        "lmeval_component_count": 0,
        "lm_eval_summary_path": str(path) if path is not None else "",
    }
    for task in LMEVAL_TASK_KEYS:
        out[f"lmeval_final_{task}"] = None
        out[f"lmeval_delta_{task}"] = None
        out[f"lmeval_stderr_{task}"] = None
    if path is None:
        return out

    summary = safe_json(path)
    metrics = summary.get("metrics", {}) if isinstance(summary, Mapping) else {}
    if not isinstance(metrics, Mapping):
        return out
    registry_baselines = dict(getattr(model_spec, "lm_eval_baselines_percent", {}) or {})

    finals: dict[str, float | None] = {}
    deltas: dict[str, float | None] = {}
    for task in LMEVAL_TASK_KEYS:
        block = metrics.get(task, {})
        if not isinstance(block, Mapping):
            block = {}
        baseline = try_float(registry_baselines.get(task))
        if baseline is None:
            baseline = try_float(block.get("baseline_percent"))

        final = _percent_value(block)
        summary_delta = try_float(block.get("delta_percent"))
        if final is None and baseline is not None and summary_delta is not None:
            final = baseline + summary_delta
        delta = (
            final - baseline
            if final is not None and baseline is not None
            else summary_delta
        )
        stderr = _raw_stderr(block)
        finals[task] = final
        deltas[task] = delta
        out[f"lmeval_final_{task}"] = final
        out[f"lmeval_delta_{task}"] = delta
        out[f"lmeval_stderr_{task}"] = stderr

    prior_block = metrics.get("prior_task_avg", {})
    if not isinstance(prior_block, Mapping):
        prior_block = {}
    prior_final = _percent_value(prior_block)
    valid_prior_finals = [finals[key] for key in PRIOR_TASK_KEYS if finals[key] is not None]
    if prior_final is None and valid_prior_finals:
        prior_final = float(statistics.fmean(valid_prior_finals))

    prior_baseline = try_float(registry_baselines.get("prior_task_avg"))
    if prior_baseline is None:
        prior_baseline = try_float(prior_block.get("baseline_percent"))
    if prior_baseline is None:
        component_baselines = [
            try_float(registry_baselines.get(key)) for key in PRIOR_TASK_KEYS
        ]
        component_baselines = [x for x in component_baselines if x is not None]
        if component_baselines:
            prior_baseline = float(statistics.fmean(component_baselines))

    prior_summary_delta = try_float(prior_block.get("delta_percent"))
    prior_delta = (
        prior_final - prior_baseline
        if prior_final is not None and prior_baseline is not None
        else prior_summary_delta
    )
    if prior_delta is None:
        valid_deltas = [deltas[key] for key in PRIOR_TASK_KEYS if deltas[key] is not None]
        if valid_deltas:
            prior_delta = float(statistics.fmean(valid_deltas))

    out["lmeval_final"] = prior_final
    out["lmeval"] = prior_delta
    out["lmeval_component_count"] = sum(
        1 for key in PRIOR_TASK_KEYS if finals[key] is not None or deltas[key] is not None
    )
    return out


# ---------------------------------------------------------------------------
# Per-seed table
# ---------------------------------------------------------------------------


def build_per_seed_row(
    run: RunDescriptor,
    model_spec: Any,
    records: Sequence[Mapping[str, Any]],
    datasets: Sequence[str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    dataset_baselines = dict(getattr(model_spec, "dataset_base_accuracy", {}) or {})
    final_by_dataset = {
        dataset: final_accuracy_for_dataset(run, dataset, args)
        for dataset in datasets
    }
    delta_by_dataset = {
        dataset: absolute_delta(final_by_dataset[dataset], dataset_baselines.get(dataset))
        for dataset in datasets
    }

    target_final = final_by_dataset.get(run.phase)
    target_delta = delta_by_dataset.get(run.phase)
    learning = target_final if args.learning_metric == "final_accuracy" else target_delta

    ent_first, ent_last, ent_delta = endpoint_summary(records, "entropy")
    len_first, len_last, len_delta = endpoint_summary(records, "completions/mean_length")

    row: dict[str, Any] = {
        "protocol": "single",
        "new_name": run.experiment,
        "context_suffix": run.context_suffix,
        "version_prefix": run.version_prefix,
        "version_suffix": run.version_suffix,
        "run_version": run.version,
        "training_entropy_first3_mean": ent_first,
        "training_entropy_last3_mean": ent_last,
        "training_entropy_delta": ent_delta,
        "training_response_length_first3_mean": len_first,
        "training_response_length_last3_mean": len_last,
        "training_response_length_delta": len_delta,
        "LR": format_learning_rate(run.learning_rate),
        "num_train_epochs": run.num_train_epochs,
        "learning_metric": args.learning_metric,
        "learning": learning,
        "lmeval_aggregation": "mean_delta",
        "lmeval_component_count": 0,
        "trained_dataset": run.phase,
        "final_accuracy": target_final,
        "accuracy_delta": target_delta,
    }
    for dataset in datasets:
        row[f"final_{dataset}_accuracy"] = final_by_dataset.get(dataset)
        row[f"accuracy_delta_{dataset}"] = delta_by_dataset.get(dataset)
        row[f"relative_delta_{dataset}"] = relative_delta(
            final_by_dataset.get(dataset), dataset_baselines.get(dataset)
        )

    row.update(load_lmeval_features(run, model_spec, args))
    if args.include_debug_columns:
        row.update(
            {
                "num_runs": 1.0,
                "version_dirs": str(run.version_dir),
                # load_lmeval_features already owns lm_eval_summary_path
            }
        )
    else:
        row.pop("lm_eval_summary_path", None)
    return row


def per_seed_fieldnames(
    datasets: Sequence[str],
    include_debug_columns: bool,
) -> list[str]:
    fields = list(PER_SEED_BASE_FIELDS)
    for dataset in datasets:
        fields.extend(
            [
                f"final_{dataset}_accuracy",
                f"accuracy_delta_{dataset}",
                f"relative_delta_{dataset}",
            ]
        )
    fields.extend(["lmeval", "lmeval_final"])
    for task in LMEVAL_TASK_KEYS:
        fields.extend(
            [
                f"lmeval_final_{task}",
                f"lmeval_delta_{task}",
                f"lmeval_stderr_{task}",
            ]
        )
    if include_debug_columns:
        fields.extend(["num_runs", "version_dirs", "lm_eval_summary_path"])
    return fields


# ---------------------------------------------------------------------------
# Per-run processing
# ---------------------------------------------------------------------------


def process_run(
    run: RunDescriptor,
    datasets: Sequence[str],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], set[str], list[str]]:
    warnings: list[str] = []
    model_spec, model_warnings = resolve_run_model_spec(run, args.model_name_or_path)
    warnings.extend(model_warnings)

    raw_records, step_source, load_warnings = load_training_records(run, args)
    warnings.extend(load_warnings)
    records, duplicates = normalize_and_deduplicate_records(raw_records)
    if duplicates:
        warnings.append(f"{run.run_id}: deduplicated {duplicates} repeated training steps")

    step_rows, metric_columns = build_step_rows(run, records, step_source)
    accuracy_points, _ = load_accuracy_points(run, args, model_spec)
    length_points, _ = load_length_points(run, args)
    checkpoint_rows = merge_checkpoint_rows(
        run,
        accuracy_points,
        length_points,
        training_total_steps(records),
    )
    per_seed_row = build_per_seed_row(run, model_spec, records, datasets, args)

    if not records:
        warnings.append(f"{run.run_id}: no usable per-step training records")
    if not checkpoint_rows:
        warnings.append(f"{run.run_id}: no checkpoint dynamics found")
    return per_seed_row, step_rows, checkpoint_rows, metric_columns, warnings


# ---------------------------------------------------------------------------
# CLI and orchestration
# ---------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build raw-ish single-phase per-seed, per-step, and per-checkpoint CSVs. "
            "All discovered single-phase runs are always included."
        )
    )
    parser.add_argument(
        "--outputs-root",
        required=True,
        type=Path,
        help="Root containing <experiment>/<version>/... single-phase runs.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Destination directory for the three CSV files.",
    )
    parser.add_argument(
        "--model",
        "--model-name-or-path",
        dest="model_name_or_path",
        default=None,
        help=(
            "Fallback/consistency-check base model. Per-run experiment_config.initial_model "
            "takes precedence. Passing this is recommended when one outputs root contains "
            "runs from one model."
        ),
    )
    parser.add_argument(
        "--learning-metric",
        choices=("final_accuracy", "accuracy_delta"),
        default="accuracy_delta",
        help=(
            "Compatibility value stored in learning_metric/learning. Both final_accuracy "
            "and accuracy_delta are always written as dedicated columns. Default: accuracy_delta."
        ),
    )
    parser.add_argument(
        "--include-debug-columns",
        action="store_true",
        help="Add num_runs, version_dirs, and lm_eval_summary_path to the per-seed CSV.",
    )

    parser.add_argument("--experiment-config-name", default="experiment_config.json")
    parser.add_argument("--training-log-stats-name", default="training_log_stats.json")
    parser.add_argument("--training-eval-curve-name", default="training_eval_curve.json")
    parser.add_argument("--response-length-name", default="response_length_evolution.json")
    parser.add_argument("--lm-eval-summary-name", default="lm_eval_summary.json")
    parser.add_argument("--log-glob", default="*.log")
    parser.add_argument(
        "--raw-log-fallback",
        action="store_true",
        default=True,
        help="Parse phase .log files when training_log_stats.json is absent. Default: true.",
    )
    parser.add_argument(
        "--no-raw-log-fallback",
        dest="raw_log_fallback",
        action="store_false",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Parallel run readers. Default: {DEFAULT_WORKERS}.",
    )
    parser.add_argument(
        "--float-significant-digits",
        type=int,
        default=12,
        help="Significant digits used when serializing floats. Default: 12.",
    )
    parser.add_argument("--fail-on-error", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    if args.float_significant_digits <= 0:
        raise ValueError("--float-significant-digits must be positive")

    runs, warnings = discover_runs(args)
    datasets = list(get_evaluation_dataset_names())
    for phase in sorted({run.phase for run in runs}):
        if phase not in datasets:
            datasets.append(phase)
    if not runs:
        raise RuntimeError(
            f"No single-phase runs discovered under {args.outputs_root.resolve()}"
        )

    per_seed_rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    step_metric_columns: set[str] = set()
    errors: list[str] = []

    def consume(
        run: RunDescriptor,
        result: tuple[
            dict[str, Any],
            list[dict[str, Any]],
            list[dict[str, Any]],
            set[str],
            list[str],
        ],
    ) -> None:
        per_seed, steps, checkpoints, metric_columns, run_warnings = result
        per_seed_rows.append(per_seed)
        step_rows.extend(steps)
        checkpoint_rows.extend(checkpoints)
        step_metric_columns.update(metric_columns)
        warnings.extend(run_warnings)

    if args.workers == 1:
        for run in runs:
            try:
                consume(run, process_run(run, datasets, args))
            except Exception as exc:  # per-run fault isolation
                errors.append(f"{run.run_id}: {type(exc).__name__}: {exc}")
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(process_run, run, datasets, args): run for run in runs
            }
            for future in as_completed(futures):
                run = futures[future]
                try:
                    consume(run, future.result())
                except Exception as exc:  # per-run fault isolation
                    errors.append(f"{run.run_id}: {type(exc).__name__}: {exc}")

    per_seed_rows.sort(
        key=lambda r: (
            str(r.get("trained_dataset", "")),
            str(r.get("new_name", "")),
            str(r.get("context_suffix", "")),
            str(r.get("version_prefix", "")),
            str(r.get("run_version", "")),
        )
    )
    step_rows.sort(
        key=lambda r: (
            str(r.get("run_id", "")),
            int(try_float(r.get("reported_step")) or 0),
        )
    )
    checkpoint_rows.sort(
        key=lambda r: (
            str(r.get("run_id", "")),
            int(try_float(r.get("checkpoint_step")) or 0),
        )
    )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    per_seed_path = output_dir / PER_SEED_CSV
    steps_path = output_dir / STEPS_CSV
    checkpoints_path = output_dir / CHECKPOINTS_CSV

    write_csv_rows(
        per_seed_path,
        per_seed_rows,
        per_seed_fieldnames(datasets, args.include_debug_columns),
        args.float_significant_digits,
    )
    write_csv_rows(
        steps_path,
        step_rows,
        [*STEP_BASE_FIELDS, *sorted(step_metric_columns)],
        args.float_significant_digits,
    )
    write_csv_rows(
        checkpoints_path,
        checkpoint_rows,
        CHECKPOINT_BASE_FIELDS,
        args.float_significant_digits,
    )

    if not args.quiet:
        print(f"Single-phase runs discovered: {len(runs)}")
        print(f"Per-seed rows written:       {len(per_seed_rows)}")
        print(f"Per-step rows written:       {len(step_rows)}")
        print(f"Checkpoint rows written:     {len(checkpoint_rows)}")
        print(f"Saved: {per_seed_path}")
        print(f"Saved: {steps_path}")
        print(f"Saved: {checkpoints_path}")
        if warnings:
            print(f"Warnings: {len(warnings)}", file=sys.stderr)
            for message in warnings[:20]:
                print(f"  - {message}", file=sys.stderr)
            if len(warnings) > 20:
                print(f"  ... {len(warnings) - 20} more warnings", file=sys.stderr)
        if errors:
            print(f"Errors: {len(errors)}", file=sys.stderr)
            for message in errors:
                print(f"  - {message}", file=sys.stderr)

    return 1 if errors and args.fail_on_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
