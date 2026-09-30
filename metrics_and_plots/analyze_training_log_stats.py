#!/usr/bin/env python3
"""
Analyze per-step training statistics from SDFT/distillation training logs.

This utility is a log-only companion to analyze_response_lengths.py.  It keeps
its lightweight philosophy and flexible input handling:

- it never runs model inference and never needs a GPU;
- it can be called on one experiment root, one method folder, or an outputs root;
- it resolves the phase order independently for each experiment;
- it reads phase-level .log files when they exist;
- it writes one JSON summary per experiment and an optional index for parent runs.

Unlike analyze_response_lengths.py, this script does not inspect checkpoints and
does not read eval_*_responses.json files.  Instead, it parses training-log
dictionary lines such as:

    {'loss': 0.2266, 'grad_norm': 12.125, 'learning_rate': 0.0, ...}

and stores all parsed per-step fields.  Numeric values are also flattened and
summarized so they are easy to plot later.

Outputs written inside each experiment root
-------------------------------------------
    training_log_stats.json
    training_log_stats.csv                    # optional, enabled by default
    training_log_stats_plot.png               # optional, enabled by default
    training_log_stats_plot.pdf               # optional, enabled by default

When --input points to a parent folder, an index is also written next to the
input path:

    training_log_stats_index.json
    training_log_stats_index.csv

Examples
--------
    python metrics_and_plots/analyze_training_log_stats.py \
      --input outputs/bwd_student_ema/v2s1317

    python metrics_and_plots/analyze_training_log_stats.py \
      --input outputs/bwd_student_ema

    python metrics_and_plots/analyze_training_log_stats.py \
      --input outputs \
      --out-name training_log_stats.json

The parser is intentionally conservative by default: it only keeps dictionary
literals that look like training metric records.  Use --parse-any-dict to keep
any dictionary literal that contains numeric values.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import re
import shlex
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

# Support direct execution from ``metrics_and_plots/`` while importing shared
# project modules from the repository root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset_adapters import (
    get_evaluation_dataset_names,
    get_standard_phase_sequence,
)

KNOWN_DATASETS = list(
    get_evaluation_dataset_names()
)
DEFAULT_PHASE_ORDER = list(
    get_standard_phase_sequence("normal")
)

CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")
VERSION_SUFFIX_RE = re.compile(
    r"^(?:v\d+|lr(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?)(?:_?s\d+)?_?(?P<suffix>.*)$",
    re.IGNORECASE,
)
TRAIN_PROGRESS_RE = re.compile(r"(?P<step>\d+)\s*/\s*(?P<total>\d+)")

DEFAULT_TRAINING_METRIC_MARKERS = [
    "loss",
    "grad_norm",
    "learning_rate",
    "num_tokens",
    "epoch",
    "rewards",
    "kl_approx",
    "entropy",
    "completions/mean_length",
    "sampling/importance_sampling_ratio/mean",
]

DEFAULT_PLOT_METRIC_KEYS = ["loss", "entropy"]
PLOT_X_CHOICES = ["global_fraction", "global_step", "phase_fraction", "reported_step", "epoch"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze per-step metric dictionaries from SDFT training logs."
    )
    parser.add_argument(
        "--input",
        "--experiment_root",
        dest="input_path",
        required=True,
        type=Path,
        help="Experiment root, method folder, outputs root, or a single .log file.",
    )
    parser.add_argument(
        "--dataset_order",
        nargs="+",
        default=None,
        help=(
            "Fallback phase order. By default this does NOT override per-experiment "
            "experiment_config/suffix/command inference. Use --force-dataset-order to force it."
        ),
    )
    parser.add_argument(
        "--force-dataset-order",
        action="store_true",
        help="Force --dataset_order for every experiment. Normally you should not use this on parent folders.",
    )
    parser.add_argument(
        "--out-name",
        default="training_log_stats.json",
        help="JSON file written inside each experiment root.",
    )
    parser.add_argument(
        "--flat-csv-name",
        default="training_log_stats.csv",
        help="Flat per-record CSV written inside each experiment root when --write-flat-csv is enabled.",
    )
    parser.add_argument(
        "--generate-plots",
        action="store_true",
        default=True,
        help="Generate PNG/PDF training-stat plots inside each experiment root. Default: true.",
    )
    parser.add_argument("--no-generate-plots", dest="generate_plots", action="store_false")
    parser.add_argument(
        "--plot-prefix",
        default="training_log_stats",
        help="Prefix for plot files written inside each experiment root. Default: training_log_stats.",
    )
    parser.add_argument(
        "--plot-metric-keys",
        nargs="+",
        default=DEFAULT_PLOT_METRIC_KEYS,
        help=(
            "Numeric metric keys to plot from the parsed log dicts. "
            "Default: loss entropy. Use slash keys directly, e.g. "
            "completions/mean_length kl_approx rewards."
        ),
    )
    parser.add_argument(
        "--plot-x",
        choices=PLOT_X_CHOICES,
        default="global_fraction",
        help=(
            "X-axis for the main plot. Default: global_fraction. "
            "phase_fraction gives each phase its own 0..1 x-axis, like response_length training-log plots."
        ),
    )
    parser.add_argument(
        "--plot-moving-average",
        type=int,
        default=1,
        help="Optional centered moving-average window for plotted values. Default: 1, no smoothing.",
    )
    parser.add_argument("--recursive", action="store_true", default=True)
    parser.add_argument("--no-recursive", dest="recursive", action="store_false")
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument(
        "--log-glob",
        default="*.log",
        help="Glob used to find log files inside each phase directory. Default: *.log.",
    )
    parser.add_argument(
        "--parse-any-dict",
        action="store_true",
        help="Keep every dictionary literal with at least --min-numeric-keys numeric values.",
    )
    parser.add_argument(
        "--metric-marker-keys",
        nargs="+",
        default=DEFAULT_TRAINING_METRIC_MARKERS,
        help=(
            "Keys that identify a parsed dict as a training-metric record. "
            "Ignored when --parse-any-dict is set."
        ),
    )
    parser.add_argument(
        "--min-numeric-keys",
        type=int,
        default=2,
        help="Minimum number of numeric scalar fields required in a parsed metric dict. Default: 2.",
    )
    parser.add_argument(
        "--include-raw-line",
        action="store_true",
        help="Store the full raw log line for every parsed record. Disabled by default to keep JSON smaller.",
    )
    parser.add_argument(
        "--write-flat-csv",
        action="store_true",
        default=True,
        help="Also write a flat CSV with one row per parsed log metric record. Default: true.",
    )
    parser.add_argument("--no-write-flat-csv", dest="write_flat_csv", action="store_false")
    parser.add_argument("--write-index", action="store_true", default=True)
    parser.add_argument("--no-write-index", dest="write_index", action="store_false")
    parser.add_argument("--index-name", default="training_log_stats_index.json")
    parser.add_argument("--fail-on-error", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_text_safe(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def safe_float(x: Any) -> float | None:
    if isinstance(x, bool):
        return None
    try:
        value = float(x)
    except Exception:
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def json_safe(value: Any) -> Any:
    """Convert values from ast.literal_eval to JSON-safe objects."""
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    return str(value)


def summarize_numeric(values: Iterable[Any]) -> dict[str, Any]:
    clean = [safe_float(v) for v in values]
    clean = [v for v in clean if v is not None]
    if not clean:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
            "first": None,
            "last": None,
        }
    return {
        "count": len(clean),
        "mean": float(mean(clean)),
        "median": float(median(clean)),
        "min": float(min(clean)),
        "max": float(max(clean)),
        "first": float(clean[0]),
        "last": float(clean[-1]),
    }


def flatten_numeric_dict(obj: dict[str, Any], prefix: str = "") -> dict[str, float]:
    """Flatten numeric scalar values from a possibly nested dictionary."""
    out: dict[str, float] = {}
    for key, value in obj.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            out.update(flatten_numeric_dict(value, prefix=full_key))
            continue
        numeric = safe_float(value)
        if numeric is not None:
            out[full_key] = numeric
    return out


def checkpoint_step(path: Path) -> int | None:
    match = CHECKPOINT_RE.match(path.name)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def is_experiment_root(path: Path, datasets: Iterable[str] = KNOWN_DATASETS) -> bool:
    if not path.is_dir():
        return False

    if (path / "experiment_config.json").exists():
        return True

    dataset_set = set(datasets)
    phase_dirs = [p for p in path.iterdir() if p.is_dir() and p.name in dataset_set]
    if not phase_dirs:
        return False

    for phase_dir in phase_dirs:
        if any(phase_dir.glob("*.log")):
            return True
        if (phase_dir / "train_command.txt").exists() or (phase_dir / "command.txt").exists():
            return True
        if any(p.is_dir() and checkpoint_step(p) is not None for p in phase_dir.iterdir()):
            return True

    return False


def discover_experiment_roots(
    input_path: Path,
    datasets: list[str],
    recursive: bool,
    max_depth: int,
) -> list[Path]:
    input_path = input_path.resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    if input_path.is_file():
        return [input_path]

    if not recursive:
        return [input_path]

    if is_experiment_root(input_path, datasets):
        return [input_path]

    roots: list[Path] = []
    seen: set[Path] = set()

    def visit(path: Path, depth: int) -> None:
        if depth > max_depth or not path.is_dir():
            return
        if path.name in datasets or CHECKPOINT_RE.match(path.name):
            return
        if is_experiment_root(path, datasets):
            resolved = path.resolve()
            if resolved not in seen:
                roots.append(resolved)
                seen.add(resolved)
            return
        try:
            children = sorted([p for p in path.iterdir() if p.is_dir()], key=lambda p: p.name)
        except PermissionError:
            return
        for child in children:
            visit(child, depth + 1)

    visit(input_path, 0)
    return roots


def read_command_file(phase_dir: Path) -> str | None:
    for name in ("train_command.txt", "command.txt"):
        path = phase_dir / name
        if path.exists():
            try:
                return read_text_safe(path)
            except Exception:
                return None
    return None


def normalize_command_text(text: str) -> str:
    return text.replace("\\\n", " ").replace("\n", " ")


def extract_arg_from_command(command: str, arg_name: str) -> str | None:
    command = normalize_command_text(command or "")
    if not command:
        return None

    try:
        tokens = shlex.split(command)
    except Exception:
        tokens = command.split()

    flag = f"--{arg_name}"
    for i, token in enumerate(tokens):
        if token == flag and i + 1 < len(tokens):
            return tokens[i + 1]
        if token.startswith(flag + "="):
            return token.split("=", 1)[1]

    return None


def infer_dependency_phase(
    current_phase: str,
    model_name: str | None,
    phase_dirs: dict[str, Path],
) -> str | None:
    if not model_name:
        return None

    for candidate, candidate_dir in phase_dirs.items():
        if candidate == current_phase:
            continue

        try:
            model_path = Path(model_name).resolve()
            resolved_candidate = candidate_dir.resolve()
            if resolved_candidate == model_path or resolved_candidate in model_path.parents:
                return candidate
        except Exception:
            pass

        if f"/{candidate}/checkpoint-" in model_name:
            return candidate

    return None


def infer_phase_order_from_commands(
    experiment_root: Path,
    datasets: list[str],
) -> tuple[list[str], bool]:
    phase_dirs = {ds: experiment_root / ds for ds in datasets if (experiment_root / ds).is_dir()}
    existing = list(phase_dirs)
    if not existing:
        return [], False

    edges: dict[str, set[str]] = defaultdict(set)
    indegree = {phase: 0 for phase in existing}
    found_dependency = False

    for phase, phase_dir in phase_dirs.items():
        command = read_command_file(phase_dir)
        model_name = extract_arg_from_command(command or "", "model_name")
        dependency = infer_dependency_phase(phase, model_name, phase_dirs)
        if dependency is None:
            continue
        found_dependency = True
        if phase not in edges[dependency]:
            edges[dependency].add(phase)
            indegree[phase] += 1

    if not found_dependency:
        return [], False

    rank = {ds: i for i, ds in enumerate(datasets)}
    queue = sorted(
        [p for p in existing if indegree[p] == 0],
        key=lambda p: (rank.get(p, 999), p),
    )
    ordered: list[str] = []

    while queue:
        node = queue.pop(0)
        ordered.append(node)
        for nxt in sorted(edges[node], key=lambda p: (rank.get(p, 999), p)):
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                queue.append(nxt)

    if len(ordered) != len(existing):
        remaining = [p for p in existing if p not in ordered]
        ordered.extend(sorted(remaining, key=lambda p: (rank.get(p, 999), p)))

    return ordered, True


def infer_phase_order_from_version_suffix(version_name: str) -> tuple[list[str], bool, str | None]:
    """Infer normal/inv/inv2 from names like v1s37, v1s37inv, v1s37inv2."""
    match = VERSION_SUFFIX_RE.match(version_name)
    if not match:
        return [], False, None

    suffix = (match.group("suffix") or "").strip("_")

    order_group = "normal" if suffix == "" else suffix
    phase_sequence = get_standard_phase_sequence(order_group)

    if phase_sequence is not None:
        return list(phase_sequence), True, suffix

    return [], False, suffix


def filter_to_existing_or_known(order: list[str], experiment_root: Path, known_datasets: list[str]) -> list[str]:
    known = set(known_datasets)
    cleaned = [x for x in order if x in known]
    if cleaned:
        return cleaned
    existing = [ds for ds in known_datasets if (experiment_root / ds).is_dir()]
    return existing or list(known_datasets)


def resolve_phase_order(
    experiment_root: Path,
    explicit_dataset_order: list[str] | None,
    default_datasets: list[str],
    force_dataset_order: bool = False,
) -> tuple[list[str], str]:
    if force_dataset_order and explicit_dataset_order:
        return filter_to_existing_or_known(explicit_dataset_order, experiment_root, default_datasets), "forced_cli_dataset_order"

    config_path = experiment_root / "experiment_config.json"
    if config_path.exists():
        try:
            cfg = load_json(config_path)
            seq = cfg.get("phase_sequence")
            if isinstance(seq, list) and seq:
                order = [str(x) for x in seq if str(x)]
                return filter_to_existing_or_known(order, experiment_root, default_datasets), "experiment_config.phase_sequence"
        except Exception:
            pass

    inferred, ok = infer_phase_order_from_commands(experiment_root, default_datasets)
    if ok and inferred:
        return filter_to_existing_or_known(inferred, experiment_root, default_datasets), "train_command_dependencies"

    suffix_order, suffix_ok, suffix = infer_phase_order_from_version_suffix(experiment_root.name)
    if suffix_ok and suffix_order:
        return filter_to_existing_or_known(suffix_order, experiment_root, default_datasets), f"version_suffix:{suffix or 'normal'}"

    if explicit_dataset_order:
        return filter_to_existing_or_known(explicit_dataset_order, experiment_root, default_datasets), "fallback_cli_dataset_order"

    # When metadata does not reveal an order, prefer the canonical normal tasks
    # first, while still recognizing additional registered phase folders.
    fallback_order = DEFAULT_PHASE_ORDER + [
        dataset
        for dataset in default_datasets
        if dataset not in DEFAULT_PHASE_ORDER
    ]

    existing = [
        dataset
        for dataset in fallback_order
        if (experiment_root / dataset).is_dir()
    ]

    if existing:
        return existing, "existing_phase_dirs_default_order"

    return list(DEFAULT_PHASE_ORDER), "default_order"


def extract_first_dict_literal(line: str) -> dict[str, Any] | None:
    """Extract the first balanced-looking dictionary literal from a log line."""
    start = line.find("{")
    end = line.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    try:
        obj = ast.literal_eval(line[start : end + 1])
    except Exception:
        return None

    return obj if isinstance(obj, dict) else None


def parse_progress_step(line: str) -> tuple[int | None, int | None]:
    matches = TRAIN_PROGRESS_RE.findall(line)
    if not matches:
        return None, None
    step_s, total_s = matches[-1]
    try:
        return int(step_s), int(total_s)
    except ValueError:
        return None, None


def looks_like_training_metric_record(
    obj: dict[str, Any],
    *,
    metric_marker_keys: list[str],
    min_numeric_keys: int,
    parse_any_dict: bool,
) -> bool:
    numeric = flatten_numeric_dict(obj)
    if len(numeric) < min_numeric_keys:
        return False
    if parse_any_dict:
        return True
    return any(key in obj for key in metric_marker_keys)


def find_log_files(phase_dir: Path, log_glob: str) -> list[Path]:
    if not phase_dir.is_dir():
        return []
    return sorted(phase_dir.glob(log_glob), key=lambda p: p.name)


def parse_training_log(
    log_path: Path,
    phase: str,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    last_progress_step: int | None = None
    last_progress_total: int | None = None

    try:
        with log_path.open("r", encoding="utf-8", errors="replace") as f:
            for line_no, line in enumerate(f, start=1):
                progress_step, progress_total = parse_progress_step(line)
                if progress_step is not None:
                    last_progress_step = progress_step
                if progress_total is not None:
                    last_progress_total = progress_total

                if "{" not in line or "}" not in line:
                    continue

                obj = extract_first_dict_literal(line)
                if obj is None:
                    continue

                if not looks_like_training_metric_record(
                    obj,
                    metric_marker_keys=args.metric_marker_keys,
                    min_numeric_keys=args.min_numeric_keys,
                    parse_any_dict=args.parse_any_dict,
                ):
                    continue

                idx = len(records) + 1
                step = last_progress_step if last_progress_step is not None else idx
                numeric_metrics = flatten_numeric_dict(obj)
                metrics = json_safe(obj)

                rec: dict[str, Any] = {
                    "phase": phase,
                    "source_log": str(log_path),
                    "line_number": line_no,
                    "phase_step_index": idx,
                    "reported_step": step,
                    "reported_total_steps": last_progress_total,
                    "epoch": safe_float(obj.get("epoch")),
                    "learning_rate": safe_float(obj.get("learning_rate")),
                    "metrics": metrics,
                    "numeric_metrics": numeric_metrics,
                }
                if args.include_raw_line:
                    rec["raw_line"] = line.rstrip("\n")

                # Duplicate numeric fields at the top level for easy JSON/CSV use.
                for key, value in numeric_metrics.items():
                    rec[key] = value

                records.append(rec)
    except Exception as exc:
        return [{"phase": phase, "source_log": str(log_path), "error": repr(exc)}]

    totals = [int(r["reported_total_steps"]) for r in records if isinstance(r.get("reported_total_steps"), int)]
    final_total = max(totals) if totals else (len(records) if records else None)

    for r in records:
        total = final_total or len(records) or 1
        step = r.get("reported_step")
        if not isinstance(step, int) or step <= 0:
            step = int(r["phase_step_index"])
        r["phase_total_steps"] = total
        r["phase_fraction"] = min(max(step / total, 0.0), 1.0) if total else None

    return records


def add_global_step_fields(phases: dict[str, Any], phase_order: list[str]) -> dict[str, int]:
    """Add global_step and global_fraction using log-derived phase lengths."""
    phase_totals: dict[str, int] = {}
    for phase in phase_order:
        records = phases.get(phase, {}).get("training_log_records", [])
        totals = [r.get("phase_total_steps") for r in records if isinstance(r.get("phase_total_steps"), int)]
        if totals:
            phase_totals[phase] = max(totals)
        elif records:
            phase_totals[phase] = len(records)
        else:
            phase_totals[phase] = 0

    offsets: dict[str, int] = {}
    running = 0
    for phase in phase_order:
        offsets[phase] = running
        running += phase_totals.get(phase, 0)

    total_global = running or 1
    for phase in phase_order:
        offset = offsets.get(phase, 0)
        for rec in phases.get(phase, {}).get("training_log_records", []):
            step = rec.get("reported_step")
            if not isinstance(step, int) or step <= 0:
                step = int(rec.get("phase_step_index") or 0)
            global_step = offset + step
            rec["phase_offset"] = offset
            rec["global_step"] = global_step
            rec["global_fraction"] = min(max(global_step / total_global, 0.0), 1.0)

    return offsets


def build_combined_training_log_series(phases: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for phase_obj in phases.values():
        for rec in phase_obj.get("training_log_records", []):
            if "error" not in rec:
                rows.append(dict(rec))
    rows.sort(
        key=lambda r: (
            r.get("global_step") is None,
            r.get("global_step") or 0,
            r.get("source_log", ""),
            r.get("line_number") or 0,
        )
    )
    return rows


def summarize_numeric_records(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_key: dict[str, list[float]] = defaultdict(list)
    for rec in records:
        numeric = rec.get("numeric_metrics", {})
        if not isinstance(numeric, dict):
            continue
        for key, value in numeric.items():
            v = safe_float(value)
            if v is not None:
                by_key[key].append(v)
    return {key: summarize_numeric(values) for key, values in sorted(by_key.items())}


def readable_metric_name(metric_key: str) -> str:
    """Convert a raw log key into a compact human-readable plot label."""
    return metric_key.replace("/", " / ").replace("_", " ")


def metric_value_from_record(record: dict[str, Any], metric_key: str) -> float | None:
    """Read a numeric metric from either top-level duplicated fields or numeric_metrics."""
    value = safe_float(record.get(metric_key))
    if value is not None:
        return value
    numeric = record.get("numeric_metrics", {})
    if isinstance(numeric, dict):
        return safe_float(numeric.get(metric_key))
    return None


def moving_average_points(points: list[tuple[float, float]], window: int) -> list[tuple[float, float]]:
    """Centered moving average for already-sorted (x, y) points."""
    if window <= 1 or len(points) <= 2:
        return points
    half = max(0, window // 2)
    smoothed: list[tuple[float, float]] = []
    for i, (x, _) in enumerate(points):
        lo = max(0, i - half)
        hi = min(len(points), i + half + 1)
        ys = [points[j][1] for j in range(lo, hi)]
        smoothed.append((x, float(mean(ys))))
    return smoothed


def collect_plot_points(
    records: list[dict[str, Any]],
    *,
    metric_key: str,
    x_key: str,
    moving_average_window: int,
) -> list[tuple[float, float]]:
    """Collect sorted numeric points for one metric and one x-axis."""
    points: list[tuple[float, float]] = []
    for record in records:
        x = safe_float(record.get(x_key))
        y = metric_value_from_record(record, metric_key)
        if x is None or y is None:
            continue
        points.append((x, y))
    points.sort(key=lambda p: p[0])
    return moving_average_points(points, moving_average_window)


def plot_training_log_stats(
    experiment_root: Path,
    combined_log: list[dict[str, Any]],
    phase_order: list[str],
    plot_prefix: str,
    metric_keys: list[str],
    x_key: str,
    moving_average_window: int,
) -> list[str]:
    """Plot selected training-log statistics and return written file paths.

    The format intentionally mirrors analyze_response_lengths.py: one PNG and
    one PDF are written directly inside the experiment root. By default this
    plots loss and entropy, grouped by phase, using global_fraction on the x-axis.
    """
    png = experiment_root / f"{plot_prefix}_plot.png"
    pdf = experiment_root / f"{plot_prefix}_plot.pdf"
    experiment_root.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 6))
    plotted_any = False

    for metric_key in metric_keys:
        for phase in phase_order:
            phase_records = [r for r in combined_log if r.get("phase") == phase]
            points = collect_plot_points(
                phase_records,
                metric_key=metric_key,
                x_key=x_key,
                moving_average_window=moving_average_window,
            )
            if not points:
                continue
            xs, ys = zip(*points)
            plotted_any = True
            label = f"{phase} {readable_metric_name(metric_key)}"
            ax.plot(xs, ys, marker=".", linewidth=1.2, label=label)

    ax.set_xlabel(readable_metric_name(x_key))
    ax.set_ylabel("Training log metric value")
    title_root = f"{experiment_root.parent.name}/{experiment_root.name}"
    ax.set_title(f"Training log statistics: {title_root}")
    ax.grid(True, alpha=0.3)

    if plotted_any:
        ax.legend(fontsize=8)
    else:
        ax.text(
            0.5,
            0.5,
            "No requested training-log metrics found",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    fig.tight_layout()
    fig.savefig(png, dpi=200)
    fig.savefig(pdf)
    plt.close(fig)
    return [str(png), str(pdf)]


def plot_training_log_stats_by_metric(
    experiment_root: Path,
    combined_log: list[dict[str, Any]],
    phase_order: list[str],
    plot_prefix: str,
    metric_keys: list[str],
    x_key: str,
    moving_average_window: int,
) -> dict[str, list[str]]:
    """Write one additional PNG/PDF pair per requested metric.

    This keeps the combined plot compact while still making scale-sensitive
    metrics such as loss and entropy easy to inspect independently.
    """
    written: dict[str, list[str]] = {}
    per_metric_dir = experiment_root / f"{plot_prefix}_metric_plots"
    per_metric_dir.mkdir(parents=True, exist_ok=True)

    for metric_key in metric_keys:
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", metric_key).strip("_") or "metric"
        png = per_metric_dir / f"{slug}_plot.png"
        pdf = per_metric_dir / f"{slug}_plot.pdf"

        fig, ax = plt.subplots(figsize=(11, 6))
        plotted_any = False
        for phase in phase_order:
            phase_records = [r for r in combined_log if r.get("phase") == phase]
            points = collect_plot_points(
                phase_records,
                metric_key=metric_key,
                x_key=x_key,
                moving_average_window=moving_average_window,
            )
            if not points:
                continue
            xs, ys = zip(*points)
            plotted_any = True
            ax.plot(xs, ys, marker=".", linewidth=1.3, label=phase)

        ax.set_xlabel(readable_metric_name(x_key))
        ax.set_ylabel(readable_metric_name(metric_key))
        title_root = f"{experiment_root.parent.name}/{experiment_root.name}"
        ax.set_title(f"{readable_metric_name(metric_key)}: {title_root}")
        ax.grid(True, alpha=0.3)
        if plotted_any:
            ax.legend(title="phase", fontsize=8)
        else:
            ax.text(
                0.5,
                0.5,
                f"No {metric_key} records found",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

        fig.tight_layout()
        fig.savefig(png, dpi=200)
        fig.savefig(pdf)
        plt.close(fig)
        written[metric_key] = [str(png), str(pdf)]

    return written


def write_flat_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write one row per log record with all numeric metric keys as columns."""
    base_fields = [
        "experiment_root",
        "experiment_name",
        "version",
        "phase",
        "source_log",
        "line_number",
        "phase_step_index",
        "reported_step",
        "reported_total_steps",
        "phase_total_steps",
        "phase_fraction",
        "phase_offset",
        "global_step",
        "global_fraction",
        "epoch",
        "learning_rate",
    ]
    metric_fields = sorted(
        {
            key
            for row in rows
            for key in (row.get("numeric_metrics", {}) or {}).keys()
        }
    )
    fieldnames = base_fields + [key for key in metric_fields if key not in set(base_fields)]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            flat = {key: row.get(key) for key in base_fields}
            for key, value in (row.get("numeric_metrics", {}) or {}).items():
                flat[key] = value
            writer.writerow(flat)


def write_index_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "experiment_root",
        "status",
        "output_json",
        "output_csv",
        "output_plots",
        "phase_order",
        "phase_order_source",
        "training_log_points",
        "num_numeric_keys",
        "warnings",
        "error",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def analyze_single_log_file(log_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    """Special case for --input pointing directly to one .log file."""
    phase = log_path.parent.name if log_path.parent else "unknown_phase"
    records = parse_training_log(log_path, phase=phase, args=args)

    phases = {
        phase: {
            "phase": phase,
            "phase_dir": str(log_path.parent),
            "exists": True,
            "log_files": [str(log_path)],
            "training_log_records": records,
            "numeric_key_summary": summarize_numeric_records(records),
        }
    }
    phase_offsets = add_global_step_fields(phases, [phase])
    combined = build_combined_training_log_series(phases)

    root = log_path.parent
    summary = {
        "created_at": now_iso(),
        "script": "analyze_training_log_stats.py",
        "input_log_file": str(log_path),
        "experiment_root": str(root),
        "experiment_name": root.parent.name if root.parent else "",
        "version": root.name,
        "phase_order": [phase],
        "phase_order_source": "single_log_file_parent_name",
        "phase_offsets": phase_offsets,
        "warnings": [],
        "phases": phases,
        "combined_training_log_series": combined,
        "numeric_key_summary": summarize_numeric_records(combined),
    }

    out_path = root / args.out_name
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    csv_path = None
    if args.write_flat_csv:
        csv_path = root / args.flat_csv_name
        for rec in combined:
            rec["experiment_root"] = str(root)
            rec["experiment_name"] = summary["experiment_name"]
            rec["version"] = summary["version"]
        write_flat_csv(csv_path, combined)

    plots: dict[str, Any] = {}
    if args.generate_plots:
        plots["combined"] = plot_training_log_stats(
            root,
            combined,
            [phase],
            args.plot_prefix,
            args.plot_metric_keys,
            args.plot_x,
            args.plot_moving_average,
        )
        plots["per_metric"] = plot_training_log_stats_by_metric(
            root,
            combined,
            [phase],
            args.plot_prefix,
            args.plot_metric_keys,
            args.plot_x,
            args.plot_moving_average,
        )

    summary["plots"] = plots
    summary["plot_metric_keys"] = args.plot_metric_keys
    summary["plot_x"] = args.plot_x
    summary["plot_moving_average"] = args.plot_moving_average
    summary["output_json"] = str(out_path)
    summary["output_csv"] = str(csv_path) if csv_path is not None else ""

    # Re-write JSON after plot generation so the JSON contains the plot paths.
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary


def analyze_experiment(experiment_root: Path, args: argparse.Namespace) -> dict[str, Any]:
    experiment_root = experiment_root.resolve()

    if experiment_root.is_file():
        return analyze_single_log_file(experiment_root, args)

    phase_order, phase_order_source = resolve_phase_order(
        experiment_root=experiment_root,
        explicit_dataset_order=args.dataset_order,
        default_datasets=KNOWN_DATASETS,
        force_dataset_order=args.force_dataset_order,
    )

    phases_out: dict[str, Any] = {}
    warnings: list[str] = []

    for phase in phase_order:
        phase_dir = experiment_root / phase
        phase_obj: dict[str, Any] = {
            "phase": phase,
            "phase_dir": str(phase_dir),
            "exists": phase_dir.exists(),
            "log_files": [],
            "training_log_records": [],
            "numeric_key_summary": {},
        }

        if not phase_dir.is_dir():
            warnings.append(f"Missing phase directory: {phase_dir}")
            phases_out[phase] = phase_obj
            continue

        log_files = find_log_files(phase_dir, args.log_glob)
        phase_obj["log_files"] = [str(p) for p in log_files]

        for log_file in log_files:
            phase_obj["training_log_records"].extend(parse_training_log(log_file, phase=phase, args=args))

        error_records = [r for r in phase_obj["training_log_records"] if "error" in r]
        if error_records:
            warnings.append(f"Errors while reading logs for phase {phase}: {len(error_records)}")

        phase_obj["numeric_key_summary"] = summarize_numeric_records(phase_obj["training_log_records"])
        phases_out[phase] = phase_obj

    phase_offsets = add_global_step_fields(phases_out, phase_order)
    combined_log = build_combined_training_log_series(phases_out)

    for rec in combined_log:
        rec["experiment_root"] = str(experiment_root)
        rec["experiment_name"] = experiment_root.parent.name
        rec["version"] = experiment_root.name

    summary = {
        "created_at": now_iso(),
        "script": "analyze_training_log_stats.py",
        "experiment_root": str(experiment_root),
        "experiment_name": experiment_root.parent.name,
        "version": experiment_root.name,
        "phase_order": phase_order,
        "phase_order_source": phase_order_source,
        "phase_offsets": phase_offsets,
        "log_glob": args.log_glob,
        "metric_marker_keys": args.metric_marker_keys,
        "min_numeric_keys": args.min_numeric_keys,
        "parse_any_dict": bool(args.parse_any_dict),
        "warnings": warnings,
        "phases": phases_out,
        "combined_training_log_series": combined_log,
        "numeric_key_summary": summarize_numeric_records(combined_log),
    }

    out_path = experiment_root / args.out_name
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    csv_path = None
    if args.write_flat_csv:
        csv_path = experiment_root / args.flat_csv_name
        write_flat_csv(csv_path, combined_log)

    plots: dict[str, Any] = {}
    if args.generate_plots:
        plots["combined"] = plot_training_log_stats(
            experiment_root,
            combined_log,
            phase_order,
            args.plot_prefix,
            args.plot_metric_keys,
            args.plot_x,
            args.plot_moving_average,
        )
        plots["per_metric"] = plot_training_log_stats_by_metric(
            experiment_root,
            combined_log,
            phase_order,
            args.plot_prefix,
            args.plot_metric_keys,
            args.plot_x,
            args.plot_moving_average,
        )

    summary["plots"] = plots
    summary["plot_metric_keys"] = args.plot_metric_keys
    summary["plot_x"] = args.plot_x
    summary["plot_moving_average"] = args.plot_moving_average
    summary["output_json"] = str(out_path)
    summary["output_csv"] = str(csv_path) if csv_path is not None else ""

    # Re-write JSON after plot generation so the JSON contains the plot paths.
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    if not args.quiet:
        print(
            f"[done] {experiment_root}: "
            f"phase_order={' '.join(phase_order)} ({phase_order_source}), "
            f"log_points={len(combined_log)}, json={out_path}"
        )

    return summary


def main() -> None:
    args = parse_args()
    input_path = args.input_path.resolve()

    if args.force_dataset_order and not args.dataset_order:
        raise ValueError("--force-dataset-order requires --dataset_order.")

    roots = discover_experiment_roots(
        input_path,
        KNOWN_DATASETS,
        args.recursive,
        args.max_depth,
    )

    if not roots:
        raise RuntimeError(f"No experiment roots found under: {input_path}")

    if not args.quiet:
        print(f"[discover] input={input_path}")
        print(f"[discover] experiments={len(roots)}")
        for root in roots[:20]:
            print(f"  - {root}")
        if len(roots) > 20:
            print(f"  ... {len(roots) - 20} more")

    index_rows: list[dict[str, Any]] = []
    had_error = False

    for root in roots:
        try:
            summary = analyze_experiment(root, args)
            index_rows.append(
                {
                    "experiment_root": str(root),
                    "status": "ok",
                    "output_json": summary.get("output_json"),
                    "output_csv": summary.get("output_csv"),
                    "output_plots": json.dumps(summary.get("plots", {}), ensure_ascii=False),
                    "phase_order": " ".join(summary.get("phase_order", [])),
                    "phase_order_source": summary.get("phase_order_source"),
                    "training_log_points": len(summary.get("combined_training_log_series", [])),
                    "num_numeric_keys": len(summary.get("numeric_key_summary", {})),
                    "warnings": len(summary.get("warnings", [])),
                }
            )
        except Exception as exc:
            had_error = True
            index_rows.append({"experiment_root": str(root), "status": "error", "error": repr(exc)})
            print(f"[error] {root}: {repr(exc)}", file=sys.stderr)
            if args.fail_on_error:
                raise

    if args.write_index:
        index = {
            "created_at": now_iso(),
            "input_path": str(input_path),
            "num_experiments": len(roots),
            "num_ok": sum(1 for r in index_rows if r.get("status") == "ok"),
            "num_error": sum(1 for r in index_rows if r.get("status") == "error"),
            "rows": index_rows,
        }
        index_path = input_path / args.index_name if input_path.is_dir() else input_path.parent / args.index_name
        with index_path.open("w", encoding="utf-8") as f:
            json.dump(index, f, indent=2, ensure_ascii=False)

        csv_path = index_path.with_suffix(".csv")
        write_index_csv(csv_path, index_rows)

        if not args.quiet:
            print(f"[index] {index_path}")
            print(f"[index-csv] {csv_path}")

    if had_error:
        sys.exit(1)


if __name__ == "__main__":
    main()
