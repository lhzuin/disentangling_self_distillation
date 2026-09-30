#!/usr/bin/env python3
"""
Analyze response-length evolution during sequential SDFT/distillation training.

This utility is intentionally lightweight: it never runs model inference and it
never requires a GPU.  It reads already-produced phase logs and
`eval_<dataset>_responses.json` files, then writes one JSON summary and two
plots per experiment.

Outputs written inside each experiment root
-------------------------------------------
    response_length_evolution.json
    response_length_checkpoint_plot.png
    response_length_checkpoint_plot.pdf
    response_length_training_log_plot.png
    response_length_training_log_plot.pdf

It can be called on:
    * one experiment root, e.g. outputs/bwd_student_ema/v2s1317
    * one method folder, e.g. outputs/bwd_student_ema
    * an outputs root, e.g. outputs

Phase-order handling
--------------------
The phase order is resolved independently for each experiment.  This is important
because normal, inv and inv2 runs may coexist in the same parent folder.

Priority per experiment:
    1. --force-dataset-order + --dataset_order, only when explicitly requested
    2. experiment_config.json["phase_sequence"]
    3. train_command.txt dependency inference via --model_name
    4. version suffix inference: normal / inv / inv2
    5. --dataset_order as fallback only
    6. existing phase folders in default order
    7. default order: tooluse science

Examples
--------
    python metrics_and_plots/analyze_response_lengths.py \
      --input outputs/bwd_student_ema/v2s1317

    python metrics_and_plots/analyze_response_lengths.py \
      --input outputs/bwd_student_ema

    python metrics_and_plots/analyze_response_lengths.py \
      --input outputs \
      --length-mode whitespace

Integration recommendation
--------------------------
Run after checkpoint evaluation and before checkpoint cleanup, so the
`eval_*_responses.json` files are available.
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

# Unordered universe of dataset folder names that the analyzer recognizes.
KNOWN_DATASETS = list(
    get_evaluation_dataset_names()
)

# Ordered fallback used only when the experiment itself does not reveal an order.
DEFAULT_PHASE_ORDER = list(
    get_standard_phase_sequence("normal")
)

CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")
VERSION_SUFFIX_RE = re.compile(
    r"^(?:v\d+|lr(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?)(?:_?s\d+)?_?(?P<suffix>.*)$",
    re.IGNORECASE,
)
TRAIN_PROGRESS_RE = re.compile(r"(?P<step>\d+)\s*/\s*(?P<total>\d+)")
RESPONSE_FILE_RE = re.compile(r"^eval_(?P<dataset>.+)_responses(?P<original>_original)?\.json$")

LOG_LENGTH_KEYS = [
    "completions/mean_length",
    "completions/min_length",
    "completions/max_length",
    "completions/clipped_ratio",
    "completions/mean_terminated_length",
    "completions/min_terminated_length",
    "completions/max_terminated_length",
]


class OptionalTokenizer:
    """Optional Hugging Face tokenizer wrapper with graceful fallback."""

    def __init__(
        self,
        tokenizer_name_or_path: str | None,
        *,
        quiet: bool = False,
        local_files_only: bool = True,
    ):
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.tokenizer = None
        self.error = None

        if tokenizer_name_or_path is None:
            return

        try:
            from transformers import AutoTokenizer  # type: ignore

            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name_or_path,
                trust_remote_code=True,
                local_files_only=local_files_only,
            )
            if not quiet:
                print(f"[tokenizer] loaded: {tokenizer_name_or_path}")
        except Exception as exc:  # environment-dependent optional path
            self.error = repr(exc)
            self.tokenizer = None
            if not quiet:
                print(
                    f"[tokenizer] could not load {tokenizer_name_or_path!r}; "
                    f"falling back when allowed. Error: {self.error}",
                    file=sys.stderr,
                )

    @property
    def available(self) -> bool:
        return self.tokenizer is not None

    def count(self, text: str) -> int | None:
        if self.tokenizer is None:
            return None
        try:
            return int(len(self.tokenizer.encode(text, add_special_tokens=False)))
        except Exception:
            return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze training/eval response-length evolution for SDFT experiments."
    )
    parser.add_argument(
        "--input",
        "--experiment_root",
        dest="input_path",
        required=True,
        type=Path,
        help="Experiment root or parent folder.",
    )
    parser.add_argument(
        "--dataset_order",
        nargs="+",
        default=None,
        help=(
            "Fallback phase order. By default this does NOT override per-experiment "
            "experiment_config/suffix inference. Use --force-dataset-order to force it."
        ),
    )
    parser.add_argument(
        "--force-dataset-order",
        action="store_true",
        help="Force --dataset_order for every experiment. Normally you should NOT use this on parent folders.",
    )
    parser.add_argument(
        "--eval_subset",
        nargs="+",
        default=None,
        help="Eval datasets to look for. Default: all known datasets in the resolved phase order.",
    )
    parser.add_argument("--out-name", default="response_length_evolution.json")
    parser.add_argument("--plot-prefix", default="response_length")
    parser.add_argument("--recursive", action="store_true", default=True)
    parser.add_argument("--no-recursive", dest="recursive", action="store_false")
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument(
        "--length-mode",
        choices=["chars", "whitespace", "hf_tokens", "hf_tokens_if_available"],
        default="whitespace",
        help=(
            "Main eval response-length unit used in checkpoint plots. "
            "JSON always stores chars and whitespace, plus HF tokens when available."
        ),
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="Optional tokenizer name/path for HF token counts.",
    )
    parser.add_argument(
        "--auto-tokenizer-from-experiment",
        action="store_true",
        help=(
            "If --tokenizer is omitted, try a local tokenizer found inside the experiment "
            "or initial_model from experiment_config.json. Disabled by default to avoid slow downloads."
        ),
    )
    parser.add_argument(
        "--tokenizer-local-files-only",
        action="store_true",
        default=True,
        help="Load tokenizer with local_files_only=True. Default: true.",
    )
    parser.add_argument(
        "--allow-tokenizer-download",
        dest="tokenizer_local_files_only",
        action="store_false",
        help="Allow transformers to download tokenizer files if needed.",
    )
    parser.add_argument("--response-field", default="response")
    parser.add_argument("--ignore-original", action="store_true")
    parser.add_argument("--write-index", action="store_true", default=True)
    parser.add_argument("--no-write-index", dest="write_index", action="store_false")
    parser.add_argument("--index-name", default="response_length_analysis_index.json")
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


def checkpoint_step(path: Path) -> int | None:
    match = CHECKPOINT_RE.match(path.name)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def is_checkpoint_dir(path: Path) -> bool:
    return path.is_dir() and checkpoint_step(path) is not None


def sorted_checkpoints(phase_dir: Path) -> list[Path]:
    if not phase_dir.is_dir():
        return []
    return sorted(
        [p for p in phase_dir.iterdir() if is_checkpoint_dir(p)],
        key=lambda p: checkpoint_step(p) or -1,
    )


def is_experiment_root(
    path: Path,
    datasets: Iterable[str] = KNOWN_DATASETS,
) -> bool:
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
        if sorted_checkpoints(phase_dir):
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
    """Keep order stable, but do not introduce invalid names."""
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
    """Resolve the phase order independently for one experiment."""
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


def phase_dirs_for_order(experiment_root: Path, phase_order: list[str]) -> dict[str, Path]:
    return {phase: experiment_root / phase for phase in phase_order if (experiment_root / phase).is_dir()}


def phase_offsets_from_checkpoints(phase_dirs: dict[str, Path], phase_order: list[str]) -> dict[str, int]:
    offsets: dict[str, int] = {}
    running = 0
    for phase in phase_order:
        if phase not in phase_dirs:
            continue
        offsets[phase] = running
        ckpts = sorted_checkpoints(phase_dirs[phase])
        if ckpts:
            running += checkpoint_step(ckpts[-1]) or 0
    return offsets


def extract_first_dict_literal(line: str) -> dict[str, Any] | None:
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


def safe_float(x: Any) -> float | None:
    try:
        value = float(x)
    except Exception:
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def parse_training_log(log_path: Path, phase: str) -> list[dict[str, Any]]:
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

                if "completions/mean_length" not in line:
                    continue

                obj = extract_first_dict_literal(line)
                if obj is None or "completions/mean_length" not in obj:
                    continue

                idx = len(records) + 1
                step = last_progress_step if last_progress_step is not None else idx
                rec: dict[str, Any] = {
                    "phase": phase,
                    "source_log": str(log_path),
                    "line_number": line_no,
                    "phase_step_index": idx,
                    "reported_step": step,
                    "reported_total_steps": last_progress_total,
                    "epoch": safe_float(obj.get("epoch")),
                    "learning_rate": safe_float(obj.get("learning_rate")),
                    "loss": safe_float(obj.get("loss")),
                    "num_tokens": safe_float(obj.get("num_tokens")),
                }

                for key in LOG_LENGTH_KEYS:
                    rec[key.replace("completions/", "")] = safe_float(obj.get(key))

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


def find_log_files(phase_dir: Path) -> list[Path]:
    if not phase_dir.is_dir():
        return []
    return sorted(phase_dir.glob("*.log"), key=lambda p: p.name)


def response_text_from_record(record: Any, response_field: str) -> str | None:
    if not isinstance(record, dict):
        return None
    value = record.get(response_field)
    if isinstance(value, str):
        return value
    if value is None:
        return None
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return str(value)


def whitespace_token_count(text: str) -> int:
    return len(text.split())


def summarize_numeric(values: list[int | float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": float(mean(values)),
        "median": float(median(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def choose_tokenizer_path(experiment_root: Path, explicit: str | None, auto_from_experiment: bool) -> str | None:
    if explicit:
        return explicit

    if not auto_from_experiment:
        return None

    # Prefer a local tokenizer under existing checkpoint directories.
    for candidate in experiment_root.rglob("tokenizer_config.json"):
        parent = candidate.parent
        if parent.is_dir():
            return str(parent)

    config_path = experiment_root / "experiment_config.json"
    if config_path.exists():
        try:
            cfg = load_json(config_path)
            initial_model = cfg.get("initial_model")
            if isinstance(initial_model, str) and initial_model.strip():
                return initial_model.strip()
        except Exception:
            pass

    return None


def get_length_field(length_mode: str, tokenizer_available: bool) -> str:
    if length_mode == "chars":
        return "char_length"
    if length_mode == "whitespace":
        return "whitespace_token_length"
    if length_mode == "hf_tokens":
        return "hf_token_length"
    if length_mode == "hf_tokens_if_available":
        return "hf_token_length" if tokenizer_available else "whitespace_token_length"
    raise ValueError(f"Unknown length mode: {length_mode}")


def analyze_response_file(
    path: Path,
    response_field: str,
    tokenizer: OptionalTokenizer,
    main_length_field: str,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "num_records": 0,
        "num_responses_with_text": 0,
        "length_field_used_for_main_mean": main_length_field,
        "main_mean_length": None,
        "char_length": summarize_numeric([]),
        "whitespace_token_length": summarize_numeric([]),
        "hf_token_length": summarize_numeric([]),
    }

    if not path.exists():
        return out

    try:
        data = load_json(path)
    except Exception as exc:
        out["error"] = f"could_not_parse_json: {repr(exc)}"
        return out

    if not isinstance(data, list):
        out["error"] = "json_is_not_a_list"
        return out

    texts = [t for t in (response_text_from_record(r, response_field) for r in data) if t is not None]
    char_lengths = [len(t) for t in texts]
    whitespace_lengths = [whitespace_token_count(t) for t in texts]
    hf_lengths: list[int] = []

    if tokenizer.available:
        for t in texts:
            n = tokenizer.count(t)
            if n is not None:
                hf_lengths.append(n)

    out["num_records"] = len(data)
    out["num_responses_with_text"] = len(texts)
    out["char_length"] = summarize_numeric(char_lengths)
    out["whitespace_token_length"] = summarize_numeric(whitespace_lengths)
    out["hf_token_length"] = summarize_numeric(hf_lengths)

    main_summary = out.get(main_length_field)
    if isinstance(main_summary, dict):
        out["main_mean_length"] = main_summary.get("mean")

    return out


def analyze_checkpoint(
    ckpt: Path,
    phase: str,
    phase_offset: int,
    eval_datasets: list[str],
    response_field: str,
    tokenizer: OptionalTokenizer,
    main_length_field: str,
    ignore_original: bool,
) -> dict[str, Any]:
    step = checkpoint_step(ckpt) or 0
    record: dict[str, Any] = {
        "checkpoint_dir": str(ckpt),
        "phase": phase,
        "phase_step": step,
        "global_step": phase_offset + step,
        "eval_response_lengths": {},
    }

    for dataset in eval_datasets:
        paths = [ckpt / f"eval_{dataset}_responses.json"]
        if not ignore_original:
            paths.append(ckpt / f"eval_{dataset}_responses_original.json")

        summaries = []
        for path in paths:
            if path.exists():
                summary = analyze_response_file(path, response_field, tokenizer, main_length_field)
                summary["dataset"] = dataset
                summary["is_original"] = path.name.endswith("_original.json")
                summaries.append(summary)

        if summaries:
            primary = next((s for s in summaries if not s.get("is_original")), summaries[0])
            primary = dict(primary)
            primary["all_files"] = summaries
            record["eval_response_lengths"][dataset] = primary
        else:
            record["eval_response_lengths"][dataset] = {
                "dataset": dataset,
                "exists": False,
                "path": str(ckpt / f"eval_{dataset}_responses.json"),
                "main_mean_length": None,
                "length_field_used_for_main_mean": main_length_field,
            }

    return record


def discover_eval_datasets(experiment_root: Path, phase_order: list[str], explicit_eval_subset: list[str] | None) -> list[str]:
    if explicit_eval_subset:
        return list(explicit_eval_subset)

    discovered: list[str] = []
    seen = set()

    # Preserve phase-order first.
    for ds in phase_order:
        if ds not in seen:
            discovered.append(ds)
            seen.add(ds)

    # Add any additional eval_*_responses files found in checkpoints.
    for path in experiment_root.rglob("eval_*_responses*.json"):
        match = RESPONSE_FILE_RE.match(path.name)
        if not match:
            continue
        dataset = match.group("dataset")
        if dataset not in seen:
            discovered.append(dataset)
            seen.add(dataset)

    return discovered


def build_combined_checkpoint_series(
    phases: dict[str, Any],
    eval_datasets: list[str],
) -> dict[str, list[dict[str, Any]]]:
    combined: dict[str, list[dict[str, Any]]] = {ds: [] for ds in eval_datasets}

    for phase, phase_obj in phases.items():
        for ckpt in phase_obj.get("checkpoints", []):
            for dataset, summary in ckpt.get("eval_response_lengths", {}).items():
                combined.setdefault(dataset, []).append(
                    {
                        "phase": phase,
                        "phase_step": ckpt.get("phase_step"),
                        "global_step": ckpt.get("global_step"),
                        "checkpoint_dir": ckpt.get("checkpoint_dir"),
                        "dataset": dataset,
                        "main_mean_length": summary.get("main_mean_length"),
                        "length_field": summary.get("length_field_used_for_main_mean"),
                        "num_records": summary.get("num_records"),
                        "num_responses_with_text": summary.get("num_responses_with_text"),
                        "path": summary.get("path"),
                        "exists": summary.get("exists", False),
                    }
                )

    for dataset in combined:
        combined[dataset].sort(key=lambda r: (r.get("global_step") is None, r.get("global_step") or 0))

    return combined


def build_combined_log_series(phases: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for phase_obj in phases.values():
        for rec in phase_obj.get("training_log_records", []):
            if rec.get("mean_length") is not None:
                rows.append(dict(rec))
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def plot_checkpoint_lengths(
    experiment_root: Path,
    combined: dict[str, list[dict[str, Any]]],
    phase_offsets: dict[str, int],
    phase_order: list[str],
    plot_prefix: str,
    main_length_field: str,
) -> list[str]:
    plotted_any = False
    plt.figure(figsize=(11, 6))

    for dataset, points in combined.items():
        usable = [p for p in points if p.get("main_mean_length") is not None]
        xs = [p["global_step"] for p in usable]
        ys = [p["main_mean_length"] for p in usable]
        if not xs:
            continue
        plotted_any = True
        plt.plot(xs, ys, marker="o", linewidth=1.8, label=f"{dataset} eval responses")

    for phase in phase_order[1:]:
        if phase in phase_offsets:
            plt.axvline(phase_offsets[phase], linestyle="--", alpha=0.5, label=f"start {phase}")

    plt.xlabel("Global checkpoint step")
    plt.ylabel(f"Average response length ({main_length_field})")
    plt.title(f"Eval response length over checkpoints: {experiment_root.parent.name}/{experiment_root.name}")
    plt.grid(True, alpha=0.3)

    if plotted_any:
        plt.legend()
    else:
        plt.text(
            0.5,
            0.5,
            "No eval response files found",
            ha="center",
            va="center",
            transform=plt.gca().transAxes,
        )

    plt.tight_layout()
    png = experiment_root / f"{plot_prefix}_checkpoint_plot.png"
    pdf = experiment_root / f"{plot_prefix}_checkpoint_plot.pdf"
    plt.savefig(png, dpi=200)
    plt.savefig(pdf)
    plt.close()
    return [str(png), str(pdf)]


def plot_training_log_lengths(
    experiment_root: Path,
    combined_log: list[dict[str, Any]],
    phase_order: list[str],
    plot_prefix: str,
) -> list[str]:
    plt.figure(figsize=(11, 6))
    plotted_any = False

    for phase in phase_order:
        points = [r for r in combined_log if r.get("phase") == phase and r.get("mean_length") is not None]
        if not points:
            continue
        points.sort(key=lambda r: (r.get("phase_fraction") is None, r.get("phase_fraction") or 0.0))
        xs = [r.get("phase_fraction") for r in points]
        ys = [r.get("mean_length") for r in points]
        plotted_any = True
        plt.plot(xs, ys, marker=".", linewidth=1.2, label=f"{phase} train completions")

    plt.xlabel("Fraction of phase training")
    plt.ylabel("Completion mean length from training log")
    plt.title(f"Training completion length by phase: {experiment_root.parent.name}/{experiment_root.name}")
    plt.grid(True, alpha=0.3)

    if plotted_any:
        plt.legend()
    else:
        plt.text(
            0.5,
            0.5,
            "No log completion-length records found",
            ha="center",
            va="center",
            transform=plt.gca().transAxes,
        )

    plt.tight_layout()
    png = experiment_root / f"{plot_prefix}_training_log_plot.png"
    pdf = experiment_root / f"{plot_prefix}_training_log_plot.pdf"
    plt.savefig(png, dpi=200)
    plt.savefig(pdf)
    plt.close()
    return [str(png), str(pdf)]


def analyze_experiment(experiment_root: Path, args: argparse.Namespace) -> dict[str, Any]:
    experiment_root = experiment_root.resolve()

    phase_order, phase_order_source = resolve_phase_order(
        experiment_root=experiment_root,
        explicit_dataset_order=args.dataset_order,
        default_datasets=KNOWN_DATASETS,
        force_dataset_order=args.force_dataset_order,
    )
    eval_datasets = discover_eval_datasets(experiment_root, phase_order, args.eval_subset)
    phase_dirs = phase_dirs_for_order(experiment_root, phase_order)
    phase_offsets = phase_offsets_from_checkpoints(phase_dirs, phase_order)

    tokenizer_path = choose_tokenizer_path(
        experiment_root,
        explicit=args.tokenizer,
        auto_from_experiment=args.auto_tokenizer_from_experiment,
    )
    tokenizer = OptionalTokenizer(
        tokenizer_path,
        quiet=args.quiet,
        local_files_only=args.tokenizer_local_files_only,
    )
    main_length_field = get_length_field(args.length_mode, tokenizer.available)

    if args.length_mode == "hf_tokens" and not tokenizer.available:
        raise RuntimeError("length-mode=hf_tokens requested but no tokenizer could be loaded.")

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
            "checkpoints": [],
        }

        if not phase_dir.is_dir():
            warnings.append(f"Missing phase directory: {phase_dir}")
            phases_out[phase] = phase_obj
            continue

        log_files = find_log_files(phase_dir)
        phase_obj["log_files"] = [str(p) for p in log_files]
        for log_file in log_files:
            phase_obj["training_log_records"].extend(parse_training_log(log_file, phase=phase))

        phase_offset = phase_offsets.get(phase, 0)
        for ckpt in sorted_checkpoints(phase_dir):
            phase_obj["checkpoints"].append(
                analyze_checkpoint(
                    ckpt=ckpt,
                    phase=phase,
                    phase_offset=phase_offset,
                    eval_datasets=eval_datasets,
                    response_field=args.response_field,
                    tokenizer=tokenizer,
                    main_length_field=main_length_field,
                    ignore_original=args.ignore_original,
                )
            )

        phases_out[phase] = phase_obj

    combined_checkpoint = build_combined_checkpoint_series(phases_out, eval_datasets)
    combined_log = build_combined_log_series(phases_out)
    plots = {
        "checkpoint": plot_checkpoint_lengths(
            experiment_root,
            combined_checkpoint,
            phase_offsets,
            phase_order,
            args.plot_prefix,
            main_length_field,
        ),
        "training_log": plot_training_log_lengths(
            experiment_root,
            combined_log,
            phase_order,
            args.plot_prefix,
        ),
    }

    summary = {
        "created_at": now_iso(),
        "script": "analyze_response_lengths.py",
        "experiment_root": str(experiment_root),
        "experiment_name": experiment_root.parent.name,
        "version": experiment_root.name,
        "phase_order": phase_order,
        "phase_order_source": phase_order_source,
        "eval_datasets": eval_datasets,
        "phase_offsets": phase_offsets,
        "length_mode_requested": args.length_mode,
        "main_length_field": main_length_field,
        "tokenizer_path_attempted": tokenizer_path,
        "tokenizer_available": tokenizer.available,
        "tokenizer_error": tokenizer.error,
        "warnings": warnings,
        "phases": phases_out,
        "combined_checkpoint_series": combined_checkpoint,
        "combined_training_log_series": combined_log,
        "plots": plots,
    }

    out_path = experiment_root / args.out_name
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    summary["output_json"] = str(out_path)

    if not args.quiet:
        n_logs = len(combined_log)
        n_ckpt_points = sum(
            1
            for pts in combined_checkpoint.values()
            for p in pts
            if p.get("main_mean_length") is not None
        )
        print(
            f"[done] {experiment_root}: "
            f"phase_order={' '.join(phase_order)} ({phase_order_source}), "
            f"log_points={n_logs}, checkpoint_response_points={n_ckpt_points}, json={out_path}"
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

    index_rows = []
    had_error = False

    for root in roots:
        try:
            summary = analyze_experiment(root, args)
            index_rows.append(
                {
                    "experiment_root": str(root),
                    "status": "ok",
                    "output_json": summary.get("output_json"),
                    "phase_order": " ".join(summary.get("phase_order", [])),
                    "phase_order_source": summary.get("phase_order_source"),
                    "eval_datasets": " ".join(summary.get("eval_datasets", [])),
                    "main_length_field": summary.get("main_length_field"),
                    "training_log_points": len(summary.get("combined_training_log_series", [])),
                    "checkpoint_response_points": sum(
                        1
                        for pts in summary.get("combined_checkpoint_series", {}).values()
                        for p in pts
                        if p.get("main_mean_length") is not None
                    ),
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
        write_csv(
            csv_path,
            index_rows,
            [
                "experiment_root",
                "status",
                "output_json",
                "phase_order",
                "phase_order_source",
                "eval_datasets",
                "main_length_field",
                "training_log_points",
                "checkpoint_response_points",
                "warnings",
                "error",
            ],
        )

        if not args.quiet:
            print(f"[index] {index_path}")
            print(f"[index-csv] {csv_path}")

    if had_error:
        sys.exit(1)


if __name__ == "__main__":
    main()
