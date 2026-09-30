#!/usr/bin/env python3
"""Calibrate a model on the original (non-contradicted) Math Contradiction task.

The ``math_contradiction`` dataset stores two aligned worlds in each row:

* the default transformed world (e.g. base 9), used by the contradiction task;
* the original decimal world under ``original_*`` fields.

This script evaluates the *non-privileged student* on the original decimal
question only.  It deliberately reuses the project's normal model-registry,
vLLM, chat-template, Math Contradiction adapter, and verification paths.

For every row, evaluation constructs an explicit original-world view:

    messages    <- original_messages
    problem     <- original_problem
    answer      <- original_answer
    output_text <- original_output_text (when available)
    base        <- 10

The explicit projection is important.  Some registered model families rebuild
chat prompts from the raw row during evaluation; changing only the answer would
risk sending the transformed/base-N prompt while scoring against the decimal
answer.

Typical use
-----------
Base Qwen2.5 on the same eval split used by the contradiction experiments::

    python -m data_utils.math_contradiction.calibrate_math_contradiction_original \
      --model qwen2.5-7b \
      --seeds 31 37 717

A trained student checkpoint::

    python -m data_utils.math_contradiction.calibrate_math_contradiction_original \
      --model outputs/.../checkpoint-120 \
      --seeds 31

The default generation temperature is 0, matching project dataset baseline
calibration.  Consequently, identical results across seeds are expected and are
not an error; the multi-seed wrapper is retained for protocol symmetry and for
non-zero-temperature diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import random
import re
import shlex
import statistics
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


DEFAULT_SEEDS = (31, 37, 717)
DATASET_NAME = "math_contradiction"
REPORT_VERSION = 1


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(message: str) -> None:
    print(f"[{timestamp()}] {message}", flush=True)


def locate_repo_root(script_path: Path) -> Path:
    """Find the project root by walking upward from the script location."""
    required = ("model_registry.py", "dataset_adapters.py", "eval_lib.py")

    for candidate in (script_path.parent, *script_path.parents):
        if all((candidate / name).is_file() for name in required):
            return candidate.resolve()

    raise FileNotFoundError(
        f"Could not locate repository root from {script_path}. Expected {required}."
    )


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def package_versions() -> dict[str, str | None]:
    return {
        name: package_version(name)
        for name in (
            "torch",
            "transformers",
            "vllm",
            "tokenizers",
            "accelerate",
            "datasets",
            "trl",
            "math-verify",
        )
    }


def git_provenance(repo_root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"commit": None, "dirty": None}
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty_output = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        result = {"commit": commit or None, "dirty": bool(dirty_output.strip())}
    except (OSError, subprocess.SubprocessError):
        pass
    return result


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent), text=True
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except Exception:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass
        raise


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def safe_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    return slug.strip("._-") or "model"



def model_run_slug(requested_model: str, spec: Any) -> str:
    """Return a collision-resistant output slug, including local checkpoints."""
    path = Path(requested_model).expanduser()
    if path.exists():
        resolved = str(path.resolve())
        digest = hashlib.sha256(resolved.encode("utf-8")).hexdigest()[:10]
        return f"{safe_slug(spec.key)}__{safe_slug(path.name)}__{digest}"
    return safe_slug(spec.key)

def set_host_seeds(seed: int) -> None:
    """Seed host RNGs without touching CUDA before vLLM starts."""
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass


def summary_stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            "n": 0,
            "mean": None,
            "sample_std": None,
            "min": None,
            "max": None,
            "values": [],
        }
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "sample_std": statistics.stdev(values) if len(values) >= 2 else 0.0,
        "min": min(values),
        "max": max(values),
        "values": list(values),
    }


def validate_seeds(seeds: list[int]) -> list[int]:
    if not seeds:
        raise ValueError("At least one seed is required.")
    if len(seeds) != len(set(seeds)):
        raise ValueError(f"Seeds must be unique, got {seeds}.")
    if any(seed < 0 for seed in seeds):
        raise ValueError(f"Seeds must be non-negative integers, got {seeds}.")
    return seeds


def worker_environment(seed: int, visible_devices: str) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = visible_devices
    env["VISIBLE_DEVICES"] = visible_devices
    env["PYTHONHASHSEED"] = str(seed)
    env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    env["WANDB_MODE"] = "offline"
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    # Preserve the caller's corporate/custom CA setup if present.
    request_bundle = env.get("REQUESTS_CA_BUNDLE")
    if request_bundle:
        env.setdefault("SSL_CERT_FILE", request_bundle)
        env.setdefault("CURL_CA_BUNDLE", request_bundle)
    return env


def stream_subprocess(
    *,
    command: list[str],
    cwd: Path,
    env: dict[str, str],
    log_path: Path,
    command_path: Path,
    dry_run: bool,
) -> None:
    command_path.parent.mkdir(parents=True, exist_ok=True)
    command_path.write_text(shlex.join(command) + "\n", encoding="utf-8")
    log(f"Command: {shlex.join(command)}")
    if dry_run:
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", buffering=1) as log_file:
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
        returncode = process.wait()
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, command)


def shutdown_vllm_engine(llm: Any, *, timeout_seconds: float = 30.0) -> None:
    """Best-effort explicit vLLM shutdown, matching baseline calibration."""
    if llm is None:
        return

    llm_engine = getattr(llm, "llm_engine", None)
    engine_core = getattr(llm_engine, "engine_core", None)
    for obj in (engine_core, llm_engine, llm):
        if obj is None:
            continue
        for method_name in ("shutdown", "close"):
            method = getattr(obj, method_name, None)
            if not callable(method):
                continue
            try:
                if method_name == "shutdown":
                    try:
                        method(timeout=timeout_seconds)
                    except TypeError:
                        method()
                else:
                    method()
            except Exception as exc:
                log(
                    "Warning: vLLM shutdown via "
                    f"{obj.__class__.__name__}.{method_name}() failed: {exc}"
                )
                continue
            return


def resolve_split_path(adapter: Any, split: str, data_path: Path | None) -> Path:
    if data_path is not None:
        return data_path.expanduser().resolve()
    if split == "train":
        return Path(adapter.train_path)
    if split == "eval":
        return Path(adapter.eval_path)
    if split == "test":
        return Path(adapter.eval_path).parent / "test_data"
    raise ValueError(f"Unsupported split: {split!r}")


def _require_nonempty(row: dict[str, Any], key: str, row_index: int) -> Any:
    value = row.get(key)
    if value in (None, "", []):
        raise ValueError(
            f"Row {row_index} has no usable {key!r}; cannot construct base-10 control."
        )
    return value


def project_original_world(row: dict[str, Any], row_index: int) -> dict[str, Any]:
    """Return a copy whose canonical task fields describe the decimal world."""
    original_messages = _require_nonempty(row, "original_messages", row_index)
    original_problem = str(_require_nonempty(row, "original_problem", row_index))
    original_answer = str(_require_nonempty(row, "original_answer", row_index))

    if not isinstance(original_messages, list) or not all(
        isinstance(message, dict)
        and "role" in message
        and "content" in message
        for message in original_messages
    ):
        raise TypeError(
            f"Row {row_index} original_messages is not a valid chat-message list."
        )

    projected = dict(row)
    projected["messages"] = [dict(message) for message in original_messages]
    projected["problem"] = original_problem
    projected["answer"] = original_answer
    projected["base"] = 10

    # Not used for inference/scoring, but keeping the canonical solution in the
    # same world makes the projected row internally coherent for downstream
    # inspection and any future adapter hooks.
    original_output = row.get("original_output_text")
    if original_output not in (None, ""):
        projected["output_text"] = str(original_output)

    return projected


def grouped_accuracy(
    response_records: list[dict[str, Any]], key: str
) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[int]] = {}
    for record in response_records:
        value = record.get(key)
        label = "<missing>" if value in (None, "") else str(value)
        buckets.setdefault(label, []).append(int(record["score"]))

    return {
        label: {
            "accuracy": statistics.fmean(scores) if scores else 0.0,
            "num_correct": int(sum(scores)),
            "num_total": len(scores),
        }
        for label, scores in sorted(buckets.items())
    }


def _worker(args: argparse.Namespace, repo_root: Path) -> None:
    """Evaluate one seed in a fresh process to isolate vLLM/CUDA state."""
    sys.path.insert(0, str(repo_root))

    from datasets import Dataset, load_from_disk
    from dataset_adapters import get_dataset_adapter
    from eval_lib import (
        _prepare_vllm_eval_prompts,
        generate_responses_vllm,
        load_model_and_tokenizer_vllm,
    )
    from model_registry import resolve_model_spec

    seed = int(args._worker_seed)
    set_host_seeds(seed)

    adapter = get_dataset_adapter(DATASET_NAME)
    split_path = resolve_split_path(adapter, args.split, args.data_path)
    if not split_path.exists():
        raise FileNotFoundError(f"Dataset split not found: {split_path}")

    source_dataset = load_from_disk(str(split_path))
    if args.limit is not None:
        limit = min(int(args.limit), len(source_dataset))
        source_dataset = source_dataset.select(range(limit))

    source_rows = source_dataset.to_list()
    projected_rows = [
        project_original_world(dict(row), row_index)
        for row_index, row in enumerate(source_rows)
    ]
    original_dataset = Dataset.from_list(projected_rows)

    spec = resolve_model_spec(args.model)
    effective_max_model_len = (
        args.max_model_len
        if args.max_model_len is not None
        else adapter.default_max_model_len
    )
    max_new_tokens = (
        args.max_new_tokens
        if args.max_new_tokens is not None
        else adapter.default_max_new_tokens
    )

    seed_dir = Path(args.run_dir) / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    log(
        f"Original-math seed={seed}; model={spec.key}; split={args.split}; "
        f"rows={len(original_dataset)}; max_model_len={effective_max_model_len}; "
        f"max_new_tokens={max_new_tokens}"
    )

    llm = None
    tokenizer = None
    try:
        llm, tokenizer = load_model_and_tokenizer_vllm(
            model_path=args.model,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=effective_max_model_len,
            tensor_parallel_size=args.tensor_parallel_size,
            seed=seed,
        )

        # Reuse the Math Contradiction adapter after replacing its canonical
        # fields with original-world fields.  References therefore become
        # {"answer": original_answer, "base": 10}, and the existing verifier
        # performs exact base-10 scoring.
        eval_output = adapter.build_eval_output(original_dataset, tokenizer)
        model_prompts = _prepare_vllm_eval_prompts(
            adapter=adapter,
            tokenizer=tokenizer,
            eval_output=eval_output,
            model_spec=spec,
        )
        responses = generate_responses_vllm(
            llm=llm,
            tokenizer=tokenizer,
            prompts=model_prompts,
            max_new_tokens=max_new_tokens,
            temperature=args.temperature,
        )

        summary = adapter.evaluate(responses, eval_output.references)
        response_records = adapter.build_response_records(
            prompts=eval_output.prompts,
            responses=responses,
            references=eval_output.references,
            scores=summary["per_sample_scores"],
            raw_examples=eval_output.raw_examples,
        )

        # Preserve the transformed counterpart as provenance without ever
        # exposing it to generation.
        for index, record in enumerate(response_records):
            source = source_rows[index]
            record["evaluation_world"] = "original_base10"
            record["transformed_problem"] = source.get("problem", "")
            record["transformed_answer"] = source.get("answer", "")
            record["transformed_base"] = source.get("base")

        breakdowns = {
            "module": grouped_accuracy(response_records, "module"),
            "difficulty": grouped_accuracy(response_records, "difficulty"),
            "bucket": grouped_accuracy(response_records, "bucket"),
        }

        results_payload = {
            **summary,
            "evaluation_world": "original_base10",
            "split": args.split,
            "split_path": str(split_path),
            "config": {
                "model": args.model,
                "model_key": spec.key,
                "seed": seed,
                "temperature": args.temperature,
                "max_new_tokens": max_new_tokens,
                "max_model_len": effective_max_model_len,
                "tensor_parallel_size": args.tensor_parallel_size,
                "gpu_memory_utilization": args.gpu_memory_utilization,
                "limit": args.limit,
            },
            "breakdowns": breakdowns,
        }
        atomic_write_json(seed_dir / "original_math_results.json", results_payload)
        atomic_write_json(seed_dir / "original_math_responses.json", response_records)

        worker_summary = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "seed": seed,
            "model": {
                "requested": args.model,
                "key": spec.key,
                "hf_repo_id": spec.hf_repo_id,
                "family": spec.family.value,
            },
            "evaluation_world": "original_base10",
            "split": args.split,
            "split_path": str(split_path),
            "accuracy": float(summary["accuracy"]),
            "num_correct": int(summary["num_correct"]),
            "num_total": int(summary["num_total"]),
            "breakdowns": breakdowns,
            "protocol": {
                "dataset": DATASET_NAME,
                "prompt_source": "original_messages",
                "problem_source": "original_problem",
                "reference_source": "original_answer",
                "scoring_base": 10,
                "privileged_context": False,
                "generation": "eval_lib.generate_responses_vllm",
                "prompt_preparation": "eval_lib._prepare_vllm_eval_prompts",
                "scoring": "MathContradictionAdapter.evaluate with base=10 references",
                "temperature": args.temperature,
                "max_new_tokens": max_new_tokens,
                "effective_max_model_len": effective_max_model_len,
                "vllm_seed": seed,
            },
            "packages": package_versions(),
        }
        atomic_write_json(seed_dir / "seed_summary.json", worker_summary)
    finally:
        shutdown_vllm_engine(llm)
        try:
            del tokenizer
        except Exception:
            pass


def aggregate_group(
    seed_summaries: list[dict[str, Any]], group_name: str
) -> dict[str, Any]:
    labels: set[str] = set()
    for summary in seed_summaries:
        labels.update((summary.get("breakdowns", {}).get(group_name, {}) or {}).keys())

    result: dict[str, Any] = {}
    for label in sorted(labels):
        per_seed: dict[str, dict[str, Any]] = {}
        accuracies: list[float] = []
        totals: set[int] = set()
        for summary in seed_summaries:
            seed = str(summary["seed"])
            item = (summary.get("breakdowns", {}).get(group_name, {}) or {}).get(label)
            if item is None:
                continue
            per_seed[seed] = item
            accuracies.append(float(item["accuracy"]))
            totals.add(int(item["num_total"]))
        result[label] = {
            "accuracy": summary_stats(accuracies),
            "num_rows_per_seed": sorted(totals),
            "per_seed": per_seed,
        }
    return result


def build_report(
    *,
    args: argparse.Namespace,
    repo_root: Path,
    spec: Any,
    run_dir: Path,
) -> dict[str, Any]:
    seed_summaries = [
        load_json(run_dir / f"seed_{seed}" / "seed_summary.json")
        for seed in args.seeds
    ]
    accuracies = [float(summary["accuracy"]) for summary in seed_summaries]
    num_totals = {int(summary["num_total"]) for summary in seed_summaries}
    if len(num_totals) != 1:
        raise RuntimeError(f"Seed runs disagree on evaluated row count: {num_totals}")

    return {
        "report_version": REPORT_VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model": {
            "requested": args.model,
            "key": spec.key,
            "hf_repo_id": spec.hf_repo_id,
            "family": spec.family.value,
        },
        "dataset": DATASET_NAME,
        "evaluation_world": "original_base10",
        "split": args.split,
        "seeds": list(args.seeds),
        "num_rows": next(iter(num_totals)),
        "accuracy": summary_stats(accuracies),
        "per_seed": {
            str(summary["seed"]): {
                "accuracy": summary["accuracy"],
                "num_correct": summary["num_correct"],
                "num_total": summary["num_total"],
            }
            for summary in seed_summaries
        },
        "breakdowns": {
            group_name: aggregate_group(seed_summaries, group_name)
            for group_name in ("module", "difficulty", "bucket")
        },
        "protocol": {
            "task": (
                "Solve the original decimal question without privileged context; "
                "score the final boxed answer against original_answer in base 10."
            ),
            "row_projection": {
                "messages": "original_messages",
                "problem": "original_problem",
                "answer": "original_answer",
                "output_text": "original_output_text when present",
                "base": 10,
            },
            "temperature": args.temperature,
            "max_new_tokens": args.max_new_tokens,
            "max_model_len_override": args.max_model_len,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "tensor_parallel_size": args.tensor_parallel_size,
            "limit": args.limit,
            "seed_policy": (
                "Python/NumPy host RNGs and vLLM engine seed use the replicate seed; "
                "CUDA is not initialized before vLLM starts."
            ),
        },
        "environment": {
            "python_executable": sys.executable,
            "python_version": sys.version.split()[0],
            "packages": package_versions(),
            "git": git_provenance(repo_root),
        },
        "notes": [
            "This is an original/base-10 control, not the transformed Math Contradiction accuracy stored in model_registry.py.",
            "No reference answer, reference solution, base-conversion rule, or teacher context is included in the student prompt.",
            "With temperature=0, zero seed variance is expected and valid.",
        ],
    }


def write_report_files(report: dict[str, Any], run_dir: Path) -> None:
    atomic_write_json(run_dir / "original_math_report.json", report)

    rows: list[dict[str, Any]] = []
    overall = report["accuracy"]
    rows.append(
        {
            "group": "overall",
            "label": "all",
            "mean_accuracy": overall["mean"],
            "sample_std": overall["sample_std"],
            "min": overall["min"],
            "max": overall["max"],
            "n_seeds": overall["n"],
            "num_rows": report["num_rows"],
        }
    )
    for group_name, group_values in report["breakdowns"].items():
        for label, payload in group_values.items():
            stats = payload["accuracy"]
            row_counts = payload["num_rows_per_seed"]
            rows.append(
                {
                    "group": group_name,
                    "label": label,
                    "mean_accuracy": stats["mean"],
                    "sample_std": stats["sample_std"],
                    "min": stats["min"],
                    "max": stats["max"],
                    "n_seeds": stats["n"],
                    "num_rows": row_counts[0] if len(row_counts) == 1 else repr(row_counts),
                }
            )

    csv_path = run_dir / "original_math_report.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "group",
                "label",
                "mean_accuracy",
                "sample_std",
                "min",
                "max",
                "n_seeds",
                "num_rows",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    mean = report["accuracy"]["mean"]
    std = report["accuracy"]["sample_std"]
    markdown = [
        "# Original Math Contradiction control",
        "",
        f"- Model: `{report['model']['key']}` (`{report['model']['requested']}`)",
        f"- Split: `{report['split']}`",
        f"- Rows: {report['num_rows']}",
        f"- Seeds: {', '.join(map(str, report['seeds']))}",
        "- World: original decimal/base-10, non-privileged student prompt",
        f"- Accuracy: **{mean:.6f}** (sample std {std:.6f})",
        "",
        "Per-seed responses and verifier details are stored under each `seed_<seed>/` directory.",
        "",
    ]
    (run_dir / "original_math_report.md").write_text("\n".join(markdown), encoding="utf-8")


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model",
        default="qwen2.5-7b",
        help=(
            "Registered model key, Hugging Face repo id, or local Trainer checkpoint. "
            "Default: qwen2.5-7b."
        ),
    )
    parser.add_argument(
        "--split",
        choices=("train", "eval", "test"),
        default="eval",
        help="Dataset split to evaluate. Default: eval.",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Optional explicit load_from_disk path; overrides --split path resolution.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Default: <repo>/baseline_results/math_contradiction_original.",
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Generation temperature. Keep 0 for baseline-compatible greedy accuracy.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional prefix limit for smoke tests. Omit for the full split.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip completed seed workers when configuration is compatible.",
    )
    parser.add_argument("--dry-run", action="store_true")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a model on the original base-10 questions aligned with the "
            "Math Contradiction dataset."
        )
    )
    add_arguments(parser)
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--_worker-seed", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--_run-dir", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args()


def worker_command(args: argparse.Namespace, seed: int, run_dir: Path) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--model",
        args.model,
        "--split",
        args.split,
        "--seeds",
        *[str(value) for value in args.seeds],
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--temperature",
        str(args.temperature),
        "--_worker",
        "--_worker-seed",
        str(seed),
        "--_run-dir",
        str(run_dir),
    ]
    if args.data_path is not None:
        command.extend(["--data-path", str(args.data_path)])
    if args.max_model_len is not None:
        command.extend(["--max-model-len", str(args.max_model_len)])
    if args.max_new_tokens is not None:
        command.extend(["--max-new-tokens", str(args.max_new_tokens)])
    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])
    return command


def config_signature(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": config.get("model"),
        "split": config.get("split"),
        "data_path": config.get("data_path"),
        "seeds": config.get("seeds"),
        "temperature": config.get("temperature"),
        "gpu_memory_utilization": config.get("gpu_memory_utilization"),
        "tensor_parallel_size": config.get("tensor_parallel_size"),
        "max_model_len": config.get("max_model_len"),
        "max_new_tokens": config.get("max_new_tokens"),
        "limit": config.get("limit"),
        "packages": config.get("packages"),
        "git_commit": (config.get("git") or {}).get("commit"),
    }


def validate_resume_config(config_path: Path, new_config: dict[str, Any]) -> None:
    if not config_path.is_file():
        return
    old_config = load_json(config_path)
    if config_signature(old_config) != config_signature(new_config):
        raise RuntimeError(
            "Existing calibration_config.json is incompatible with this run. "
            "Use a different --output-root or rerun with --no-resume after "
            "intentionally replacing the old outputs.\n"
            f"Existing: {config_signature(old_config)}\n"
            f"Requested: {config_signature(new_config)}"
        )


def main() -> None:
    args = parse_args()
    script_path = Path(__file__).resolve()
    repo_root = locate_repo_root(script_path)
    sys.path.insert(0, str(repo_root))

    if args._worker:
        if args._worker_seed is None or args._run_dir is None:
            raise ValueError("Internal worker mode requires --_worker-seed and --_run-dir.")
        args.run_dir = Path(args._run_dir).resolve()
        _worker(args, repo_root)
        return

    from model_registry import check_tensor_parallel_compatibility, resolve_model_spec

    args.seeds = validate_seeds(list(args.seeds))
    if not (0.0 < args.gpu_memory_utilization <= 1.0):
        raise ValueError("--gpu-memory-utilization must be in (0, 1].")
    if args.tensor_parallel_size <= 0:
        raise ValueError("--tensor-parallel-size must be positive.")
    if args.max_model_len is not None and args.max_model_len <= 0:
        raise ValueError("--max-model-len must be positive when provided.")
    if args.max_new_tokens is not None and args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive when provided.")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive when provided.")
    if not math.isfinite(args.temperature) or args.temperature < 0:
        raise ValueError("--temperature must be a finite non-negative number.")

    spec = resolve_model_spec(args.model)
    check_tensor_parallel_compatibility(spec, args.tensor_parallel_size)

    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root is not None
        else repo_root / "baseline_results" / "math_contradiction_original"
    )
    run_dir = output_root / model_run_slug(args.model, spec)
    run_dir.mkdir(parents=True, exist_ok=True)
    args.run_dir = run_dir

    visible_devices = os.environ.get(
        "VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    ).strip()
    if not visible_devices:
        raise ValueError("VISIBLE_DEVICES/CUDA_VISIBLE_DEVICES resolved to empty.")

    config = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model": {
            "requested": args.model,
            "key": spec.key,
            "hf_repo_id": spec.hf_repo_id,
            "family": spec.family.value,
        },
        "dataset": DATASET_NAME,
        "evaluation_world": "original_base10",
        "split": args.split,
        "data_path": str(args.data_path.expanduser().resolve()) if args.data_path else None,
        "seeds": list(args.seeds),
        "temperature": args.temperature,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "tensor_parallel_size": args.tensor_parallel_size,
        "max_model_len": args.max_model_len,
        "max_new_tokens": args.max_new_tokens,
        "limit": args.limit,
        "visible_devices": visible_devices,
        "packages": package_versions(),
        "git": git_provenance(repo_root),
    }
    config_path = run_dir / "calibration_config.json"
    if args.resume:
        validate_resume_config(config_path, config)
    atomic_write_json(config_path, config)

    log(f"Repository root: {repo_root}")
    log(f"Model: {spec.key} ({spec.hf_repo_id})")
    log(f"Dataset: {DATASET_NAME}, original/base-10 world, split={args.split}")
    log(f"Seeds: {args.seeds}")
    log(f"Output: {run_dir}")

    for seed in args.seeds:
        done_path = run_dir / f"seed_{seed}" / "seed_summary.json"
        if args.resume and done_path.is_file():
            log(f"Skipping completed seed {seed}: {done_path}")
            continue

        command = worker_command(args, seed, run_dir)
        stream_subprocess(
            command=command,
            cwd=repo_root,
            env=worker_environment(seed, visible_devices),
            log_path=run_dir / "logs" / f"seed_{seed}.log",
            command_path=run_dir / "commands" / f"seed_{seed}.txt",
            dry_run=args.dry_run,
        )

    if args.dry_run:
        log("Dry run complete; no aggregate report was written.")
        return

    report = build_report(args=args, repo_root=repo_root, spec=spec, run_dir=run_dir)
    write_report_files(report, run_dir)
    log(f"Wrote report: {run_dir / 'original_math_report.json'}")
    log(f"Wrote CSV: {run_dir / 'original_math_report.csv'}")
    log(f"Wrote Markdown: {run_dir / 'original_math_report.md'}")


if __name__ == "__main__":
    main()
