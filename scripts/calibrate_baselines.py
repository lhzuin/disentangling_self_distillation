#!/usr/bin/env python3
"""Calibrate untouched-model baselines for SDFT datasets and lm-eval.

The script measures a registered *base model before training* using the same
project evaluation machinery used for checkpoints:

* dataset metrics use ``eval_lib.load_model_and_tokenizer_vllm`` and
  ``eval_lib.run_loaded_eval``;
* model-family behavior comes from ``model_registry.py``;
* lm-eval model arguments reuse ``scripts.run_lmeval.build_model_args``;
* lm-eval metric names/extraction reuse ``scripts.summarize_lmeval``.

Each seed runs in a fresh subprocess. This avoids cross-run CUDA/vLLM state and
makes each replicate independently reproducible. The final report contains the
per-seed measurements, mean, sample standard deviation, range, environment
provenance, and registry-ready baseline dictionaries.

Typical use
-----------
    python -m scripts.calibrate_baselines \\
      --model ministral-3-3b \\
      --datasets tooluse science math_contradiction spatial_contradiction2 spatial_standard2 \\
      --seeds 31 37 717

The companion ``scripts/run_baseline_calibration.sh`` provides exactly this
Ministral-3B calibration as its no-argument default while remaining reusable.
"""

from __future__ import annotations

import argparse
import csv
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
DEFAULT_LM_EVAL_TASKS = (
    "hellaswag",
    "mmlu",
    "truthfulqa",
    "winogrande",
    "humaneval",
    "ifeval",
)
REPORT_VERSION = 1


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(message: str) -> None:
    print(f"[{timestamp()}] {message}", flush=True)


def locate_repo_root(script_path: Path) -> Path:
    """Find the repository root from a script living under ``scripts/``."""
    candidates = (script_path.parent.parent, script_path.parent)
    required = ("model_registry.py", "dataset_adapters.py", "eval_lib.py")
    for candidate in candidates:
        if all((candidate / name).is_file() for name in required):
            return candidate.resolve()
    raise FileNotFoundError(
        f"Could not locate repository root from {script_path}. Expected: {required}."
    )


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def package_versions() -> dict[str, str | None]:
    names = (
        "lm_eval",
        "torch",
        "transformers",
        "vllm",
        "tokenizers",
        "accelerate",
        "datasets",
        "peft",
        "trl",
    )
    return {name: package_version(name) for name in names}


def git_provenance(repo_root: Path) -> dict[str, Any]:
    """Best-effort git provenance without making git a hard dependency."""
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
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return slug.strip("._-") or "model"


def set_host_seeds(seed: int) -> None:
    """Seed host RNGs without initializing accelerator state.

    vLLM owns accelerator initialization inside its engine process. Touching
    ``torch.cuda`` before vLLM starts can make a fork-based engine launch fail
    with ``Cannot re-initialize CUDA in forked subprocess``.

    Python and NumPy are safe to seed here. Accelerator/model RNGs are seeded
    through vLLM's explicit ``seed`` argument; lm-eval additionally receives
    its own Python/NumPy/Torch/few-shot seeds in ``simple_evaluate``.
    """
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass


def parse_batch_size(value: str) -> str | int:
    text = str(value).strip()
    if not text:
        return "auto"
    if "auto" in text:
        return text
    try:
        parsed = int(text)
    except ValueError as exc:
        raise ValueError(
            f"--lm-eval-batch-size must be 'auto' or a positive integer, got {value!r}."
        ) from exc
    if parsed <= 0:
        raise ValueError("--lm-eval-batch-size must be positive.")
    return parsed


def mean_or_none(values: Iterable[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return statistics.fmean(clean) if clean else None


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


def validate_unique_positive_seeds(seeds: list[int]) -> list[int]:
    if not seeds:
        raise ValueError("At least one seed is required.")
    if len(seeds) != len(set(seeds)):
        raise ValueError(f"Seeds must be unique, got {seeds}.")
    if any(seed < 0 for seed in seeds):
        raise ValueError(f"Seeds must be non-negative integers, got {seeds}.")
    return seeds


def max_model_len_for_datasets(dataset_names: list[str], get_dataset_adapter: Any) -> int | None:
    """Match run_all_checkpoint_evals.py: one engine sized for all requested datasets."""
    values = [
        get_dataset_adapter(name).default_max_model_len
        for name in dataset_names
        if get_dataset_adapter(name).default_max_model_len is not None
    ]
    return max(values) if values else None


def worker_environment(seed: int, visible_devices: str) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = visible_devices
    env["VISIBLE_DEVICES"] = visible_devices
    env["PYTHONHASHSEED"] = str(seed)

    # These isolated workers are protected by the module's ``__main__`` guard.
    # Use spawn explicitly so a future import that inspects CUDA cannot poison
    # vLLM's engine process through inherited forked CUDA state.
    env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    env["WANDB_MODE"] = "offline"
    env["HF_ALLOW_CODE_EVAL"] = "1"
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    # Respect the caller's CA configuration; never hardcode a machine/company path.
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


def shutdown_vllm_engine(
    llm: Any,
    *,
    timeout_seconds: float = 30.0,
) -> None:
    """Shut down a standalone vLLM engine and its background processes.

    vLLM V1 runs EngineCore in a background process. Explicit shutdown avoids
    leaving that process alive when a calibration worker exits, which can keep
    inherited stdout pipes open and prevent the parent orchestrator from
    observing worker completion.
    """
    if llm is None:
        return

    llm_engine = getattr(llm, "llm_engine", None)
    engine_core = getattr(llm_engine, "engine_core", None)

    # Prefer the EngineCore client because it owns the background EngineCore
    # process in vLLM V1. Fall back to higher-level hooks for compatibility
    # with alternative vLLM layouts.
    for obj in (engine_core, llm_engine, llm):
        if obj is None:
            continue

        for method_name in ("shutdown", "close"):
            method = getattr(obj, method_name, None)
            if not callable(method):
                continue

            log(
                f"Shutting down vLLM via "
                f"{obj.__class__.__name__}.{method_name}()"
            )

            try:
                if method_name == "shutdown":
                    try:
                        method(timeout=timeout_seconds)
                    except TypeError:
                        # Compatibility with versions whose shutdown()
                        # does not expose a timeout argument.
                        method()
                else:
                    method()
            except Exception as exc:
                log(
                    f"Warning: vLLM shutdown via "
                    f"{obj.__class__.__name__}.{method_name}() failed: {exc}"
                )
                continue

            log("vLLM shutdown complete")
            return

    log("Warning: no explicit vLLM shutdown hook was found")


def _dataset_worker(args: argparse.Namespace, repo_root: Path) -> None:
    sys.path.insert(0, str(repo_root))
    from dataset_adapters import get_dataset_adapter, get_evaluation_dataset_names
    from eval_lib import load_model_and_tokenizer_vllm, run_loaded_eval
    from model_registry import resolve_model_spec

    seed = int(args._worker_seed)
    set_host_seeds(seed)

    available = set(get_evaluation_dataset_names())
    unknown = [name for name in args.datasets if name not in available]
    if unknown:
        raise ValueError(f"Unknown/non-evaluation dataset(s): {unknown}. Available: {sorted(available)}")

    spec = resolve_model_spec(args.model)
    seed_dir = Path(args.run_dir) / "dataset_eval" / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    effective_max_model_len = (
        args.max_model_len
        if args.max_model_len is not None
        else max_model_len_for_datasets(args.datasets, get_dataset_adapter)
    )

    log(
        f"Dataset baseline seed={seed}; model={spec.key}; datasets={args.datasets}; "
        f"max_model_len={effective_max_model_len}"
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

        compact_results: dict[str, Any] = {}
        for dataset_name in args.datasets:
            adapter = get_dataset_adapter(dataset_name)
            summary = run_loaded_eval(
                dataset_name=dataset_name,
                llm=llm,
                tokenizer=tokenizer,
                model_path=args.model,
                output_dir=str(seed_dir),
                max_new_tokens=adapter.default_max_new_tokens,
                temperature=args.temperature,
            )
            compact_results[dataset_name] = {
                "accuracy": float(summary["accuracy"]),
                "num_correct": int(summary["num_correct"]),
                "num_total": int(summary["num_total"]),
                "max_new_tokens": int(adapter.default_max_new_tokens),
                "max_model_len": adapter.default_max_model_len,
            }

        worker_summary = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "seed": seed,
            "model": {
                "requested": args.model,
                "key": spec.key,
                "hf_repo_id": spec.hf_repo_id,
                "family": spec.family.value,
            },
            "protocol": {
                "source": "eval_lib.run_loaded_eval",
                "temperature": args.temperature,
                "gpu_memory_utilization": args.gpu_memory_utilization,
                "tensor_parallel_size": args.tensor_parallel_size,
                "effective_max_model_len": effective_max_model_len,
                "vllm_seed": seed,
            },
            "packages": package_versions(),
            "datasets": compact_results,
        }
        atomic_write_json(seed_dir / "dataset_seed_summary.json", worker_summary)
    finally:
        shutdown_vllm_engine(llm)

def _lm_eval_worker(args: argparse.Namespace, repo_root: Path) -> None:
    sys.path.insert(0, str(repo_root))

    from model_registry import (
        check_tensor_parallel_compatibility,
        resolve_model_spec,
        vllm_eval_engine_kwargs,
    )
    from scripts.run_lmeval import (
        build_model_args,
        ensure_expected_lm_eval_version,
        ensure_nltk_resources,
    )

    seed = int(args._worker_seed)
    set_host_seeds(seed)
    lm_eval_version = ensure_expected_lm_eval_version(dry_run=False)

    spec = resolve_model_spec(args.model)
    check_tensor_parallel_compatibility(spec, args.tensor_parallel_size)
    model_args, _ = build_model_args(
        spec=spec,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.lm_eval_max_model_len,
        vllm_eval_engine_kwargs=vllm_eval_engine_kwargs,
    )
    # lm-eval's vLLM backend otherwise defaults to seed=1234. Make the model
    # seed explicit so the requested replicate seed controls every RNG layer.
    model_args["seed"] = seed

    task_string = ",".join(args.lm_eval_tasks)
    ensure_nltk_resources(task_string, dry_run=False)

    from lm_eval import simple_evaluate
    from lm_eval.utils import handle_non_serializable

    batch_size = parse_batch_size(args.lm_eval_batch_size)
    log(
        f"lm-eval baseline seed={seed}; model={spec.key}; tasks={args.lm_eval_tasks}; "
        f"model_args={model_args}"
    )

    results = simple_evaluate(
        model="vllm",
        model_args=model_args,
        tasks=list(args.lm_eval_tasks),
        batch_size=batch_size,
        apply_chat_template=args.lm_eval_apply_chat_template,
        confirm_run_unsafe_code=True,
        random_seed=seed,
        numpy_random_seed=seed,
        torch_random_seed=seed,
        fewshot_random_seed=seed,
        log_samples=False,
    )
    if results is None:
        raise RuntimeError("lm_eval.simple_evaluate returned no results on the main process.")

    seed_dir = Path(args.run_dir) / "lm_eval" / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    raw_path = seed_dir / "lm_eval_results.json"
    with raw_path.open("w", encoding="utf-8") as handle:
        json.dump(
            results,
            handle,
            indent=2,
            ensure_ascii=False,
            default=handle_non_serializable,
        )
        handle.write("\n")

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "seed": seed,
        "model": {
            "requested": args.model,
            "key": spec.key,
            "hf_repo_id": spec.hf_repo_id,
            "family": spec.family.value,
        },
        "protocol": {
            "lm_eval_version": lm_eval_version,
            "backend": "vllm",
            "model_args": model_args,
            "tasks": list(args.lm_eval_tasks),
            "batch_size": batch_size,
            "apply_chat_template": args.lm_eval_apply_chat_template,
            "random_seed": seed,
            "numpy_random_seed": seed,
            "torch_random_seed": seed,
            "fewshot_random_seed": seed,
            "confirm_run_unsafe_code": True,
        },
        "packages": package_versions(),
        "raw_results": str(raw_path),
    }
    atomic_write_json(seed_dir / "lm_eval_environment.json", metadata)


def _aggregate(args: argparse.Namespace, repo_root: Path) -> dict[str, Any]:
    sys.path.insert(0, str(repo_root))
    from model_registry import resolve_model_spec
    from scripts.summarize_lmeval import (
        METRIC_SPECS,
        PRIOR_TASK_KEYS,
        get_metric_stderr,
        get_metric_value,
    )

    spec = resolve_model_spec(args.model)
    run_dir = Path(args.run_dir)

    dataset_aggregate: dict[str, Any] = {}
    if args.run_dataset_eval:
        for dataset_name in args.datasets:
            per_seed = []
            values = []
            expected_total: int | None = None
            for seed in args.seeds:
                summary_path = run_dir / "dataset_eval" / f"seed_{seed}" / "dataset_seed_summary.json"
                if not summary_path.is_file():
                    raise FileNotFoundError(f"Missing dataset seed summary: {summary_path}")
                record = load_json(summary_path)["datasets"][dataset_name]
                total = int(record["num_total"])
                if expected_total is None:
                    expected_total = total
                elif total != expected_total:
                    raise ValueError(
                        f"Dataset {dataset_name} num_total changed across seeds: "
                        f"{expected_total} vs {total}."
                    )
                accuracy = float(record["accuracy"])
                values.append(accuracy)
                per_seed.append(
                    {
                        "seed": seed,
                        "accuracy": accuracy,
                        "num_correct": int(record["num_correct"]),
                        "num_total": total,
                    }
                )
            stats = summary_stats(values)
            dataset_aggregate[dataset_name] = {
                **stats,
                "baseline_accuracy": stats["mean"],
                "num_total": expected_total,
                "per_seed": per_seed,
            }

    lm_eval_aggregate: dict[str, Any] = {}
    if args.run_lm_eval:
        per_seed_task_values: dict[int, dict[str, float | None]] = {}
        for seed in args.seeds:
            result_path = run_dir / "lm_eval" / f"seed_{seed}" / "lm_eval_results.json"
            if not result_path.is_file():
                raise FileNotFoundError(f"Missing lm-eval result: {result_path}")
            raw = load_json(result_path)
            task_results = raw.get("results")
            if not isinstance(task_results, dict):
                raise ValueError(f"Invalid lm-eval results file: {result_path}")
            seed_values: dict[str, float | None] = {}
            for key, metric_spec in METRIC_SPECS.items():
                seed_values[key] = get_metric_value(
                    task_results,
                    metric_spec["task"],
                    metric_spec["metric"],
                )
            per_seed_task_values[seed] = seed_values

        for key, metric_spec in METRIC_SPECS.items():
            per_seed = []
            percent_values = []
            harness_stderr_percents = []
            for seed in args.seeds:
                result_path = run_dir / "lm_eval" / f"seed_{seed}" / "lm_eval_results.json"
                raw = load_json(result_path)
                task_results = raw["results"]
                value = per_seed_task_values[seed][key]
                stderr = get_metric_stderr(
                    task_results,
                    metric_spec["task"],
                    metric_spec["metric"],
                )
                value_percent = None if value is None else 100.0 * float(value)
                stderr_percent = None if stderr is None else 100.0 * float(stderr)
                if value_percent is not None:
                    percent_values.append(value_percent)
                if stderr_percent is not None:
                    harness_stderr_percents.append(stderr_percent)
                per_seed.append(
                    {
                        "seed": seed,
                        "value": value,
                        "value_percent": value_percent,
                        "harness_stderr": stderr,
                        "harness_stderr_percent": stderr_percent,
                    }
                )
            if not percent_values:
                continue
            stats = summary_stats(percent_values)
            lm_eval_aggregate[key] = {
                "display_name": metric_spec["display_name"],
                "task": metric_spec["task"],
                "metric": metric_spec["metric"],
                **stats,
                "baseline_percent": stats["mean"],
                "mean_harness_stderr_percent": mean_or_none(harness_stderr_percents),
                "per_seed": per_seed,
            }

        prior_seed_values = []
        prior_per_seed = []
        for seed in args.seeds:
            components = [
                per_seed_task_values[seed].get(key)
                for key in PRIOR_TASK_KEYS
                if per_seed_task_values[seed].get(key) is not None
            ]
            prior_value_percent = (
                100.0 * statistics.fmean(float(value) for value in components)
                if components
                else None
            )
            if prior_value_percent is not None:
                prior_seed_values.append(prior_value_percent)
            prior_per_seed.append(
                {
                    "seed": seed,
                    "value_percent": prior_value_percent,
                    "components": list(PRIOR_TASK_KEYS),
                }
            )
        if prior_seed_values:
            stats = summary_stats(prior_seed_values)
            lm_eval_aggregate["prior_task_avg"] = {
                "display_name": "Prior-task avg",
                "tasks": list(PRIOR_TASK_KEYS),
                "metric": "mean_percent",
                **stats,
                "baseline_percent": stats["mean"],
                "per_seed": prior_per_seed,
            }

    registry_dataset_values = {
        key: (None if value["baseline_accuracy"] is None else round(float(value["baseline_accuracy"]), 6))
        for key, value in dataset_aggregate.items()
    }
    registry_lm_eval_values = {
        key: (None if value["baseline_percent"] is None else round(float(value["baseline_percent"]), 4))
        for key, value in lm_eval_aggregate.items()
    }

    report = {
        "report_version": REPORT_VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model": {
            "requested": args.model,
            "key": spec.key,
            "hf_repo_id": spec.hf_repo_id,
            "family": spec.family.value,
            "existing_registry_dataset_baselines": dict(spec.dataset_base_accuracy),
            "existing_registry_lm_eval_baselines_percent": dict(spec.lm_eval_baselines_percent),
        },
        "seeds": list(args.seeds),
        "protocol": {
            "dataset_eval": {
                "implementation": "eval_lib.load_model_and_tokenizer_vllm + run_loaded_eval",
                "datasets": list(args.datasets),
                "temperature": args.temperature,
                "gpu_memory_utilization": args.gpu_memory_utilization,
                "tensor_parallel_size": args.tensor_parallel_size,
                "max_model_len_override": args.max_model_len,
                "seed_policy": (
                    "Python/NumPy host RNGs and the vLLM engine seed are set to "
                    "the replicate seed; CUDA is intentionally not initialized "
                    "before vLLM starts"
                ),
            },
            "lm_eval": {
                "implementation": "lm_eval.simple_evaluate via lm-eval 0.4.12 API",
                "tasks": list(args.lm_eval_tasks),
                "batch_size": args.lm_eval_batch_size,
                "apply_chat_template": args.lm_eval_apply_chat_template,
                "gpu_memory_utilization": args.gpu_memory_utilization,
                "tensor_parallel_size": args.tensor_parallel_size,
                "max_model_len": args.lm_eval_max_model_len,
                "seed_policy": (
                    "replicate seed applied to lm-eval random/numpy/torch/fewshot seeds "
                    "and the vLLM backend model seed; the calibration worker avoids "
                    "pre-initializing CUDA before vLLM starts"
                ),
            },
        },
        "environment": {
            "python_executable": sys.executable,
            "python_version": sys.version.split()[0],
            "packages": package_versions(),
            "git": git_provenance(repo_root),
        },
        "dataset_baselines": dataset_aggregate,
        "lm_eval_baselines": lm_eval_aggregate,
        "registry_values": {
            "dataset_base_accuracy": registry_dataset_values,
            "lm_eval_baselines_percent": registry_lm_eval_values,
        },
        "notes": [
            "The reported standard value is the arithmetic mean across the requested seeds.",
            "Dataset evaluation is greedy by default (temperature=0.0), so zero seed variance is expected and valid.",
            "lm-eval seed variability and lm-eval's per-task bootstrap stderr are recorded separately.",
            "Do not compare lm-eval values across different harness/task protocol versions without an explicit compatibility check.",
        ],
    }
    return report


def write_report_files(report: dict[str, Any], run_dir: Path) -> None:
    atomic_write_json(run_dir / "baseline_report.json", report)
    atomic_write_json(run_dir / "registry_values.json", report["registry_values"])

    registry_values = report["registry_values"]
    snippet = (
        "dataset_base_accuracy="
        + repr(registry_values["dataset_base_accuracy"])
        + ",\n"
        + "lm_eval_baselines_percent="
        + repr(registry_values["lm_eval_baselines_percent"])
        + ",\n"
    )
    (run_dir / "model_registry_values.txt").write_text(snippet, encoding="utf-8")

    csv_path = run_dir / "baseline_report.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "kind",
            "metric",
            "mean",
            "sample_std",
            "min",
            "max",
            "n",
            "units",
            "per_seed_values",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for name, data in report["dataset_baselines"].items():
            writer.writerow(
                {
                    "kind": "dataset",
                    "metric": name,
                    "mean": data["mean"],
                    "sample_std": data["sample_std"],
                    "min": data["min"],
                    "max": data["max"],
                    "n": data["n"],
                    "units": "accuracy_0_to_1",
                    "per_seed_values": json.dumps(data["values"]),
                }
            )
        for name, data in report["lm_eval_baselines"].items():
            writer.writerow(
                {
                    "kind": "lm_eval",
                    "metric": name,
                    "mean": data["mean"],
                    "sample_std": data["sample_std"],
                    "min": data["min"],
                    "max": data["max"],
                    "n": data["n"],
                    "units": "percent_0_to_100",
                    "per_seed_values": json.dumps(data["values"]),
                }
            )

    markdown_lines = [
        f"# Baseline calibration — {report['model']['key']}",
        "",
        f"- HF model: `{report['model']['hf_repo_id']}`",
        f"- Seeds: `{', '.join(map(str, report['seeds']))}`",
        f"- Created: `{report['created_at']}`",
        "",
        "## Dataset baselines",
        "",
        "| Dataset | Mean accuracy | Seed std | Min | Max |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, data in report["dataset_baselines"].items():
        markdown_lines.append(
            f"| {name} | {data['mean']:.6f} | {data['sample_std']:.6f} | "
            f"{data['min']:.6f} | {data['max']:.6f} |"
        )

    markdown_lines.extend(
        [
            "",
            "## lm-eval baselines",
            "",
            "| Metric | Mean (%) | Seed std (pp) | Min (%) | Max (%) |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name, data in report["lm_eval_baselines"].items():
        markdown_lines.append(
            f"| {name} | {data['mean']:.4f} | {data['sample_std']:.4f} | "
            f"{data['min']:.4f} | {data['max']:.4f} |"
        )

    markdown_lines.extend(
        [
            "",
            "## Registry-ready values",
            "",
            "```json",
            json.dumps(report["registry_values"], indent=2, ensure_ascii=False),
            "```",
            "",
            "The full JSON report contains per-seed values, counts, package versions, git provenance, and protocol details.",
            "",
        ]
    )
    (run_dir / "baseline_report.md").write_text(
        "\n".join(markdown_lines), encoding="utf-8"
    )


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model",
        required=True,
        help="Registered model key, HF repo id, or supported model path.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help="Dataset adapter names to calibrate.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEEDS),
        help=f"Replicate seeds. Default: {' '.join(map(str, DEFAULT_SEEDS))}.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Root directory for calibration outputs. Default: <repo>/baseline_results.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.6,
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--lm-eval-tasks",
        nargs="+",
        default=list(DEFAULT_LM_EVAL_TASKS),
    )
    parser.add_argument("--lm-eval-batch-size", default="auto")
    parser.add_argument("--lm-eval-max-model-len", type=int, default=None)
    parser.add_argument(
        "--lm-eval-apply-chat-template",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Default false, preserving the current SDFT lm-eval protocol.",
    )
    parser.add_argument(
        "--run-dataset-eval",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--run-lm-eval",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip seed workers whose completed output already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve configuration and print worker commands without running models.",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure multi-seed untouched-model dataset and lm-eval baselines."
    )
    add_common_arguments(parser)
    parser.add_argument("--_worker", choices=("dataset", "lm_eval"), help=argparse.SUPPRESS)
    parser.add_argument("--_worker-seed", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--_run-dir", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args()


def worker_command(args: argparse.Namespace, seed: int, worker: str, run_dir: Path) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--model",
        args.model,
        "--datasets",
        *args.datasets,
        "--seeds",
        *[str(value) for value in args.seeds],
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--temperature",
        str(args.temperature),
        "--lm-eval-tasks",
        *args.lm_eval_tasks,
        "--lm-eval-batch-size",
        str(args.lm_eval_batch_size),
        "--_worker",
        worker,
        "--_worker-seed",
        str(seed),
        "--_run-dir",
        str(run_dir),
    ]
    if args.max_model_len is not None:
        command.extend(["--max-model-len", str(args.max_model_len)])
    if args.lm_eval_max_model_len is not None:
        command.extend(["--lm-eval-max-model-len", str(args.lm_eval_max_model_len)])
    command.append(
        "--lm-eval-apply-chat-template"
        if args.lm_eval_apply_chat_template
        else "--no-lm-eval-apply-chat-template"
    )
    return command


def _config_signature(config: dict[str, Any]) -> dict[str, Any]:
    """Return only fields that determine calibration result compatibility."""
    return {
        "model": config.get("model"),
        "datasets": config.get("datasets"),
        "seeds": config.get("seeds"),
        "run_dataset_eval": config.get("run_dataset_eval"),
        "run_lm_eval": config.get("run_lm_eval"),
        "temperature": config.get("temperature"),
        "gpu_memory_utilization": config.get("gpu_memory_utilization"),
        "tensor_parallel_size": config.get("tensor_parallel_size"),
        "max_model_len": config.get("max_model_len"),
        "lm_eval_tasks": config.get("lm_eval_tasks"),
        "lm_eval_batch_size": config.get("lm_eval_batch_size"),
        "lm_eval_max_model_len": config.get("lm_eval_max_model_len"),
        "lm_eval_apply_chat_template": config.get("lm_eval_apply_chat_template"),
        "packages": config.get("packages"),
        "git_commit": (config.get("git") or {}).get("commit"),
    }


def _validate_resume_config(config_path: Path, new_config: dict[str, Any]) -> None:
    """Prevent accidental mixing of seed outputs from different protocols."""
    if not config_path.is_file():
        return
    old_config = load_json(config_path)
    if _config_signature(old_config) != _config_signature(new_config):
        raise RuntimeError(
            "Existing calibration_config.json is incompatible with this run. "
            "Use a different --output-root or rerun with --no-resume after "
            "intentionally replacing the previous calibration outputs.\n"
            f"Existing: {_config_signature(old_config)}\n"
            f"Requested: {_config_signature(new_config)}"
        )


def main() -> None:
    args = parse_args()
    script_path = Path(__file__).resolve()
    repo_root = locate_repo_root(script_path)
    sys.path.insert(0, str(repo_root))

    # Internal worker mode: no orchestration recursion.
    if args._worker:
        if args._worker_seed is None or args._run_dir is None:
            raise ValueError("Internal worker mode requires --_worker-seed and --_run-dir.")
        args._worker_seed = int(args._worker_seed)
        args.run_dir = Path(args._run_dir).resolve()
        if args._worker == "dataset":
            _dataset_worker(args, repo_root)
        else:
            _lm_eval_worker(args, repo_root)
        return

    from dataset_adapters import get_evaluation_dataset_names
    from model_registry import check_tensor_parallel_compatibility, resolve_model_spec

    args.seeds = validate_unique_positive_seeds(list(args.seeds))
    if not (0 < args.gpu_memory_utilization <= 1):
        raise ValueError("--gpu-memory-utilization must be in (0, 1].")
    if args.tensor_parallel_size <= 0:
        raise ValueError("--tensor-parallel-size must be positive.")
    if args.max_model_len is not None and args.max_model_len <= 0:
        raise ValueError("--max-model-len must be positive when provided.")
    if args.lm_eval_max_model_len is not None and args.lm_eval_max_model_len <= 0:
        raise ValueError("--lm-eval-max-model-len must be positive when provided.")

    available_datasets = set(get_evaluation_dataset_names())
    unknown = [name for name in args.datasets if name not in available_datasets]
    if unknown:
        raise ValueError(
            f"Unknown/non-evaluation dataset(s): {unknown}. Available: {sorted(available_datasets)}"
        )
    if len(args.datasets) != len(set(args.datasets)):
        raise ValueError(f"Datasets must be unique, got {args.datasets}.")

    spec = resolve_model_spec(args.model)
    check_tensor_parallel_compatibility(spec, args.tensor_parallel_size)
    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root is not None
        else repo_root / "baseline_results"
    )
    run_dir = output_root / safe_slug(spec.key)
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
        "datasets": list(args.datasets),
        "seeds": list(args.seeds),
        "run_dataset_eval": args.run_dataset_eval,
        "run_lm_eval": args.run_lm_eval,
        "temperature": args.temperature,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "tensor_parallel_size": args.tensor_parallel_size,
        "max_model_len": args.max_model_len,
        "lm_eval_tasks": list(args.lm_eval_tasks),
        "lm_eval_batch_size": args.lm_eval_batch_size,
        "lm_eval_max_model_len": args.lm_eval_max_model_len,
        "lm_eval_apply_chat_template": args.lm_eval_apply_chat_template,
        "visible_devices": visible_devices,
        "packages": package_versions(),
        "git": git_provenance(repo_root),
    }
    config_path = run_dir / "calibration_config.json"
    if args.resume:
        _validate_resume_config(config_path, config)
    atomic_write_json(config_path, config)

    log(f"Repository root: {repo_root}")
    log(f"Model: {spec.key} ({spec.hf_repo_id})")
    log(f"Datasets: {args.datasets}")
    log(f"Seeds: {args.seeds}")
    log(f"Output: {run_dir}")

    for seed in args.seeds:
        env = worker_environment(seed, visible_devices)
        if args.run_dataset_eval:
            done_path = run_dir / "dataset_eval" / f"seed_{seed}" / "dataset_seed_summary.json"
            if args.resume and done_path.is_file():
                log(f"Skipping completed dataset seed {seed}: {done_path}")
            else:
                cmd = worker_command(args, seed, "dataset", run_dir)
                stream_subprocess(
                    command=cmd,
                    cwd=repo_root,
                    env=env,
                    log_path=run_dir / "logs" / f"dataset_seed_{seed}.log",
                    command_path=run_dir / "commands" / f"dataset_seed_{seed}.txt",
                    dry_run=args.dry_run,
                )

        if args.run_lm_eval:
            done_path = run_dir / "lm_eval" / f"seed_{seed}" / "lm_eval_results.json"
            if args.resume and done_path.is_file():
                log(f"Skipping completed lm-eval seed {seed}: {done_path}")
            else:
                cmd = worker_command(args, seed, "lm_eval", run_dir)
                stream_subprocess(
                    command=cmd,
                    cwd=repo_root,
                    env=env,
                    log_path=run_dir / "logs" / f"lm_eval_seed_{seed}.log",
                    command_path=run_dir / "commands" / f"lm_eval_seed_{seed}.txt",
                    dry_run=args.dry_run,
                )

    if args.dry_run:
        log("Dry run complete; no aggregate report was written because no measurements ran.")
        return

    report = _aggregate(args, repo_root)
    write_report_files(report, run_dir)
    log(f"Wrote baseline report: {run_dir / 'baseline_report.json'}")
    log(f"Wrote Markdown report: {run_dir / 'baseline_report.md'}")
    log(f"Wrote registry-ready values: {run_dir / 'registry_values.json'}")


if __name__ == "__main__":
    main()
