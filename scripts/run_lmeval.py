#!/usr/bin/env python3
"""Run lm-eval reproducibly for an SDFT checkpoint.

This script is intentionally model-agnostic. Model-specific vLLM settings are
resolved from ``model_registry.py`` so Qwen2.5, Qwen3.5, Ministral, and future
registered models all use one execution path.

The shell wrapper ``scripts/lmeval.sh`` exists only to select the repository's
Python interpreter and preserve the historical one-argument interface used by
``run_experiments.py``.

Environment variables
---------------------
VISIBLE_DEVICES
    GPUs exposed to lm-eval. Falls back to CUDA_VISIBLE_DEVICES, then ``0``.
GPU_MEMORY_UTILIZATION
    vLLM GPU-memory fraction. Default: ``0.80``.
TASKS
    Comma-separated lm-eval task list. Historical default is preserved.
LM_EVAL_BATCH_SIZE
    lm-eval batch-size setting. Default: ``auto``.
LM_EVAL_TENSOR_PARALLEL_SIZE
    vLLM tensor-parallel size for lm-eval. Default: ``1``.
LM_EVAL_MAX_MODEL_LEN
    Optional vLLM maximum model length.
LM_EVAL_EXPECTED_VERSION
    Expected lm_eval package version. Default: ``0.4.12``.
LM_EVAL_ALLOW_VERSION_MISMATCH
    Set to a truthy value to warn instead of failing on an lm_eval version
    mismatch. Default: false.
LM_EVAL_LOG_LEVEL
    lm-eval logging level. Default: ``INFO``.
LM_EVAL_LIMIT
    Optional lm-eval ``--limit`` value for smoke tests only.
LM_EVAL_APPLY_CHAT_TEMPLATE
    Optional opt-in to ``--apply_chat_template``. Default: false, preserving
    the historical SDFT lm-eval protocol.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_TASKS = "hellaswag,mmlu,truthfulqa,winogrande,humaneval,ifeval"
DEFAULT_EXPECTED_LM_EVAL_VERSION = "0.4.12"
PROTOCOL_ID = "sdft-lm-eval-v0.4.12-vllm-v1"


def locate_repo_root(script_path: Path) -> Path:
    """Locate the repository root without hardcoding usernames or paths."""
    candidates = (script_path.parent, script_path.parent.parent)

    for candidate in candidates:
        if (
            (candidate / "model_registry.py").is_file()
            and (candidate / "scripts" / "summarize_lmeval.py").is_file()
        ):
            return candidate

    raise FileNotFoundError(
        "Could not locate the repository root from "
        f"{script_path}. Expected model_registry.py at the repository root "
        "and scripts/summarize_lmeval.py."
    )


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(message: str) -> None:
    print(f"[{timestamp()}] {message}", flush=True)


def str_to_bool(value: str | bool | int | None) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value != 0
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def package_version(distribution_name: str) -> str | None:
    try:
        return importlib.metadata.version(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def ensure_expected_lm_eval_version(*, dry_run: bool) -> str | None:
    actual = package_version("lm_eval")
    expected = os.environ.get(
        "LM_EVAL_EXPECTED_VERSION",
        DEFAULT_EXPECTED_LM_EVAL_VERSION,
    ).strip()

    if dry_run:
        return actual

    if actual is None:
        raise RuntimeError(
            "lm_eval is not installed in the Python environment running "
            f"{Path(sys.executable)}. Install lm_eval[ifeval]=={expected}."
        )

    if expected and actual != expected:
        message = (
            f"Expected lm_eval=={expected} for protocol {PROTOCOL_ID}, "
            f"but found lm_eval=={actual} in {sys.executable}."
        )
        if str_to_bool(os.environ.get("LM_EVAL_ALLOW_VERSION_MISMATCH")):
            log(f"WARNING: {message}")
        else:
            raise RuntimeError(
                message
                + " Set LM_EVAL_ALLOW_VERSION_MISMATCH=1 only if this version "
                "difference is intentional and will be recorded with the run."
            )

    return actual


def parse_positive_float(name: str, default: str) -> float:
    raw = os.environ.get(name, default)
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be numeric, got {raw!r}.") from exc
    if not 0 < value <= 1:
        raise ValueError(f"{name} must be in (0, 1], got {value}.")
    return value


def parse_positive_int(name: str, default: str) -> int:
    raw = os.environ.get(name, default)
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}.") from exc
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")
    return value


def parse_optional_positive_int(name: str) -> int | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}.") from exc
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")
    return value


def normalize_tasks(raw_tasks: str) -> str:
    tasks = [part.strip() for part in raw_tasks.replace(" ", ",").split(",") if part.strip()]
    if not tasks:
        raise ValueError("TASKS resolved to an empty task list.")
    return ",".join(tasks)


def ensure_nltk_resources(tasks: str, *, dry_run: bool) -> None:
    """Ensure the NLTK assets required by IFEval are locally available."""
    if dry_run or "ifeval" not in set(tasks.split(",")):
        return

    try:
        import nltk
    except ImportError as exc:
        raise RuntimeError(
            "IFEval is requested but NLTK is unavailable. Install the "
            "lm_eval[ifeval] extra."
        ) from exc

    resources = {
        "punkt": "tokenizers/punkt",
        "punkt_tab": "tokenizers/punkt_tab",
    }

    for name, resource_path in resources.items():
        try:
            nltk.data.find(resource_path)
            log(f"NLTK resource already available: {name}")
            continue
        except LookupError:
            pass

        log(f"Downloading missing NLTK resource: {name}")
        if not nltk.download(name, quiet=False):
            raise RuntimeError(f"Failed to download NLTK resource: {name}")

        try:
            nltk.data.find(resource_path)
        except LookupError as exc:
            raise RuntimeError(
                f"NLTK resource {name!r} is still unavailable after download."
            ) from exc

    log("All required NLTK resources are available.")


def serialize_model_arg(value: Any) -> str:
    """Serialize a scalar model argument for lm-eval's key=value parser.

    Registry vLLM engine kwargs are deliberately restricted to scalar values
    here. Failing loudly is preferable to silently mis-parsing nested values
    containing commas.
    """
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, (str, int, float)):
        text = str(value)
        if "," in text:
            raise ValueError(
                "lm-eval model arguments containing commas are not supported "
                f"by this runner: {text!r}"
            )
        return text
    raise TypeError(
        "Unsupported non-scalar lm-eval model argument "
        f"{value!r} ({type(value).__name__})."
    )


def build_model_args(
    *,
    spec: Any,
    gpu_memory_utilization: float,
    tensor_parallel_size: int,
    max_model_len: int | None,
    vllm_eval_engine_kwargs: Any,
) -> tuple[dict[str, Any], str]:
    args: dict[str, Any] = {
        "pretrained": spec.hf_repo_id,
        "gpu_memory_utilization": gpu_memory_utilization,
    }
    args.update(vllm_eval_engine_kwargs(spec))

    if spec.trust_remote_code:
        args["trust_remote_code"] = True

    if tensor_parallel_size != 1:
        args["tensor_parallel_size"] = tensor_parallel_size

    if max_model_len is not None:
        args["max_model_len"] = max_model_len

    serialized = ",".join(
        f"{key}={serialize_model_arg(value)}" for key, value in args.items()
    )
    return args, serialized


def subprocess_environment(visible_devices: str) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = visible_devices
    env["WANDB_MODE"] = "offline"
    env["HF_ALLOW_CODE_EVAL"] = "1"
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("LMEVAL_LOG_LEVEL", "INFO")

    # Reuse a caller-provided CA bundle consistently without hardcoding a
    # machine- or company-specific certificate path.
    requests_bundle = env.get("REQUESTS_CA_BUNDLE")
    if requests_bundle:
        env.setdefault("SSL_CERT_FILE", requests_bundle)
        env.setdefault("CURL_CA_BUNDLE", requests_bundle)

    return env


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def write_command_file(path: Path, command: list[str], cwd: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"# Command launched at {timestamp()}\n")
        handle.write(f"# Working directory: {cwd}\n\n")
        handle.write(shlex.join(command))
        handle.write("\n")


def run_streamed_command(
    *,
    name: str,
    command: list[str],
    cwd: Path,
    env: dict[str, str],
    log_path: Path,
    command_path: Path,
    dry_run: bool,
) -> None:
    write_command_file(command_path, command, cwd)
    log(f"Starting: {name}")
    log(f"Saved command to: {command_path}")
    log(f"Log file: {log_path}")

    if dry_run:
        log(f"DRY RUN: {shlex.join(command)}")
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

    log(f"Finished: {name}")


def build_environment_metadata(
    *,
    model_path: Path,
    spec: Any,
    model_args: dict[str, Any],
    tasks: str,
    batch_size: str,
    visible_devices: str,
    apply_chat_template: bool,
    lm_eval_version: str | None,
) -> dict[str, Any]:
    distributions = [
        "lm_eval",
        "torch",
        "transformers",
        "vllm",
        "tokenizers",
        "accelerate",
        "datasets",
        "peft",
    ]
    versions = {name: package_version(name) for name in distributions}

    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "protocol_id": PROTOCOL_ID,
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "packages": versions,
        "expected_lm_eval_version": os.environ.get(
            "LM_EVAL_EXPECTED_VERSION",
            DEFAULT_EXPECTED_LM_EVAL_VERSION,
        ),
        "actual_lm_eval_version": lm_eval_version,
        "checkpoint_path": str(model_path),
        "resolved_model": {
            "key": spec.key,
            "family": spec.family.value,
            "hf_repo_id": spec.hf_repo_id,
        },
        "lm_eval": {
            "backend": "vllm",
            "model_args": model_args,
            "tasks": tasks.split(","),
            "batch_size": batch_size,
            "apply_chat_template": apply_chat_template,
            "confirm_run_unsafe_code": True,
        },
        "cuda_visible_devices": visible_devices,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run modern lm-eval on one SDFT checkpoint."
    )
    parser.add_argument("model_path", type=Path)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve configuration and write provenance files without running lm-eval.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    script_path = Path(__file__).resolve()
    repo_root = locate_repo_root(script_path)

    # Import the project's registry only after locating the repository root.
    sys.path.insert(0, str(repo_root))
    from model_registry import (  # noqa: PLC0415
        check_tensor_parallel_compatibility,
        resolve_model_spec,
        vllm_eval_engine_kwargs,
    )

    model_path = args.model_path.expanduser().resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {model_path}")

    summary_script = repo_root / "scripts" / "summarize_lmeval.py"
    if not summary_script.is_file():
        raise FileNotFoundError(f"Missing summary script: {summary_script}")

    lm_eval_version = ensure_expected_lm_eval_version(dry_run=args.dry_run)

    visible_devices = os.environ.get(
        "VISIBLE_DEVICES",
        os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
    ).strip()
    if not visible_devices:
        raise ValueError("VISIBLE_DEVICES/CUDA_VISIBLE_DEVICES resolved to an empty value.")

    gpu_memory_utilization = parse_positive_float(
        "GPU_MEMORY_UTILIZATION",
        "0.80",
    )
    tensor_parallel_size = parse_positive_int(
        "LM_EVAL_TENSOR_PARALLEL_SIZE",
        "1",
    )
    max_model_len = parse_optional_positive_int("LM_EVAL_MAX_MODEL_LEN")
    tasks = normalize_tasks(os.environ.get("TASKS", DEFAULT_TASKS))
    batch_size = os.environ.get("LM_EVAL_BATCH_SIZE", "auto").strip() or "auto"
    apply_chat_template = str_to_bool(os.environ.get("LM_EVAL_APPLY_CHAT_TEMPLATE"))

    spec = resolve_model_spec(str(model_path))
    check_tensor_parallel_compatibility(spec, tensor_parallel_size)
    model_args, model_args_string = build_model_args(
        spec=spec,
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        vllm_eval_engine_kwargs=vllm_eval_engine_kwargs,
    )

    lm_eval_out = model_path / "lm_eval"
    lm_eval_out.mkdir(parents=True, exist_ok=True)

    exp_root = model_path.parent.parent
    checkpoint_name = model_path.name
    phase_name = model_path.parent.name
    experiment_name = exp_root.name
    run_name = f"{experiment_name}_{phase_name}_{checkpoint_name}_lm_eval"

    log(f"Repository root: {repo_root}")
    log(f"Python: {sys.executable}")
    log(f"lm_eval version: {lm_eval_version or 'not installed (dry-run)'}")
    log(f"Checkpoint: {model_path}")
    log(
        "Resolved model: "
        f"key={spec.key}, family={spec.family.value}, "
        f"vllm_model_args={model_args}"
    )
    log(f"Tasks: {tasks}")

    ensure_nltk_resources(tasks, dry_run=args.dry_run)

    env = subprocess_environment(visible_devices)

    metadata = build_environment_metadata(
        model_path=model_path,
        spec=spec,
        model_args=model_args,
        tasks=tasks,
        batch_size=batch_size,
        visible_devices=visible_devices,
        apply_chat_template=apply_chat_template,
        lm_eval_version=lm_eval_version,
    )
    metadata_path = lm_eval_out / "lm_eval_environment.json"
    write_json(metadata_path, metadata)
    log(f"Saved environment metadata to: {metadata_path}")

    # Use ``python -m lm_eval`` rather than relying on whichever ``lm_eval``
    # console script happens to be first on PATH. This guarantees the harness
    # runs in the same interpreter selected by lmeval.sh.
    lm_eval_command = [
        sys.executable,
        "-m",
        "lm_eval",
        "run",
        "--model",
        "vllm",
        "--model_args",
        model_args_string,
        "--output_path",
        str(lm_eval_out),
        "--confirm_run_unsafe_code",
        "--batch_size",
        batch_size,
        "--tasks",
        tasks,
    ]

    if apply_chat_template:
        lm_eval_command.append("--apply_chat_template")

    limit = os.environ.get("LM_EVAL_LIMIT", "").strip()
    if limit:
        lm_eval_command.extend(["--limit", limit])

    run_streamed_command(
        name=run_name,
        command=lm_eval_command,
        cwd=repo_root,
        env=env,
        log_path=lm_eval_out / f"{run_name}.log",
        command_path=lm_eval_out / "lm_eval_command.txt",
        dry_run=args.dry_run,
    )

    if args.dry_run:
        return

    summary_command = [
        sys.executable,
        "-m",
        "scripts.summarize_lmeval",
        "--checkpoint-path",
        str(model_path),
        "--lm-eval-dir",
        str(lm_eval_out),
        "--exp-root",
        str(exp_root),
        "--output-name",
        "lm_eval_summary.json",
    ]

    run_streamed_command(
        name=f"{run_name}_summary",
        command=summary_command,
        cwd=repo_root,
        env=env,
        log_path=exp_root / f"{run_name}_summary.log",
        command_path=exp_root / "lm_eval_summary_command.txt",
        dry_run=False,
    )


if __name__ == "__main__":
    main()
