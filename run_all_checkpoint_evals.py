"""Evaluate all checkpoints in one experiment and build training curves.

The script discovers phase checkpoints, evaluates the requested project datasets
through ``eval_lib.py``/``dataset_adapters.py``, and writes per-checkpoint result
files plus combined JSON/PNG/PDF training curves. Base-model step-zero accuracies
are resolved from ``experiment_config.json`` through the model-aware dataset
metric API; unmeasured model/dataset baselines remain absent rather than falling
back to another model.
"""

import gc
import time
import argparse
import json
import math
import os
import re
import shlex
from collections import defaultdict, deque
from pathlib import Path

import torch
import matplotlib.pyplot as plt

from eval_lib import load_model_and_tokenizer_vllm, run_loaded_eval
from dataset_adapters import (
    get_dataset_adapter,
    get_dataset_metric_defaults,
    get_evaluation_dataset_names,
)




CHECKPOINT_PATTERN = re.compile(r"checkpoint-(\d+)$")
DEFAULT_DATASET_ORDER = list(get_evaluation_dataset_names())

PHASE_LINE_COLOR_CYCLE = [
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:gray",
]

LEARNING_RATE_RUN_RE = re.compile(
    r"^lr(?P<value>(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?)(?:$|_s\d+(?:_|$))",
    re.IGNORECASE,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run evaluations for all checkpoints in one experiment folder and build combined graphs."
    )
    parser.add_argument(
        "--experiment_root",
        type=str,
        required=True,
        help="Path to one experiment root, e.g. outputs/method/run_name",
    )
    
    parser.add_argument(
        "--cuda_visible_devices",
        type=str,
        default=None,
        help="Value for CUDA_VISIBLE_DEVICES passed to eval subprocesses",
    )

    parser.add_argument(
        "--gpu_memory_utilization",
        "--gpu-memory-utilization",
        dest="gpu_memory_utilization",
        type=float,
        default=0.6,
        help="Fraction of GPU memory vLLM may use. Default: 0.6.",
    )

    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help=(
            "Tensor-parallel size for the standalone vLLM evaluation engine. "
            "Unlike training (which needs torchrun/accelerate to establish "
            "the process group), vLLM manages its own worker processes here, "
            "so this can be raised without any other launcher changes. "
            "Default: 1."
        ),
    )

    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip an evaluation if its output json already exists",
    )
    parser.add_argument(
        "--dataset_order",
        nargs="+",
        default=None,
        help=(
            "Optional train/eval dataset order, e.g. "
            "--dataset_order tooluse science. "
            "If omitted, the script tries to infer the order from train_command.txt/command.txt. "
            "If inference fails, it falls back to the registered evaluation-dataset order."
        ),
    )
    parser.add_argument(
        "--eval_subset",
        nargs="+",
        default=None,
        help=(
            "Optional list of datasets to evaluate on every checkpoint. "
            "If omitted, evaluate datasets whose phase folders currently exist. "
            "If provided, this list is used as the eval dataset "
            "subset even if some corresponding phase folders do not exist yet, "
            "which is useful for per-phase eval + cleanup."
        ),
    )
    # parser.add_argument("--tooluse_max_new_tokens", type=int, default=1024)
    # parser.add_argument("--science_max_new_tokens", type=int, default=2048)
    for dataset_name in DEFAULT_DATASET_ORDER:
        parser.add_argument(
            f"--{dataset_name}_max_new_tokens",
            type=int,
            default=None,
            help=(
                f"Optional max-new-token override for {dataset_name}; "
                "defaults to the dataset adapter value."
            ),
        )
    parser.add_argument("--temperature", type=float, default=0.0)
    return parser.parse_args()

def max_model_len_for_all_datasets(eval_datasets: list[str]) -> int | None:
    lengths = []
    for dataset_name in eval_datasets:
        adapter = get_dataset_adapter(dataset_name)
        if adapter.default_max_model_len is not None:
            lengths.append(adapter.default_max_model_len)
    return max(lengths) if lengths else None


def extract_checkpoint_step(path: Path) -> int:
    match = CHECKPOINT_PATTERN.match(path.name)
    if not match:
        raise ValueError(f"Invalid checkpoint folder name: {path}")
    return int(match.group(1))


def sorted_checkpoints(folder: Path):
    checkpoints = [
        p for p in folder.iterdir()
        if p.is_dir() and CHECKPOINT_PATTERN.match(p.name)
    ]
    return sorted(checkpoints, key=extract_checkpoint_step)


def read_command_file(phase_dir: Path) -> str | None:
    for name in ["train_command.txt", "command.txt"]:
        path = phase_dir / name
        if path.exists():
            try:
                return path.read_text(errors="ignore")
            except Exception:
                return None
    return None


def normalize_command_text(text: str) -> str:
    return text.replace("\\\n", " ").replace("\n", " ")


def extract_arg_from_command(command: str, arg_name: str) -> str | None:
    """
    Extracts --arg_name value from a shell command.

    Handles:
      --model_name VALUE
      --model_name=VALUE
    """
    if not command:
        return None

    command = normalize_command_text(command)

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
    """
    If current phase was initialized from a checkpoint inside another phase folder,
    return that previous phase.
    """
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

    return None


def infer_phase_order(experiment_root: Path, datasets: list[str]) -> tuple[list[str], bool]:
    """
    Try to infer phase order from train_command.txt or command.txt.

    Returns:
      (phase_order, inferred_successfully)

    If no dependency edges can be found, inferred_successfully=False.
    """
    phase_dirs = find_phase_dirs(experiment_root, datasets)
    existing = list(phase_dirs.keys())

    if not existing:
        return [], False

    model_names = {}

    for phase, phase_dir in phase_dirs.items():
        command = read_command_file(phase_dir)
        model_names[phase] = extract_arg_from_command(command or "", "model_name")

    edges = defaultdict(set)
    indegree = {phase: 0 for phase in existing}
    found_dependency = False

    for phase, model_name in model_names.items():
        dependency = infer_dependency_phase(
            current_phase=phase,
            model_name=model_name,
            phase_dirs=phase_dirs,
        )

        if dependency is None:
            continue

        found_dependency = True

        if phase not in edges[dependency]:
            edges[dependency].add(phase)
            indegree[phase] += 1

    if not found_dependency:
        return [], False

    dataset_rank = {dataset: i for i, dataset in enumerate(datasets)}

    queue = deque(
        sorted(
            [phase for phase in existing if indegree[phase] == 0],
            key=lambda x: (dataset_rank.get(x, 10_000), x),
        )
    )

    ordered = []

    while queue:
        node = queue.popleft()
        ordered.append(node)

        for nxt in sorted(edges[node], key=lambda x: (dataset_rank.get(x, 10_000), x)):
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                queue.append(nxt)

    if len(ordered) != len(existing):
        remaining = [phase for phase in existing if phase not in ordered]
        ordered.extend(sorted(remaining, key=lambda x: (dataset_rank.get(x, 10_000), x)))

    return ordered, True


def resolve_dataset_order(args, experiment_root: Path) -> list[str]:
    """
    Priority:
      1. Explicit --dataset_order
      2. Inferred order from command files
      3. DEFAULT_DATASET_ORDER
    """
    if args.dataset_order is not None:
        return list(args.dataset_order)

    inferred_order, ok = infer_phase_order(
        experiment_root=experiment_root,
        datasets=DEFAULT_DATASET_ORDER,
    )

    if ok and inferred_order:
        print(f"Inferred dataset order: {inferred_order}")
        return inferred_order

    print(f"Could not infer dataset order. Falling back to default: {DEFAULT_DATASET_ORDER}")
    return list(DEFAULT_DATASET_ORDER)


def find_phase_dirs(experiment_root: Path, dataset_order: list[str]):
    """
    Return only the phases that actually exist.
    Missing phases are treated as 'not performed'.
    """
    phase_dirs = {}
    children = [p for p in experiment_root.iterdir() if p.is_dir()]

    for phase in dataset_order:
        matches = [p for p in children if p.name == phase]
        if len(matches) > 1:
            raise RuntimeError(
                f"Expected at most 1 folder named '{phase}' in {experiment_root}, found {len(matches)}"
            )
        if len(matches) == 1:
            phase_dirs[phase] = matches[0]

    if not phase_dirs:
        raise RuntimeError(
            f"No recognized phase folders found in {experiment_root}. "
            f"Expected any subset of: {dataset_order}"
        )

    return phase_dirs


def results_filename(dataset_name: str) -> str:
    return f"eval_{dataset_name}_results.json"


def max_new_tokens_for_dataset(args, dataset_name: str) -> int:
    adapter = get_dataset_adapter(dataset_name)
    override = getattr(args, f"{dataset_name}_max_new_tokens", None)
    return (
        int(override)
        if override is not None
        else adapter.default_max_new_tokens
    )


def cleanup_vllm(llm=None, tokenizer=None, sleep_seconds: float = 3.0):
    """
    Best-effort cleanup for running multiple vLLM LLM instances sequentially
    in the same Python process.

    This helper releases vLLM/distributed/CUDA state between checkpoint loads.
    Subprocess isolation remains the strongest option when complete process-level
    isolation is required.
    """
    # 1. Try public-ish / version-dependent shutdown hooks.
    if llm is not None:
        candidates = [
            llm,
            getattr(llm, "llm_engine", None),
            getattr(getattr(llm, "llm_engine", None), "engine_core", None),
            getattr(getattr(llm, "llm_engine", None), "model_executor", None),
        ]

        for obj in candidates:
            if obj is None:
                continue

            for method_name in ("shutdown", "close"):
                method = getattr(obj, method_name, None)
                if callable(method):
                    try:
                        print(f"Calling {obj.__class__.__name__}.{method_name}()")
                        method()
                    except Exception as e:
                        print(
                            f"Warning: {obj.__class__.__name__}.{method_name}() failed: {e}"
                        )

    # 2. Try vLLM distributed cleanup.
    try:
        from vllm.distributed.parallel_state import (
            destroy_model_parallel,
            destroy_distributed_environment,
        )

        try:
            destroy_model_parallel()
        except Exception as e:
            print(f"Warning: destroy_model_parallel() failed: {e}")

        try:
            destroy_distributed_environment()
        except Exception as e:
            print(f"Warning: destroy_distributed_environment() failed: {e}")

    except Exception as e:
        print(f"Warning: vLLM distributed cleanup unavailable/failed: {e}")

    # 3. Drop refs.
    try:
        del llm
    except Exception:
        pass

    try:
        del tokenizer
    except Exception:
        pass

    # 4. Python + CUDA cleanup.
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass

    # 5. Give vLLM multiprocessing cleanup a moment.
    if sleep_seconds > 0:
        time.sleep(sleep_seconds)

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def maybe_run_all_evals_for_checkpoint(checkpoint_dir: Path, args, eval_datasets: list[str]):
    pending_datasets = []

    for dataset_name in eval_datasets:
        json_path = checkpoint_dir / results_filename(dataset_name)

        if args.skip_existing and json_path.exists():
            print(f"Skipping existing {dataset_name} eval for {checkpoint_dir}")
            continue

        pending_datasets.append(dataset_name)

    if not pending_datasets:
        print(f"All evals already exist for {checkpoint_dir}")
        return

    effective_max_model_len = max_model_len_for_all_datasets(eval_datasets)

    print("\n" + "#" * 100)
    print(f"Loading model once for checkpoint: {checkpoint_dir}")
    print(f"Datasets to run: {pending_datasets}")
    print(f"max_model_len: {effective_max_model_len}")
    print("#" * 100)

    llm = None
    tokenizer = None

    try:
        llm, tokenizer = load_model_and_tokenizer_vllm(
            model_path=str(checkpoint_dir),
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=effective_max_model_len,
            tensor_parallel_size=args.tensor_parallel_size,
        )

        for dataset_name in pending_datasets:
            summary = run_loaded_eval(
                dataset_name=dataset_name,
                llm=llm,
                tokenizer=tokenizer,
                model_path=str(checkpoint_dir),
                output_dir=str(checkpoint_dir),
                max_new_tokens=max_new_tokens_for_dataset(args, dataset_name),
                temperature=args.temperature,
            )

            print("\n" + "=" * 60)
            print("Evaluation Results:")
            print(f"  Dataset: {dataset_name}")
            print(f"  Total samples: {summary['num_total']}")
            print(f"  Correct: {summary['num_correct']}")
            print(f"  Accuracy: {summary['accuracy']:.4f} ({summary['accuracy'] * 100:.2f}%)")
            print("=" * 60)

    finally:
        cleanup_vllm(llm, tokenizer)


def load_accuracy(json_path: Path) -> float | None:
    if not json_path.exists():
        return None
    with open(json_path, "r") as f:
        data = json.load(f)
    return float(data["accuracy"])


def resolve_experiment_baselines(
    experiment_root: Path,
) -> tuple[dict[str, float | None], str | None]:
    """Resolve base-model dataset baselines for one experiment.

    Experiments normally record their base model in ``experiment_config.json``
    under ``initial_model``. The model-aware dataset-adapter API resolves the
    corresponding values from the model registry. If model metadata is absent,
    ``get_dataset_metric_defaults()`` uses the project default model
    (Qwen2.5-7B).

    A malformed existing config is treated as an error rather than silently
    substituting the default model, which could produce scientifically
    incorrect baseline deltas.
    """
    config_path = experiment_root / "experiment_config.json"

    if not config_path.is_file():
        return get_dataset_metric_defaults("baseline"), None

    try:
        with config_path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON in experiment config: {config_path}"
        ) from exc

    if not isinstance(config, dict):
        raise ValueError(
            f"Experiment config must contain a JSON object: {config_path}"
        )

    initial_model = config.get("initial_model")
    if initial_model is None:
        return get_dataset_metric_defaults("baseline"), None

    if not isinstance(initial_model, str):
        raise ValueError(
            f"experiment_config.json field 'initial_model' must be a string: "
            f"{config_path}"
        )

    initial_model = initial_model.strip()
    if not initial_model:
        return get_dataset_metric_defaults("baseline"), None
    baselines = get_dataset_metric_defaults(
        "baseline",
        model_name_or_path=initial_model,
    )
    return baselines, initial_model


def build_curve_data(
    phase_dirs: dict[str, Path],
    phase_order: list[str],
    eval_datasets: list[str],
    baseline_eval_accuracy: dict[str, float | None],
):
    checkpoints_by_phase = {}
    existing_phases = [phase for phase in phase_order if phase in phase_dirs]

    for phase in existing_phases:
        folder = phase_dirs[phase]
        ckpts = sorted_checkpoints(folder)
        if not ckpts:
            print(f"Warning: no checkpoints found in {folder}; skipping phase '{phase}'")
            continue
        checkpoints_by_phase[phase] = ckpts

    if not checkpoints_by_phase:
        raise RuntimeError("No checkpoints found in any existing phase folders.")

    phase_offsets = {}
    running_offset = 0
    for phase in existing_phases:
        if phase not in checkpoints_by_phase:
            continue
        phase_offsets[phase] = running_offset
        last_step = extract_checkpoint_step(checkpoints_by_phase[phase][-1])
        running_offset += last_step

    curve = {
        "baseline": {},
        "phases": {},
        "combined": {dataset_name: [] for dataset_name in eval_datasets},
        "phase_offsets": phase_offsets,
    }

    for dataset_name in eval_datasets:
        baseline_acc = baseline_eval_accuracy.get(dataset_name)

        if baseline_acc is not None:
            curve["baseline"][dataset_name] = {
                "global_step": 0,
                "accuracy": baseline_acc,
            }
            curve["combined"][dataset_name].append((0, baseline_acc))

    for phase in existing_phases:
        if phase not in checkpoints_by_phase:
            continue

        curve["phases"][phase] = []

        for ckpt in checkpoints_by_phase[phase]:
            phase_step = extract_checkpoint_step(ckpt)
            global_step = phase_offsets[phase] + phase_step

            record = {
                "checkpoint_dir": str(ckpt),
                "phase": phase,
                "phase_step": phase_step,
                "global_step": global_step,
            }

            for dataset_name in eval_datasets:
                acc = load_accuracy(ckpt / results_filename(dataset_name))
                record[f"{dataset_name}_eval_accuracy"] = acc
                curve["combined"][dataset_name].append((global_step, acc))

            curve["phases"][phase].append(record)

    return curve

def _positive_float(value) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) and parsed > 0 else None


def experiment_learning_rate(experiment_root: Path) -> float | None:
    """Resolve LR from run metadata, then from an explicit ``lr...`` label."""

    config_path = experiment_root / "experiment_config.json"
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
    except (OSError, json.JSONDecodeError, TypeError):
        config = {}

    if isinstance(config, dict):
        common_args = config.get("common_args")
        if isinstance(common_args, dict):
            value = _positive_float(common_args.get("learning_rate"))
            if value is not None:
                return value
        value = _positive_float(config.get("learning_rate"))
        if value is not None:
            return value

    match = LEARNING_RATE_RUN_RE.match(experiment_root.name)
    return _positive_float(match.group("value")) if match else None


def format_learning_rate(value: float) -> str:
    mantissa, exponent = f"{value:.3E}".split("E")
    return f"{mantissa.rstrip('0').rstrip('.')}e{int(exponent)}"


def format_experiment_title(experiment_root: Path) -> str:
    base_title = f"{experiment_root.parent.name}/{experiment_root.name}"

    lr = experiment_learning_rate(experiment_root)

    if lr is None:
        return base_title

    return f"{base_title} (LR: {format_learning_rate(lr)})"


def plot_curve(
    curve_data,
    experiment_root: Path,
    phase_order: list[str],
    eval_dataset_order: list[str],
):
    plt.figure(figsize=(11, 6))

    dataset_colors = {
        dataset_name: PHASE_LINE_COLOR_CYCLE[i % len(PHASE_LINE_COLOR_CYCLE)]
        for i, dataset_name in enumerate(phase_order)
    }

    for dataset_name in eval_dataset_order:
        points = sorted(curve_data["combined"][dataset_name], key=lambda x: x[0])
        x_vals = [x for x, y in points if y is not None]
        y_vals = [y for x, y in points if y is not None]

        if x_vals:
            plt.plot(
                x_vals,
                y_vals,
                marker="o",
                color=dataset_colors.get(dataset_name),
                label=f"{dataset_name} eval",
            )

    offsets = curve_data["phase_offsets"]

    for phase in phase_order[1:]:
        if phase in offsets:
            plt.axvline(
                offsets[phase],
                linestyle="--",
                color=dataset_colors.get(phase),
                label=f"Start of {phase} training",
            )

    plt.xlabel("Global checkpoint step")
    plt.ylabel("Accuracy")
    plt.title(f"Evaluation during sequential training: {format_experiment_title(experiment_root)}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    png_path = experiment_root / "training_eval_curve.png"
    pdf_path = experiment_root / "training_eval_curve.pdf"

    plt.savefig(png_path, dpi=200)
    plt.savefig(pdf_path)
    plt.close()

    print(f"Saved plot to {png_path}")
    print(f"Saved plot to {pdf_path}")


def main():
    args = parse_args()
    experiment_root = Path(args.experiment_root).resolve()

    if not experiment_root.exists():
        raise FileNotFoundError(f"Experiment root does not exist: {experiment_root}")

    baseline_eval_accuracy, baseline_model = resolve_experiment_baselines(
        experiment_root
    )

    dataset_order = resolve_dataset_order(args, experiment_root)
    phase_dirs = find_phase_dirs(experiment_root, dataset_order)

    if args.eval_subset is None:
        # Evaluate datasets whose phase folders currently exist. This supports
        # partial/manual experiment roots without requiring an explicit subset.
        eval_dataset_order = [
            dataset for dataset in dataset_order
            if dataset in phase_dirs
        ]
    else:
        # Evaluate the explicit subset on every checkpoint, even if some of
        # those phase folders do not exist yet. This is useful for per-phase
        # evaluation before cleanup. Example: after phase 1,
        # only the phase-1 folder exists, but we still want to evaluate phase-1
        # checkpoints on all intended eval datasets before cleanup removes old
        # checkpoint weights.
        requested_eval_subset = set(args.eval_subset)

        unknown = [
            dataset for dataset in args.eval_subset
            if dataset not in dataset_order
        ]
        if unknown:
            raise ValueError(
                f"--eval_subset contains dataset(s) not present in --dataset_order/"
                f"resolved dataset order: {unknown}. "
                f"Resolved dataset_order={dataset_order}"
            )

        eval_dataset_order = [
            dataset for dataset in dataset_order
            if dataset in requested_eval_subset
        ]

    print(f"Experiment root: {experiment_root}")
    if baseline_model is None:
        print("Baseline model: default (qwen2.5-7b)")
    else:
        print(f"Baseline model: {baseline_model}")
    available_baselines = [
        f"{name}={value:.6f}"
        for name, value in baseline_eval_accuracy.items()
        if value is not None
    ]
    print(
        "Dataset baselines: "
        + (", ".join(available_baselines) if available_baselines else "[none registered]")
    )
    print(f"Dataset train/eval order: {dataset_order}")

    for phase in dataset_order:
        if phase in phase_dirs:
            print(f"{phase:10s}: {phase_dirs[phase]}")
        else:
            print(f"{phase:10s}: [missing, skipped]")

    all_checkpoints = []
    existing_phases = [phase for phase in dataset_order if phase in phase_dirs]

    for phase in existing_phases:
        ckpts = sorted_checkpoints(phase_dirs[phase])
        if not ckpts:
            print(f"Warning: no checkpoints found in {phase_dirs[phase]}; skipping phase '{phase}'")
            continue
        all_checkpoints.extend(ckpts)

    if not all_checkpoints:
        raise RuntimeError("No checkpoints found in any existing phase folders.")

    for ckpt in all_checkpoints:
        maybe_run_all_evals_for_checkpoint(ckpt, args, eval_dataset_order)

    curve_data = build_curve_data(
        phase_dirs=phase_dirs,
        phase_order=dataset_order,
        eval_datasets=eval_dataset_order,
        baseline_eval_accuracy=baseline_eval_accuracy,
    )
    curve_data["baseline_metadata"] = {
        "model_name_or_path": baseline_model,
        "source": (
            "model_registry"
            if baseline_model is not None
            else "default_model_registry"
        ),
    }

    curve_json_path = experiment_root / "training_eval_curve.json"
    with open(curve_json_path, "w") as f:
        json.dump(curve_data, f, indent=2)
    print(f"Saved curve data to {curve_json_path}")

    plot_curve(
        curve_data=curve_data,
        experiment_root=experiment_root,
        phase_order=dataset_order,
        eval_dataset_order=eval_dataset_order,
    )


if __name__ == "__main__":
    main()
