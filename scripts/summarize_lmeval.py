#!/usr/bin/env python3
"""Summarize lm-eval-harness output with model-aware reference baselines.

The script locates the raw lm-eval JSON for one checkpoint, extracts the project
benchmark metrics, resolves the experiment's initial model from
``experiment_config.json``, and writes a compact ``lm_eval_summary.json`` with
baseline/checkpoint/delta percentages. If model metadata is absent, the project
default model's lm-eval baselines are used; an explicitly identified model with
no registered baselines remains unscored rather than borrowing another model's
values.
"""

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from model_registry import DEFAULT_MODEL_KEY, MODEL_REGISTRY, resolve_model_spec

# Default-model lm-eval baselines used when an experiment has no model metadata.
BASELINES_PERCENT = MODEL_REGISTRY[DEFAULT_MODEL_KEY].lm_eval_baselines_percent


METRIC_SPECS = {
    "hellaswag": {
        "display_name": "HellaSwag",
        "task": "hellaswag",
        "metric": "acc_norm",
        "higher_is_better": True,
    },
    "mmlu": {
        "display_name": "MMLU",
        "task": "mmlu",
        "metric": "acc",
        "higher_is_better": True,
    },
    "truthfulqa_mc2": {
        "display_name": "TruthfulQA (mc2)",
        "task": "truthfulqa_mc2",
        "metric": "acc",
        "higher_is_better": True,
    },
    "winogrande": {
        "display_name": "Winogrande",
        "task": "winogrande",
        "metric": "acc",
        "higher_is_better": True,
    },
    "ifeval": {
        "display_name": "IFEval",
        "task": "ifeval",
        "metric": "prompt_level_strict_acc",
        "higher_is_better": True,
    },
    "humaneval": {
        "display_name": "HumanEval",
        "task": "humaneval",
        "metric": "pass@1",
        "higher_is_better": True,
    },
}


PRIOR_TASK_KEYS = [
    "hellaswag",
    "mmlu",
    "truthfulqa_mc2",
    "winogrande",
    "ifeval",
]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def is_lmeval_result_json(path: Path) -> bool:
    try:
        data = load_json(path)
    except Exception:
        return False

    return isinstance(data, dict) and isinstance(data.get("results"), dict)


def find_lmeval_result_json(lm_eval_dir: Path) -> Path:
    candidates = sorted(lm_eval_dir.rglob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)

    valid = []
    for path in candidates:
        if path.name in {"lm_eval_summary.json", "lmeval_summary.json"}:
            continue
        if is_lmeval_result_json(path):
            valid.append(path)

    if not valid:
        raise FileNotFoundError(f"No valid lm_eval result JSON found under: {lm_eval_dir}")

    return valid[0]


def get_metric_value(task_results: dict[str, Any], task_name: str, metric_name: str) -> Optional[float]:
    if task_name not in task_results:
        return None

    task_dict = task_results[task_name]

    possible_keys = [
        metric_name,
        f"{metric_name},none",
        f"{metric_name},create_test",
    ]

    for key in possible_keys:
        if key in task_dict:
            return safe_float(task_dict[key])

    for key, value in task_dict.items():
        if key.startswith(metric_name):
            return safe_float(value)

    return None


def get_metric_stderr(task_results: dict[str, Any], task_name: str, metric_name: str) -> Optional[float]:
    if task_name not in task_results:
        return None

    task_dict = task_results[task_name]

    possible_keys = [
        f"{metric_name}_stderr",
        f"{metric_name}_stderr,none",
        f"{metric_name}_stderr,create_test",
    ]

    for key in possible_keys:
        if key in task_dict:
            return safe_float(task_dict[key])

    for key, value in task_dict.items():
        if key.startswith(f"{metric_name}_stderr"):
            return safe_float(value)

    return None


def safe_float(x: Any) -> Optional[float]:
    try:
        value = float(x)
    except (TypeError, ValueError):
        return None

    if math.isnan(value) or math.isinf(value):
        return None

    return value


def percent(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return 100.0 * value


def rounded(value: Optional[float], ndigits: int = 4) -> Optional[float]:
    if value is None:
        return None
    return round(value, ndigits)


def derive_exp_root_from_checkpoint(checkpoint_path: Path) -> Path:
    # Expected:
    #   exp_root / phase / checkpoint-XXX
    #
    # Example:
    #   outputs/fwd_student_ema/v2s7_old/science/checkpoint-250
    #   -> outputs/fwd_student_ema/v2s7_old
    return checkpoint_path.parent.parent


def resolve_lm_eval_baselines_percent(exp_root: Path) -> dict[str, float]:
    """Return lm-eval baselines for the experiment's explicitly selected model.

    Experiment folders without recorded model metadata use the project default
    model (Qwen2.5-7B-Instruct) baselines.

    Once an experiment explicitly records ``initial_model``, however, an empty
    baseline registry entry means "not measured yet" and must remain empty.
    Silently substituting Qwen2.5 baselines for Ministral or Qwen3.5 would make
    the reported deltas scientifically invalid.
    """

    config_path = exp_root / "experiment_config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return dict(BASELINES_PERCENT)

    initial_model = config.get("initial_model")
    if not initial_model:
        return dict(BASELINES_PERCENT)

    spec = resolve_model_spec(str(initial_model))
    return dict(spec.lm_eval_baselines_percent)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint-path",
        required=True,
        help="Path to the evaluated checkpoint, e.g. .../science/checkpoint-250",
    )
    parser.add_argument(
        "--lm-eval-dir",
        default=None,
        help="Directory containing lm_eval outputs. Default: checkpoint_path/lm_eval",
    )
    parser.add_argument(
        "--exp-root",
        default=None,
        help="Experiment root where the compact summary JSON should be stored. Default: parent of phase dir.",
    )
    parser.add_argument(
        "--output-name",
        default="lm_eval_summary.json",
        help="Name of compact summary JSON written in exp_root.",
    )
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint_path).resolve()
    lm_eval_dir = Path(args.lm_eval_dir).resolve() if args.lm_eval_dir else checkpoint_path / "lm_eval"
    exp_root = Path(args.exp_root).resolve() if args.exp_root else derive_exp_root_from_checkpoint(checkpoint_path)
    baselines_percent = resolve_lm_eval_baselines_percent(exp_root)

    result_json_path = find_lmeval_result_json(lm_eval_dir)
    raw = load_json(result_json_path)
    task_results = raw["results"]

    metrics: dict[str, Any] = {}

    for key, spec in METRIC_SPECS.items():
        value = get_metric_value(task_results, spec["task"], spec["metric"])
        stderr = get_metric_stderr(task_results, spec["task"], spec["metric"])

        value_percent = percent(value)
        stderr_percent = percent(stderr)

        baseline_percent = baselines_percent.get(key)
        delta_percent = None
        if value_percent is not None and baseline_percent is not None:
            delta_percent = value_percent - baseline_percent

        metrics[key] = {
            "display_name": spec["display_name"],
            "task": spec["task"],
            "metric": spec["metric"],
            "value": rounded(value, 6),
            "value_percent": rounded(value_percent, 4),
            "stderr": rounded(stderr, 6),
            "stderr_percent": rounded(stderr_percent, 4),
            "baseline_percent": rounded(baseline_percent, 4),
            "delta_percent": rounded(delta_percent, 4),
            "higher_is_better": spec["higher_is_better"],
        }

    prior_values = [
        metrics[key]["value_percent"]
        for key in PRIOR_TASK_KEYS
        if metrics.get(key, {}).get("value_percent") is not None
    ]

    prior_avg_percent = None
    if prior_values:
        prior_avg_percent = sum(prior_values) / len(prior_values)

    prior_baseline_percent = baselines_percent.get("prior_task_avg")
    prior_delta_percent = None
    if prior_avg_percent is not None and prior_baseline_percent is not None:
        prior_delta_percent = prior_avg_percent - prior_baseline_percent

    metrics["prior_task_avg"] = {
        "display_name": "Prior-task avg",
        "tasks": PRIOR_TASK_KEYS,
        "metric": "mean_percent",
        "value": None,
        "value_percent": rounded(prior_avg_percent, 4),
        "stderr": None,
        "stderr_percent": None,
        "baseline_percent": rounded(prior_baseline_percent, 4),
        "delta_percent": rounded(prior_delta_percent, 4),
        "higher_is_better": True,
    }

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "checkpoint_path": str(checkpoint_path),
        "experiment_root": str(exp_root),
        "lm_eval_dir": str(lm_eval_dir),
        "source_lm_eval_json": str(result_json_path),
        "notes": {
            "ifeval_metric": "prompt_level_strict_acc",
            "hellaswag_metric": "acc_norm",
            "truthfulqa_metric": "truthfulqa_mc2 acc",
            "values_are_percent_for_table_fields": True,
        },
        "metrics": metrics,
        "table_rows": [
            {
                "metric": metrics[key]["display_name"],
                "baseline_percent": metrics[key]["baseline_percent"],
                "checkpoint_percent": metrics[key]["value_percent"],
                "delta_percent": metrics[key]["delta_percent"],
            }
            for key in [
                "hellaswag",
                "mmlu",
                "truthfulqa_mc2",
                "winogrande",
                "ifeval",
                "prior_task_avg",
                "humaneval",
            ]
        ],
    }

    exp_root.mkdir(parents=True, exist_ok=True)
    output_path = exp_root / args.output_name

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Wrote compact lm_eval summary to: {output_path}")


if __name__ == "__main__":
    main()
