#!/usr/bin/env python3

import sys
import json
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset_adapters import get_dataset_adapter
from eval_lib import run_standalone_eval


DATASET = "spatial_contradiction2"
MODEL = "Qwen/Qwen2.5-7B-Instruct"

OUTPUT_DIR = PROJECT_ROOT / "baselines" / "spatial_contradiction2_qwen25_7b"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    adapter = get_dataset_adapter(DATASET)

    print("=" * 80)
    print("Spatial Contradiction 2 — untouched baseline")
    print(f"Model:              {MODEL}")
    print(f"Dataset:            {DATASET}")
    print(f"Eval path:          {adapter.eval_path}")
    print(f"Temperature:        0.0")
    print(f"Max new tokens:     {adapter.default_max_new_tokens}")
    print(f"Max model length:   {adapter.default_max_model_len}")
    print("=" * 80)

    # This is the same evaluation path used by checkpoint evaluation:
    # - adapter-defined chat template
    # - vLLM
    # - greedy generation (temperature=0)
    # - adapter-defined max_new_tokens
    # - exact SpatialCoordinateVerifier scoring
    summary = run_standalone_eval(
        dataset_name=DATASET,
        model_path=MODEL,
        output_dir=str(OUTPUT_DIR),
        temperature=0.0,
        gpu_memory_utilization=0.8,
    )

    # run_standalone_eval already saves full per-example response records.
    responses_path = OUTPUT_DIR / adapter.responses_filename

    with responses_path.open("r", encoding="utf-8") as f:
        records = json.load(f)

    per_hop = defaultdict(lambda: {"num_total": 0, "num_correct": 0})

    for record in records:
        hop = int(record["hop_count"])
        per_hop[hop]["num_total"] += 1
        per_hop[hop]["num_correct"] += int(bool(record["correct"]))

    per_hop_summary = {}

    for hop in sorted(per_hop):
        total = per_hop[hop]["num_total"]
        correct = per_hop[hop]["num_correct"]

        per_hop_summary[str(hop)] = {
            "num_total": total,
            "num_correct": correct,
            "accuracy": correct / total if total else 0.0,
        }

    final_summary = {
        "dataset": DATASET,
        "model": MODEL,
        "eval_path": adapter.eval_path,
        "temperature": 0.0,
        "max_new_tokens": adapter.default_max_new_tokens,
        "max_model_len": adapter.default_max_model_len,
        "accuracy": float(summary["accuracy"]),
        "num_correct": int(summary["num_correct"]),
        "num_total": int(summary["num_total"]),
        "per_hop": per_hop_summary,
    }

    summary_path = OUTPUT_DIR / "baseline_summary.json"

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(final_summary, f, indent=2)

    print("\n" + "=" * 80)
    print("BASELINE RESULTS")
    print("=" * 80)
    print(
        f"Overall: {final_summary['num_correct']}/{final_summary['num_total']} "
        f"= {100 * final_summary['accuracy']:.2f}%"
    )

    print("\nPer hop:")
    for hop, stats in per_hop_summary.items():
        print(
            f"  hop {hop}: "
            f"{stats['num_correct']:3d}/{stats['num_total']:3d} "
            f"= {100 * stats['accuracy']:6.2f}%"
        )

    print("\nUse this value in DatasetSpec:")
    print(
        "baseline_accuracy="
        f"{final_summary['accuracy']:.6f}"
    )

    print(f"\nSummary saved to:   {summary_path}")
    print(f"Responses saved to: {responses_path}")
    print(
        f"Eval results:        "
        f"{OUTPUT_DIR / adapter.results_filename}"
    )


if __name__ == "__main__":
    main()
