# /// script
# requires-python = ">=3.10"
# dependencies = ["pandas>=2.0", "pyarrow>=15"]
# ///
"""Load runs.parquet (built by collect_llm_experiments.py) and add analysis-ready columns.

All plotting/statistics code should import ``load_runs`` from here.
"""

import os
from pathlib import Path

import pandas as pd

DEFAULT_PARQUET = Path(__file__).resolve().parent / "data" / "runs.parquet"

LMEVAL_COMPONENTS = ["hellaswag", "mmlu", "truthfulqa_mc2", "winogrande", "ifeval"]
MODEL_ALIASES = {
    "Qwen/Qwen2.5-7B-Instruct": "qwen2.5-7b",
    "mistralai/Ministral-3-3B-Instruct-2512-BF16": "ministral-3-3b",
    "mistralai/Ministral-3-3B-Instruct-2512": "ministral-3-3b",
}
# Datasets the untrained model already handles (no learning signal) — dropped by default.
EXCLUDED_DATASETS = {"openr1_math", "openr1_math_qwen3", "medreason"}
# The 2-seed campaign the paper uses everywhere. Extra seeds (3-seed campaign for the
# rebuttal) are dropped by default until every cell has them; override with the
# SDFT_SEEDS env var ("all" or a comma list) or the seeds argument.
CANONICAL_SEEDS = (13, 31)
BASELINE_METHODS = {"cpt", "ce_dataset_sft"}
CONTRADICTORY_DATASETS = {"math_contradiction", "spatial_contradiction", "spatial_contradiction2"}

# Factor columns used across plots/tests (all categorical).
FACTORS = ["model", "dataset", "method", "direction", "role", "ref_update", "ema_rate",
           "context", "lr", "epochs", "seed"]


def read_table(path: Path, table: str) -> pd.DataFrame:
    """Read only the rows of one table kind; a parquet without the column is all runs."""
    import pyarrow.parquet as pq
    if "table" not in pq.read_schema(path).names:
        df = pd.read_parquet(path)
        return df if table == "runs" else df.iloc[0:0]
    return pd.read_parquet(path, filters=[("table", "==", table)]).reset_index(drop=True)


def _seed_filter(seeds=None):
    """Resolve the seed restriction: None → CANONICAL_SEEDS / SDFT_SEEDS; 'all' → no filter."""
    seeds = seeds if seeds is not None else os.environ.get("SDFT_SEEDS") or CANONICAL_SEEDS
    if isinstance(seeds, str):
        if seeds.lower() == "all":
            return None
        seeds = [int(s) for s in seeds.split(",")]
    return set(seeds)


def load_runs(path: Path | str | None = None, exclude_datasets=EXCLUDED_DATASETS,
              retention: str | None = None, seeds=None) -> pd.DataFrame:
    """retention: only 'all' here (the 5-benchmark prior-task average) — the paper's
    figures and tables use nothing else; the no_ifeval / ifeval_only splits live in
    old/load_runs.py with the exploration scripts.
    seeds: restriction to a seed set — default CANONICAL_SEEDS (or SDFT_SEEDS), 'all' disables."""
    path = Path(path or os.environ.get("SDFT_RUNS_PARQUET") or DEFAULT_PARQUET)
    retention = retention or "all"
    if retention != "all":
        raise ValueError("retention splits other than 'all' retired to old/load_runs.py")
    comps = LMEVAL_COMPONENTS
    df = read_table(path, "runs")
    if exclude_datasets:
        df = df[~df["trained_dataset"].isin(exclude_datasets)].reset_index(drop=True)
    keep = _seed_filter(seeds)
    if keep is not None:
        df = df[df["seed"].isin(keep)].reset_index(drop=True)

    # Short, uniform names for the main factors.
    df["model"] = df["model_key"].map(
        lambda m: MODEL_ALIASES.get(m, m.removeprefix("model:").split("/")[-1].lower()))
    df["dataset"] = df["trained_dataset"]
    df["method"] = df["new_name"]
    df["context"] = df["context_suffix"].replace(
        # the alpha = 0.5 arms carry an "_ema050_default" method suffix whose "default"
        # leaks into the context tag: they are ordinary default-context runs
        {"dataset_default": "default"})
    df["lr"] = df["LR"].astype(float)
    df["epochs"] = df["num_train_epochs"].astype(int)
    df["is_baseline"] = df["method"].isin(BASELINE_METHODS)
    df["contradictory"] = df["dataset"].isin(CONTRADICTORY_DATASETS)
    # Compact method label without the ema-rate suffix (rate is its own factor).
    df["method_base"] = df["method"].str.replace(r"ema\d{3}$", "ema", regex=True)

    # Forgetting: mean of the lm-eval component deltas (percentage points; negative = forgot).
    delta_cols = [f"lmeval_delta_{c}" for c in LMEVAL_COMPONENTS if f"lmeval_delta_{c}" in df]
    df["forgetting"] = df[delta_cols].mean(axis=1, skipna=False)
    # Absolute scales (percent), with the untrained model's value as reference.
    df["task_acc"] = df["final_accuracy"].astype(float) * 100
    df["task_acc_baseline"] = (df["final_accuracy"] - df["accuracy_delta"]).astype(float) * 100
    df["task_acc_baseline"] = df.groupby(["model", "dataset"])["task_acc_baseline"].transform("median")
    vals = df[[f"lmeval_{c}_value_percent" for c in comps]].astype(float)
    bases = df[[f"lmeval_{c}_baseline_percent" for c in comps]].astype(float)
    df["lmeval_avg"] = vals.mean(axis=1, skipna=False)
    df["lmeval_avg_baseline"] = bases.mean(axis=1, skipna=False)
    df["retention_set"] = retention
    # Learning: accuracy delta on the trained task (fraction) — kept for reference.
    df["learning"] = df["learning"].astype(float)
    return df


def summarize_scale(df: pd.DataFrame) -> str:
    """Text overview of what the table contains (for quick sanity checks)."""
    lines = [f"runs: {len(df)}   usable (learning & forgetting present): "
             f"{df[['learning', 'forgetting']].dropna().shape[0]}"]
    for col in ["model", "dataset", "method", "context", "lr", "epochs", "seed"]:
        vc = df[col].value_counts()
        lines.append(f"{col} ({len(vc)}): " + ", ".join(f"{k}={v}" for k, v in vc.head(12).items())
                     + (" ..." if len(vc) > 12 else ""))
    cell = ["model", "dataset", "method", "context", "lr", "epochs"]
    seeds = df.groupby(cell)["seed"].nunique()
    lines.append(f"cells (model,dataset,method,context,lr,epochs): {len(seeds)}; "
                 f"seeds per cell: {seeds.value_counts().sort_index().to_dict()}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(summarize_scale(load_runs()))
