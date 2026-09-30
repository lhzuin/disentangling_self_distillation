#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["pandas>=2.0", "pyarrow>=15"]
# ///
"""Collect per-seed experiment CSVs and per-run JSONs into a single parquet.

Run on the machine holding the outputs folders referenced by the CSV's
``version_dirs`` column (no flag needed there), or anywhere with --outputs-root
to remap the path prefix. Rebuilds the full table on every invocation.

Also works fully locally (no outputs tree): when a run's lm_eval_summary.json is
missing, the per-benchmark lm-eval columns are reconstructed from the CSV's
``lmeval_final_* / lmeval_delta_* / lmeval_stderr_*`` columns (baseline = final
− delta), and --model-key SUBSTR=KEY assigns the model from the CSV path, e.g.:
    python figures/collect_llm_experiments.py \
        metrics_out/qwen25_7b/all_methods/complete/context_single_phase_all_per_seed.csv \
        metrics_out/ministral3b/all_methods/complete/context_single_phase_all_per_seed.csv \
        --dynamics-checkpoints metrics_out/*/all_methods/complete/single_phase_training_dynamics_checkpoints.csv \
        --dynamics-steps metrics_out/*/all_methods/complete/single_phase_training_dynamics_steps.csv \
        --model-key qwen25_7b=qwen2.5-7b --model-key ministral3b=ministral-3-3b \
        --out-dir figures/data
Only the cfg_* columns (unused downstream) then stay empty.

Optionally folds the intermediate training-dynamics tables (per-checkpoint
accuracies, per-time-bin step metrics) into the same parquet. A ``table``
column separates the row kinds: "runs", "dynamics_checkpoints",
"dynamics_steps". Dynamics rows are matched to their run (to inherit
``model_key`` and the config-based ``ema_rate``) through
``run_id == <last two components of version_dirs>``.

Usage (needs pandas + pyarrow; `uv run` also works via the inline metadata):
    python figures/collect_llm_experiments.py \
        metrics_out/summaries/context_per_dataset/all_methods/complete/context_single_phase_all_per_seed.csv \
        metrics_out/ministral3b/all_methods/complete/context_single_phase_all_per_seed.csv \
        --dynamics-checkpoints \
            metrics_out/summaries/context_per_dataset/all_methods/complete/single_phase_training_dynamics_checkpoints.csv \
            metrics_out/ministral3b/all_methods/complete/single_phase_training_dynamics_checkpoints.csv \
        --dynamics-steps \
            metrics_out/summaries/context_per_dataset/all_methods/complete/single_phase_training_dynamics_steps.csv \
            metrics_out/ministral3b/all_methods/complete/single_phase_training_dynamics_steps.csv \
        --out-dir figures/data \
        [--outputs-root OLD=NEW]

Only the seeds listed in --seeds (default: 13 31, the two grid seeds) are kept,
for runs and dynamics alike; rows from other, partial seeds are discarded and
reported. Everything else (methods, learning rates, ... outside the fixed grid)
is kept.

Writes DIR/runs.parquet and prints a summary of join coverage.
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd

DEFAULT_MODEL_KEY = "qwen2.5-7b"
BASELINE_METHODS = {"cpt", "ce_dataset_sft"}
DEFAULT_EMA_RATE = 0.02
DEFAULT_SEEDS = [13, 31]  # the two grid seeds; other seeds are partial reruns
RUN_SEED_PATTERN = (
    r"^(?:v\d+|lr(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?)_?s(\d+)"
)


def run_seed(value) -> int | None:
    """Extract a seed from current ``lr...`` or legacy ``vN`` run names."""

    match = re.match(RUN_SEED_PATTERN, str(value))
    return int(match.group(1)) if match else None


def seed_filter(df: pd.DataFrame, seeds: list[int]) -> tuple[pd.DataFrame, dict]:
    """Keep rows whose ``seed`` is in ``seeds``; report what was discarded."""
    keep = df["seed"].isin(seeds)
    discarded = df.loc[~keep, "seed"]
    counts = {("?" if pd.isna(k) else int(k)): int(v)
              for k, v in discarded.value_counts(dropna=False).items()}
    return df[keep].reset_index(drop=True), counts


def load_json(path: Path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None


def coerce_column(series: pd.Series) -> pd.Series:
    """JSON fields mix int/str across runs; make each column uniformly numeric or string."""
    if series.dtype != object:
        return series
    non_null = series.dropna()
    if non_null.empty:
        return series
    if non_null.map(lambda v: isinstance(v, bool)).all():
        return series.astype("boolean")
    numeric = pd.to_numeric(non_null, errors="coerce")
    if numeric.notna().all():
        return pd.to_numeric(series, errors="coerce")
    return series.where(series.isna(), series.astype(str))


def flatten_config(cfg: dict) -> dict:
    out = {}
    for key, value in cfg.items():
        if key == "common_args":
            if isinstance(value, dict):
                out["cfg_common_num_train_epochs"] = value.get("num_train_epochs")
        elif isinstance(value, (list, tuple)):
            out[f"cfg_{key}"] = "->".join(str(v) for v in value)
        elif isinstance(value, (str, int, float, bool)) or value is None:
            out[f"cfg_{key}"] = value
    return out


def flatten_lmeval(summary: dict) -> dict:
    out = {}
    for name, entry in (summary.get("metrics") or {}).items():
        if not isinstance(entry, dict):
            continue
        for field in ("value_percent", "stderr_percent", "baseline_percent", "delta_percent"):
            out[f"lmeval_{name}_{field}"] = entry.get(field)
    return out


def lmeval_from_csv(row: pd.Series) -> dict:
    """Rebuild the lmeval_* percent columns from the per-seed CSV itself.

    CSV lmeval_final_* / lmeval_delta_* are already in percent; lmeval_stderr_*
    is a fraction. baseline = final - delta."""
    out = {}
    for col in row.index:
        if not col.startswith("lmeval_final_"):
            continue
        name = col.removeprefix("lmeval_final_")
        final, delta = row.get(col), row.get(f"lmeval_delta_{name}")
        if pd.isna(final):
            continue
        out[f"lmeval_{name}_value_percent"] = final
        out[f"lmeval_{name}_delta_percent"] = delta
        out[f"lmeval_{name}_baseline_percent"] = final - delta if pd.notna(delta) else None
        stderr = row.get(f"lmeval_stderr_{name}")
        out[f"lmeval_{name}_stderr_percent"] = stderr * 100 if pd.notna(stderr) else None
    return out


def parse_method(method: str, cfg: dict) -> dict:
    """Factor columns (direction/role/ref_update/ema_rate) from a method name + optional config."""
    if method in BASELINE_METHODS:
        return {"direction": "baseline", "role": None, "ref_update": None, "ema_rate": None}
    out = {
        "direction": "fwd" if method.startswith("fwd") else "bwd" if method.startswith("bwd") else None,
        "role": "teacher" if "teacher" in method else "student" if "student" in method else None,
        "ref_update": "frozen" if "frozen" in method else "ema" if "ema" in method else None,
    }
    if cfg.get("ref_model_mixup_alpha") not in (None, ""):
        out["ema_rate"] = float(cfg["ref_model_mixup_alpha"])
    elif out["ref_update"] == "ema":
        # name-based fallback, inferred: suffix is a zero-padded rate (005 -> 0.05)
        m = re.search(r"ema(\d{3})", method)
        out["ema_rate"] = int(m.group(1)) / 100 if m else DEFAULT_EMA_RATE
    else:
        # configs record ref_model_mixup_alpha = 0.0 for frozen references
        out["ema_rate"] = 0.0 if out["ref_update"] == "frozen" else None
    return out


def derive_columns(row: pd.Series, cfg: dict, model_map: list[tuple[str, str]]) -> dict:
    out = parse_method(str(row.get("new_name", "")), cfg)

    if cfg.get("seed") not in (None, ""):
        out["seed"] = int(cfg["seed"])
    else:
        out["seed"] = run_seed(row.get("run_version", ""))

    csv_model = row.get("model")
    csv_model = None if pd.isna(csv_model) or csv_model == "" else str(csv_model)
    mapped = next((key for sub, key in model_map if sub in str(row.get("source_csv", ""))), None)
    out["model_key"] = (cfg.get("model_identity") or cfg.get("initial_model")
                        or csv_model or mapped or DEFAULT_MODEL_KEY)
    return out


def run_key(path: str) -> str:
    """'<...>/<method_folder>/<version_dir>' -> '<method_folder>/<version_dir>' (== dynamics run_id)."""
    parts = [p for p in str(path).split("/") if p]
    return "/".join(parts[-2:])


def collect(csv_paths: list[Path], remaps: list[tuple[str, str]],
            model_map: list[tuple[str, str]], seeds: list[int]) -> tuple[pd.DataFrame, dict]:
    frames = []
    for path in csv_paths:
        df = pd.read_csv(path)
        df["source_csv"] = str(path)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    n_read = len(df)
    df = df.drop_duplicates(subset="version_dirs", keep="first").reset_index(drop=True)
    df["seed"] = pd.to_numeric(
        df["run_version"].astype(str).str.extract(RUN_SEED_PATTERN, expand=False),
        errors="coerce",
    )
    df, seed_discarded = seed_filter(df, seeds)
    df = df.drop(columns=["seed"])  # re-derived per row below (config value preferred)

    extras = []
    n_config = n_lmeval = 0
    for _, row in df.iterrows():
        run_dir = str(row.get("version_dirs", ""))
        for old, new in remaps:
            if run_dir.startswith(old):
                run_dir = new + run_dir[len(old):]
                break
        run_dir = Path(run_dir)

        cfg = load_json(run_dir / "experiment_config.json") or {}
        lmeval = load_json(run_dir / "lm_eval_summary.json") or {}
        n_config += bool(cfg)
        n_lmeval += bool(lmeval)

        extra = {"run_dir_resolved": str(run_dir), "config_found": bool(cfg), "lmeval_found": bool(lmeval),
                 "lmeval_source": "json" if lmeval else "csv"}
        extra.update(flatten_config(cfg))
        extra.update(flatten_lmeval(lmeval) if lmeval else lmeval_from_csv(row))
        extra.update(derive_columns(row, cfg, model_map))
        extras.append(extra)

    extra_df = pd.DataFrame(extras)
    combined = pd.concat([df, extra_df], axis=1)
    stats = {"rows_read": n_read, "rows_written": len(combined),
             "configs_found": n_config, "lmevals_found": n_lmeval,
             "seed_discarded": seed_discarded}
    return combined, stats


def norm_path(p) -> str:
    return "/".join(x for x in str(p).split("/") if x)


def collect_dynamics(csv_paths: list[Path], kind: str, runs_df: pd.DataFrame,
                     seeds: list[int], ckpt_map: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Read training-dynamics CSVs ('checkpoints' or 'steps'), attach run metadata, dedupe.

    ``run_id`` ('<method_folder>/<version_dir>') is NOT unique across models, so rows
    are matched to the runs table by full path (``checkpoint_dir``) when available.
    Steps rows carry no path: they inherit the mapping of the checkpoints file living
    in the same source folder (``ckpt_map[(source_dir, run_id)]``), and only fall back
    to run_id when it is unambiguous in the runs table.
    """
    frames = []
    for path in csv_paths:
        d = pd.read_csv(path)
        d["source_csv"] = str(path)
        d["source_dir"] = str(Path(path).parent)
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    n_read = len(d)
    d["seed"] = pd.to_numeric(
        d["run_id"].astype(str).str.split("/").str[-1].str.extract(
            RUN_SEED_PATTERN,
            expand=False,
        ),
        errors="coerce",
    )
    d, seed_discarded = seed_filter(d, seeds)
    if kind == "checkpoints":
        step_col = "checkpoint_step"
    else:  # binned summaries use time_bin; raw per-step logs use global_step
        step_col = next(c for c in ("time_bin", "global_step", "reported_step") if c in d.columns)

    # Lookups from the runs table.
    path_lookup = {}    # normalized version_dirs -> (model_key, cfg ema_rate)
    root_models = {}    # outputs root -> models seen under it
    by_run_id = {}      # run_id -> [(model_key, ema_rate), ...]
    for vd, mk, er in zip(runs_df["version_dirs"], runs_df["model_key"], runs_df["ema_rate"]):
        key = norm_path(vd)
        path_lookup[key] = (mk, er)
        root_models.setdefault("/".join(key.split("/")[:-2]), set()).add(mk)
        by_run_id.setdefault(run_key(vd), []).append((mk, er))
    root_model = {k: next(iter(v)) for k, v in root_models.items() if len(v) == 1}
    rk_unique = {k: v[0] for k, v in by_run_id.items() if len(v) == 1}

    has_ckpt_dir = "checkpoint_dir" in d.columns
    cols = d[["run_id", "source_dir"] + (["checkpoint_dir"] if has_ckpt_dir else [])]
    models, rates, matcheds = [], [], []
    for rec in cols.itertuples(index=False):
        model = rate = None
        matched = False
        ckpt_dir = getattr(rec, "checkpoint_dir", None) if has_ckpt_dir else None
        if isinstance(ckpt_dir, str) and ckpt_dir:
            # checkpoint_dir = <run_dir>/<dataset>/checkpoint-N
            run_dir = "/".join(norm_path(ckpt_dir).split("/")[:-2])
            hit = path_lookup.get(run_dir)
            if hit is not None:
                model, rate = hit
                matched = True
            else:  # run absent from the per-seed CSVs: at least infer the model from the root
                model = root_model.get("/".join(run_dir.split("/")[:-2]))
        if model is None and ckpt_map:
            hit = ckpt_map.get((rec.source_dir, str(rec.run_id)))
            if hit is not None:
                model, rate, matched = hit
        if model is None:
            hit = rk_unique.get(str(rec.run_id))
            if hit is not None:
                model, rate = hit
                matched = True
        models.append(model)
        rates.append(rate)
        matcheds.append(matched)
    d["model_key"] = pd.Series(models, index=d.index, dtype=object)
    d["_cfg_rate"] = pd.Series(rates, index=d.index, dtype=object)
    d["run_matched"] = matcheds
    # Baseline checkpoint rows have no checkpoint_dir: fill from sibling rows of the
    # same run in the same source file (fills only missing values).
    grp = d.groupby(["source_csv", "run_id"], sort=False)
    d["model_key"] = grp["model_key"].transform(lambda s: s.ffill().bfill())
    d["_cfg_rate"] = grp["_cfg_rate"].transform(lambda s: s.ffill().bfill())
    d["run_matched"] = grp["run_matched"].transform("max")

    parsed = d["experiment"].map(lambda m: parse_method(str(m), {}))
    for field in ("direction", "role", "ref_update", "ema_rate"):
        d[field] = parsed.map(lambda pr: pr[field])
    d["ema_rate"] = d["_cfg_rate"].where(d["_cfg_rate"].notna(), d["ema_rate"])

    # Dedupe one row per (model, run, step); source_csv breaks ties for rows without
    # a model so two different-model files with a colliding run_id are both kept.
    d["_model"] = d["model_key"].astype(object).where(d["model_key"].notna(), d["source_csv"])
    d = (d.drop_duplicates(subset=["_model", "run_id", step_col], keep="first")
          .reset_index(drop=True))

    # (source_dir, run_id) -> (model, cfg rate, matched), for the steps file of the same folder.
    out_map = {}
    known = d[d["model_key"].notna()].drop_duplicates(["source_dir", "run_id"])
    for sdir, rid, mk, cr, mt in zip(known["source_dir"], known["run_id"], known["model_key"],
                                     known["_cfg_rate"], known["run_matched"]):
        out_map[(sdir, str(rid))] = (mk, cr, bool(mt))

    d = d.drop(columns=["_model", "_cfg_rate", "source_dir"])
    d["table"] = f"dynamics_{kind}"

    unmatched = sorted(d.loc[~d["run_matched"], "run_id"].astype(str).unique())
    stats = {"rows_read": n_read, "rows_written": len(d),
             "rows_matched": int(d["run_matched"].sum()),
             "model_known": int(d["model_key"].notna().sum()),
             "seed_discarded": seed_discarded,
             "unmatched_run_ids": unmatched, "out_map": out_map}
    return d, stats


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csvs", nargs="+", type=Path, help="per-seed summary CSV file(s)")
    parser.add_argument("--dynamics-checkpoints", nargs="+", type=Path, default=[], metavar="CSV",
                        help="single_phase_training_dynamics_checkpoints.csv file(s)")
    parser.add_argument("--dynamics-steps", nargs="+", type=Path, default=[], metavar="CSV",
                        help="single_phase_training_dynamics_steps.csv file(s)")
    parser.add_argument("--out-dir", required=True, type=Path, help="output directory for runs.parquet")
    parser.add_argument("--outputs-root", action="append", default=[], metavar="OLD=NEW",
                        help="remap a version_dirs path prefix (repeatable)")
    parser.add_argument("--model-key", action="append", default=[], metavar="SUBSTR=KEY",
                        help="model key for rows whose source CSV path contains SUBSTR, "
                             "used when experiment_config.json is unavailable (repeatable)")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS,
                        help="seeds to keep, for runs and dynamics alike; rows from other "
                             "(partial) seeds are discarded (default: %(default)s)")
    args = parser.parse_args()

    remaps = []
    for spec in args.outputs_root:
        old, sep, new = spec.partition("=")
        if not sep:
            parser.error(f"--outputs-root must be OLD=NEW, got: {spec}")
        remaps.append((old, new))

    model_map = []
    for spec in args.model_key:
        sub, sep, key = spec.partition("=")
        if not sep:
            parser.error(f"--model-key must be SUBSTR=KEY, got: {spec}")
        model_map.append((sub, key))

    runs_df, stats = collect(args.csvs, remaps, model_map, args.seeds)
    runs_df["table"] = "runs"

    tables = [runs_df]
    dyn_stats = {}
    ckpt_map: dict = {}
    for kind, paths in (("checkpoints", args.dynamics_checkpoints), ("steps", args.dynamics_steps)):
        if paths:
            ddf, dstats = collect_dynamics(paths, kind, runs_df, args.seeds, ckpt_map)
            ckpt_map.update(dstats.pop("out_map"))
            tables.append(ddf)
            dyn_stats[kind] = dstats

    df = pd.concat(tables, ignore_index=True) if len(tables) > 1 else runs_df
    for col in df.columns:
        df[col] = coerce_column(df[col])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "runs.parquet"
    df.to_parquet(out_path, engine="pyarrow", index=False)

    n = stats["rows_written"]
    print(f"Seed filter: keeping only seeds {args.seeds}.")
    all_discarded = {"runs": stats["seed_discarded"],
                     **{f"dynamics_{k}": s["seed_discarded"] for k, s in dyn_stats.items()}}
    for name, counts in all_discarded.items():
        if counts:
            detail = ", ".join(f"seed {k}: {v} rows" for k, v in sorted(counts.items(), key=str))
            print(f"  {name}: discarded {sum(counts.values())} rows from "
                  f"{len(counts)} partial seed(s) ({detail})")
        else:
            print(f"  {name}: nothing discarded")
    print(f"runs: rows read {stats['rows_read']}  (after seed filter + dedup on version_dirs: {n})")
    print(f"experiment_config.json found: {stats['configs_found']}/{n}   missing: {n - stats['configs_found']}")
    print(f"lm_eval_summary.json  found: {stats['lmevals_found']}/{n}   missing: {n - stats['lmevals_found']}")
    for kind, s in dyn_stats.items():
        nd = s["rows_written"]
        print(f"dynamics_{kind}: rows read {s['rows_read']}  (after dedup: {nd})   "
              f"matched to a run: {s['rows_matched']}/{nd}   model known: {s['model_known']}/{nd}")
        if s["unmatched_run_ids"]:
            shown = s["unmatched_run_ids"][:5]
            print(f"  unmatched run_ids ({len(s['unmatched_run_ids'])} unique), e.g.: " + "; ".join(shown))
    print("\nRows per table:")
    print(df["table"].value_counts().to_string())
    print("\nRuns rows per model_key:")
    print(runs_df["model_key"].value_counts().to_string())
    print("\nRuns rows per trained_dataset:")
    print(runs_df["trained_dataset"].value_counts().to_string())
    print("\nRuns rows per method (new_name):")
    print(runs_df["new_name"].value_counts().to_string())
    print(f"\nWrote {out_path}  ({len(df)} rows, {len(df.columns)} columns)")


if __name__ == "__main__":
    main()
