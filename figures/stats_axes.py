#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["pandas>=2.0", "pyarrow>=15", "scipy>=1.11", "matplotlib>=3.8"]
# ///
"""Per-(model, task) Wilcoxon signed-rank tests along one grid axis.

Binary axes (role, kl) implement the paper's protocol sentence: "Significance
along one binary axis is assessed per model and task with a Wilcoxon
signed-rank test over matched pairs of runs, identical except for that axis."
Pairs live on the canonical grid (3 LRs x 5 couplings incl. frozen x 2 KL x
2 seeds, canonical epochs, default context).

The ema axis (Sec. 6.2) is the coupling axis: each EMA rate (0.02..0.5) is
tested against the frozen teacher on the CONSTANT-DOSE slice (same number of
optimization steps for every task: 4 epochs, 6 for chemistry), pairs matched
on lr x KL x rollout source: 12 per (model, task, rate) when the slice is
complete (0.5 is off-grid and partial while runs land).

Differences are (A - B) for acquisition (task accuracy gain, pp) and retention
(lm-eval change, pp); seeds averaged within a config before pairing (the paper
protocol); two-sided p-values, Holm-corrected within each (model, metric[, rate])
family of testable tasks (n >= 6).

    python figures/stats_axes.py [role|kl|ema|all]     # default: all

Writes out/stats/stats_<axis>.csv, prints each table, and for the ema axis also
prints the LaTeX rows of stats_coupling_table.tex (tex_rows below: pure formatting
-- Holm stars + phantoms, 1/2 decimals, commented alpha=0.50 rows -- no statistics).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from load_runs import load_runs
from constants import CANON_EP, DISPLAY

OUT = Path(__file__).resolve().parent / "out" / "stats"

GRID_LRS = [5e-06, 1e-05, 2e-05]
GRID_RATES = [0.0, 0.02, 0.05, 0.10, 0.25]  # 0.0 = frozen reference
TASKS = ["tooluse", "science", "spatial_standard2", "spatial_contradiction2",
         "math_contradiction"]

# binary axes: column, (side A, side B); differences are A - B
AXES = {
    "role": ("role", ("teacher", "student")),
    "kl": ("direction", ("fwd", "bwd")),  # forward - reverse
}

# constant dose (Sec. 6.2): same number of optimization steps for every task
EQ_EP = {"tooluse": 4, "science": 6}  # others: 4
EMA_RATES = [0.02, 0.05, 0.10, 0.25, 0.5]  # each tested against frozen (0.0)


def constant_dose(df):
    """Runs of the constant-dose slice, grid LRs, coupling ladder incl. 0.5."""
    rate = df.ema_rate.where(df.ref_update != "frozen", 0.0)
    return df[(df.context == "default")
              & df.direction.isin(["fwd", "bwd"]) & df.role.notna()
              & df.lr.isin(GRID_LRS)
              & (df.epochs == df.dataset.map(EQ_EP).fillna(4))
              & rate.isin([0.0] + EMA_RATES)].assign(rate=rate)


def canonical_grid(df):
    """Runs of the canonical 20-config grid (both rollout sources, both KLs)."""
    rate = df.ema_rate.where(df.ref_update != "frozen", 0.0)
    return df[(df.context == "default")
              & df.direction.isin(["fwd", "bwd"]) & df.role.notna()
              & df.lr.isin(GRID_LRS)
              & (df.epochs == df.dataset.map(CANON_EP))
              & rate.isin(GRID_RATES)].assign(rate=rate)


def holm(p):
    """Holm step-down adjustment (returns adjusted p-values, same order)."""
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def test_row(model, task, p):
    row = {"model": model, "task": task, "n_pairs": len(p)}
    for name in ("acq", "ret"):
        d = (p[f"{name}_a"] - p[f"{name}_b"]).values
        row[f"median_d_{name}"] = float(np.median(d))
        row[f"p_{name}"] = (wilcoxon(d).pvalue if len(d) >= 6
                            and not np.allclose(d, 0) else np.nan)
    return row


def per_task_with_holm(pairs):
    """One row per (model, task), Holm within each (model, metric) family of
    testable tasks (untestable rows, n < 6, keep p = NaN)."""
    res = pd.DataFrame([test_row(model, DISPLAY.get(task, task), p)
                        for (model, task), p in pairs.groupby(["model", "dataset"])])
    for metric in ("acq", "ret"):
        res[f"p_{metric}_holm"] = np.nan
        for model, idx in res.groupby("model").groups.items():
            pv = res.loc[idx, f"p_{metric}"]
            ok = pv.notna()
            res.loc[pv.index[ok], f"p_{metric}_holm"] = holm(pv[ok].values)
    return res


def pooled_rows(pairs, by_lr=False):
    """ALL / CONTRADICTORY / ORDINARY rows per model (single tests, raw p);
    by_lr adds one all-tasks row per (model, lr) — the per-lr escalation rows
    of the KL table (Sec. 6.3: the Ministral flip grows with the lr)."""
    contra = pairs.index.get_level_values("dataset").isin(
        ["spatial_contradiction2", "math_contradiction"])
    pooled = pd.DataFrame(
        [test_row(model, "ALL", p) for model, p in pairs.groupby("model")]
        + [test_row(model, "CONTRADICTORY", p)
           for model, p in pairs[contra].groupby("model")]
        + [test_row(model, "ORDINARY", p)
           for model, p in pairs[~contra].groupby("model")]
        + ([test_row(model, f"ALL (lr {lr:g})", p)
            for (model, lr), p in pairs.groupby(["model", "lr"])] if by_lr
           else []))
    for metric in ("acq", "ret"):
        pooled[f"p_{metric}_holm"] = pooled[f"p_{metric}"]
    return pooled


def run_axis(axis):
    df = load_runs(retention="all").dropna(subset=["task_acc", "lmeval_avg"])

    if axis == "ema":
        # coupling axis: each rate vs the frozen teacher, constant-dose slice
        df = constant_dose(df)
        df = df.assign(acq=df.task_acc - df.task_acc_baseline,
                       ret=df.lmeval_avg - df.lmeval_avg_baseline)
        match = ["model", "dataset", "lr", "role", "direction"]
        cell = df.groupby(match + ["rate"])[["acq", "ret"]].mean()
        frozen = cell.xs(0.0, level="rate")
        blocks = []
        for r in EMA_RATES:
            pairs = (cell.xs(r, level="rate")
                     .join(frozen, lsuffix="_a", rsuffix="_b", how="inner"))
            if pairs.empty:
                continue
            block = pd.concat([per_task_with_holm(pairs), pooled_rows(pairs)],
                              ignore_index=True)
            block.insert(0, "rate", r)
            blocks.append(block)
        res = pd.concat(blocks, ignore_index=True)
        header = ("axis = ema: differences are rate - frozen, constant dose "
                  "(4 epochs, 6 for chemistry); two-sided Wilcoxon, Holm "
                  "within (model, metric, rate)")
    else:
        col, (side_a, side_b) = AXES[axis]
        df = canonical_grid(df)
        df = df.assign(acq=df.task_acc - df.task_acc_baseline,
                       ret=df.lmeval_avg - df.lmeval_avg_baseline)
        # matched pairs: identical on everything except the tested axis
        match = ["model", "dataset", "lr", "epochs", "rate"] + [
            c for c in ("direction", "role") if c != col]
        a = df[df[col] == side_a].set_index(match)[["acq", "ret"]]
        b = df[df[col] == side_b].set_index(match)[["acq", "ret"]]
        # a config cell can hold re-runs: average duplicates before pairing
        a = a.groupby(level=match).mean()
        b = b.groupby(level=match).mean()
        pairs = a.join(b, lsuffix="_a", rsuffix="_b", how="inner")
        res = pd.concat([per_task_with_holm(pairs), pooled_rows(pairs, by_lr=True)],
                        ignore_index=True)
        header = (f"axis = {axis}: differences are {side_a} - {side_b}; "
                  "two-sided Wilcoxon, Holm within (model, metric)")

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"stats_{axis}.csv"
    res.to_csv(path, index=False)

    print(header + "\n")
    show = res.assign(**{c: res[c].round(4) for c in res.columns if c.startswith("p_")},
                      **{c: res[c].round(2) for c in res.columns if c.startswith("median")})
    print(show.to_string(index=False))
    print(f"\nwrote {path}")
    if axis == "ema":
        print(tex_rows(path))


# ---- LaTeX rendering of the coupling table (pure formatting, no statistics) ---------
# per-task rows -> Holm p, grouped rows -> raw p; n labels from the alpha=.02 counts.
TEX_TASKS = ["tool-alpaca", "chemistry", "spatial", "spatial-contradiction",
         "math-contradiction", "ORDINARY", "CONTRADICTORY", "ALL"]
LABEL = {"ORDINARY": "ordinary", "CONTRADICTORY": "contradictory", "ALL": "all tasks"}
RATES = [0.02, 0.05, 0.10, 0.25, 0.50]


def stars(p):
    if pd.isna(p):
        return "\\phantom{***}"
    s = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    pad = "\\phantom{" + "*" * (3 - len(s)) + "}" if len(s) < 3 else ""
    return (s + pad) if s or pad else ""


def cell(med, p, dec):
    if pd.isna(med):
        return "--\\phantom{***}"
    return f"${med:+.{dec}f}$" + stars(p)


def tex_rows(csv_path=OUT / "stats_ema.csv"):
    """The tabular body of stats_coupling_table.tex, as one string."""
    df = pd.read_csv(csv_path)
    out = []
    for task in TEX_TASKS:
        grouped = task in LABEL
        sub = df[df.task == task]
        n = {}
        for m in ("qwen2.5-7b", "ministral-3-3b"):
            r = sub[(sub.model == m) & (sub.rate == 0.02)]
            n[m] = int(r.n_pairs.iloc[0]) if len(r) else 0
        name = f"{LABEL.get(task, task)} ({n['qwen2.5-7b']}/{n['ministral-3-3b']})"
        first = True
        for rate in RATES:
            cells = []
            for m in ("qwen2.5-7b", "ministral-3-3b"):
                r = sub[(sub.model == m) & (sub.rate == rate)]
                if len(r) == 0:
                    cells += ["--\\phantom{***}", "--\\phantom{***}"]
                    continue
                r = r.iloc[0]
                pa = r.p_acq if grouped else r.p_acq_holm
                pr = r.p_ret if grouped else r.p_ret_holm
                cells += [cell(r.median_d_acq, pa, 1), cell(r.median_d_ret, pr, 2)]
            lead = name if first else ""
            comment = "%" if rate == 0.50 else ""
            pad = " " * max(1, 29 - len(lead) - len(comment))
            cells = [c.ljust(23) for c in cells]
            tail = " % alpha=0.50 commented out (off-grid; kept on the figure, dropped from the table)" if rate == 0.50 else ""
            out.append(f"{comment}{lead}{pad}& ${rate:.2f}$ & "
                       + " & ".join(cells).rstrip() + " \\\\" + tail)
            first = False
        out.append("\\midrule" if task != "ALL" else "\\bottomrule")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("axis", nargs="?", default="all",
                    choices=sorted(AXES) + ["ema", "all"],
                    help="which table to refresh (default: all three)")
    args = ap.parse_args()
    for axis in (sorted(AXES) + ["ema"] if args.axis == "all" else [args.axis]):
        run_axis(axis)
        print()


if __name__ == "__main__":
    main()
