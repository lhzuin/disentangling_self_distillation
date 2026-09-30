# /// script
# requires-python = ">=3.10"
# dependencies = ["pandas>=2.0", "pyarrow>=15", "matplotlib==3.11.1", "scipy>=1.11"]
# ///
"""Fig 2 (teacher coupling): the paper pair at print height + the 7 appendix combos.

    uv run --offline figures/fig2.py
"""

from fig_common import (BLUE, BUDGET_LR, FINAL, GOLD_SWITCH, GREY_SWITCH, ORANGE, ROOT,
                        cells_table)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CSV = ROOT / "results/v14_final/aggregate/ema_paper_curves.csv"
# model panel slice (finite autoregressive model) and its curve styling
FIG2_MODEL = dict(lam=2.5, kl="forward", rollout="teacher", c=(0.9, 0.6, 0.3))
STYLE = {0.3: ("tab:blue", "o"), 0.6: ("tab:orange", "s"), 0.9: ("tab:green", "^")}
# experimental tasks: colour families follow the model curves (same hue, tasks of
# similar context informativeness close in shade)
TASK_STYLE = {"tooluse": ("#2ca02c", "^"), "science": ("#93d693", "v"),
              "spatial_standard2": ("#1f77b4", "o"), "spatial_contradiction2": ("#77aed2", "D"),
              "math_contradiction": ("#ff7f0e", "s")}
EQUAL_DOSE_EP = {"tooluse": 4, "science": 6}  # constant-dose slice; every other task: 4
SHRINK = 0.25  # print height = 1.7 in minus this (empirical floor: the ylabel clips lower)
# Every number inside the drawing code below is STYLE only (sizes, margins, fonts,
# legend and annotation positions, axis cosmetics): moving it moves ink, never points.


def draw_model(ax, compact=False):
    df = pd.read_csv(CSV)
    df = df[(abs(df["lambda"] - FIG2_MODEL["lam"]) < 1e-6)
            & (df.kl_direction == FIG2_MODEL["kl"])
            & (df.rollout_source == FIG2_MODEL["rollout"])]
    alphas = sorted(df.ema_alpha.unique())
    # geometric grid + 0 -> even index spacing
    x_of = {a: i for i, a in enumerate(alphas)}
    for c in FIG2_MODEL["c"]:
        s = df[abs(df.c - c) < 1e-9].sort_values("ema_alpha")
        if s.empty:
            continue
        y = s.new_match_gain__mean * 100
        se = s.new_match_gain__std / np.sqrt(s.new_match_gain__count) * 100
        xs = [x_of[a] for a in s.ema_alpha]
        colr, mk = STYLE.get(c, (None, "o"))
        ax.fill_between(xs, y - 1.96 * se, y + 1.96 * se, color=colr, alpha=0.15, lw=0, zorder=2)
        ax.plot(xs, y, "-", marker=mk, ms=2.8 if compact else 4.5, lw=1.1 if compact else 1.6,
                color=colr, label=f"$\\kappa$ = {c:g}", zorder=3)
        i_best = int(y.values.argmax())  # alpha*: emphasised marker instead of a label
        ax.scatter([xs[i_best]], [y.iloc[i_best]], s=26 if compact else 70, marker=mk, color=colr,
                   edgecolors="black", linewidths=0.8 if compact else 1.0, zorder=4)
        print(f"c={c:g}: alpha*={s.ema_alpha.iloc[i_best]:g}  acq={y.iloc[i_best]:.1f} pp "
              f"(alpha=0: {y.iloc[0]:.1f}, alpha=max: {y.iloc[-1]:.1f})")
    ax.axhline(0, color="grey", lw=0.7, zorder=1)
    # even index spacing with round grid values: same visual rhythm as the experiment
    # panel's 0/0.02/0.05/0.1/0.25 (the two alpha ranges differ; the analogy is the shape)
    shown = [0.0, 0.0025, 0.01, 0.04, 0.16]
    ax.set_xticks([x_of[a] for a in shown], ["0", "0.0025", "0.01", "0.04", "0.16"])
    fs, ft = (8.5, 7.5) if compact else (10, 10)
    ax.set_xlabel("EMA rate $\\alpha$ (0 = frozen)", fontsize=fs)
    ax.set_ylabel("acquisition (pp)" if compact else "new-task acquisition (pp)", fontsize=fs)
    ax.tick_params(labelsize=ft)
    ax.grid(lw=0.3, alpha=0.4)
    ax.legend(fontsize=7 if compact else 9, frameon=False,
              loc="upper right" if compact else "lower left",  # compact: top right is empty
              handlelength=1.4, handletextpad=0.4, labelspacing=0.3)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def draw_experiment(ax, role="student", kl="reverse", force_lr=None,
                    equal_dose=False, sftnorm=False, compact=False,
                    model="qwen2.5-7b"):
    """One model, one rollout source, one KL: mean gain (pts) per EMA rate at the fixed
    (LR, epochs) of plot_cube --commute-lr (same selection rule, values match that figure)."""
    from load_runs import load_runs
    from constants import DISPLAY

    df = load_runs(retention="all").dropna(subset=["task_acc", "lmeval_avg"])
    df = df[(df.context == "default") & (df.model == model)]
    df = df.assign(learn=df.task_acc - df.task_acc_baseline)
    sft = df[df.method == "ce_dataset_sft"]
    dirn = "bwd" if kl == "reverse" else "fwd"
    stu = df[(df.role == role) & df.direction.isin(["fwd", "bwd"])]
    # tick ladder from the rates that can actually appear (this KL side, this LR):
    # avoids empty slots when e.g. 0.5 only exists for another combo
    pool = stu[(stu.ref_update == "ema") & (stu.direction == dirn)]
    if force_lr is not None:
        pool = pool[pool.lr == force_lr]
    if equal_dose:
        pool = pool[pool.epochs == pool.dataset.map(EQUAL_DOSE_EP).fillna(4)]
    all_rates = sorted(pool.ema_rate.dropna().unique())
    ticks = [0.0] + all_rates
    x_of = {r: i for i, r in enumerate(ticks)}
    ymax = 0.0
    for d, (colr, mk) in TASK_STYLE.items():
        sub = stu[stu.dataset == d]
        anchor = sub[(sub.direction == "bwd") & (sub.ref_update == "frozen")]
        if anchor.empty:
            continue
        by_lr = anchor.groupby("lr").learn.mean().sort_values(ascending=False)
        ema = sub[sub.ref_update == "ema"]
        cov = ema.groupby(["lr", "epochs"]).ema_rate.nunique()
        cov_lr = cov.groupby(level=0).max() if len(cov) else pd.Series(dtype=int)
        need = min(3, cov.max()) if len(cov) else 0
        lr = (force_lr if force_lr is not None else
              next((l for l in by_lr.index if cov_lr.get(l, 0) >= need), by_lr.index[0]))
        if lr not in by_lr.index:
            continue
        anc = anchor[anchor.lr == lr]
        ep_anchor = anc.groupby("epochs").learn.mean().idxmax()
        eps = (cov.loc[lr] if len(cov) and lr in cov.index.get_level_values(0)
               else pd.Series(dtype=int))
        ep = (sorted(eps.items(), key=lambda t: (-t[1], t[0] != ep_anchor))[0][0]
              if len(eps) else ep_anchor)
        if equal_dose:
            # same optimisation dose for every task: same number of steps
            ep = EQUAL_DOSE_EP.get(d, 4)
        elif force_lr is not None:
            # canonical horizons at the shared LR; both spatial variants at 4 so the pair
            # stays directly comparable
            ep = {"spatial_standard2": 4, "spatial_contradiction2": 4}.get(d, ep)
        at = sub[(sub.direction == dirn) & (sub.lr == lr)]
        s0 = at[(at.ref_update == "frozen") & (at.epochs == ep)]
        if s0.empty and not equal_dose:
            # dose-controlled variants stay strict: no silent fallback to another epoch
            s0 = at[(at.ref_update == "frozen") & (at.epochs == ep_anchor)]
        arms = [(0.0, s0.learn.mean())] if len(s0) else []
        for r in all_rates:
            s = at[(at.ref_update == "ema") & (at.ema_rate == r) & (at.epochs == ep)]
            if len(s):
                arms.append((r, s.learn.mean()))
        if not arms:
            continue
        if sftnorm:  # 1 = the best SFT gain of the task over its own (lr, epochs) grid
            sden = sft[sft.dataset == d].groupby(["lr", "epochs"]).learn.mean().max()
            arms = [(r, v / sden) for r, v in arms]
        xs = [x_of[r] for r, _ in arms]
        ys = [v for _, v in arms]
        ax.plot(xs, ys, "-", marker=mk, ms=2.6 if compact else 4.5,
                lw=1.0 if compact else 1.6, color=colr,
                label=(DISPLAY.get(d, d).replace("-contradiction", "-contra.") if compact
                       else f"{DISPLAY.get(d, d)} (lr {lr:g}, {ep:g} ep)"), zorder=3)
        i_best = max(range(len(ys)), key=lambda i: ys[i])
        ymax = max(ymax, ys[i_best])  # alpha*: emphasised marker instead of a label
        ax.scatter([xs[i_best]], [ys[i_best]], s=24 if compact else 70, marker=mk, color=colr,
                   edgecolors="black", linewidths=0.8 if compact else 1.0, zorder=4)
    ax.axhline(0, color="grey", lw=0.7, zorder=1)
    if sftnorm:
        ax.axhline(1.0, color="grey", lw=0.9, ls=":", zorder=1)
        ax.annotate("best SFT", (0.99, 1.0), fontsize=7, color="#888", va="bottom", ha="right",
                    xycoords=("axes fraction", "data"))
    ax.set_xticks(range(len(ticks)), ["0"] + [f"{r:g}" for r in all_rates])
    # floor low enough to host the legend; deep collapses (tool-alpaca) still exit
    # sftnorm: keep the best-SFT line (y = 1) in view even when every curve stays below it
    ax.set_ylim(*((-1.2, max(ymax + 0.2, 1.08)) if sftnorm else (-48, ymax + 8)))
    fs, ft = (8.5, 7.5) if compact else (10, 10)
    ax.set_xlabel("EMA rate $\\alpha$ (0 = frozen)", fontsize=fs)
    ax.set_ylabel(("acquisition / best SFT" if compact else "new-task acquisition / best SFT gain")
                  if sftnorm else
                  ("acquisition (pp)" if compact else "new-task acquisition (pts)"), fontsize=fs)
    ax.tick_params(labelsize=ft)
    ax.grid(lw=0.3, alpha=0.4)
    hs, ls = ax.get_legend_handles_labels()
    if compact and len(hs) == 5:  # 3 rows x 2 cols read row-wise: greens on top, spatial pair, math
        order = [0, 2, 4, 1, 3]   # matplotlib fills columns first
        hs, ls = [hs[i] for i in order], [ls[i] for i in order]
    ax.legend(hs, ls, fontsize=6.8 if compact else 8, frameon=False, loc="lower left",
              handlelength=1.4, handletextpad=0.4, labelspacing=0.3,
              ncol=2 if compact else 1, columnspacing=0.8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)




def fig2_panel(exp_model, rollout, kl, shrink=None):
    """One Fig-2 coupling panel: the (qwen, teacher, forward) combo is the paper pair
    (experiment + model halves); the 7 other combos are appendix ablation panels
    (experiment half only). Same equal-dose + budget-LR rule everywhere."""
    canonical = exp_model == "qwen2.5-7b" and rollout == "teacher" and kl == "forward"
    outdir = FINAL
    if not canonical:
        outdir = outdir / "appendix"
    outdir.mkdir(parents=True, exist_ok=True)
    # 50/50 halves, fixed margins in inches (no tight crop) so both PDFs share height
    # and axes position
    W, H, LM, RM, TM, BM = 2.7, 1.7 - (shrink or 0), 0.50, 0.05, 0.06, 0.36
    f1 = plt.figure(figsize=(W, H))
    draw_experiment(f1.gca(), rollout, kl, force_lr=BUDGET_LR[exp_model],
                    equal_dose=True, sftnorm=True, compact=True, model=exp_model)
    f1.subplots_adjust(left=LM / W, right=1 - RM / W, top=1 - TM / H, bottom=BM / H)
    tag = ("" if canonical else
           f"_{exp_model.split('-')[0].split('2')[0]}_{rollout}_{kl}")
    name = f"2_coupling{tag}_experiment.pdf"
    f1.savefig(outdir / name)
    plt.close(f1)
    print("wrote", outdir / name)
    if canonical:
        f2 = plt.figure(figsize=(W, H))
        draw_model(f2.gca(), compact=True)
        f2.subplots_adjust(left=LM / W, right=1 - RM / W, top=1 - TM / H, bottom=BM / H)
        f2.savefig(outdir / "2_coupling_model.pdf")
        plt.close(f2)
        print("wrote", outdir / "2_coupling_model.pdf")



if __name__ == "__main__":
    fig2_panel("qwen2.5-7b", "teacher", "forward", shrink=SHRINK)
    for exp_model in ["qwen2.5-7b", "ministral-3-3b"]:
        for rollout in ["teacher", "student"]:
            for kl in ["forward", "reverse"]:
                if (exp_model, rollout, kl) != ("qwen2.5-7b", "teacher", "forward"):
                    fig2_panel(exp_model, rollout, kl)
