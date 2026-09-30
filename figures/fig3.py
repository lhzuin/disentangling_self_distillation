# /// script
# requires-python = ">=3.10"
# dependencies = ["pandas>=2.0", "pyarrow>=15", "matplotlib==3.11.1", "scipy>=1.11"]
# ///
"""Fig 3 (loss axis, KL x coupling): the paper 2x3 at print height, its student-rollout
variant, and the other-tasks experiment-only grids (teacher and student).

    uv run --offline figures/fig3.py
"""

from pathlib import Path

from fig_common import (BLUE, BUDGET_LR, FINAL, GOLD_SWITCH, GREY_SWITCH, ORANGE, ROOT,
                        cells_table)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# panel lrs and model-column parameters of the main 2x3 (qwen row always at 2e-05)
FIG3 = dict(loss_task="science", ministral_lr=2e-05, ministral_lr_contra=1e-05,
            lam=2.5, kappa=0.6, kappa_bottom=0.9, rho_bottom=1.0)
FIG3_OTHERS = dict(tasks=["tooluse", "spatial_standard2", "spatial_contradiction2"],
                   ministral_lr=5e-06,
                   ministral_lr_task={"spatial_standard2": 2e-05,
                                      "spatial_contradiction2": 2e-05})
# SFT envelope: the 7 declared lrs (dose-matched: canonical epochs, no extra-epoch runs)
SFT_LRS = [2e-06, 3.25e-06, 5e-06, 1e-05, 2e-05, 3.25e-05, 5e-05]
SHRINK = 0.8  # print height = 3.6 in minus this (empirical floor: row ticks collide lower)
# Every number inside the drawing code below is STYLE only (sizes, margins, fonts,
# legend and annotation positions, axis cosmetics): moving it moves ink, never points.


def sweep(ax, start, pts, color, span, floor=None, compact=False):
    """EMA sweep from `start` through rates ascending up to the best-learning one (big marker).

    The argmax includes `start` (the frozen ref): when no EMA rate beats it, no leg is drawn and
    None is returned -- the route simply stays at `start`. Rates past the optimum are faint dots.
    On-path coordinates are appended to `span` (used to frame the axes on the routes).
    floor: arms whose mean lands below it (below the untrained model = collapsed runs) are
    excluded from the route and the argmax; they stay as ordinary faint dots.
    compact: print-size rendering — smaller markers/lines, no per-dot rate labels."""
    s_faint, lw = (10, 1.2) if compact else (26, 2)
    if pts.empty:
        return None
    if floor is not None:
        bad = pts[pts.learn < floor]
        if len(bad):
            ax.scatter(bad.keep, bad.learn, s=s_faint, c=color, alpha=0.35, edgecolors="none",
                       zorder=2)
            if not compact:
                for _, r in bad.iterrows():
                    ax.annotate(f"{r.ref:g}", (r.keep, r.learn), fontsize=6, color=color,
                                alpha=0.8, xytext=(3, -7), textcoords="offset points")
            pts = pts[pts.learn >= floor]
        if pts.empty:
            return None
    pts = pts.sort_values("ref")
    ax.scatter(pts.keep, pts.learn, s=s_faint, c=color, alpha=0.35, edgecolors="none", zorder=2)
    if not compact:
        for _, r in pts.iterrows():
            ax.annotate(f"{r.ref:g}", (r.keep, r.learn), fontsize=6, color=color, alpha=0.8,
                        xytext=(3, -7), textcoords="offset points")
    if pts.learn.max() <= start.learn:
        return None
    opt = pts.learn.idxmax()
    on_path = pts.loc[:opt]  # ascending rates up to the optimum
    span += list(zip(on_path.keep, on_path.learn))
    xs = [start.keep] + list(on_path.keep)
    ys = [start.learn] + list(on_path.learn)
    ax.plot(xs, ys, "-", color=color, lw=lw, zorder=3)
    return pts.loc[opt]


def draw_panel(ax, at, base, s_sub, tb, show_lr=False, compact=False, labels=True,
               alpha_pos=None):
    """One commute diagram: cells `at` a fixed (model, dataset, role, lr), anchored at `base`
    (the bwd + frozen row), with the SFT grid `s_sub` as reference and `tb` the untrained accuracy.
    show_lr annotates each key point's own LR (for the best-LR-per-point variant).
    compact: print-size rendering (small markers, no faint-dot labels, ticks 6pt);
    labels=False keeps only the alpha* values next to the optima (a first labelled panel
    plays the legend for the rest). alpha_pos hand-places those values per panel:
    {"bwd"|"fwd": (dx_pt, dy_pt, ha, va)} — no white boxes, each number goes where its
    panel has room."""
    s_key, s_opt, s_faint, lw, fs, off = ((32, 40, 10, 1.2, 7.0, 3) if compact
                                          else (90, 110, 26, 2, 6.5, 4))
    # short switch arrows (near-coinciding optima) must still peek out between markers
    sh, ms = (1.5, 7) if compact else (4, 10)

    def lr_note(row):
        if show_lr:
            ax.annotate(f"lr {row.lr:g}", (row.keep, row.learn), fontsize=5.5, color="#888",
                        xytext=(4, -13), textcoords="offset points")

    def note(txt, xy, colr, dy=None, ha="left", va="baseline", dx=None):
        if dx is None:
            dx = off if ha == "left" else (-off if ha == "right" else 0)
        ax.annotate(txt, xy, fontsize=fs, color=colr, ha=ha, va=va, linespacing=1.15,
                    xytext=(dx, dy if dy is not None else off), textcoords="offset points",
                    zorder=6)
    ax.axhline(tb, color="#777", lw=0.7 if compact else 0.9, ls="--", zorder=0)
    ax.axvline(0, color="#777", lw=0.7 if compact else 0.9, ls="--", zorder=0)
    if tb:  # tb == 0 when the caller already renormalised learning to acquisition
        ax.annotate(f"untrained {tb:.1f}", (0.02, tb), fontsize=6.5, color="#777",
                    va="bottom", xycoords=("axes fraction", "data"))
    span = [(base.keep, base.learn)]

    # blue: tune EMA first (stay reverse KL), then switch KL
    bwd_opt = sweep(ax, base, at[(at.direction == "bwd") & at.ref.notna()], BLUE, span,
                    floor=tb, compact=compact)
    # orange: switch KL first (forward + frozen), then tune EMA
    fwd_frozen = at[(at.direction == "fwd") & at.ref.isna()]
    fwd_start, f0 = base, None
    if not fwd_frozen.empty:
        f0 = fwd_frozen.iloc[0]
        ax.annotate("", xy=(f0.keep, f0.learn), xytext=(base.keep, base.learn),
                    arrowprops=dict(arrowstyle="-|>", color=GREY_SWITCH, lw=lw,
                                    shrinkA=sh, shrinkB=sh, mutation_scale=ms), zorder=4)
        ax.scatter([f0.keep], [f0.learn], s=s_key, marker="s", c=ORANGE, edgecolors="black",
                   linewidths=0.8, zorder=5)
        if labels:
            if compact:  # legend panel: two-line label to the LEFT of the square
                note("forward\nfrozen", (f0.keep, f0.learn), ORANGE, dy=0,
                     ha="right", va="center")
            else:
                note("forward frozen", (f0.keep, f0.learn), ORANGE)
        lr_note(f0)
        fwd_start = f0
        span.append((f0.keep, f0.learn))
    fwd_opt = sweep(ax, fwd_start, at[(at.direction == "fwd") & at.ref.notna()], ORANGE, span,
                    floor=tb, compact=compact)
    # the KL switch leg of the blue route; when frozen beats every EMA on a side, that
    # side's route simply stays put (arrow from/to the frozen point)
    blue_from = base if bwd_opt is None else bwd_opt
    target = f0 if fwd_opt is None else fwd_opt
    if target is not None:
        ax.annotate("", xy=(target.keep, target.learn), xytext=(blue_from.keep, blue_from.learn),
                    arrowprops=dict(arrowstyle="-|>", color=GOLD_SWITCH, lw=lw,
                                    shrinkA=sh, shrinkB=sh, mutation_scale=ms), zorder=4)

    ax.scatter([base.keep], [base.learn], s=s_key, c="black", zorder=6)
    if labels:
        if compact:  # legend panel: below the black point, nudged up and left so the
            # text neither clips the right spine nor touches the x axis
            ax.annotate("reverse frozen", (base.keep, base.learn), fontsize=fs,
                        color="black", ha="center", va="top", linespacing=1.15,
                        xytext=(-14, -2), textcoords="offset points", zorder=6)
        else:
            note("reverse frozen", (base.keep, base.learn), "black")
    lr_note(base)
    if bwd_opt is not None:
        ax.scatter([bwd_opt.keep], [bwd_opt.learn], s=s_key, c=BLUE, edgecolors="black",
                   linewidths=0.8, zorder=6)
        if labels and compact:  # legend panel: two-line label above the blue point
            note(f"reverse\nEMA {bwd_opt.ref:g}", (bwd_opt.keep, bwd_opt.learn), BLUE,
                 dy=4.5, ha="center", va="bottom")
        elif compact:  # alpha* value, hand-placed where the panel has room
            dx, dy, ha, va = (alpha_pos or {}).get("bwd", (0, 4.5, "center", "bottom"))
            note(f"{bwd_opt.ref:g}", (bwd_opt.keep, bwd_opt.learn), BLUE,
                 dx=dx, dy=dy, ha=ha, va=va)
        else:
            note(f"reverse EMA {bwd_opt.ref:g}" if labels else f"{bwd_opt.ref:g}",
                 (bwd_opt.keep, bwd_opt.learn), BLUE)
        lr_note(bwd_opt)
    if fwd_opt is not None:
        ax.scatter([fwd_opt.keep], [fwd_opt.learn], s=s_opt, marker="^", c=ORANGE,
                   edgecolors="black", linewidths=0.8, zorder=6)
        if labels and compact:
            # two-line label to the LEFT of the triangle (the right side runs into
            # the blue route)
            note(f"forward\nEMA {fwd_opt.ref:g}", (fwd_opt.keep, fwd_opt.learn), ORANGE,
                 dy=0, ha="right", va="center")
        elif compact:  # alpha* value, hand-placed where the panel has room
            dx, dy, ha, va = (alpha_pos or {}).get("fwd", (-off, -5.5, "right", "top"))
            note(f"{fwd_opt.ref:g}", (fwd_opt.keep, fwd_opt.learn), ORANGE,
                 dx=dx, dy=dy, ha=ha, va=va)
        else:
            note(f"forward EMA {fwd_opt.ref:g}" if labels else f"{fwd_opt.ref:g}",
                 (fwd_opt.keep, fwd_opt.learn), ORANGE, dy=-10)
        lr_note(fwd_opt)

    # reference point outside the fixed-LR routes: best-acquisition SFT
    sb_ext = None
    if not s_sub.empty:
        sb = s_sub.loc[s_sub.learn.idxmax()]
        # Pareto envelope of the SFT LR grid (best learning at each retention level)
        env = s_sub.sort_values("keep", ascending=False)
        env = env[env.learn > env.learn.cummax().shift(fill_value=-np.inf)]
        ax.plot(env.keep, env.learn, "-", drawstyle="steps-post", color="#63975f",
                alpha=0.25, lw=1.6 if compact else 2.2, zorder=0.5)
        faint = s_sub.drop(sb.name).sort_values("lr")
        ax.scatter(faint.keep, faint.learn, s=s_faint, c="#63975f", alpha=0.3, edgecolors="none", zorder=1)
        if not compact:
            for _, r in faint.iterrows():
                ax.annotate(f"{r.lr:g}", (r.keep, r.learn), fontsize=6, color="#63975f",
                            alpha=0.75, xytext=(3, -7), textcoords="offset points")
        ax.scatter([sb.keep], [sb.learn], s=s_key, c="#63975f", edgecolors="black", linewidths=0.8, zorder=6)
        if labels:
            note("SFT" if compact else f"SFT (lr {sb.lr:g}, {sb.epochs:g} ep)",
                 (sb.keep, sb.learn), "#3d6b39")
        span.append((sb.keep, sb.learn))
        # past the best point the frontier is flat: worse-retention SFT runs exist but never
        # learn more, so the envelope continues horizontally out of the frame
        sb_ext = sb if (s_sub.keep < sb.keep).any() else None

    # frame the axes on the routes, SFT best and the untrained/no-forgetting lines; stray faint
    # dots (extreme EMA rates or SFT LRs) may fall outside on purpose
    xs = [p[0] for p in span] + [0]
    ys = [p[1] for p in span] + [tb]
    ax.set_xlim(min(xs) - 0.5, max(xs) + 0.4)
    ax.set_ylim(min(ys) - 2, max(ys) + 4)
    if sb_ext is not None:
        ax.plot([ax.get_xlim()[0], sb_ext.keep], [sb_ext.learn] * 2, "-", color="#63975f",
                alpha=0.25, lw=2.2, zorder=0.5)
    ax.grid(True, color="#eee", lw=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=7.5 if compact else 8)
    if compact:
        ax.locator_params(nbins=5)


def draw_model_commute(ax, lam, kappa, rollout, fs=8, rho_phi=None,
                       compact=False, stack_labels=False):
    """One model-side commute diagram (Fig 3's model column) on `ax`: at (lambda, kappa,
    rollout, rho_phi), both routes from (reverse KL + frozen) to (forward KL + optimal EMA).
    Blue: tune the reverse-KL EMA first, then switch KL; orange: switch KL first, then tune.
    Axes are ADDITIONAL acquisition / retention relative to the reverse + frozen origin.
    Data: the V14.2 canonical main grid extract (E85, full 18-alpha Sec-4 ladder, 96 paired
    seeds, rho_phi in {0, 0.5, 1}, default model lr); rho values outside it fall back to
    the coarse E76 grid archived in old/. The lr/rho sweep variants retired to
    old/plot_figures.py."""
    df = pd.read_csv(Path(__file__).resolve().parent / "data/e85_main_grid_curves.csv")
    df = df[abs(df.rho_phi - rho_phi) < 1e-4]
    if df.empty:  # coarse E76 grid, retired to old/ (unused by the current figures)
        df = pd.read_csv(Path(__file__).resolve().parent / "old/e76_rho_curves.csv")
        df = df[abs(df.rho_phi - rho_phi) < 1e-4]
    df = df[(abs(df["lambda"] - lam) < 1e-4) & (abs(df.c - kappa) < 1e-9)
            & (df.rollout_source == rollout)]
    ser = {}
    for kl in ["reverse", "forward"]:
        g = df[df.kl_direction == kl].sort_values("ema_alpha")
        ser[kl] = g.assign(acq=g.new_match_gain__mean * 100,
                           ret=-g.old_match_forgetting__mean * 100)
    o = ser["reverse"][ser["reverse"].ema_alpha == 0].iloc[0]

    def rel(row):
        return row.ret - o.ret, row.acq - o.acq

    s_key, s_opt, s_faint, lw, osc = (32, 40, 10, 1.2, 0.6) if compact else (80, 100, 26, 2, 1)
    sh, ms = (1.5, 7) if compact else (4, 10)  # same arrowheads as the experiment panels
    ax.axhline(0, color="grey", lw=0.8, zorder=1)
    ax.axvline(0, color="grey", lw=0.8, zorder=1)

    paths = {}
    for kl, colr in [("reverse", BLUE), ("forward", ORANGE)]:
        g = ser[kl]
        i_star = g.acq.idxmax()
        upto = g[g.ema_alpha <= g.loc[i_star].ema_alpha]
        pts = [rel(r) for _, r in upto.iterrows()]
        paths[kl] = (pts, rel(g.loc[i_star]), rel(g[g.ema_alpha == 0].iloc[0]))
        print(f"lam={lam:g} kap={kappa:g} {rollout} rho={rho_phi:g} "
              f"{kl}: alpha* = {g.loc[i_star].ema_alpha:g}, "
              f"add acq {rel(g.loc[i_star])[1]:+.2f}, add ret {rel(g.loc[i_star])[0]:+.2f}")

    rev_pts, rev_star, _ = paths["reverse"]
    fwd_pts, fwd_star, fwd0 = paths["forward"]
    # blue route: reverse EMA sweep then the KL switch
    ax.plot([p[0] for p in rev_pts], [p[1] for p in rev_pts], "-", color=BLUE, lw=lw, zorder=3)
    ax.scatter([p[0] for p in rev_pts[1:-1]], [p[1] for p in rev_pts[1:-1]], s=s_faint, color=BLUE,
               alpha=0.4, edgecolors="none", zorder=3)
    ax.annotate("", xy=fwd_star, xytext=rev_star,
                arrowprops=dict(arrowstyle="-|>", color=GOLD_SWITCH, lw=lw,
                                shrinkA=sh, shrinkB=sh, mutation_scale=ms), zorder=4)
    # orange route: switch KL first (grey arrow), then the forward sweep
    fwd_sweep = [p for p in fwd_pts if p != fwd0] and fwd_pts
    ax.annotate("", xy=fwd0, xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=GREY_SWITCH, lw=lw,
                                shrinkA=sh, shrinkB=sh, mutation_scale=ms), zorder=4)
    ax.plot([p[0] for p in fwd_sweep], [p[1] for p in fwd_sweep], "-", color=ORANGE, lw=lw, zorder=3)
    ax.scatter([p[0] for p in fwd_sweep[1:-1]], [p[1] for p in fwd_sweep[1:-1]], s=s_faint,
               color=ORANGE, alpha=0.4, edgecolors="none", zorder=3)
    # key points
    ax.scatter([0], [0], s=s_key, color="black", zorder=5)
    ax.scatter([rev_star[0]], [rev_star[1]], s=s_key, color=BLUE, edgecolors="black",
               linewidths=0.8, zorder=5)
    ax.scatter([fwd0[0]], [fwd0[1]], s=s_key, marker="s", color=ORANGE, edgecolors="black",
               linewidths=0.8, zorder=5)
    ax.scatter([fwd_star[0]], [fwd_star[1]], s=s_opt, marker="^", color=ORANGE,
               edgecolors="black", linewidths=0.8, zorder=5)
    # print layout: the markers themselves are the legend (same coding as the
    # experiment panels, labelled once there); only the two route names stay.
    if stack_labels:
        # no route names on this panel: unreadable at print size and redundant
        # with the labels kept on the other model panel
        pass
    else:
        # above the square: the left side would run into the corner note
        ax.annotate("switch KL first", fwd0, xytext=(0, 5), textcoords="offset points",
                    fontsize=fs, color=GREY_SWITCH, ha="center", va="bottom",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=0.4))
        ax.annotate("tune EMA first", rev_pts[len(rev_pts) // 2], xytext=(0, 4),
                    textcoords="offset points", fontsize=fs, color=BLUE, ha="center")
    ax.grid(lw=0.3, alpha=0.4)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    if compact:
        ax.tick_params(labelsize=7.5)
        ax.locator_params(nbins=5)


def loss_axis_figure(role="teacher", others=False, shrink=None):
    """Draft of the LOSS-AXIS figure, 2x3. Rows = models at their budget LR: qwen at 2e-05
    (open commute squares), ministral at 5e-06 (squares closed onto a line). Columns = one
    ordinary task, math-contradiction, and the model counterpart (default model lr on the
    qwen row; lr/4 -- the small-step regime -- on the ministral row). Teacher rollouts,
    canonical epochs everywhere."""
    from load_runs import load_runs
    from constants import DISPLAY, CANON_EP

    ROLE = role
    cfg = FIG3_OTHERS if others else FIG3
    no_model = others  # the other-tasks variant is an experiment-only grid
    ROWS = [("qwen2.5-7b", 2e-05), ("ministral-3-3b", cfg["ministral_lr"])]
    # per-task override for the ministral row (the unstable regime is task-dependent:
    # chemistry shows it at 2e-05, math-contra already at 1e-05)
    MIN_LR_TASK = {"math_contradiction": cfg.get("ministral_lr_contra", cfg["ministral_lr"])}
    MIN_LR_TASK.update(cfg.get("ministral_lr_task", {}))
    ORDER = ["tooluse", "science", "spatial_standard2", "spatial_contradiction2",
             "spatial_contradiction", "math_contradiction"]
    wanted = (set(cfg["tasks"]) if others else {cfg["loss_task"], "math_contradiction"})
    TASKS = [t for t in ORDER if t in wanted]
    df = load_runs(retention="all").dropna(subset=["task_acc", "lmeval_avg"])
    df = df[df.context == "default"]
    # SFT envelope, dose-matched: default epochs per task, the SFT_LRS grid.
    # Extra-epoch and off-grid SFT runs are excluded, matching tab:app:sft-crossing (giving
    # SFT more steps than self-distillation would confound "better" with "trained longer").
    sft = cells_table(df[(df.method == "ce_dataset_sft") & df.lr.isin(SFT_LRS)
                         & (df.epochs == df.dataset.map(CANON_EP))]
                      .assign(direction="sft"))
    task_base = df.groupby(["model", "dataset"]).task_acc_baseline.first().to_dict()
    cells = cells_table(df[(df.role == ROLE) & df.direction.isin(["fwd", "bwd"])])

    # print export at the Fig-1/2 print width (5.4 in). Compact styling: the top-left
    # panel keeps the full point labels and plays the legend for the grid; everywhere
    # else the optima keep only their alpha value. lambda = 2.5 on both model panels
    # and epochs are canonical — said in the caption, not in the titles.
    MODEL_SHORT = {"qwen2.5-7b": "Qwen2.5-7B", "ministral-3-3b": "Ministral-3-3B"}
    # the main figure is (teacher, science+math-contra, with the model column);
    # role/task variants are appendix ablations
    canonical = ROLE == "teacher" and not others
    suffix = ("_others" if others else "") + ("" if canonical else f"_{ROLE}")
    outdir = FINAL
    if not canonical:
        outdir = outdir / "appendix"
    outdir.mkdir(parents=True, exist_ok=True)
    n_exp = len(TASKS) if no_model else 2
    W, H = 5.4, 3.6 - (shrink or 0)
    LM, RM, TM, BM = 0.55, 0.03, 0.16, 0.40
    # the two experiment columns share their y quantity: tight gap between them;
    # the model column (own Delta ylabel) keeps a wider one
    G01, G12 = 0.16, 0.40
    pw = ((W - LM - RM - (n_exp - 1) * G01) / n_exp if no_model
          else (W - LM - RM - G01 - G12) / 3)
    def build(axes):
        """Draw the panels onto a 2 x (n_exp [+ model]) grid of axes."""
        for i, (model, lr) in enumerate(ROWS):
            for j, d in enumerate(TASKS[:n_exp]):
                ax = axes[i][j]
                lr_d = (MIN_LR_TASK.get(d, lr) if model == "ministral-3-3b" else lr)
                tb = task_base[(model, d)]
                sub = cells[(cells.model == model) & (cells.dataset == d)]
                at = sub[(sub.lr == lr_d) & (sub.epochs == CANON_EP[d])].assign(
                    learn=lambda s: s.learn - tb)
                fr = at[(at.direction == "bwd") & at.ref.isna()]
                s_sub = sft[(sft.model == model) & (sft.dataset == d)].assign(
                    learn=lambda s: s.learn - tb)
                # the qwen math-contra panel has well-separated points: it carries
                # the full point labels and plays the legend for the whole grid.
                # alpha* values hand-placed per panel (no white boxes): each number
                # sits close to its optimum, on the side free of route segments
                APOS = {
                    ("qwen2.5-7b", "science"):
                        {"bwd": (0, 5, "center", "bottom"),
                         "fwd": (-4, 0, "right", "center")},
                    ("ministral-3-3b", "science"):
                        {"bwd": (0, 5, "center", "bottom"),
                         "fwd": (4, 2, "left", "center")},
                    ("ministral-3-3b", "math_contradiction"):
                        {"bwd": (-4, 0, "right", "center"),
                         "fwd": (4, 0, "left", "center")},
                }
                draw_panel(ax, at, fr.iloc[0], s_sub, 0.0, compact=True,
                           labels=(i == 0 and j == 1),
                           alpha_pos=APOS.get((model, d)))
                # column titles once (top row); per-panel LR as a corner note
                if i == 0:
                    ax.set_title(DISPLAY.get(d, d), fontsize=8, pad=3)
                ax.text(0.03, 0.03, f"lr {lr_d:g}", transform=ax.transAxes,
                        fontsize=6, color="#555", zorder=7,
                        bbox=dict(facecolor="white", edgecolor="none",
                                  alpha=0.75, pad=0.6))
                if j == 0:
                    ax.set_ylabel("acquisition (pp)", fontsize=8.5)
                if i == 1:
                    ax.set_xlabel("retention (pp)", fontsize=8.5)
            if no_model:
                continue
            ax = axes[i][n_exp]
            kap, rho = ((FIG3["kappa"], 0.5) if i == 0
                        else (FIG3["kappa_bottom"], FIG3["rho_bottom"]))
            note = "$\\kappa$ = %g\n$\\rho_\\varphi$ = %g" % (kap, rho)
            # top panel: rho_phi passed explicitly (0.5) so both rows read the same
            # E85 main-grid campaign (same 96 seeds) instead of ema_paper_curves
            draw_model_commute(ax, FIG3["lam"], kap, ROLE,
                               rho_phi=(FIG3["rho_bottom"] if i == 1 else 0.5),
                               fs=6, compact=True, stack_labels=(i == 1))
            if i == 0:
                ax.set_title("model", fontsize=8, pad=3)
            else:
                ax.set_xlabel("$\\Delta$ retention (pp)", fontsize=8.5)
            # bottom-left on two lines, aligned with the experiment panels' lr notes
            ax.text(0.03, 0.03, note, transform=ax.transAxes, fontsize=6,
                    color="#555", va="bottom", zorder=7, linespacing=1.35,
                    bbox=dict(facecolor="white", edgecolor="none",
                              alpha=0.75, pad=0.6))
            ax.set_ylabel("$\\Delta$ acquisition (pp)", fontsize=8.5)
        if not no_model:  # shared union scale on the two model panels
            x0, x1 = zip(axes[0][n_exp].get_xlim(), axes[1][n_exp].get_xlim())
            y0, y1 = zip(axes[0][n_exp].get_ylim(), axes[1][n_exp].get_ylim())
            for ax in (axes[0][n_exp], axes[1][n_exp]):
                ax.set_xlim(min(x0), max(x1))
                ax.set_ylim(min(y0), max(y1))
        # align per column: a global align_ylabels merges gridspec columns 0 and
        # drags the model Delta-ylabels onto the left-column ones
        axes[0][0].figure.align_ylabels([axes[0][0], axes[1][0]])
        if not no_model:
            axes[0][n_exp].figure.align_ylabels([axes[0][n_exp], axes[1][n_exp]])
        # row labels (model names), rotated in the left margin of their figure,
        # at the same physical distance (0.09 in) whatever the figure width
        f_exp = axes[0][0].figure
        fw = f_exp.get_size_inches()[0]
        for i, (model, _) in enumerate(ROWS):
            pos = axes[i][0].get_position()
            f_exp.text(0.09 / fw, (pos.y0 + pos.y1) / 2, MODEL_SHORT[model],
                       rotation=90, ha="center", va="center", fontsize=8)

    if no_model:
        # ---- experiment-only grid (2 x n_exp), one PDF, full print width
        fig = plt.figure(figsize=(W, H))
        gs = fig.add_gridspec(2, n_exp, left=LM / W, right=1 - RM / W,
                              top=1 - TM / H, bottom=BM / H,
                              wspace=G01 / pw, hspace=0.16)
        build([[fig.add_subplot(gs[i, j]) for j in range(n_exp)] for i in range(2)])
        path = outdir / f"3_loss_axis{suffix}.pdf"
        fig.savefig(path)
        plt.close(fig)
        print("wrote", path)
        return

    # ---- single 2x3 PDF
    fig = plt.figure(figsize=(W, H))
    gs_exp = fig.add_gridspec(2, 2, left=LM / W, right=(LM + 2 * pw + G01) / W,
                              top=1 - TM / H, bottom=BM / H,
                              wspace=G01 / pw, hspace=0.16)
    gs_mod = fig.add_gridspec(2, 1, left=(W - RM - pw) / W, right=1 - RM / W,
                              top=1 - TM / H, bottom=BM / H, hspace=0.16)
    build([[fig.add_subplot(gs_exp[i, 0]), fig.add_subplot(gs_exp[i, 1]),
            fig.add_subplot(gs_mod[i, 0])] for i in range(2)])
    path = outdir / f"3_loss_axis{suffix}.pdf"
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)

    # ---- same content split in two PDFs for LaTeX subfigure assembly (Fig-1 style:
    # fixed margins in inches, same height, axes at the same vertical positions;
    # include both at scale=1)
    RM1, LM2 = 0.05, G12 - 0.05  # the G12 gap splits into exp right + model left margin
    W1, W2 = LM + 2 * pw + G01 + RM1, LM2 + pw + RM
    f1 = plt.figure(figsize=(W1, H))
    g1 = f1.add_gridspec(2, 2, left=LM / W1, right=(LM + 2 * pw + G01) / W1,
                         top=1 - TM / H, bottom=BM / H, wspace=G01 / pw, hspace=0.16)
    f2 = plt.figure(figsize=(W2, H))
    g2 = f2.add_gridspec(2, 1, left=LM2 / W2, right=1 - RM / W2,
                         top=1 - TM / H, bottom=BM / H, hspace=0.16)
    build([[f1.add_subplot(g1[i, 0]), f1.add_subplot(g1[i, 1]),
            f2.add_subplot(g2[i, 0])] for i in range(2)])
    for f, name in [(f1, f"3_loss_axis{suffix}_experiment.pdf"),
                    (f2, f"3_loss_axis{suffix}_model.pdf")]:
        f.savefig(outdir / name)
        plt.close(f)
        print("wrote", outdir / name)
    print(f"split sizes: experiment {W1:g} x {H:g} in, model {W2:g} x {H:g} in; "
          f"LaTeX subfigure widths at ICLR textwidth 5.5 in: "
          f"{W1 / 5.5:.3f}\\linewidth and {W2 / 5.5:.3f}\\linewidth (scale=1)")



if __name__ == "__main__":
    loss_axis_figure("teacher", shrink=SHRINK)
    loss_axis_figure("student")
    loss_axis_figure("teacher", others=True)
    loss_axis_figure("student", others=True)
