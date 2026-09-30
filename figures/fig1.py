# /// script
# requires-python = ">=3.10"
# dependencies = ["pandas>=2.0", "pyarrow>=15", "matplotlib==3.11.1", "scipy>=1.11"]
# ///
"""Fig 1 (rollout source): the forward-KL paper pair + the reverse-KL appendix pair.

    uv run --offline figures/fig1.py
"""

from fig_common import (BLUE, BUDGET_LR, FINAL, GOLD_SWITCH, GREY_SWITCH, ORANGE, ROOT,
                        cells_table)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODEL_KAPPA = 0.3    # model panel: context strength of the lambda sweep
FIG1_LAM_MIN = 0.7   # model panel: drop lambda = 0.625 (flat teacher-student pair)
# Every number inside the drawing code below is STYLE only (sizes, margins, fonts,
# legend and annotation positions, axis cosmetics): moving it moves ink, never points.


def rollout_axis_figure(kl, shrink=None):
    """Experiment <-> model figure for the ROLLOUT-SOURCE axis: blue = student rollouts,
    orange = teacher rollouts, one pair per task (experiment) / per lambda (model), grey
    connectors; top = acquisition, bottom = retention. The function exports the two halves
    as untitled PDFs (out/final/1_rollout_<direction>_{experiment,model}.pdf) for LaTeX
    assembly."""
    from load_runs import load_runs
    from constants import CATEGORIES, DISPLAY, CANON_EP

    # one LR per model: the largest at which every run of the grid stays within the
    # 5 pp retention budget (see the LR-selection rule in the paper's setup section)
    MODEL_LR = BUDGET_LR
    DIRN = "bwd" if kl == "reverse" else "fwd"
    ROLE_C = {"student": BLUE, "teacher": ORANGE}
    roles = ["student", "teacher"]

    df = load_runs(retention="all").dropna(subset=["task_acc", "lmeval_avg"])
    df = df[df.context == "default"]
    cat = {d for c in CATEGORIES for d in CATEGORIES[c] if d != "fictionalqa"}
    df = df[df.dataset.isin(cat)]
    cells_of = {r: cells_table(df[(df.role == r) & df.direction.isin(["fwd", "bwd"])])
                for r in roles}
    task_base = df.groupby(["model", "dataset"]).task_acc_baseline.first().to_dict()
    datasets = [d for c in CATEGORIES for d in CATEGORIES[c] if d in cat]

    # ---- experiment data: one (task, model) pair at the per-model LR and canonical epochs
    exp_pts, fallback_tasks = [], set()
    for model, filled in [("qwen2.5-7b", True), ("ministral-3-3b", False)]:
        for j, d in enumerate(datasets):
            subs = {r: cells_of[r][(cells_of[r].model == model) & (cells_of[r].dataset == d)]
                    for r in roles}
            lr = MODEL_LR[model]
            pts, fell_back = {}, False
            for r in roles:
                avail = subs[r][(subs[r].lr == lr) & (subs[r].direction == DIRN)
                                & subs[r].ref.notna()]
                fb_ep = (avail.groupby("epochs").ref.nunique().idxmax()
                         if len(avail) else None)
                for ep_try, fb in [(CANON_EP.get(d), False), (fb_ep, True)]:
                    if ep_try is None:
                        continue
                    # frozen (ref = NaN) is a candidate too: alpha* may be 0
                    c = subs[r][(subs[r].lr == lr) & (subs[r].epochs == ep_try)
                                & (subs[r].direction == DIRN)]
                    if len(c):
                        pts[r] = c.loc[c.learn.idxmax()]
                        fell_back |= fb
                        break
            if len(pts) < 2:
                continue
            if fell_back:
                fallback_tasks.add(d)
            exp_pts.append((j, filled, {r: (pts[r].learn - task_base[(model, d)], pts[r].keep)
                                        for r in roles}))

    # ---- model data: one pair per lambda (kappa = --model-kappa, per-point alpha*),
    # from trajectory_paper_curves (fine alpha* grid, default model lr)
    cur_all = pd.read_csv(ROOT / "results/v14_final/aggregate/trajectory_paper_curves.csv")
    cur_all = cur_all.assign(rollout=cur_all.teacher_fraction.map(
        {0.0: "student", 1.0: "teacher"}))
    cur0 = cur_all[cur_all.kl_direction == kl]
    lams = [l for l in sorted(cur0["lambda"].unique()) if l > FIG1_LAM_MIN]
    # retention here is the model's absolute forgetting, i.e. vs the initial (untrained)
    # model -- the same reading as the experiment panel. Only the Fig-3 commute diagrams
    # use a Delta, and there the origin is the drawn starting point of the routes.
    vals_k = {}
    for kappa in sorted(cur0.c.unique()):
        cur = cur0[cur0.c == kappa]
        vals = {r: {"acq": [], "ret": []} for r in roles}
        for lam in lams:
            sl = cur[cur["lambda"] == lam]
            for r in roles:
                sf = sl[sl.rollout == r]
                row = sf.loc[sf.new_match_gain__mean.idxmax()]
                vals[r]["acq"].append(row.new_match_gain__mean * 100)
                vals[r]["ret"].append(-row.old_match_forgetting__mean * 100)
        vals_k[kappa] = vals
    x = np.log2(lams)

    def style(ax):
        ax.grid(lw=0.3, alpha=0.4)
        ax.tick_params(labelsize=8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    def draw_exp(ax_a, ax_r, compact=False):
        fs, ft, fleg = (8.5, 7.5, 7) if compact else (9, 8, 7.5)
        smk, lwc, lwe, mleg = (15, 0.7, 0.9, 3.2) if compact else (44, 1.1, 1.3, 6)
        def centered_pair(ax, xp, dv, filled):
            # student anchored at 0, teacher at the paired difference
            ax.plot([xp] * 2, [0, dv], "-", color="#bbb", lw=lwc, zorder=2)
            for r, v in [("student", 0.0), ("teacher", dv)]:
                ax.scatter([xp], [v], s=smk, marker="o",
                           color=ROLE_C[r] if filled else "white",
                           edgecolors=ROLE_C[r], linewidths=lwe, zorder=3)

        for j, filled, pr in exp_pts:
            off = -0.09 if filled else 0.09
            da = pr["teacher"][0] - pr["student"][0]
            dr = pr["teacher"][1] - pr["student"][1]
            centered_pair(ax_r, j + off, dr, filled)
            centered_pair(ax_a, j + off, da, filled)
        for ax in (ax_a, ax_r):
            ax.axhline(0, color="grey", lw=0.6 if compact else 0.8, zorder=1)
            style(ax)
            ax.grid(axis="y", lw=0.3, alpha=0.4)
            ax.tick_params(labelsize=ft)
        ax_a.set_ylabel("acquisition (pp)\nteacher $-$ student",
                        fontsize=(fs - 1) if compact else fs)
        ax_r.set_ylabel("retention (pp)\nteacher $-$ student",
                        fontsize=(fs - 1) if compact else fs)
        # symmetric around 0: don't let autoscale dramatise tiny diffs
        for ax in (ax_r, ax_a):
            m = max(map(abs, ax.get_ylim()))
            ax.set_ylim(-m, m)
        # two task groups, not a sequence: dashed separator + group labels (headroom on top)
        n_ord = sum(d in CATEGORIES["non_contradictory"] for d in datasets)
        for ax in (ax_a, ax_r):
            ax.axvline(n_ord - 0.5, color="#999", lw=0.6, ls=(0, (3, 2)), zorder=1)
        lo, hi = ax_a.get_ylim()
        ax_a.set_ylim(lo, lo + (hi - lo) * 1.16)
        for xc, lab in [((n_ord - 1) / 2, "ordinary"),
                        ((n_ord + len(datasets) - 1) / 2, "contradictory")]:
            ax_a.text(xc, 0.975, lab, transform=ax_a.get_xaxis_transform(), ha="center",
                      va="top", fontsize=ft, color="#444", zorder=5)
        ax_a.tick_params(labelbottom=False)
        ax_r.set_xticks(range(len(datasets)),
                        [DISPLAY.get(d, d).replace("-contradiction", "-\ncontra.")
                         + ("*" if d in fallback_tasks else "") for d in datasets], fontsize=ft)
        # hand-placed key: exact dot-text alignment (CM metrics shift legends)
        entries = [(0.36, 0.315, ROLE_C["student"], ROLE_C["student"], "student rollouts"),
                   (0.36, 0.235, ROLE_C["teacher"], ROLE_C["teacher"], "teacher rollouts"),
                   (0.36, 0.155, "#666", "#666", "qwen (filled)"),
                   (0.36, 0.075, "white", "#666", "ministral (hollow)")]
        for xd, yd, fc, ec, lab in entries:
            ax_a.scatter([xd], [yd], transform=ax_a.transAxes, s=smk, marker="o",
                         color=fc, edgecolors=ec, linewidths=lwe, zorder=5)
            ax_a.text(xd + 0.025, yd, lab, transform=ax_a.transAxes, fontsize=fleg,
                      ha="left", va="center", zorder=5)

    def draw_mod(ax_ma, ax_mr, kappas=None, compact=False):
        kappas = kappas if kappas is not None else (MODEL_KAPPA,)
        fs, ft = (8.5, 7.5) if compact else (9, 8)
        smk, lwc, lwe = (14, 0.7, 0.9) if compact else (40, 1.1, 1.2)
        off_vals = np.linspace(-0.14, 0.14, len(kappas)) if len(kappas) > 1 else [0.0]
        offs = {k: o for k, o in zip(kappas, off_vals)}
        for kappa in kappas:
            vals = vals_k[kappa]
            filled = {0.3: 1.0, 0.6: 0.55, 0.9: 0.0}.get(kappa, 1.0)
            def centered_pairs_mod(ax, dv):
                for i, xi in enumerate(x):
                    ax.plot([xi + offs[kappa]] * 2, [0, dv[i]], "-", color="#bbb",
                            lw=lwc, zorder=2)
                for r, ys in [("student", [0.0] * len(x)), ("teacher", dv)]:
                    ax.scatter(x + offs[kappa], ys, s=smk, marker="o", color=ROLE_C[r],
                               alpha=1.0 if filled != 0.55 else 0.55,
                               edgecolors=ROLE_C[r], linewidths=lwe, zorder=3)

            dv = [t - s for t, s in zip(vals["teacher"]["ret"], vals["student"]["ret"])]
            centered_pairs_mod(ax_mr, dv)
            ax_mr.axhline(0, color="grey", lw=0.7, zorder=1)
            dva = [t - s for t, s in zip(vals["teacher"]["acq"], vals["student"]["acq"])]
            centered_pairs_mod(ax_ma, dva)
            ax_ma.axhline(0, color="grey", lw=0.7, zorder=1)
        for ax in (ax_ma, ax_mr):
            style(ax)
            ax.tick_params(labelsize=ft)
        ax_ma.tick_params(labelbottom=False)
        ax_mr.set_xticks(x, [f"{round(l, 1):g}" for l in lams], fontsize=ft)
        # symmetric around 0 before choosing the ticks
        for ax in (ax_mr, ax_ma):
            m = max(map(abs, ax.get_ylim()))
            ax.set_ylim(-m, m)
        # exactly two round ticks on the model retention axis
        lo, hi = ax_mr.get_ylim()
        cand = [t for t in plt.MaxNLocator(nbins=3, steps=[1, 2, 5, 10]).tick_values(lo, hi)
                if lo <= t <= hi]
        if len(cand) >= 2:
            ax_mr.set_yticks([cand[0], cand[-1]])
        ax_mr.set_xlabel("initial-policy concentration $\\lambda$", fontsize=fs)
        ax_ma.set_ylabel("acquisition (pp)\nteacher $-$ student",
                         fontsize=(fs - 1) if compact else fs)
        ax_mr.set_ylabel("retention (pp)\nteacher $-$ student",
                         fontsize=(fs - 1) if compact else fs)
        if compact:  # standalone key, hand-placed
            for yd, r in [(0.20, "student"), (0.08, "teacher")]:
                ax_ma.scatter([0.33], [yd], transform=ax_ma.transAxes, s=smk, marker="o",
                              color=ROLE_C[r], zorder=5)
                ax_ma.text(0.365, yd, f"{r} rollouts", transform=ax_ma.transAxes, fontsize=7,
                           ha="left", va="center", zorder=5)

    # forward KL is the paper figure; the reverse variant goes to final/appendix/.
    # The KL direction is always spelled out in the file name.
    outdir = FINAL
    prefix = f"1_rollout_{kl}_"
    if kl == "reverse":
        outdir = outdir / "appendix"
    outdir.mkdir(parents=True, exist_ok=True)
    # true print size: included at these widths in the paper, fonts stay at scale 1
    # Fixed margins in inches (no tight crop) so that the two PDFs have exactly the same
    # height and their axes sit at the same vertical positions when included at scale=1.
    H, RM, TM, BM = 2.1 - (shrink or 0), 0.04, 0.05, 0.40
    LM = 0.62  # room for the two-line retention ylabel
    for name, w, draw in [(f"{prefix}experiment.pdf", 3.15, draw_exp),
                          (f"{prefix}model.pdf", 2.25, draw_mod)]:
        f = plt.figure(figsize=(w, H))
        g = f.add_gridspec(2, 1, height_ratios=[2.2, 1], hspace=0.16,
                           left=LM / w, right=1 - RM / w, top=1 - TM / H, bottom=BM / H)
        a_top = f.add_subplot(g[0])
        draw(a_top, f.add_subplot(g[1], sharex=a_top), compact=True)
        f.align_ylabels()
        f.savefig(outdir / name)
        plt.close(f)
        print("wrote", outdir / name)



if __name__ == "__main__":
    for kl in ["forward", "reverse"]:
        rollout_axis_figure(kl)
