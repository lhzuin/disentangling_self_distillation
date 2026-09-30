# This repository provides the code for reproducing our main figures and statistical tables

Everything runs off **one parquet file** and is re-runnable end to end:

```
metrics_out/ CSVs ──collect_llm_experiments.py──▶ data/runs.parquet ──load_runs.py──▶ fig*.py / stats_axes.py
(training machine                      (LLM runs + training               (figures in out/,
 or local copy)                         dynamics, one file)                stats CSVs + LaTeX rows)
```

- The **outputs tree** (raw run folders) lives only on the training machine; this
  folder needs only the parquet. The parquet can be rebuilt fully locally from the
  `metrics_out/{qwen25_7b,ministral3b}/all_methods/complete/` CSVs.
- All analysis code reads the parquet **through `load_runs.py`** — never directly.
- The finite-autoregressive-model curves do NOT go through the parquet: Fig 3's model
  panels read `data/e85_main_grid_curves.csv` — an exact extract of the E85 block of
  `results/v14_final/aggregate/cell_summary.csv.gz` (V14.2 main grid, 96 seeds), rebuilt
  with **`collect_controlled_model.py`** after any re-aggregation. Fig 1's and Fig 2's model panels
  read `results/v14_final/aggregate/` directly (`kl_lr_alpha_curves.csv`,
  `trajectory_paper_curves.csv`, `ema_paper_curves.csv`).
- No network needed: `uv run --offline figures/<script>.py` (deps are inline PEP-723
  headers or already cached).

The repository contains the figure-generation code, not the experiment artifacts.
Before running the figure scripts, provide `figures/data/runs.parquet` and the
controlled-model aggregate CSVs under `results/v14_final/aggregate/`, or rebuild
them with the collectors below from the corresponding metrics and toy-model
outputs. Generated PDFs and statistics are written under `figures/out/`.

## Rebuilding the parquet (when new runs land)

```bash
uv run --offline figures/collect_llm_experiments.py \
  metrics_out/qwen25_7b/all_methods/complete/context_single_phase_all_per_seed.csv \
  metrics_out/ministral3b/all_methods/complete/context_single_phase_all_per_seed.csv \
  --model-key qwen25_7b=qwen2.5-7b --model-key ministral3b=ministral-3-3b \
  --out-dir figures/data
```

Then regenerate the three paper figures and the stats tables below (their protocols are
frozen; only the numbers move).

## What makes the paper figures (out/final/)

`out/final/` holds the print PDFs included by the paper, `out/final/appendix/` the
ablation panels. One subcommand per figure family regenerates main + appendix:

```bash
uv run --offline figures/fig1.py   # rollout source: paper pair + reverse-KL appendix pair
uv run --offline figures/fig2.py   # teacher coupling: paper pair + 7 appendix combos
uv run --offline figures/fig3.py   # loss axis: paper 2x3 + student + other-tasks grids
```

| Figure | Files |
|---|---|
| Fig 1 rollout source | `1_rollout_forward_{experiment,model}.pdf`; appendix: `1_rollout_reverse_*` |
| Fig 2 teacher coupling | `2_coupling_{experiment,model}.pdf`; appendix: `2_coupling_{qwen\|ministral}_{role}_{kl}_experiment.pdf` |
| Fig 3 loss axis (KL × coupling) | `3_loss_axis{,_experiment,_model}.pdf`; appendix: `3_loss_axis_student{,_experiment,_model}.pdf`, `3_loss_axis_others.pdf`, `3_loss_axis_others_student.pdf` |

There is nothing to configure, and data and style are kept apart: each `fig*.py`
declares its DATA choices (learning rates, epochs, tasks, SFT grid, model-panel slice,
print height) as named constants at the top; every number inside the drawing code is
style only (sizes, margins, fonts, legend and annotation positions, axis clipping) —
moving one moves ink, never points. The main figures come out at their print height
(the empirical floor before labels and ticks collide); appendix panels always keep the
full-height axes. matplotlib is pinned (3.11.1): 3.11.2 silently drops single-point
scatter groups from the PDFs.

## What makes the statistics tables (appendix)

All from **`stats_axes.py`** (Wilcoxon signed-rank on matched pairs, seeds averaged):

```bash
uv run --offline figures/stats_axes.py          # all three axes (or: role | kl | ema)
```

| Table | Axis | CSV → LaTeX |
|---|---|---|
| `stats_rollout_table.tex` (teacher − student) | `role` | `out/stats/stats_role.csv`, rows transcribed by hand |
| `stats_kl_table.tex` (forward − reverse, incl. per-lr rows) | `kl` | `out/stats/stats_kl.csv`, by hand |
| `stats_coupling_table.tex` (each EMA rate − frozen) | `ema` | `out/stats/stats_ema.csv`; the run also prints the ready-to-splice LaTeX rows |

Binary axes pair on the canonical grid (canonical epochs, 3 lrs); the `ema` axis pairs on
the constant-dose slice (4 epochs, 6 for chemistry). Per-task rows use Holm-adjusted p,
grouped ordinary/contradictory/all rows use raw p; stars \*/\*\*/\*\*\* at .05/.01/.001.

## The scripts, one by one

### Pipeline (used by everything)

- **`collect_llm_experiments.py`** — in: per-seed CSVs (+ optional dynamics CSVs, + per-run JSONs when
  the outputs tree is present); out: `data/runs.parquet` (full rebuild, seeds 13/31 kept,
  a `table` column separates runs / dynamics_checkpoints / dynamics_steps).
- **`load_runs.py`** — in: the parquet; out (import, no CLI): analysis-ready DataFrames.
  `load_runs()` adds `acq` inputs (task_acc, baselines), the 5-benchmark `lmeval_avg`,
  and parsed factors (role, direction, ema_rate, lr, epochs, context). The only
  sanctioned reader of the parquet.
- **`constants.py`** — the grid constants shared by every script here: `CATEGORIES`
  (ordinary / contradictory / novel-facts task partition), `DISPLAY` (internal ids →
  paper names), `CANON_EP` (canonical epochs per task).

### Paper figures and stats

- **`fig1.py` / `fig2.py` / `fig3.py`** — one standalone script per paper figure,
  no arguments: each regenerates its whole family (main + appendix variants).
  Experiment panels come from the parquet; model panels from the E85 extract (Fig 3)
  and the `results/v14_final/aggregate/` curve CSVs (Figs 1-2).
- **`fig_common.py`** — what the figure scripts share: the Computer-Modern typography,
  the route colour coding (blue/orange routes, grey/gold KL switches), the per-model
  budget learning rate, and `cells_table` (per-configuration cell means, Figs 1 and 3).
- **`stats_axes.py`** — the Wilcoxon tests behind the three appendix tables (see above):
  one optional positional argument (`role`/`kl`/`ema`, default all three), writes
  `out/stats/stats_<axis>.csv` and prints each result table. After the `ema` axis it
  also prints the ready-to-splice LaTeX rows of `stats_coupling_table.tex` (a clearly
  marked formatting-only section at the bottom of the file — Holm stars + phantoms,
  commented α=0.50 rows; the role/KL tables are stable and stay hand-transcribed).
