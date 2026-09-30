"""Shared setup for the paper-figure scripts fig1.py / fig2.py / fig3.py.

Typography (Computer Modern without a TeX install), the route colour coding, the
per-model budget learning rate (Sec. 4), and the per-configuration cell means that
Figs 1 and 3 both read. Figure-specific parameters live in their own fig*.py.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# LaTeX-looking typography without requiring a TeX install: Computer Modern via mathtext,
# CM serif for text (DejaVu kept as glyph fallback). Greek letters must go through
# mathtext ($\alpha$, $\lambda$, ...): cmr10 has no unicode greek.
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["cmr10", "CMU Serif", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.unicode_minus": False,
    "axes.formatter.use_mathtext": True,
})
import numpy as np

ROOT = Path(__file__).resolve().parent.parent     # repo root
FINAL = Path(__file__).resolve().parent / "out" / "final"

BUDGET_LR = {"qwen2.5-7b": 2e-05, "ministral-3-3b": 5e-06}  # Sec-4 per-model lr

# route colour coding, shared by the Fig-1 pairs and the Fig-3 commute panels:
# blue = student rollouts / reverse-KL route, orange = teacher rollouts / forward-KL
# route; the KL-switch legs get their own colours (grey: at the frozen level, gold:
# at the EMA optima)
BLUE, ORANGE = "#2a78d6", "#eb6834"
GREY_SWITCH, GOLD_SWITCH = "#909090", "#c9a227"


def cells_table(df):
    """Per-configuration cell means: one row per (model, dataset, KL direction, coupling,
    lr, epochs), seeds averaged; ref = the EMA rate, NaN for the frozen teacher."""
    df = df.assign(learn=df.task_acc, keep=df.lmeval_avg - df.lmeval_avg_baseline,
                   ref=np.where(df.ref_update.eq("frozen"), np.nan, df.ema_rate))
    key = ["model", "dataset", "direction", "ref", "lr", "epochs"]
    return df.groupby(key, observed=True, dropna=False).agg(
        learn=("learn", "mean"), keep=("keep", "mean"), n=("learn", "size")).reset_index()
