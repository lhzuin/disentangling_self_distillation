#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["pandas>=2.0"]
# ///
"""Rebuild data/e85_main_grid_curves.csv from results/v14_final/aggregate/cell_summary.csv.gz.

The E85 block (V14.2 canonical main grid: 96 paired seeds, kappas {0.3,0.6,0.9} x
rho_phi {0,0.5,1} x 7 lambdas x full 18-alpha ladder) lives inside cell_summary as
rows whose cell_name is e85_p{rho}_l{lam}_k{kappa}_{rollout}_a{alpha}_{kl}, with 'p'
standing for the decimal point. This extract keeps only the two plotted measures;
fig3.py's model panels read it. Rerun after any V14 re-aggregation
(verified: extract == source on all 4536 rows).

    uv run --offline figures/collect_controlled_model.py
"""
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
SRC = ROOT.parent / "results/v14_final/aggregate/cell_summary.csv.gz"
OUT = ROOT / "data/e85_main_grid_curves.csv"
PAT = re.compile(r"^e85_p(?P<rho>[0-9p]+)_l(?P<lam>[0-9p]+)_k(?P<kap>[0-9p]+)"
                 r"_(?P<roll>teacher|student)_a(?P<a>[0-9p]+(?:e-?\d+)?)"
                 r"_(?P<kl>forward|reverse)$")


def num(s):
    return float(s.replace("p", "."))


def main():
    src = pd.read_csv(SRC, usecols=["cell_name", "new_match_gain__mean",
                                    "old_match_forgetting__mean"])
    src = src[src.cell_name.str.startswith("e85_")]
    rows = []
    for _, r in src.iterrows():
        m = PAT.match(r.cell_name)
        if not m:
            raise SystemExit(f"unparsed cell_name: {r.cell_name}")
        rows.append({"lambda": num(m["lam"]), "c": num(m["kap"]), "rho_phi": num(m["rho"]),
                     "kl_direction": m["kl"], "rollout_source": m["roll"],
                     "ema_alpha": num(m["a"]),
                     "new_match_gain__mean": r.new_match_gain__mean,
                     "old_match_forgetting__mean": r.old_match_forgetting__mean,
                     "source_block": "V14_E85_main_grid"})
    df = pd.DataFrame(rows).sort_values(
        ["lambda", "c", "rho_phi", "kl_direction", "rollout_source", "ema_alpha"])
    df.to_csv(OUT, index=False)
    print(f"wrote {OUT} ({len(df)} rows)")


if __name__ == "__main__":
    main()
