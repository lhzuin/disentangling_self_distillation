#!/usr/bin/env python3
"""V14.1 stability-envelope extension for the final contextual-SD toy model (dynamic multi-GPU recovery runner).

Purpose
=======
V14.0 established the three paper-facing mechanisms.  This extension is a
hypothesis-driven closure experiment motivated by the LLM observation that, in
some faster-adaptation regimes, forward KL can outperform reverse KL on both
new-task acquisition and old-task retention.

The extension deliberately keeps the V14 model itself frozen.  It reuses
``contextual_sd_toy_v14.py`` unchanged and varies only paper-facing / optimizer
controls already present in V14:

* learning rate,
* initial-policy concentration lambda,
* context strength kappa (stored in the V14 engine as ``c``),
* feature overlap rho_phi,
* KL direction,
* EMA teacher update rate alpha.

Teacher trajectories are used throughout to keep the extension focused and to
limit the added grid.  H=4, K=8, D=64, rho_W=0 and beta=0 remain unchanged.
All paper-facing cells use 96 paired task seeds and exact occupancy.

Blocks
======
E83  LR-only stability extension at rho_phi=.5.  We first declare the complete
     desired stability surface, then automatically remove every scientific cell
     that already exists anywhere in the V14.0 E70--E82 registry.  The remaining
     cells extend LR above the original range and fill only genuinely missing
     low-alpha/LR combinations.

E84  LR x representation-overlap stability extension.  The same automatic
     registry subtraction is applied to the high-overlap candidate surface, so
     existing E70/E76 cells are reused rather than rerun.  Only genuinely new
     combinations are scheduled.

Aggregation
===========
``--aggregate`` writes a V14.1-new-results-only aggregation directory
(default: ``<output-root>/aggregate_v14_1``).  Existing V14.0 cells are never
copied or rerun there.  ``--update-global-aggregate`` merges the new state into
the existing global V14 aggregate, which remains the place where old and new
evidence are analyzed together.

``--update-global-aggregate`` idempotently merges the V14.1 cell tables into the
existing V14.0 aggregate directory and copies the new specialized tables there.
It also creates ``kl_lr_alpha_curves_extended.csv`` by combining the old E72
rho_phi=.5 curves with E83/E84, without modifying the original E72 CSV.

Execution / recovery
====================
The runner is seed-sharded.  Every shard is written to an independent directory
under ``<block>/shards/``.  Complete shards are skipped on rerun; incomplete
shards are deleted and recomputed.  The canonical block-level ``runs.csv`` and
``history.csv`` are rebuilt deterministically from complete shards, so a crash
never requires discarding completed seeds.

V14.1 also supports multi-GPU execution by assigning independent seed shards to
explicit devices via ``--devices cuda:0,cuda:1,...``.  Parallel execution uses a
dynamic shared queue: whenever a device finishes a shard it immediately claims
the next pending shard.  Faster GPUs therefore perform more shards instead of
idling behind slower workers.  Device names may be repeated deliberately (for
example ``...,cuda:6,cuda:6``) to run multiple independent workers on one GPU;
do this only when GPU compute utilization, not merely free memory, leaves useful
headroom.  ``--shard-order reverse`` or ``--shard-indices 6,5,4`` can still be
used to restrict/order the queue.  Atomic claim files prevent duplicate work.

Recovery is cell-aware as well as shard-aware.  If an earlier run used a larger
seed shard (for example all 96 seeds) and stopped after completing only part of
the cell grid, ``--recover-partial`` (default) automatically salvages every cell
that has a complete row for every seed in a requested new shard.  The old shard
is never modified.  Recovered cells are split into the new seed-shard layout and
only the missing cells are computed.  The same mechanism also salvages completed
cells from interrupted recovery attempts, so changing from 96-seed to 16-seed
shards does not require discarding already-written work.

Typical workflow
================
  python toy_model/explore_v14_1_stability_multigpu_dynamic.py --self-test
  python toy_model/explore_v14_1_stability_multigpu_dynamic.py --aggregation-self-test
  python toy_model/explore_v14_1_stability_multigpu_dynamic.py --engine-self-test
  python toy_model/explore_v14_1_stability_multigpu_dynamic.py --list

  # Smoke outside the evidence folder.
  python toy_model/explore_v14_1_stability_multigpu_dynamic.py --run all --smoke \
      --output-root results/v14_1_smoke --seed-shard-size 4
  python toy_model/explore_v14_1_stability_multigpu_dynamic.py --aggregate \
      --output-root results/v14_1_smoke

  # Evidence runs live beside V14.0 (E70--E82).
  python toy_model/explore_v14_1_stability_multigpu_dynamic.py --run V14_E83_lr_stability \
      --output-root results/v14_final --seed-shard-size 16
  python toy_model/explore_v14_1_stability_multigpu_dynamic.py --run V14_E84_lr_rhophi_stability \
      --output-root results/v14_final --seed-shard-size 16

  python toy_model/explore_v14_1_stability_multigpu_dynamic.py --aggregate \
      --output-root results/v14_final
  python toy_model/explore_v14_1_stability_multigpu_dynamic.py --update-global-aggregate \
      --output-root results/v14_final
"""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Empty, Queue
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


DEFAULT_V14_EXPLORATION = Path("toy_model/explore_v14_final.py")
DEFAULT_ENGINE = Path("toy_model/contextual_sd_toy_v14.py")
DEFAULT_OUTPUT_ROOT = Path("results/v14_final")

N_MAIN = 96
BASE_H = 4
BASE_K = 8
BASE_D = 64
BASE_STEPS = 200
ORIGINAL_V14_ROWS = 3_018_816
EXPECTED_E83_CANDIDATES = 4_914
EXPECTED_E83_EXISTING = 2_730
EXPECTED_E83_NEW = 2_184
EXPECTED_E84_CANDIDATES = 14_040
EXPECTED_E84_EXISTING = 480
EXPECTED_E84_NEW = 13_560

# V14.0 grids kept verbatim where possible.
LAMBDA_FINE = (
    0.625, 0.883883476, 1.25, 1.767766953,
    2.5, 3.535533906, 5.0,
)
# The high-overlap experiment omits only the two weakest-prior values, where
# V14 already showed very small forward/reverse separation.  Every retained
# value is an established V14 lambda grid point.
LAMBDA_STABILITY = (1.25, 1.767766953, 2.5, 3.535533906, 5.0)
KAPPA_MAIN = (0.30, 0.60, 0.90)

LR_V14 = (
    0.00025, 0.000353553391, 0.0005, 0.000707106781,
    0.001, 0.001414213562, 0.002, 0.002828427125, 0.004,
)
# Continue the sqrt(2) schedule beyond V14.0 to probe the stability boundary.
LR_HIGH = (0.005656854249, 0.008, 0.011313708499, 0.016)
LR_STABILITY = LR_V14 + LR_HIGH

# Retain the whole E72 route up to .04 while adding low-alpha points needed
# for kappa=.9.  This is much denser around the stable region than a simple
# frozen/.0025/.02 anchor grid.
EMA_STABILITY = (
    0.0, 0.000625, 0.000883883476, 0.00125,
    0.0025, 0.005, 0.01, 0.02, 0.04,
)
# Existing angle-spaced V14 values plus one deliberately predeclared midpoint
# between 30 and 15 degrees, where the previous results place the retention
# sign transition.  rho=.5 remains in E72/E83 and is not duplicated in E84.
RHO_PHI_STABILITY = (
    round(math.cos(math.radians(30.0)), 12),     # .866025...
    round(math.cos(math.radians(22.5)), 12),     # .923879... (new midpoint)
    round(math.cos(math.radians(15.0)), 12),     # .965925...
    1.0,
)

# A small, preregistered subset receives history traces.  The endpoint metrics
# already record drawdown for all cells; histories are only needed to visualize
# the temporal instability mechanism without exploding disk usage.
HISTORY_RHOS = {
    0.5,
    round(math.cos(math.radians(30.0)), 12),
    round(math.cos(math.radians(22.5)), 12),
    round(math.cos(math.radians(15.0)), 12),
}
HISTORY_LAMBDA = 3.535533906
HISTORY_KAPPA = 0.90
HISTORY_ALPHAS = (0.0, 0.000883883476, 0.0025, 0.005)
HISTORY_LRS = (0.000707106781, 0.001, 0.001414213562, 0.002, 0.004, 0.008)

# All genuinely new E83/E84 cells share a fresh 96-seed family.  Existing V14.0
# cells are reused from their original blocks and are not rerun merely to force
# cross-block seed pairing.  New-new comparisons remain paired.
SEED_START_STABILITY = 161_000
SEED_FAMILY = "v14_1_stability_161k"


@dataclass(frozen=True)
class ExtensionBlock:
    name: str
    question: str
    prediction: str
    cells: Tuple[Dict[str, object], ...]
    candidate_cells: int
    excluded_existing_cells: int
    existing_source_occurrences: Tuple[Tuple[str, int], ...] = ()
    seed_start: int = SEED_START_STABILITY
    n_seeds: int = N_MAIN
    seed_family: str = SEED_FAMILY

    @property
    def n_runs(self) -> int:
        return len(self.cells) * self.n_seeds


_V14 = None


def _load_module(path: Path, name: str):
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_v14_exploration(path: Path):
    global _V14
    if _V14 is None:
        _V14 = _load_module(path, "explore_v14_final_for_v14_1")
    return _V14


def _close(a: float, b: float, atol: float = 1e-12) -> bool:
    return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=atol)


def _token(x: float) -> str:
    return f"{float(x):.12g}".replace("-", "m").replace(".", "p")


def _history_cell(rho: float, lam: float, kappa: float, alpha: float, lr: float) -> bool:
    return (
        any(_close(rho, r) for r in HISTORY_RHOS)
        and _close(lam, HISTORY_LAMBDA)
        and _close(kappa, HISTORY_KAPPA)
        and any(_close(alpha, a) for a in HISTORY_ALPHAS)
        and any(_close(lr, x) for x in HISTORY_LRS)
    )


NON_SCIENTIFIC_CELL_KEYS = {"name", "analysis_block", "variant", "record_history"}


def _scientific_signature(cell_: Mapping[str, object]) -> str:
    """Canonical scientific identity used to subtract the V14.0 registry.

    ``teacher_fraction`` is semantically inert for endpoint rollout sources in
    the frozen backend, but some historical V14 blocks left the cell-factory
    default (0.5) while others stored the endpoint value explicitly.  Normalize
    it here so those are correctly recognized as the same experiment.
    """
    payload = dict(cell_)
    for key in NON_SCIENTIFIC_CELL_KEYS:
        payload.pop(key, None)
    rollout = payload.get("rollout_source")
    if rollout == "student":
        payload["teacher_fraction"] = 0.0
    elif rollout == "teacher":
        payload["teacher_fraction"] = 1.0
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _original_registry_index(v14) -> Dict[str, List[Tuple[str, str]]]:
    index: Dict[str, List[Tuple[str, str]]] = {}
    for block in v14.BLOCKS:
        for cell_ in block.cells:
            index.setdefault(_scientific_signature(cell_), []).append(
                (str(block.name), str(cell_["name"]))
            )
    return index


def _subtract_existing_cells(
    cells: Sequence[Dict[str, object]],
    original_index: Mapping[str, Sequence[Tuple[str, str]]],
) -> Tuple[List[Dict[str, object]], int, Tuple[Tuple[str, int], ...]]:
    """Keep only candidate cells absent from every V14.0 scientific block."""
    new_cells: List[Dict[str, object]] = []
    excluded = 0
    source_counts: Dict[str, int] = {}
    for cell_ in cells:
        hits = original_index.get(_scientific_signature(cell_))
        if hits:
            excluded += 1
            for block_name, _ in hits:
                source_counts[block_name] = source_counts.get(block_name, 0) + 1
        else:
            new_cells.append(cell_)
    return new_cells, excluded, tuple(sorted(source_counts.items()))


def build_blocks(v14) -> Tuple[ExtensionBlock, ...]:
    """Declare desired surfaces, then subtract all experiments already in V14.0."""
    original_index = _original_registry_index(v14)

    e83_candidates: List[Dict[str, object]] = []
    for lam in LAMBDA_FINE:
        for kappa in KAPPA_MAIN:
            for lr in LR_STABILITY:
                for alpha in EMA_STABILITY:
                    for kl in ("forward", "reverse"):
                        e83_candidates.append(v14.cell(
                            f"e83_l{_token(lam)}_k{_token(kappa)}_lr{_token(lr)}_a{_token(alpha)}_{kl}",
                            "V14_E83_lr_stability",
                            rho_phi=0.50, lam=lam, c=kappa,
                            kl=kl, rollout="teacher", teacher_fraction=1.0,
                            alpha=alpha, lr=lr, estimator="exact",
                            train_steps=BASE_STEPS,
                            record_history=_history_cell(0.5, lam, kappa, alpha, lr),
                            variant="lr_only_stability_surface",
                        ))

    e84_candidates: List[Dict[str, object]] = []
    for rho in RHO_PHI_STABILITY:
        for lam in LAMBDA_STABILITY:
            for kappa in KAPPA_MAIN:
                for lr in LR_STABILITY:
                    for alpha in EMA_STABILITY:
                        hist = _history_cell(rho, lam, kappa, alpha, lr)
                        for kl in ("forward", "reverse"):
                            e84_candidates.append(v14.cell(
                                f"e84_p{_token(rho)}_l{_token(lam)}_k{_token(kappa)}_lr{_token(lr)}_a{_token(alpha)}_{kl}",
                                "V14_E84_lr_rhophi_stability",
                                rho_phi=rho, lam=lam, c=kappa,
                                kl=kl, rollout="teacher", teacher_fraction=1.0,
                                alpha=alpha, lr=lr, estimator="exact",
                                train_steps=BASE_STEPS, record_history=hist,
                                variant="lr_x_rho_phi_stability_surface",
                            ))

    e83, e83_excluded, e83_sources = _subtract_existing_cells(e83_candidates, original_index)
    e84, e84_excluded, e84_sources = _subtract_existing_cells(e84_candidates, original_index)

    return (
        ExtensionBlock(
            "V14_E83_lr_stability",
            "Which LR/alpha combinations missing from V14.0 complete the rho_phi=.5 teacher-rollout stability surface?",
            "Reusing E72/E73 for existing points, the added cells should locate whether faster adaptation can move the reference-overlap system toward a forward-wins-both regime.",
            tuple(e83), len(e83_candidates), e83_excluded, e83_sources,
        ),
        ExtensionBlock(
            "V14_E84_lr_rhophi_stability",
            "How do learning rate and shared-feature interference jointly move the forward/reverse acquisition-retention ordering beyond the already-run V14 points?",
            "Increasing rho_phi should move the retention sign transition to lower LR. Existing E70/E76 points are reused; only missing combinations are run.",
            tuple(e84), len(e84_candidates), e84_excluded, e84_sources,
        ),
    )


def registry_summary(blocks: Sequence[ExtensionBlock]) -> Dict[str, object]:
    cells = sum(len(b.cells) for b in blocks)
    runs = sum(b.n_runs for b in blocks)
    candidates = sum(b.candidate_cells for b in blocks)
    excluded = sum(b.excluded_existing_cells for b in blocks)
    return {
        "version": "14.1",
        "policy": "schedule only scientific cells absent from the V14.0 registry",
        "seed_family": SEED_FAMILY,
        "n_blocks": len(blocks),
        "candidate_cells_before_v14_dedup": candidates,
        "excluded_existing_v14_cells": excluded,
        "n_new_cells": cells,
        "n_new_per_seed_rows": runs,
        "original_v14_rows": ORIGINAL_V14_ROWS,
        "extension_over_original_pct": 100.0 * runs / ORIGINAL_V14_ROWS,
        "combined_over_original_pct": 100.0 * (ORIGINAL_V14_ROWS + runs) / ORIGINAL_V14_ROWS,
        "blocks": {
            b.name: {
                "candidate_cells": b.candidate_cells,
                "excluded_existing_cells": b.excluded_existing_cells,
                "new_cells": len(b.cells),
                "seeds": b.n_seeds,
                "new_rows": b.n_runs,
                "seed_start": b.seed_start,
                "existing_source_occurrences": dict(b.existing_source_occurrences),
            }
            for b in blocks
        },
    }


def run_static_self_tests(v14, blocks: Sequence[ExtensionBlock]) -> Dict[str, object]:
    if N_MAIN != 96:
        raise AssertionError("V14.1 paper cells must use 96 task seeds")
    if set(KAPPA_MAIN) != {0.3, 0.6, 0.9}:
        raise AssertionError("kappa grid drifted from V14 CONTEXT_MAIN")
    if not set(LAMBDA_STABILITY).issubset(set(LAMBDA_FINE)):
        raise AssertionError("V14.1 lambda values must come from LAMBDA_FINE")
    if len(set(LR_STABILITY)) != len(LR_STABILITY) or tuple(sorted(LR_STABILITY)) != LR_STABILITY:
        raise AssertionError("LR_STABILITY must be unique and increasing")
    ratios = np.asarray(LR_HIGH[1:]) / np.asarray(LR_HIGH[:-1])
    if np.max(np.abs(ratios - math.sqrt(2.0))) > 5e-6:
        raise AssertionError("high-LR continuation is not sqrt(2)-spaced")

    original_index = _original_registry_index(v14)
    names: set[str] = set()
    scientific: set[str] = set()
    for b in blocks:
        if b.n_seeds != 96:
            raise AssertionError(f"{b.name} does not use 96 seeds")
        for c in b.cells:
            name = str(c["name"])
            if name in names:
                raise AssertionError(f"duplicate cell name: {name}")
            names.add(name)
            sig = _scientific_signature(c)
            if sig in original_index:
                raise AssertionError(
                    f"V14.1 scheduled an experiment that already exists in V14.0: {name}; "
                    f"sources={original_index[sig]}"
                )
            if sig in scientific:
                raise AssertionError(f"duplicate scientific cell inside V14.1: {name}")
            scientific.add(sig)
            if c["rollout_source"] != "teacher":
                raise AssertionError(f"non-teacher trajectory leaked into {name}")
            if c["estimator"] != "exact":
                raise AssertionError(f"non-exact estimator leaked into {name}")
            s = c["structure"]
            if (int(s["horizon"]), int(s["vocab_size"]), int(s["feature_dim"])) != (4, 8, 64):
                raise AssertionError(f"structural change leaked into {name}: {s}")
            ab = c["ablation"]
            if not _close(float(ab["readout_compatibility"]), 0.0):
                raise AssertionError("rho_W must remain zero")
            if not _close(float(ab["support_placement_bias"]), 0.0):
                raise AssertionError("beta must remain zero")

    e83 = next(b for b in blocks if b.name == "V14_E83_lr_stability")
    e84 = next(b for b in blocks if b.name == "V14_E84_lr_rhophi_stability")
    expected = {
        e83.name: (EXPECTED_E83_CANDIDATES, EXPECTED_E83_EXISTING, EXPECTED_E83_NEW),
        e84.name: (EXPECTED_E84_CANDIDATES, EXPECTED_E84_EXISTING, EXPECTED_E84_NEW),
    }
    for b in (e83, e84):
        candidate, excluded, new = expected[b.name]
        if (b.candidate_cells, b.excluded_existing_cells, len(b.cells)) != (candidate, excluded, new):
            raise AssertionError(
                f"{b.name} V14 subtraction changed: got "
                f"{(b.candidate_cells, b.excluded_existing_cells, len(b.cells))}, "
                f"expected {(candidate, excluded, new)}"
            )
    if any(not _close(float(c["core"]["feature_overlap"]), 0.5) for c in e83.cells):
        raise AssertionError("E83 must be rho_phi=.5 only")
    if any(float(c["core"]["feature_overlap"]) <= 0.5 for c in e84.cells):
        raise AssertionError("E84 must contain only high-overlap values")

    # Compatibility contract with the original cell factory.
    probe = v14.cell("probe", "probe", rho_phi=.5, lam=2.5, c=.6, kl="reverse", rollout="teacher")
    if set(probe["core"]) != {"feature_overlap", "initial_policy_concentration", "context_strength"}:
        raise AssertionError("unexpected V14 core surface")

    return {
        "status": "PASS",
        **registry_summary(blocks),
        "scheduled_overlap_with_v14": 0,
        "history_cells": sum(bool(c["record_history"]) for b in blocks for c in b.cells),
    }


def smoke_subset(block: ExtensionBlock) -> List[Dict[str, object]]:
    """Small motifs containing only cells that survive V14.0 subtraction."""
    groups: Dict[Tuple[object, ...], Dict[str, Dict[str, object]]] = {}
    for c in block.cells:
        key = (
            float(c["core"]["feature_overlap"]),
            float(c["core"]["initial_policy_concentration"]),
            float(c["core"]["context_strength"]),
            float(c["learning_rate"]),
            float(c["ema_alpha"]),
        )
        groups.setdefault(key, {})[str(c["kl_direction"])] = c
    paired = [(k, v) for k, v in groups.items() if {"forward", "reverse"}.issubset(v)]
    if not paired:
        return list(block.cells[:8])

    # Spread across LR and, for E84, rho_phi.  This is plumbing only.
    paired.sort(key=lambda kv: (kv[0][0], kv[0][3], kv[0][4]))
    indices = sorted({0, len(paired)//4, len(paired)//2, 3*len(paired)//4, len(paired)-1})
    out: List[Dict[str, object]] = []
    for i in indices:
        _, pair = paired[i]
        out.extend([pair["forward"], pair["reverse"]])
    return out[:10]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_block_spec(path: Path, block: ExtensionBlock, cells: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "v14_1_registry_block": block.name,
        "seed_family": block.seed_family,
        "question": block.question,
        "prediction": block.prediction,
        "cells": list(cells),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _csv_row_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("rb") as f:
        # subtract header; robust for ordinary engine CSV output (no embedded newlines)
        return max(0, sum(1 for _ in f) - 1)


def _validate_shard(shard_dir: Path, expected_rows: int, expected_seed_start: int,
                    expected_n_seeds: int, expected_cells: int) -> Tuple[bool, str]:
    runs = shard_dir / "runs.csv"
    manifest = shard_dir / "manifest.json"
    if not runs.exists() or not manifest.exists():
        return False, "missing runs.csv or manifest.json"
    rows = _csv_row_count(runs)
    if rows != expected_rows:
        return False, f"row count {rows} != {expected_rows}"
    try:
        key = pd.read_csv(runs, usecols=["cell_name", "task_seed"])
    except Exception as exc:
        return False, f"cannot read shard keys: {exc}"
    if len(key) != expected_rows:
        return False, "parsed row count mismatch"
    if key.duplicated(["cell_name", "task_seed"]).any():
        return False, "duplicate (cell_name, task_seed) rows"
    seeds = sorted(pd.to_numeric(key["task_seed"]).astype(int).unique().tolist())
    wanted = list(range(expected_seed_start, expected_seed_start + expected_n_seeds))
    if seeds != wanted:
        return False, f"task seeds {seeds[:3]}... do not match {wanted[:3]}..."
    if key["cell_name"].nunique() != expected_cells:
        return False, f"cell count {key['cell_name'].nunique()} != {expected_cells}"
    return True, "complete"


def _tee_subprocess(cmd: Sequence[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            list(cmd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
        ret = proc.wait()
    if ret != 0:
        raise subprocess.CalledProcessError(ret, list(cmd))


def _merge_csv_files(paths: Sequence[Path], out: Path) -> None:
    """Streaming, deterministic CSV concatenation with header verification."""
    paths = [p for p in paths if p.exists()]
    if not paths:
        if out.exists():
            out.unlink()
        return
    tmp = out.with_suffix(out.suffix + ".tmp")
    header: Optional[bytes] = None
    with tmp.open("wb") as w:
        for i, p in enumerate(paths):
            with p.open("rb") as r:
                this_header = r.readline()
                if header is None:
                    header = this_header
                    w.write(this_header)
                elif this_header != header:
                    raise ValueError(f"CSV header mismatch while merging {p}")
                shutil.copyfileobj(r, w, length=1 << 20)
    os.replace(tmp, out)


def _rebuild_block_outputs(block_dir: Path, shard_dirs: Sequence[Path]) -> None:
    runs = [d / "runs.csv" for d in shard_dirs if (d / "runs.csv").exists()]
    histories = [d / "history.csv" for d in shard_dirs if (d / "history.csv").exists()]
    _merge_csv_files(runs, block_dir / "runs.csv")
    _merge_csv_files(histories, block_dir / "history.csv")


_SHARD_NAME_RE = re.compile(r"^seeds_(\d+)_(\d+)$")


def _seed_range_from_name(path: Path) -> Optional[Tuple[int, int]]:
    m = _SHARD_NAME_RE.match(path.name)
    if not m:
        return None
    lo, hi = int(m.group(1)), int(m.group(2))
    if hi < lo:
        return None
    return lo, hi


def _read_recovery_csv(path: Path) -> Optional[pd.DataFrame]:
    """Best-effort reader for a possibly interrupted CSV.

    Recovery is deliberately conservative later: a cell is reused only when all
    requested seeds are present exactly once.  ``on_bad_lines='skip'`` therefore
    only allows us to ignore a torn final line; it cannot make an incomplete cell
    eligible for reuse.
    """
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        return pd.read_csv(path, on_bad_lines="skip")
    except Exception as exc:
        print(f"[recovery] could not read {path}: {exc}; ignoring it", flush=True)
        return None


def _compatible_source_cell_names(
    source_dir: Path,
    current_cells_by_name: Mapping[str, Mapping[str, object]],
) -> set[str]:
    """Return source cells whose scientific specification matches this run.

    We require the source's experiment_spec.json.  This prevents accidentally
    reusing rows from a directory with the same cell names but different
    scientific settings.
    """
    spec_path = source_dir / "experiment_spec.json"
    if not spec_path.exists() and source_dir.name.startswith("attempt_"):
        # Recovery attempts store their exact missing-cell spec beside the
        # attempt directory (attempt_XXXX_spec.json).  Supporting this fallback
        # is what lets a later repartition (e.g. 16 seeds -> 6 seeds) salvage
        # cells written by an interrupted recovery attempt.
        sibling = source_dir.parent / f"{source_dir.name}_spec.json"
        if sibling.exists():
            spec_path = sibling
    if not spec_path.exists():
        print(
            f"[recovery] {source_dir} has runs.csv but no compatible experiment spec; "
            "not reusing it automatically.",
            flush=True,
        )
        return set()
    try:
        payload = json.loads(spec_path.read_text(encoding="utf-8"))
        source_cells = {
            str(c["name"]): c for c in payload.get("cells", []) if "name" in c
        }
    except Exception as exc:
        print(f"[recovery] invalid spec in {source_dir}: {exc}; ignoring it", flush=True)
        return set()

    compatible: set[str] = set()
    for name, current in current_cells_by_name.items():
        old = source_cells.get(name)
        if old is not None and _scientific_signature(old) == _scientific_signature(current):
            compatible.add(name)
    return compatible


def _recovery_frames_from_source(
    source_dir: Path,
    current_cells_by_name: Mapping[str, Mapping[str, object]],
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], set[str]]:
    compatible = _compatible_source_cell_names(source_dir, current_cells_by_name)
    if not compatible:
        return None, None, set()
    runs = _read_recovery_csv(source_dir / "runs.csv")
    if runs is None or not {"cell_name", "task_seed"}.issubset(runs.columns):
        return None, None, set()
    runs = runs[runs["cell_name"].astype(str).isin(compatible)].copy()
    histories = _read_recovery_csv(source_dir / "history.csv")
    if histories is not None:
        if {"cell_name", "task_seed"}.issubset(histories.columns):
            histories = histories[histories["cell_name"].astype(str).isin(compatible)].copy()
        else:
            histories = None
    return runs, histories, compatible


def _rows_equal_ignoring_source(g: pd.DataFrame) -> bool:
    """Whether duplicate recovered rows agree on all scientific/output columns."""
    cols = [c for c in g.columns if c != "__recovery_source"]
    if len(g) <= 1:
        return True
    # Converting to strings is intentional: these are rows read from CSV, and
    # exact textual equality is stricter than a numerical tolerance for recovery.
    canon = g[cols].astype(str).drop_duplicates()
    return len(canon) == 1


def _collect_complete_recovered_cells(
    *,
    source_frames: Sequence[Tuple[Path, pd.DataFrame, Optional[pd.DataFrame]]],
    cells: Sequence[Mapping[str, object]],
    expected_seeds: Sequence[int],
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], List[str], List[str]]:
    """Collect only cells that are fully recoverable for ``expected_seeds``.

    A recovered endpoint cell must have exactly one non-conflicting runs.csv row
    for every requested seed.  Cells that record histories are accepted only if
    history rows exist for every requested seed as well.  Anything ambiguous is
    recomputed rather than silently trusted.
    """
    expected = tuple(int(x) for x in expected_seeds)
    expected_set = set(expected)
    order = {str(c["name"]): i for i, c in enumerate(cells)}
    needs_history = {
        str(c["name"]) for c in cells if bool(c.get("record_history", False))
    }

    run_parts: List[pd.DataFrame] = []
    hist_parts: List[pd.DataFrame] = []
    sources_used: List[str] = []
    for source_dir, runs, hist in source_frames:
        r = runs.copy()
        r["task_seed"] = pd.to_numeric(r["task_seed"], errors="coerce")
        r = r[r["task_seed"].isin(expected_set)].copy()
        if len(r):
            r["task_seed"] = r["task_seed"].astype(int)
            r["__recovery_source"] = str(source_dir)
            run_parts.append(r)
            sources_used.append(str(source_dir))
        if hist is not None and len(hist):
            h = hist.copy()
            h["task_seed"] = pd.to_numeric(h["task_seed"], errors="coerce")
            h = h[h["task_seed"].isin(expected_set)].copy()
            if len(h):
                h["task_seed"] = h["task_seed"].astype(int)
                h["__recovery_source"] = str(source_dir)
                hist_parts.append(h)

    if not run_parts:
        return pd.DataFrame(), None, [], []

    all_runs = pd.concat(run_parts, ignore_index=True, sort=False)
    unsafe_cells: set[str] = set()
    duplicate_mask = all_runs.duplicated(["cell_name", "task_seed"], keep=False)
    if duplicate_mask.any():
        for (_, _), g in all_runs[duplicate_mask].groupby(["cell_name", "task_seed"], sort=False):
            if not _rows_equal_ignoring_source(g):
                unsafe_cells.add(str(g.iloc[0]["cell_name"]))

    # Deterministic source priority: first listed source wins identical duplicates.
    all_runs = all_runs.drop_duplicates(["cell_name", "task_seed"], keep="first")
    complete: List[str] = []
    for name, g in all_runs.groupby("cell_name", sort=False):
        name = str(name)
        if name in unsafe_cells or name not in order:
            continue
        seeds = set(pd.to_numeric(g["task_seed"]).astype(int).tolist())
        if seeds == expected_set and len(g) == len(expected):
            complete.append(name)

    all_hist: Optional[pd.DataFrame]
    if hist_parts:
        all_hist = pd.concat(hist_parts, ignore_index=True, sort=False)
        # Duplicate history records from two identical sources are harmless.
        hist_key_candidates = [
            c for c in ("cell_name", "task_seed", "step", "iteration", "record_step")
            if c in all_hist.columns
        ]
        if len(hist_key_candidates) >= 2:
            all_hist = all_hist.drop_duplicates(hist_key_candidates, keep="first")
    else:
        all_hist = None

    # History-bearing cells are only reusable if their history is also present
    # for every seed.  This keeps paper-facing time-series outputs complete.
    if needs_history:
        hist_ok: set[str] = set()
        if all_hist is not None:
            for name, g in all_hist.groupby("cell_name", sort=False):
                seeds = set(pd.to_numeric(g["task_seed"], errors="coerce").dropna().astype(int))
                if expected_set.issubset(seeds):
                    hist_ok.add(str(name))
        complete = [n for n in complete if n not in needs_history or n in hist_ok]

    complete_set = set(complete)
    recovered_runs = all_runs[all_runs["cell_name"].astype(str).isin(complete_set)].copy()
    recovered_runs.drop(columns=["__recovery_source"], inplace=True, errors="ignore")
    recovered_runs["__cell_order"] = recovered_runs["cell_name"].astype(str).map(order)
    recovered_runs.sort_values(["__cell_order", "task_seed"], inplace=True, kind="stable")
    recovered_runs.drop(columns=["__cell_order"], inplace=True)
    recovered_runs.reset_index(drop=True, inplace=True)

    recovered_hist: Optional[pd.DataFrame] = None
    if all_hist is not None and complete_set:
        recovered_hist = all_hist[all_hist["cell_name"].astype(str).isin(complete_set)].copy()
        recovered_hist.drop(columns=["__recovery_source"], inplace=True, errors="ignore")
        recovered_hist["__cell_order"] = recovered_hist["cell_name"].astype(str).map(order)
        sort_cols = ["__cell_order", "task_seed"]
        for extra in ("step", "iteration", "record_step"):
            if extra in recovered_hist.columns:
                sort_cols.append(extra)
                break
        recovered_hist.sort_values(sort_cols, inplace=True, kind="stable")
        recovered_hist.drop(columns=["__cell_order"], inplace=True)
        recovered_hist.reset_index(drop=True, inplace=True)

    complete.sort(key=lambda n: order[n])
    return recovered_runs, recovered_hist, complete, sorted(set(sources_used))


def _write_partial_recovery_shard(
    *,
    shard_dir: Path,
    full_spec_path: Path,
    runs: pd.DataFrame,
    history: Optional[pd.DataFrame],
    recovered_cells: Sequence[str],
    sources: Sequence[str],
    expected_seeds: Sequence[int],
) -> None:
    """Materialize reusable rows in a new shard without marking it complete."""
    if not len(runs):
        return
    shard_dir.mkdir(parents=True, exist_ok=True)
    # A previous manifest may belong to a failed/incompatible layout.  A partial
    # recovered shard must never advertise itself as complete.
    try:
        (shard_dir / "manifest.json").unlink()
    except FileNotFoundError:
        pass
    runs_tmp = shard_dir / "runs.csv.recovery_tmp"
    runs.to_csv(runs_tmp, index=False)
    os.replace(runs_tmp, shard_dir / "runs.csv")
    if history is not None and len(history):
        hist_tmp = shard_dir / "history.csv.recovery_tmp"
        history.to_csv(hist_tmp, index=False)
        os.replace(hist_tmp, shard_dir / "history.csv")
    shutil.copy2(full_spec_path, shard_dir / "experiment_spec.json")
    # Do not create manifest.json here: _validate_shard must continue treating
    # this as incomplete until all cells are present.
    recovery_manifest = {
        "status": "partial_recovery",
        "created_unix": time.time(),
        "seed_start": int(expected_seeds[0]),
        "seed_end": int(expected_seeds[-1]),
        "n_recovered_cells": len(recovered_cells),
        "n_recovered_rows": int(len(runs)),
        "sources": list(sources),
    }
    (shard_dir / "recovery_manifest.json").write_text(
        json.dumps(recovery_manifest, indent=2, sort_keys=True), encoding="utf-8"
    )


def _migrate_overlapping_legacy_shards(
    *,
    block_dir: Path,
    cells: Sequence[Mapping[str, object]],
    indexed_shards: Sequence[Tuple[int, Tuple[int, int]]],
    full_spec_path: Path,
) -> None:
    """Split useful rows from old larger shards into the current shard layout.

    The source directories are read-only.  For the common migration from one
    interrupted 96-seed shard to six 16-seed shards, every fully written cell in
    the 96-seed file is copied into all six new shard directories, filtered to
    their seed ranges.  The original 96-seed directory remains untouched.
    """
    shards_root = block_dir / "shards"
    if not shards_root.exists():
        return
    target_names = {
        f"seeds_{start}_{start + n - 1}" for _, (start, n) in indexed_shards
    }
    current_cells_by_name = {str(c["name"]): c for c in cells}

    legacy_dirs: List[Path] = []
    for child in shards_root.iterdir():
        if not child.is_dir() or child.name.startswith("."):
            continue
        rng = _seed_range_from_name(child)
        if rng is None or child.name in target_names:
            continue
        lo, hi = rng
        if any(not (hi < s or lo > s + n - 1) for _, (s, n) in indexed_shards):
            legacy_dirs.append(child)

    # Also salvage interrupted *recovery attempts* from an older shard layout.
    # Their rows live under .recovery_work/seeds_A_B/attempt_XXXX and otherwise
    # would be invisible when repartitioning, for example 16-seed -> 6-seed.
    work_root = shards_root / ".recovery_work"
    if work_root.exists():
        for seed_dir in sorted(p for p in work_root.iterdir() if p.is_dir()):
            rng = _seed_range_from_name(seed_dir)
            if rng is None:
                continue
            lo, hi = rng
            if not any(not (hi < s or lo > s + n - 1) for _, (s, n) in indexed_shards):
                continue
            # Exact-layout attempts are handled by _partial_sources_for_shard;
            # only older/different layouts need to be injected globally here.
            if seed_dir.name in target_names:
                continue
            legacy_dirs.extend(
                p for p in seed_dir.glob("attempt_*")
                if p.is_dir()
            )

    if not legacy_dirs:
        return

    loaded: List[Tuple[Path, pd.DataFrame, Optional[pd.DataFrame]]] = []
    for src in sorted(set(legacy_dirs), key=str):
        runs, hist, compatible = _recovery_frames_from_source(src, current_cells_by_name)
        if runs is None or not len(runs):
            continue
        label = src.name if src.parent.name != ".recovery_work" else str(src.relative_to(shards_root))
        print(
            f"[recovery] found overlapping source {label}: "
            f"{len(runs)} readable compatible rows across "
            f"{runs['cell_name'].nunique()} cells",
            flush=True,
        )
        loaded.append((src, runs, hist))
    if not loaded:
        return

    for idx, (start, n) in indexed_shards:
        expected_seeds = list(range(start, start + n))
        target = shards_root / f"seeds_{start}_{start + n - 1}"
        target_ok, _ = _validate_shard(
            target, len(cells) * n, start, n, len(cells)
        )
        if target_ok:
            continue
        sources = list(loaded)
        # Preserve any already-materialized partial target rows too.
        if target.exists():
            tr, th, _ = _recovery_frames_from_source(target, current_cells_by_name)
            if tr is not None and len(tr):
                sources.insert(0, (target, tr, th))
        rr, rh, complete, used = _collect_complete_recovered_cells(
            source_frames=sources, cells=cells, expected_seeds=expected_seeds
        )
        if complete:
            _write_partial_recovery_shard(
                shard_dir=target,
                full_spec_path=full_spec_path,
                runs=rr,
                history=rh,
                recovered_cells=complete,
                sources=used,
                expected_seeds=expected_seeds,
            )
            print(
                f"[recovery] shard {idx}/{len(indexed_shards)} "
                f"{start}:{start+n-1}: recovered {len(complete)}/{len(cells)} cells "
                f"({len(rr)} rows); only {len(cells)-len(complete)} cells remain.",
                flush=True,
            )


def _next_recovery_attempt_dir(block_dir: Path, shard_name: str) -> Path:
    root = block_dir / "shards" / ".recovery_work" / shard_name
    root.mkdir(parents=True, exist_ok=True)
    existing = []
    for p in root.glob("attempt_*"):
        try:
            existing.append(int(p.name.split("_", 1)[1]))
        except Exception:
            pass
    idx = max(existing, default=0) + 1
    return root / f"attempt_{idx:04d}"


def _partial_sources_for_shard(
    *,
    block_dir: Path,
    shard_dir: Path,
    current_cells_by_name: Mapping[str, Mapping[str, object]],
) -> List[Tuple[Path, pd.DataFrame, Optional[pd.DataFrame]]]:
    out: List[Tuple[Path, pd.DataFrame, Optional[pd.DataFrame]]] = []
    candidates: List[Path] = []
    if shard_dir.exists():
        candidates.append(shard_dir)
    work_root = block_dir / "shards" / ".recovery_work" / shard_dir.name
    if work_root.exists():
        candidates.extend(sorted(p for p in work_root.glob("attempt_*") if p.is_dir()))
    for src in candidates:
        runs, hist, _ = _recovery_frames_from_source(src, current_cells_by_name)
        if runs is not None and len(runs):
            out.append((src, runs, hist))
    return out


def _atomic_finalize_recovered_shard(
    *,
    shard_dir: Path,
    full_spec_path: Path,
    runs: pd.DataFrame,
    history: Optional[pd.DataFrame],
    manifest: Mapping[str, object],
) -> None:
    shard_dir.mkdir(parents=True, exist_ok=True)
    runs_tmp = shard_dir / "runs.csv.finalize_tmp"
    runs.to_csv(runs_tmp, index=False)
    os.replace(runs_tmp, shard_dir / "runs.csv")
    hist_path = shard_dir / "history.csv"
    if history is not None and len(history):
        hist_tmp = shard_dir / "history.csv.finalize_tmp"
        history.to_csv(hist_tmp, index=False)
        os.replace(hist_tmp, hist_path)
    elif hist_path.exists():
        hist_path.unlink()
    shutil.copy2(full_spec_path, shard_dir / "experiment_spec.json")
    manifest_tmp = shard_dir / "manifest.json.finalize_tmp"
    manifest_tmp.write_text(json.dumps(dict(manifest), indent=2, sort_keys=True), encoding="utf-8")
    # Manifest is replaced last, making it the completion marker.
    os.replace(manifest_tmp, shard_dir / "manifest.json")


def _parse_shard_indices(spec: Optional[str], n_shards: int) -> Optional[List[int]]:
    """Parse a 1-based comma-separated shard list, preserving user order."""
    if spec is None:
        return None
    out: List[int] = []
    seen: set[int] = set()
    for raw in spec.split(","):
        raw = raw.strip()
        if not raw:
            continue
        value = int(raw)
        if value < 1 or value > n_shards:
            raise ValueError(f"shard index {value} outside valid range 1..{n_shards}")
        if value not in seen:
            out.append(value)
            seen.add(value)
    if not out:
        raise ValueError("--shard-indices did not contain any shard indices")
    return out


def _claim_shard(claim_path: Path) -> bool:
    """Atomically claim a shard for cooperating V14.1 parallel runners.

    The legacy single-GPU runner predates claims and therefore does not honor
    them.  Claims prevent duplicate work among *new* parallel runners only.
    """
    claim_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(str(claim_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    try:
        payload = {
            "pid": os.getpid(),
            "created_unix": time.time(),
        }
        os.write(fd, json.dumps(payload, sort_keys=True).encode("utf-8"))
    finally:
        os.close(fd)
    return True


def _release_shard_claim(claim_path: Path) -> None:
    try:
        claim_path.unlink()
    except FileNotFoundError:
        pass


def _run_subprocess_to_log(cmd: Sequence[str], log_path: Path) -> None:
    """Run one engine process with its output isolated in a per-shard log."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(
            list(cmd), stdout=log, stderr=subprocess.STDOUT, text=True,
        )
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, list(cmd))


def _execute_one_shard(
    *,
    block: ExtensionBlock,
    cells: Sequence[Mapping[str, object]],
    block_dir: Path,
    spec_path: Path,
    engine: Path,
    dtype: str,
    device: str,
    shard_index: int,
    n_shards: int,
    shard_start: int,
    shard_n: int,
    repair_incomplete: bool,
    recover_partial: bool,
) -> Dict[str, object]:
    """Execute one seed shard, reusing any safely recoverable completed cells."""
    shard_dir = block_dir / "shards" / f"seeds_{shard_start}_{shard_start + shard_n - 1}"
    expected_rows = len(cells) * shard_n
    expected_seeds = list(range(shard_start, shard_start + shard_n))

    ok, reason = _validate_shard(
        shard_dir, expected_rows, shard_start, shard_n, len(cells)
    )
    if ok:
        return {
            "status": "already_complete", "device": device,
            "shard_index": shard_index, "shard_dir": str(shard_dir),
        }

    claim_path = block_dir / "shards" / ".claims" / f"{shard_dir.name}.claim"
    if not _claim_shard(claim_path):
        return {
            "status": "claimed_elsewhere", "device": device,
            "shard_index": shard_index, "shard_dir": str(shard_dir),
        }

    try:
        ok, reason = _validate_shard(
            shard_dir, expected_rows, shard_start, shard_n, len(cells)
        )
        if ok:
            return {
                "status": "already_complete", "device": device,
                "shard_index": shard_index, "shard_dir": str(shard_dir),
            }

        current_cells_by_name = {str(c["name"]): c for c in cells}
        recovered_runs = pd.DataFrame()
        recovered_history: Optional[pd.DataFrame] = None
        recovered_names: List[str] = []
        recovery_sources: List[str] = []

        if recover_partial:
            sources = _partial_sources_for_shard(
                block_dir=block_dir,
                shard_dir=shard_dir,
                current_cells_by_name=current_cells_by_name,
            )
            if sources:
                recovered_runs, recovered_history, recovered_names, recovery_sources = (
                    _collect_complete_recovered_cells(
                        source_frames=sources,
                        cells=cells,
                        expected_seeds=expected_seeds,
                    )
                )

        recovered_set = set(recovered_names)
        missing_cells = [c for c in cells if str(c["name"]) not in recovered_set]
        print(
            f"[{block.name}] GPU {device}: shard {shard_index}/{n_shards} "
            f"recovery={len(recovered_names)}/{len(cells)} cells; "
            f"remaining={len(missing_cells)}",
            flush=True,
        )

        if not recover_partial and shard_dir.exists():
            if not repair_incomplete:
                return {
                    "status": "incomplete_not_repaired", "device": device,
                    "shard_index": shard_index, "shard_dir": str(shard_dir),
                    "reason": reason,
                }
            print(
                f"[{block.name}] GPU {device}: recovery disabled; deleting incomplete "
                f"shard {shard_index}/{n_shards} ({reason})",
                flush=True,
            )
            shutil.rmtree(shard_dir)
            recovered_runs = pd.DataFrame()
            recovered_history = None
            recovered_names = []
            recovery_sources = []
            missing_cells = list(cells)

        attempt_dir: Optional[Path] = None
        attempt_manifest: Optional[Mapping[str, object]] = None
        computed_rows = pd.DataFrame()
        computed_history: Optional[pd.DataFrame] = None

        if missing_cells:
            attempt_dir = _next_recovery_attempt_dir(block_dir, shard_dir.name)
            missing_spec = attempt_dir.parent / f"{attempt_dir.name}_spec.json"
            write_block_spec(missing_spec, block, missing_cells)
            attempt_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(missing_spec, attempt_dir / "experiment_spec.json")
            cmd = [
                sys.executable, str(engine),
                "--spec", str(missing_spec),
                "--output-dir", str(attempt_dir),
                "--device", device,
                "--dtype", dtype,
                "--n-task-seeds", str(shard_n),
                "--seed-start", str(shard_start),
                "--train-steps", str(BASE_STEPS),
                "--record-every", "10",
                "--sampled-trajectories-per-step", "256",
                "--paired-optimization-rng",
            ]
            log_path = attempt_dir.parent / f"{attempt_dir.name}.log"
            print(
                f"[{block.name}] GPU {device}: computing {len(missing_cells)} missing "
                f"cells for seeds={shard_start}:{shard_start + shard_n - 1}",
                flush=True,
            )
            _run_subprocess_to_log(cmd, log_path)

            # The engine should have completed the full missing-cell sub-spec.
            attempt_expected = len(missing_cells) * shard_n
            ok_attempt, attempt_reason = _validate_shard(
                attempt_dir, attempt_expected, shard_start, shard_n, len(missing_cells)
            )
            if not ok_attempt:
                raise RuntimeError(
                    f"Recovery attempt finished but did not validate: {attempt_dir}: "
                    f"{attempt_reason}"
                )
            computed_rows = pd.read_csv(attempt_dir / "runs.csv")
            hp = attempt_dir / "history.csv"
            if hp.exists():
                computed_history = pd.read_csv(hp)
            mp = attempt_dir / "manifest.json"
            if mp.exists():
                try:
                    attempt_manifest = json.loads(mp.read_text(encoding="utf-8"))
                except Exception:
                    attempt_manifest = None

        run_parts = [df for df in (recovered_runs, computed_rows) if df is not None and len(df)]
        if not run_parts:
            raise RuntimeError(f"No rows available to finalize {shard_dir}")
        final_runs = pd.concat(run_parts, ignore_index=True, sort=False)
        if final_runs.duplicated(["cell_name", "task_seed"]).any():
            dup = final_runs[final_runs.duplicated(["cell_name", "task_seed"], keep=False)]
            names = sorted(set(dup["cell_name"].astype(str)))[:10]
            raise RuntimeError(
                f"Duplicate recovered/computed keys while finalizing {shard_dir}; "
                f"example cells={names}"
            )

        order = {str(c["name"]): i for i, c in enumerate(cells)}
        final_runs["__cell_order"] = final_runs["cell_name"].astype(str).map(order)
        final_runs.sort_values(["__cell_order", "task_seed"], inplace=True, kind="stable")
        final_runs.drop(columns=["__cell_order"], inplace=True)
        final_runs.reset_index(drop=True, inplace=True)

        hist_parts = [
            df for df in (recovered_history, computed_history)
            if df is not None and len(df)
        ]
        final_history: Optional[pd.DataFrame] = None
        if hist_parts:
            final_history = pd.concat(hist_parts, ignore_index=True, sort=False)
            key_cols = [
                c for c in ("cell_name", "task_seed", "step", "iteration", "record_step")
                if c in final_history.columns
            ]
            if len(key_cols) >= 2:
                final_history.drop_duplicates(key_cols, keep="first", inplace=True)
            final_history["__cell_order"] = final_history["cell_name"].astype(str).map(order)
            sort_cols = ["__cell_order", "task_seed"]
            for extra in ("step", "iteration", "record_step"):
                if extra in final_history.columns:
                    sort_cols.append(extra)
                    break
            final_history.sort_values(sort_cols, inplace=True, kind="stable")
            final_history.drop(columns=["__cell_order"], inplace=True)
            final_history.reset_index(drop=True, inplace=True)

        if len(final_runs) != expected_rows:
            raise RuntimeError(
                f"Recovered+computed shard has {len(final_runs)} rows, expected {expected_rows}"
            )
        keys = final_runs[["cell_name", "task_seed"]].copy()
        if keys["cell_name"].nunique() != len(cells):
            raise RuntimeError(
                f"Recovered+computed shard has {keys['cell_name'].nunique()} cells, "
                f"expected {len(cells)}"
            )
        seeds = sorted(pd.to_numeric(keys["task_seed"]).astype(int).unique().tolist())
        if seeds != expected_seeds:
            raise RuntimeError(
                f"Recovered+computed shard seeds {seeds[:3]}... do not match "
                f"{expected_seeds[:3]}..."
            )

        composite_manifest = {
            "v14_1_composite_recovery_shard": True,
            "created_unix": time.time(),
            "block": block.name,
            "seed_family": block.seed_family,
            "seed_start": shard_start,
            "seed_end": shard_start + shard_n - 1,
            "n_task_seeds": shard_n,
            "n_cells": len(cells),
            "expected_rows": expected_rows,
            "observed_rows": int(len(final_runs)),
            "recovered_cells": len(recovered_names),
            "computed_cells_this_attempt": len(missing_cells),
            "recovery_sources": recovery_sources,
            "engine": str(engine),
            "dtype": dtype,
            "train_steps": BASE_STEPS,
            "record_every": 10,
            "sampled_trajectories_per_step": 256,
            "paired_optimization_rng": True,
            "device_for_missing_cells": device if missing_cells else None,
            "attempt_dir": str(attempt_dir) if attempt_dir else None,
            "attempt_engine_manifest": attempt_manifest,
            "full_spec_sha256": _sha256(spec_path),
        }
        _atomic_finalize_recovered_shard(
            shard_dir=shard_dir,
            full_spec_path=spec_path,
            runs=final_runs,
            history=final_history,
            manifest=composite_manifest,
        )

        ok, reason = _validate_shard(
            shard_dir, expected_rows, shard_start, shard_n, len(cells)
        )
        if not ok:
            raise RuntimeError(
                f"Shard validation failed after recovery finalization: {shard_dir}: {reason}"
            )
        print(
            f"[{block.name}] GPU {device}: completed shard {shard_index}/{n_shards} "
            f"({len(recovered_names)} cells reused, {len(missing_cells)} computed)",
            flush=True,
        )
        return {
            "status": "complete", "device": device,
            "shard_index": shard_index, "shard_dir": str(shard_dir),
            "recovered_cells": len(recovered_names),
            "computed_cells": len(missing_cells),
        }
    finally:
        _release_shard_claim(claim_path)


def execute_block_resumable(
    block: ExtensionBlock,
    output_root: Path,
    *,
    engine: Path,
    device: str,
    devices: Optional[Sequence[str]],
    dtype: str,
    seed_shard_size: int,
    smoke: bool,
    overwrite: bool,
    repair_incomplete: bool,
    recover_partial: bool,
    shard_order: str = "forward",
    shard_indices: Optional[str] = None,
    max_shards: Optional[int] = None,
    defer_finalize: bool = False,
) -> None:
    """Run a block with resumable seed shards, optionally across many GPUs.

    Scientific semantics are unchanged: each engine subprocess receives the
    same complete cell specification and a disjoint seed shard.  Parallelism is
    therefore only across independent task-seed shards.

    ``--shard-indices`` and ``--shard-order reverse`` are particularly useful
    for filling tail shards while an older forward-only runner is still active.
    Because that legacy runner does not understand the new claim files, do not
    target a shard that the legacy process is already computing.
    """
    cells = smoke_subset(block) if smoke else list(block.cells)
    n_seeds = min(4, block.n_seeds) if smoke else block.n_seeds
    seed_start = block.seed_start
    block_dir = output_root / block.name
    if overwrite and block_dir.exists():
        shutil.rmtree(block_dir)
    block_dir.mkdir(parents=True, exist_ok=True)
    spec_path = output_root / "specs_v14_1" / f"{block.name}.json"
    write_block_spec(spec_path, block, cells)
    shutil.copy2(spec_path, block_dir / "experiment_spec.json")

    shard_size = max(1, int(seed_shard_size))
    shards: List[Tuple[int, int]] = []
    for s in range(seed_start, seed_start + n_seeds, shard_size):
        shards.append((s, min(shard_size, seed_start + n_seeds - s)))
    indexed_shards = list(enumerate(shards, start=1))

    if recover_partial:
        _migrate_overlapping_legacy_shards(
            block_dir=block_dir,
            cells=cells,
            indexed_shards=indexed_shards,
            full_spec_path=spec_path,
        )

    explicit = _parse_shard_indices(shard_indices, len(shards))
    if explicit is not None:
        by_index = dict(indexed_shards)
        schedule = [(i, by_index[i]) for i in explicit]
    else:
        schedule = list(indexed_shards)
        if shard_order == "reverse":
            schedule.reverse()
        elif shard_order != "forward":
            raise ValueError(f"unknown shard order: {shard_order}")

    # Only pending shards count toward --max-shards.  This makes repeated tail
    # invocations naturally advance inward rather than wasting the quota on
    # shards that a previous process already completed.
    pending: List[Tuple[int, Tuple[int, int]]] = []
    for idx_, (shard_start, shard_n) in schedule:
        shard_dir = block_dir / "shards" / f"seeds_{shard_start}_{shard_start + shard_n - 1}"
        ok, _ = _validate_shard(
            shard_dir, len(cells) * shard_n, shard_start, shard_n, len(cells)
        )
        if not ok:
            pending.append((idx_, (shard_start, shard_n)))
    if max_shards is not None:
        if max_shards < 1:
            raise ValueError("--max-shards must be >= 1")
        pending = pending[:max_shards]

    t0 = time.time()
    device_list = [d.strip() for d in (devices or []) if d.strip()]
    if not device_list:
        # Sequential execution uses the same cell-aware recovery machinery as
        # parallel execution; only the device assignment differs.
        for idx_, (shard_start, shard_n) in pending:
            result = _execute_one_shard(
                block=block,
                cells=cells,
                block_dir=block_dir,
                spec_path=spec_path,
                engine=engine,
                dtype=dtype,
                device=device,
                shard_index=idx_,
                n_shards=len(shards),
                shard_start=shard_start,
                shard_n=shard_n,
                repair_incomplete=repair_incomplete,
                recover_partial=recover_partial,
            )
            print(
                f"[{block.name}] {device}: shard {result['shard_index']} "
                f"status={result['status']}", flush=True,
            )

    else:
        if device_list == ["auto"]:
            raise ValueError(
                "--devices auto is ambiguous for parallel execution; provide explicit "
                "devices such as --devices cuda:0,cuda:1"
            )
        if not pending:
            print(f"[{block.name}] no selected incomplete shards remain.", flush=True)
        else:
            print(
                f"[{block.name}] parallel shard execution on {len(device_list)} device(s): "
                f"{', '.join(device_list)}; selected shards="
                f"{[i for i, _ in pending]}",
                flush=True,
            )
            # Dynamic shared queue.  A fast GPU takes another shard immediately
            # after finishing its current one, so heterogeneous devices naturally
            # receive different amounts of work.  Repeated device strings create
            # multiple worker slots on the same physical GPU intentionally.
            work_queue: Queue = Queue()
            for item in pending:
                work_queue.put(item)

            def worker(slot: int, dev: str):
                results: List[Dict[str, object]] = []
                slot_name = f"worker{slot}:{dev}"
                while True:
                    try:
                        idx_, (shard_start, shard_n) = work_queue.get_nowait()
                    except Empty:
                        break
                    try:
                        print(
                            f"[{block.name}] {slot_name}: claiming shard "
                            f"{idx_}/{len(shards)} seeds={shard_start}:{shard_start+shard_n-1}",
                            flush=True,
                        )
                        result = _execute_one_shard(
                            block=block,
                            cells=cells,
                            block_dir=block_dir,
                            spec_path=spec_path,
                            engine=engine,
                            dtype=dtype,
                            device=dev,
                            shard_index=idx_,
                            n_shards=len(shards),
                            shard_start=shard_start,
                            shard_n=shard_n,
                            repair_incomplete=repair_incomplete,
                            recover_partial=recover_partial,
                        )
                        results.append(result)
                        print(
                            f"[{block.name}] {slot_name}: shard {idx_} "
                            f"status={result['status']}; looking for more work",
                            flush=True,
                        )
                    finally:
                        work_queue.task_done()
                return results

            failures: List[BaseException] = []
            with ThreadPoolExecutor(max_workers=len(device_list)) as pool:
                futures = {
                    pool.submit(worker, slot, dev): (slot, dev)
                    for slot, dev in enumerate(device_list, start=1)
                }
                for fut in as_completed(futures):
                    slot, dev = futures[fut]
                    try:
                        fut.result()
                    except BaseException as exc:
                        failures.append(exc)
                        print(
                            f"[{block.name}] worker{slot}:{dev} FAILED: {exc}",
                            flush=True,
                        )
            if failures:
                raise RuntimeError(
                    f"{len(failures)} parallel worker(s) failed; first error: {failures[0]}"
                )

    # Validate the complete canonical shard set.  Partial/tail runs intentionally
    # leave block-level runs.csv untouched unless every shard is complete.
    complete_dirs: List[Path] = []
    incomplete: List[Tuple[int, Path, str]] = []
    for idx_, (shard_start, shard_n) in indexed_shards:
        d = block_dir / "shards" / f"seeds_{shard_start}_{shard_start + shard_n - 1}"
        ok, reason = _validate_shard(
            d, len(cells) * shard_n, shard_start, shard_n, len(cells)
        )
        if ok:
            complete_dirs.append(d)
        else:
            incomplete.append((idx_, d, reason))

    if incomplete:
        print(
            f"[{block.name}] PARTIAL: {len(complete_dirs)}/{len(shards)} shards complete. "
            f"Block-level finalization deferred. Remaining shard indices: "
            f"{[i for i, _, _ in incomplete]}",
            flush=True,
        )
        return

    if defer_finalize:
        print(
            f"[{block.name}] all shards are complete, but --defer-finalize was requested. "
            f"Run this driver again without --defer-finalize (or let the existing "
            f"legacy runner finish) to rebuild canonical outputs.",
            flush=True,
        )
        return

    _rebuild_block_outputs(block_dir, complete_dirs)
    expected_total = len(cells) * n_seeds
    observed_total = _csv_row_count(block_dir / "runs.csv")
    if observed_total != expected_total:
        raise RuntimeError(
            f"Canonical runs.csv has {observed_total} rows, expected {expected_total}"
        )

    manifest = {
        "v14_1_block": block.name,
        "seed_family": block.seed_family,
        "question": block.question,
        "prediction": block.prediction,
        "n_cells": len(cells),
        "n_task_seeds": n_seeds,
        "seed_start": seed_start,
        "seed_end": seed_start + n_seeds - 1,
        "seed_shard_size": shard_size,
        "n_shards": len(shards),
        "expected_rows": expected_total,
        "observed_rows": observed_total,
        "smoke": smoke,
        "elapsed_seconds": time.time() - t0,
        "exploration_sha256": _sha256(Path(__file__)),
        "engine": str(engine),
        "parallel_devices": device_list,
        "shard_order": shard_order,
    }
    (block_dir / "v14_1_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"[{block.name}] COMPLETE: {observed_total}/{expected_total} rows", flush=True)

def _load_extension_block(block: ExtensionBlock, output_root: Path, usecols: Optional[Sequence[str]] = None) -> pd.DataFrame:
    path = output_root / block.name / "runs.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    if usecols is None:
        out = pd.read_csv(path)
    else:
        header = pd.read_csv(path, nrows=0).columns
        cols = [c for c in usecols if c in header]
        missing = [c for c in usecols if c not in header]
        if missing:
            raise ValueError(f"{block.name} missing required columns: {missing}")
        out = pd.read_csv(path, usecols=cols)
    out["registry_block"] = block.name
    return out


def _find_small_complete_shard(block: ExtensionBlock, output_root: Path) -> Optional[Path]:
    """Return the smallest complete seed-shard runs.csv for schema/config inference.

    A complete shard contains every scientific cell but only a subset of task
    seeds, so it is enough to infer the metric/configuration schema without
    materializing the much larger canonical block runs.csv.
    """
    root = output_root / block.name / "shards"
    if not root.exists():
        return None
    candidates: List[Tuple[int, Path]] = []
    for d in root.glob("seeds_*_*"):
        if not d.is_dir() or not (d / "runs.csv").exists() or not (d / "manifest.json").exists():
            continue
        rows = _csv_row_count(d / "runs.csv")
        if rows <= 0 or rows % len(block.cells) != 0:
            continue
        n = rows // len(block.cells)
        rng = _seed_range_from_name(d)
        if rng is None or (rng[1] - rng[0] + 1) != n:
            continue
        ok, _ = _validate_shard(d, rows, rng[0], n, len(block.cells))
        if ok:
            candidates.append((rows, d / "runs.csv"))
    return min(candidates, key=lambda x: x[0])[1] if candidates else None


def _metric_summary_batched(
    path: Path,
    metric_cols: Sequence[str],
    *,
    batch_size: int = 12,
) -> pd.DataFrame:
    """Exact per-cell mean/median/std/count without loading the rich CSV at once.

    The original V14.1 implementation called ``pd.read_csv(path)`` on the full
    E84 table (>1.3M rows with every diagnostic column).  This version makes
    several narrow passes over the CSV.  Statistics are exactly the same as
    ``v14.cell_summary_table`` because every metric is still aggregated over all
    rows for each cell; only the columns are processed in batches.
    """
    metric_cols = list(metric_cols)
    if not metric_cols:
        return pd.DataFrame(columns=["cell_name"])
    pieces: List[pd.DataFrame] = []
    total_batches = math.ceil(len(metric_cols) / batch_size)
    for bi, lo in enumerate(range(0, len(metric_cols), batch_size), start=1):
        batch = metric_cols[lo:lo + batch_size]
        print(
            f"[aggregate] metric batch {bi}/{total_batches}: "
            f"{batch[0]} .. {batch[-1]}",
            flush=True,
        )
        d = pd.read_csv(path, usecols=["cell_name", *batch])
        g = d.groupby("cell_name", sort=False, dropna=False)[batch]
        part = g.agg(["mean", "median", "std", "count"]).reset_index()
        part.columns = [
            "cell_name" if a == "cell_name" else f"{a}__{b}"
            for a, b in part.columns.to_flat_index()
        ]
        pieces.append(part)
        del d, g, part

    out = pieces[0]
    for part in pieces[1:]:
        out = out.merge(part, on="cell_name", how="outer", validate="one_to_one")

    # Match the canonical V14 column ordering: all means, then medians, stds,
    # and counts, with metric columns in infer_metric_columns() order.
    ordered = ["cell_name"] + [
        f"{m}__{stat}"
        for stat in ("mean", "median", "std", "count")
        for m in metric_cols
    ]
    return out[ordered]


def _condition_id_summary(path: Path, *, chunksize: int = 250_000) -> Optional[pd.DataFrame]:
    header = pd.read_csv(path, nrows=0).columns
    if "condition_id" not in header:
        return None
    pieces: List[pd.DataFrame] = []
    for chunk in pd.read_csv(
        path, usecols=["cell_name", "condition_id"], chunksize=chunksize
    ):
        pieces.append(chunk.drop_duplicates(["cell_name", "condition_id"]))
    if not pieces:
        return None
    unique = pd.concat(pieces, ignore_index=True).drop_duplicates(
        ["cell_name", "condition_id"]
    )
    out = (
        unique.groupby("cell_name", sort=False, dropna=False)["condition_id"]
        .nunique(dropna=False)
        .rename("condition_id_nunique")
        .reset_index()
    )
    out["recovered_or_multibatch"] = out["condition_id_nunique"].fillna(0).astype(int) > 1
    return out


def _cell_tables(v14, blocks: Sequence[ExtensionBlock], output_root: Path, agg: Path) -> pd.DataFrame:
    """Memory-bounded exact canonical cell tables for the large V14.1 blocks."""
    configs: List[pd.DataFrame] = []
    summaries: List[pd.DataFrame] = []
    completion: List[Dict[str, object]] = []
    metric_batch = max(1, int(os.environ.get("V14_AGG_METRIC_BATCH", "12")))

    for b in blocks:
        path = output_root / b.name / "runs.csv"
        if not path.exists():
            completion.append({
                "block": b.name, "status": "missing",
                "expected_rows": b.n_runs, "observed_rows": 0,
                "expected_cells": len(b.cells), "observed_cells": 0,
            })
            continue

        observed = _csv_row_count(path)
        print(
            f"[aggregate] {b.name}: canonical rows={observed:,}; "
            f"expected={b.n_runs:,}",
            flush=True,
        )

        schema_path = _find_small_complete_shard(b, output_root)
        if schema_path is None:
            # Conservative fallback: enough rows to include several seeds for
            # every cell, but still far smaller than the complete block.
            schema_rows = min(observed, max(len(b.cells) * 4, 50_000))
            print(
                f"[aggregate] {b.name}: no complete shard found for schema; "
                f"sampling {schema_rows:,} canonical rows",
                flush=True,
            )
            schema_df = pd.read_csv(path, nrows=schema_rows)
        else:
            print(
                f"[aggregate] {b.name}: inferring schema/config from {schema_path}",
                flush=True,
            )
            schema_df = pd.read_csv(schema_path)

        metrics = v14.infer_metric_columns(schema_df)
        print(
            f"[aggregate] {b.name}: inferred {len(metrics)} metric columns; "
            f"building configuration table",
            flush=True,
        )
        cfg = v14.cell_configuration_table(schema_df, metrics)
        if len(cfg) != len(b.cells):
            raise RuntimeError(
                f"Schema shard/sample contains {len(cfg)} cells, expected {len(b.cells)}. "
                "Use a complete current seed shard for scalable aggregation."
            )

        # Recompute provenance over all seed chunks.  The scientific config is
        # seed-invariant, but condition_id legitimately changes with seed batch.
        prov = _condition_id_summary(path)
        cfg = cfg.drop(
            columns=["condition_id_nunique", "recovered_or_multibatch"],
            errors="ignore",
        )
        if prov is not None:
            cfg = cfg.merge(prov, on="cell_name", how="left", validate="one_to_one")

        del schema_df

        print(
            f"[aggregate] {b.name}: exact metric summaries in batches of {metric_batch}",
            flush=True,
        )
        sm = _metric_summary_batched(path, metrics, batch_size=metric_batch)

        cfg["registry_block"] = b.name
        sm["registry_block"] = b.name
        configs.append(cfg)
        summaries.append(sm)
        completion.append({
            "block": b.name,
            "status": "complete" if observed == b.n_runs else "row_count_mismatch",
            "expected_rows": b.n_runs,
            "observed_rows": observed,
            "expected_cells": len(b.cells),
            "observed_cells": int(len(cfg)),
        })

    if configs:
        pd.concat(configs, ignore_index=True, sort=False).to_csv(
            agg / "cell_configurations.csv.gz", index=False, compression="gzip"
        )
    if summaries:
        pd.concat(summaries, ignore_index=True, sort=False).to_csv(
            agg / "cell_summary.csv.gz", index=False, compression="gzip"
        )
    status = pd.DataFrame(completion)
    status.to_csv(agg / "completion_status.csv", index=False)
    return status

def _stability_columns(v14) -> List[str]:
    """Only columns actually required by the V14.1 specialized analyses.

    The old implementation pulled every public/configuration column into a
    >1.5M-row dataframe even though most were fixed metadata.  Keeping only the
    analysis keys plus outcomes cuts memory substantially without changing any
    scientific calculation.
    """
    wanted_outcomes = [
        "new_match_gain", "old_match_forgetting", "new_match_drawdown_from_peak",
        "old_match_max_drawdown", "context_utility_drawdown_from_peak",
        "final_context_context_teacher_utility_student_occupancy",
        "initial_context_geometry_jsd_student_occupancy",
        "initial_context_geometry_forward_reverse_logit_grad_cosine_student_occupancy",
        "initial_context_geometry_log_forward_reverse_logit_grad_norm_ratio_student_occupancy",
    ]
    return [
        "cell_name", "task_seed",
        "v14_core_feature_overlap",
        "v14_core_initial_policy_concentration",
        "v14_core_context_strength",
        "kl_direction", "rollout_source", "ema_alpha", "learning_rate",
        *wanted_outcomes,
    ]


def _summary_ci(df: pd.DataFrame, group_cols: Sequence[str], value_cols: Sequence[str]) -> pd.DataFrame:
    records: List[Dict[str, object]] = []
    for key, sub in df.groupby(list(group_cols), dropna=False, sort=False):
        rec = {c: v for c, v in zip(group_cols, key if isinstance(key, tuple) else (key,))}
        for col in value_cols:
            vals = pd.to_numeric(sub[col], errors="coerce").to_numpy(float)
            vals = vals[np.isfinite(vals)]
            n = len(vals)
            mean = float(np.mean(vals)) if n else float("nan")
            sd = float(np.std(vals, ddof=1)) if n > 1 else float("nan")
            se = sd / math.sqrt(n) if n > 1 else float("nan")
            rec[f"{col}__n"] = n
            rec[f"{col}__mean"] = mean
            rec[f"{col}__std"] = sd
            rec[f"{col}__ci95_low"] = mean - 1.96 * se if np.isfinite(se) else float("nan")
            rec[f"{col}__ci95_high"] = mean + 1.96 * se if np.isfinite(se) else float("nan")
        records.append(rec)
    return pd.DataFrame(records)


def _paired_kl_contrasts(df: pd.DataFrame) -> pd.DataFrame:
    idx = ["task_seed", "rho_phi", "lambda", "kappa", "ema_alpha", "learning_rate"]
    metrics = ["new_match_gain", "old_match_forgetting"]
    optional = [
        "new_match_drawdown_from_peak",
        "old_match_max_drawdown",
    ]
    metrics += [c for c in optional if c in df.columns]

    f = df[df["kl_direction"] == "forward"][idx + metrics].copy()
    r = df[df["kl_direction"] == "reverse"][idx + metrics].copy()
    paired = f.merge(
        r, on=idx, how="inner", validate="one_to_one",
        suffixes=("__forward", "__reverse"),
    )
    out = paired[idx].copy()
    out["forward_minus_reverse_acquisition"] = (
        paired["new_match_gain__forward"] - paired["new_match_gain__reverse"]
    )
    out["forward_minus_reverse_forgetting"] = (
        paired["old_match_forgetting__forward"] - paired["old_match_forgetting__reverse"]
    )
    out["forward_minus_reverse_retention"] = -out["forward_minus_reverse_forgetting"]
    if "new_match_drawdown_from_peak" in metrics:
        out["forward_minus_reverse_acquisition_drawdown"] = (
            paired["new_match_drawdown_from_peak__forward"]
            - paired["new_match_drawdown_from_peak__reverse"]
        )
    if "old_match_max_drawdown" in metrics:
        out["forward_minus_reverse_old_max_drawdown"] = (
            paired["old_match_max_drawdown__forward"]
            - paired["old_match_max_drawdown__reverse"]
        )
    return out


def _route_points_vectorized(df: pd.DataFrame, cf: pd.DataFrame) -> pd.DataFrame:
    gcols = ["task_seed", "rho_phi", "lambda", "kappa", "learning_rate"]
    vals = ["new_match_gain", "old_match_forgetting"]
    frozen = df[np.isclose(df["ema_alpha"].astype(float), 0.0, atol=1e-12)]

    def side(src: pd.DataFrame, kl: str, prefix: str, alpha_from_selected: bool) -> pd.DataFrame:
        cols = gcols + vals + (["selected_alpha"] if alpha_from_selected else [])
        x = src[src["kl_direction"] == kl][cols].copy()
        if x.duplicated(gcols).any():
            dup = int(x.duplicated(gcols, keep=False).sum())
            raise RuntimeError(f"Route table has {dup} duplicate rows for {prefix}")
        ren = {
            "new_match_gain": f"{prefix}_acquisition",
            "old_match_forgetting": f"{prefix}_forgetting",
        }
        if alpha_from_selected:
            ren["selected_alpha"] = f"{prefix}_alpha"
        return x.rename(columns=ren)

    rf = side(frozen, "reverse", "reverse_frozen", False)
    ff = side(frozen, "forward", "forward_frozen", False)
    ro = side(cf, "reverse", "reverse_optimal", True)
    fo = side(cf, "forward", "forward_optimal", True)
    wide = rf.merge(ff, on=gcols, validate="one_to_one")
    wide = wide.merge(ro, on=gcols, validate="one_to_one")
    wide = wide.merge(fo, on=gcols, validate="one_to_one")

    base_acq = wide["reverse_frozen_acquisition"]
    base_fgt = wide["reverse_frozen_forgetting"]
    frames: List[pd.DataFrame] = []
    for label, alpha_col in (
        ("reverse_frozen", None),
        ("forward_frozen", None),
        ("reverse_optimal", "reverse_optimal_alpha"),
        ("forward_optimal", "forward_optimal_alpha"),
    ):
        p = wide[gcols].copy()
        p["point"] = label
        p["selected_alpha"] = 0.0 if alpha_col is None else wide[alpha_col].astype(float)
        p["acquisition"] = wide[f"{label}_acquisition"].astype(float)
        p["forgetting"] = wide[f"{label}_forgetting"].astype(float)
        p["retention"] = -p["forgetting"]
        p["additional_acquisition_vs_reverse_frozen"] = p["acquisition"] - base_acq
        p["additional_forgetting_vs_reverse_frozen"] = p["forgetting"] - base_fgt
        p["additional_retention_vs_reverse_frozen"] = -p["additional_forgetting_vs_reverse_frozen"]
        frames.append(p)
    return pd.concat(frames, ignore_index=True)


def aggregate_stability(v14, blocks: Sequence[ExtensionBlock], output_root: Path, agg: Path) -> None:
    frames: List[pd.DataFrame] = []
    cols = _stability_columns(v14)
    for b in blocks:
        path = output_root / b.name / "runs.csv"
        if not path.exists():
            continue
        print(f"[aggregate] {b.name}: loading narrow stability columns", flush=True)
        d = v14.short_names(_load_extension_block(b, output_root, cols))
        # Categorical conversion happens after parsing and substantially shrinks
        # retained memory before the two blocks are concatenated.
        for c in ("cell_name", "kl_direction", "rollout_source", "registry_block"):
            if c in d.columns:
                d[c] = d[c].astype("category")
        frames.append(d)
    if not frames:
        return
    print("[aggregate] concatenating narrow E83/E84 stability frame", flush=True)
    df = pd.concat(frames, ignore_index=True, sort=False)
    df["kappa"] = df["c"]

    metric_cols = [
        c for c in (
            "new_match_gain", "old_match_forgetting", "new_match_drawdown_from_peak",
            "old_match_max_drawdown", "context_utility_drawdown_from_peak",
            "final_context_context_teacher_utility_student_occupancy",
            "initial_context_geometry_jsd_student_occupancy",
            "initial_context_geometry_forward_reverse_logit_grad_cosine_student_occupancy",
            "initial_context_geometry_log_forward_reverse_logit_grad_norm_ratio_student_occupancy",
        ) if c in df.columns
    ]
    curve_group = [
        "registry_block", "rho_phi", "lambda", "kappa", "rollout_source",
        "ema_alpha", "learning_rate", "kl_direction",
    ]
    print("[aggregate] stability surface curves", flush=True)
    curves = df.groupby(curve_group, dropna=False, observed=True)[metric_cols].agg(["mean", "std", "count"])
    curves.columns = [f"{a}__{b}" for a, b in curves.columns]
    curves.reset_index().to_csv(agg / "kl_stability_surface_curves.csv", index=False)

    print("[aggregate] vectorized paired forward/reverse contrasts", flush=True)
    contrasts = _paired_kl_contrasts(df)
    contrasts.to_csv(agg / "kl_stability_paired_contrasts_seed.csv.gz", index=False, compression="gzip")

    contrast_metrics = [c for c in contrasts.columns if c.startswith("forward_minus_reverse_")]
    contrast_summary = _summary_ci(
        contrasts,
        ["rho_phi", "lambda", "kappa", "ema_alpha", "learning_rate"],
        contrast_metrics,
    )
    if not contrast_summary.empty:
        contrast_summary["forward_wins_acquisition"] = contrast_summary["forward_minus_reverse_acquisition__mean"] > 0
        contrast_summary["forward_wins_retention"] = contrast_summary["forward_minus_reverse_forgetting__mean"] < 0
        contrast_summary["forward_wins_both"] = contrast_summary["forward_wins_acquisition"] & contrast_summary["forward_wins_retention"]
        contrast_summary["confident_forward_wins_both"] = (
            (contrast_summary["forward_minus_reverse_acquisition__ci95_low"] > 0)
            & (contrast_summary["forward_minus_reverse_forgetting__ci95_high"] < 0)
        )
    contrast_summary.to_csv(agg / "kl_stability_paired_summary.csv", index=False)
    flip_cols = [
        "rho_phi", "lambda", "kappa", "ema_alpha", "learning_rate",
        "forward_minus_reverse_acquisition__mean",
        "forward_minus_reverse_acquisition__ci95_low",
        "forward_minus_reverse_acquisition__ci95_high",
        "forward_minus_reverse_forgetting__mean",
        "forward_minus_reverse_forgetting__ci95_low",
        "forward_minus_reverse_forgetting__ci95_high",
        "forward_wins_acquisition", "forward_wins_retention",
        "forward_wins_both", "confident_forward_wins_both",
    ]
    contrast_summary[[c for c in flip_cols if c in contrast_summary]].to_csv(
        agg / "kl_stability_flip_map.csv", index=False
    )

    print("[aggregate] cross-fitted alpha selection", flush=True)
    cf = v14.crossfit_alpha_rows(
        df,
        ["rho_phi", "lambda", "kappa", "kl_direction", "learning_rate"],
        objective="new_match_gain",
    )
    cf.to_csv(agg / "kl_stability_crossfit_optimal_alpha_rows.csv.gz", index=False, compression="gzip")

    print("[aggregate] vectorized route points", flush=True)
    route = _route_points_vectorized(df, cf)
    route.to_csv(agg / "kl_stability_route_points_seed.csv.gz", index=False, compression="gzip")
    if not route.empty:
        route_summary = _summary_ci(
            route,
            ["rho_phi", "lambda", "kappa", "learning_rate", "point"],
            ["acquisition", "forgetting", "retention", "additional_acquisition_vs_reverse_frozen", "additional_retention_vs_reverse_frozen", "selected_alpha"],
        )
    else:
        route_summary = pd.DataFrame()
    route_summary.to_csv(agg / "kl_stability_route_points_summary.csv", index=False)

    if not contrast_summary.empty:
        cand = contrast_summary[contrast_summary["forward_wins_both"]].copy()
        cand["acquisition_advantage_pp"] = 100.0 * cand["forward_minus_reverse_acquisition__mean"]
        cand["retention_advantage_pp"] = -100.0 * cand["forward_minus_reverse_forgetting__mean"]
        cand = cand.sort_values(
            ["confident_forward_wins_both", "retention_advantage_pp", "acquisition_advantage_pp"],
            ascending=[False, False, False],
        )
        cand.to_csv(agg / "kl_stability_forward_wins_both_candidates.csv", index=False)

def aggregate(v14, blocks: Sequence[ExtensionBlock], output_root: Path, agg: Path) -> None:
    agg.mkdir(parents=True, exist_ok=True)
    print(f"[aggregate] output directory: {agg}", flush=True)
    print("[aggregate] stage 1/2: canonical cell configuration/summary tables", flush=True)
    status = _cell_tables(v14, blocks, output_root, agg)
    print("[aggregate] stage 2/2: specialized stability analyses", flush=True)
    aggregate_stability(v14, blocks, output_root, agg)
    manifest = {
        "version": "14.1",
        "registry": registry_summary(blocks),
        "seed_pairing_note": "E83 and E84 intentionally share the same 96 task seeds to enable paired rho_phi comparisons.",
        "aggregation_note": "This directory contains only V14.1 raw evidence-derived aggregates. V14.0 is merged only by --update-global-aggregate.",
        "global_update_note": "The global updater preserves original specialized V14.0 CSVs and adds an extended KL/LR/rho_phi surface instead of silently changing their schema.",
        "exploration_sha256": _sha256(Path(__file__)),
        "complete": bool(len(status) and (status["status"] == "complete").all()),
    }
    (agg / "aggregation_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    lines = ["# V14.1 aggregate status", "", json.dumps(registry_summary(blocks), indent=2), ""]
    for _, r in status.iterrows():
        lines.append(f"- {r['block']}: {r['status']} ({r['observed_rows']}/{r['expected_rows']} rows)")
    (agg / "aggregate_status.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _merge_keyed_csv(existing: Path, addition: Path, key_cols: Sequence[str], compression: Optional[str] = None) -> None:
    if not addition.exists():
        return
    add = pd.read_csv(addition)
    if existing.exists():
        old = pd.read_csv(existing)
        both = pd.concat([old, add], ignore_index=True, sort=False)
    else:
        both = add
    keys = [c for c in key_cols if c in both.columns]
    if not keys:
        raise ValueError(f"No merge keys available for {existing.name}")
    both = both.drop_duplicates(keys, keep="last")
    both.to_csv(existing, index=False, compression=compression)


def update_global_aggregate(v14_1_agg: Path, global_agg: Path) -> None:
    if not v14_1_agg.exists():
        raise FileNotFoundError(v14_1_agg)
    manifest_path = v14_1_agg / "aggregation_manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(
            f"Refusing global update: {manifest_path} is missing. "
            "Run --aggregate successfully first."
        )
    try:
        aggregate_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Refusing global update: invalid {manifest_path}: {exc}") from exc
    if not bool(aggregate_manifest.get("complete", False)):
        raise RuntimeError(
            "Refusing global update: the V14.1 aggregation manifest is not complete. "
            "This prevents stale/partial aggregate_v14_1 files from being merged globally."
        )
    global_agg.mkdir(parents=True, exist_ok=True)

    # Merge canonical cell state idempotently.
    _merge_keyed_csv(
        global_agg / "cell_configurations.csv.gz",
        v14_1_agg / "cell_configurations.csv.gz",
        ["registry_block", "cell_name"], compression="gzip",
    )
    _merge_keyed_csv(
        global_agg / "cell_summary.csv.gz",
        v14_1_agg / "cell_summary.csv.gz",
        ["registry_block", "cell_name"], compression="gzip",
    )
    _merge_keyed_csv(
        global_agg / "completion_status.csv",
        v14_1_agg / "completion_status.csv",
        ["block"], compression=None,
    )

    # Copy V14.1 specialized outputs using unique names; these are already
    # global-analysis ready and do not change any V14.0 schema in place.
    specialized = [
        "kl_stability_surface_curves.csv",
        "kl_stability_paired_contrasts_seed.csv.gz",
        "kl_stability_paired_summary.csv",
        "kl_stability_flip_map.csv",
        "kl_stability_crossfit_optimal_alpha_rows.csv.gz",
        "kl_stability_route_points_seed.csv.gz",
        "kl_stability_route_points_summary.csv",
        "kl_stability_forward_wins_both_candidates.csv",
    ]
    for name in specialized:
        src = v14_1_agg / name
        if src.exists():
            shutil.copy2(src, global_agg / name)

    # Build a schema-explicit extended curve table.  V14.1 contains only cells
    # absent from V14.0, so the old E72 table supplies the already-existing
    # rho_phi=.5 points while V14.1 contributes only new LR/alpha combinations.
    frames: List[pd.DataFrame] = []
    old = global_agg / "kl_lr_alpha_curves.csv"
    if old.exists():
        d = pd.read_csv(old)
        d["rho_phi"] = 0.5
        d["kappa"] = d["c"] if "c" in d else np.nan
        d["source_block"] = "V14_E72_kl_matched_learning"
        frames.append(d)
    new = v14_1_agg / "kl_stability_surface_curves.csv"
    if new.exists():
        d = pd.read_csv(new)
        d["source_block"] = d["registry_block"]
        frames.append(d)
    if frames:
        ext = pd.concat(frames, ignore_index=True, sort=False)
        ext.to_csv(global_agg / "kl_lr_alpha_curves_extended.csv", index=False)

    update_manifest = {
        "updated_unix": time.time(),
        "v14_1_aggregate": str(v14_1_agg),
        "global_aggregate": str(global_agg),
        "policy": "Original V14.0 specialized CSVs are preserved; canonical cell tables are merged and V14.1-specific/extended CSVs are added.",
        "v14_1_manifest_sha256": _sha256(v14_1_agg / "aggregation_manifest.json") if (v14_1_agg / "aggregation_manifest.json").exists() else None,
    }
    (global_agg / "v14_1_global_update_manifest.json").write_text(
        json.dumps(update_manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

    # Add/replace a clearly delimited status section without disturbing the old
    # report generated by V14.0.
    status_path = global_agg / "aggregate_status.md"
    text = status_path.read_text(encoding="utf-8") if status_path.exists() else "# V14 aggregate status\n"
    begin = "\n<!-- V14.1 EXTENSION BEGIN -->\n"
    end = "<!-- V14.1 EXTENSION END -->\n"
    if begin in text and end in text:
        text = text.split(begin, 1)[0] + text.split(end, 1)[1]
    ext_status = (v14_1_agg / "aggregate_status.md").read_text(encoding="utf-8") if (v14_1_agg / "aggregate_status.md").exists() else "V14.1 aggregate present.\n"
    text = text.rstrip() + begin + ext_status.rstrip() + "\n" + end
    status_path.write_text(text, encoding="utf-8")


def run_aggregation_self_test(v14, blocks: Sequence[ExtensionBlock]) -> Dict[str, object]:
    """Synthetic end-to-end regression test for aggregation and global merge."""
    with tempfile.TemporaryDirectory(prefix="v14_1_agg_test_") as td:
        root = Path(td)
        agg = root / "aggregate_v14_1"
        seeds = (1, 2, 3, 4)
        # Build tiny synthetic raw blocks with paired KL/alpha/LR/rho motifs.
        for b in blocks:
            rows: List[Dict[str, object]] = []
            rhos = (0.5,) if b.name.endswith("E83_lr_stability") else (RHO_PHI_STABILITY[0], RHO_PHI_STABILITY[-1])
            for seed in seeds:
                for rho in rhos:
                    for lr in (0.001, 0.004):
                        for a in (0.0, 0.0025):
                            for kl in ("forward", "reverse"):
                                row = v14._synthetic_row(
                                    seed, b.name, lam=3.535533906, cval=.9, rho=rho,
                                    kl=kl, rollout="teacher", frac=1.0, alpha=a, lr=lr,
                                )
                                row["cell_name"] = f"syn_{b.name}_{seed}_{rho}_{lr}_{a}_{kl}"
                                row["old_match_max_drawdown"] = row["old_match_forgetting"] + 0.01
                                rows.append(row)
            d = root / b.name
            d.mkdir(parents=True)
            pd.DataFrame(rows).to_csv(d / "runs.csv", index=False)
        # For the synthetic smoke, cell-table expected rows intentionally differ
        # from the full registry; test specialized aggregation directly.
        agg.mkdir()
        aggregate_stability(v14, blocks, root, agg)
        required = [
            "kl_stability_surface_curves.csv",
            "kl_stability_paired_contrasts_seed.csv.gz",
            "kl_stability_paired_summary.csv",
            "kl_stability_flip_map.csv",
            "kl_stability_crossfit_optimal_alpha_rows.csv.gz",
            "kl_stability_route_points_summary.csv",
        ]
        missing = [x for x in required if not (agg / x).exists()]
        if missing:
            raise AssertionError(f"aggregation self-test missing outputs: {missing}")
        # Global merge regression with minimal canonical tables.
        pd.DataFrame([{"registry_block": "old", "cell_name": "old_cell"}]).to_csv(agg / "cell_configurations.csv.gz", index=False, compression="gzip")
        pd.DataFrame([{"registry_block": "old", "cell_name": "old_cell", "x__mean": 1.0}]).to_csv(agg / "cell_summary.csv.gz", index=False, compression="gzip")
        pd.DataFrame([{"block": "old", "status": "complete"}]).to_csv(agg / "completion_status.csv", index=False)
        (agg / "aggregation_manifest.json").write_text("{}", encoding="utf-8")
        global_agg = root / "aggregate"
        global_agg.mkdir()
        pd.DataFrame([{"registry_block": "base", "cell_name": "base_cell"}]).to_csv(global_agg / "cell_configurations.csv.gz", index=False, compression="gzip")
        pd.DataFrame([{"registry_block": "base", "cell_name": "base_cell", "x__mean": 0.0}]).to_csv(global_agg / "cell_summary.csv.gz", index=False, compression="gzip")
        pd.DataFrame([{"block": "base", "status": "complete"}]).to_csv(global_agg / "completion_status.csv", index=False)
        update_global_aggregate(agg, global_agg)
        merged = pd.read_csv(global_agg / "cell_configurations.csv.gz")
        if len(merged) != 2:
            raise AssertionError("global cell merge is not idempotent/correct")
        update_global_aggregate(agg, global_agg)
        merged2 = pd.read_csv(global_agg / "cell_configurations.csv.gz")
        if len(merged2) != 2:
            raise AssertionError("second global update duplicated rows")
        return {"status": "PASS", "generated_files": len(list(agg.iterdir())), "required_checked": len(required)}


def selected_blocks(blocks: Sequence[ExtensionBlock], selector: str) -> List[ExtensionBlock]:
    if selector == "all":
        return list(blocks)
    exact = [b for b in blocks if b.name == selector]
    if exact:
        return exact
    aliases = {
        "E83": "V14_E83_lr_stability",
        "E84": "V14_E84_lr_rhophi_stability",
    }
    if selector in aliases:
        return [b for b in blocks if b.name == aliases[selector]]
    raise ValueError(f"unknown --run {selector!r}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--v14-exploration", type=Path, default=DEFAULT_V14_EXPLORATION,
                   help="Path to the original V14.0 exploration driver used as a library.")
    p.add_argument("--engine", type=Path, default=DEFAULT_ENGINE)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--aggregate-dir", type=Path, default=None,
                   help="Default: <output-root>/aggregate_v14_1")
    p.add_argument("--global-aggregate-dir", type=Path, default=None,
                   help="Default: <output-root>/aggregate")
    p.add_argument("--device", default="auto")
    p.add_argument(
        "--devices", default=None,
        help="Comma-separated worker devices for dynamic parallel seed-shard execution, e.g. cuda:0,cuda:1,cuda:2. A device may be repeated intentionally to run multiple workers on it. Overrides --device for selected shards.",
    )
    p.add_argument(
        "--shard-order", choices=("forward", "reverse"), default="forward",
        help="Order used to choose pending shards. reverse is useful for filling the tail while a legacy forward runner is active.",
    )
    p.add_argument(
        "--shard-indices", default=None,
        help="Optional 1-based comma-separated shard indices to run, e.g. 6,5,4. This takes precedence over --shard-order.",
    )
    p.add_argument(
        "--max-shards", type=int, default=None,
        help="Run at most this many selected *incomplete* shards. Useful for bounded tail filling.",
    )
    p.add_argument(
        "--defer-finalize", action="store_true",
        help="Do not rebuild block-level runs.csv/history.csv even if all shards become complete. Recommended when another legacy runner is still active.",
    )
    p.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    p.add_argument("--seed-shard-size", type=int, default=16,
                   help="Independent resumable seed shard size; 16 gives six shards for 96 seeds.")
    p.add_argument("--repair-incomplete", action=argparse.BooleanOptionalAction, default=True,
                   help="Fallback behavior when partial recovery is disabled: delete and recompute an incomplete shard.")
    p.add_argument(
        "--recover-partial", action=argparse.BooleanOptionalAction, default=True,
        help=(
            "Reuse complete cells already written in incomplete/current or overlapping legacy "
            "seed shards. This is enabled by default and supports migration such as 96 seeds -> 16 seeds."
        ),
    )
    p.add_argument("--overwrite", action="store_true",
                   help="Delete the selected block directory before running. Usually unnecessary because runs resume by shard.")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--aggregation-self-test", action="store_true")
    p.add_argument("--engine-self-test", action="store_true")
    p.add_argument("--list", action="store_true")
    p.add_argument("--run", default=None, help="all, E83, E84, or exact block name")
    p.add_argument("--aggregate", action="store_true")
    p.add_argument("--update-global-aggregate", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    v14 = load_v14_exploration(args.v14_exploration)
    blocks = build_blocks(v14)
    agg = args.aggregate_dir or (args.output_root / "aggregate_v14_1")
    global_agg = args.global_aggregate_dir or (args.output_root / "aggregate")

    if args.self_test:
        print(json.dumps(run_static_self_tests(v14, blocks), indent=2, sort_keys=True))
        return
    if args.aggregation_self_test:
        print(json.dumps(run_aggregation_self_test(v14, blocks), indent=2, sort_keys=True))
        return
    if args.engine_self_test:
        cmd = [sys.executable, str(args.engine), "--self-test"]
        print("$", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
        return
    if args.list:
        print(json.dumps(registry_summary(blocks), indent=2, sort_keys=True))
        for b in blocks:
            print(
                f"{b.name:34s} new_cells={len(b.cells):6d} excluded_existing={b.excluded_existing_cells:5d} "
                f"seeds={b.n_seeds:3d} new_rows={b.n_runs:9d}  {b.question}",
                flush=True,
            )
        return
    if args.run:
        for b in selected_blocks(blocks, args.run):
            execute_block_resumable(
                b, args.output_root,
                engine=args.engine,
                device=args.device,
                devices=(args.devices.split(",") if args.devices else None),
                dtype=args.dtype,
                seed_shard_size=args.seed_shard_size,
                smoke=args.smoke, overwrite=args.overwrite,
                repair_incomplete=args.repair_incomplete,
                recover_partial=args.recover_partial,
                shard_order=args.shard_order,
                shard_indices=args.shard_indices,
                max_shards=args.max_shards,
                defer_finalize=args.defer_finalize,
            )
    if args.aggregate:
        aggregate(v14, blocks, args.output_root, agg)
        print(f"V14.1 aggregate written to {agg}", flush=True)
    if args.update_global_aggregate:
        update_global_aggregate(agg, global_agg)
        print(f"Global V14 aggregate updated in {global_agg}", flush=True)
    if not any((args.run, args.aggregate, args.update_global_aggregate)):
        raise SystemExit(
            "Choose --self-test, --aggregation-self-test, --engine-self-test, --list, "
            "--run ..., --aggregate, or --update-global-aggregate"
        )


if __name__ == "__main__":
    main()
