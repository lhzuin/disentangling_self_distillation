#!/usr/bin/env python3
"""V14.2 canonical main-grid completion for the final finite autoregressive model.

Purpose
=======
The paper now reports a *fully crossed* finite-model grid with

    rollout source              : student, teacher                     (2)
    EMA teacher coupling alpha  : V14 EMA_FINE                        (18)
    KL direction                : reverse, forward                    (2)
    lambda                      : V14 LAMBDA_FINE                      (7)
    kappa                       : 0.3, 0.6, 0.9                       (3)
    rho_phi                     : 0, 0.5, 1                           (3)
    learning rate               : 1e-3                                (1)
    paired task seeds           : 96                                  (96)

This is 4,536 scientific cells and 435,456 per-seed training rows.

The central design choice in V14.2 is that the 96 seeds are not merely "96
seeds per cell": the *same* 96 task seeds are used for every cell in the
reported grid.  We anchor this canonical seed family at the V14 E76 public-
interaction block, whose 64 seeds already cover the first part of the desired
seed family for five EMA values.  Existing E76 rows are reused only when they
match both the scientific cell and the canonical task seed.  Everything else
is computed.  This yields a genuinely paired 96-seed factorial grid rather
than a count-only mixture of unrelated seed families from E70/E71/E73/E76.

With an intact V14 E76 block, the nominal accounting is:

    target grid rows                         435,456
    reusable E76 rows                         80,640
    genuinely new V14.2 rows                 354,816

The code does *not* assume E76 is complete.  Reuse is audited from the actual
raw E76 runs.csv, and any missing source row is recomputed automatically.

Scientific invariants
=====================
* The V14 model/engine is unchanged.
* H=4, K=8, D=64, rho_W=0, beta=0.
* Adam, exact occupancy, 200 updates, LR=1e-3.
* The target construction and all other V14 defaults remain unchanged.
* Student/teacher endpoint rollouts are encoded explicitly with teacher
  fractions 0/1 in the V14.2 spec.  Historical V14 endpoint cells sometimes
  stored an inert teacher_fraction=0.5; scientific matching normalizes this.
* No V14.1 stability-extension seed (161k family) is reused in the canonical
  grid, because doing so would destroy exact cross-cell seed pairing.

Recovery and parallel execution
===============================
Execution is seed-sharded and dynamically distributed across explicit devices.
Each requested shard is assembled from, in priority order:

1. already-completed/partially-completed V14.2 rows;
2. exact matching rows from V14 E76 for the same task seeds;
3. newly computed rows for every still-missing scientific cell.

A shard is marked complete only after it contains exactly one row for every
(cell, task_seed) pair.  Atomic claim files prevent duplicate work between
cooperating V14.2 runners.  Seed shards are automatically split at the E76
64-seed reuse boundary, so source rows are never needlessly discarded merely
because a user chose a shard size that straddles that boundary.

Aggregation
===========
``--aggregate`` writes a V14.2-only directory (default
``<output-root>/aggregate_v14_2``) containing canonical configuration/metric
summaries, exact seed-coverage audits, provenance summaries, and paired
forward-vs-reverse contrasts for the completed paper grid.

``--update-global-aggregate`` idempotently merges the canonical V14.2 cell
configuration/summary tables into the existing V14 aggregate.  Older V14/V14.1
specialized files are preserved; V14.2-specific audit/paired files are copied
under explicit ``v14_2_*`` names.

Typical workflow
================
  python toy_model/explore_v14_2_main_grid.py --self-test
  python toy_model/explore_v14_2_main_grid.py --preflight \
      --output-root results/v14_final

  # Smoke outside the evidence folder.
  python toy_model/explore_v14_2_main_grid.py --run all --smoke \
      --output-root results/v14_2_smoke --seed-shard-size 4

  # Canonical evidence grid.  Six 16-seed shards by default.
  python toy_model/explore_v14_2_main_grid.py --run all \
      --output-root results/v14_final --seed-shard-size 16 \
      --devices cuda:0,cuda:1,cuda:2,cuda:3

  python toy_model/explore_v14_2_main_grid.py --aggregate \
      --output-root results/v14_final
  python toy_model/explore_v14_2_main_grid.py --update-global-aggregate \
      --output-root results/v14_final
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
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

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Defaults / paper-facing constants
# -----------------------------------------------------------------------------

DEFAULT_V14_EXPLORATION = Path("toy_model/explore_v14_final.py")
DEFAULT_ENGINE = Path("toy_model/contextual_sd_toy_v14.py")
DEFAULT_OUTPUT_ROOT = Path("results/v14_final")

TARGET_BLOCK = "V14_E85_main_grid_completion"
SOURCE_BLOCK = "V14_E76_public_interactions"
TARGET_N_SEEDS = 96
TARGET_RHOS = (0.0, 0.5, 1.0)
TARGET_KAPPAS = (0.3, 0.6, 0.9)
TARGET_LR = 1e-3
BASE_STEPS = 200
BASE_H = 4
BASE_K = 8
BASE_D = 64

EXPECTED_TARGET_CELLS = 2 * 18 * 2 * 7 * 3 * 3  # 4,536
EXPECTED_TARGET_ROWS = EXPECTED_TARGET_CELLS * TARGET_N_SEEDS  # 435,456
EXPECTED_E76_MATCHING_CELLS = 3 * 7 * 3 * 2 * 2 * 5  # 1,260
EXPECTED_E76_REUSE_SEEDS = 64
EXPECTED_E76_REUSABLE_ROWS = EXPECTED_E76_MATCHING_CELLS * EXPECTED_E76_REUSE_SEEDS
EXPECTED_NEW_ROWS_IF_E76_COMPLETE = EXPECTED_TARGET_ROWS - EXPECTED_E76_REUSABLE_ROWS

NON_SCIENTIFIC_CELL_KEYS = {
    "name", "analysis_block", "variant", "record_history", "regime_name"
}

_V14 = None


@dataclass(frozen=True)
class MainGridBlock:
    name: str
    cells: Tuple[Dict[str, object], ...]
    seed_start: int
    n_seeds: int = TARGET_N_SEEDS
    seed_family: str = ""

    @property
    def n_runs(self) -> int:
        return len(self.cells) * self.n_seeds

    @property
    def seed_end(self) -> int:
        return self.seed_start + self.n_seeds - 1


# -----------------------------------------------------------------------------
# Imports / scientific identity
# -----------------------------------------------------------------------------

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
        _V14 = _load_module(path, "explore_v14_final_for_v14_2")
    return _V14


def _close(a: float, b: float, atol: float = 1e-12) -> bool:
    return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=atol)


def _token(x: float) -> str:
    return f"{float(x):.12g}".replace("-", "m").replace(".", "p")


def _scientific_signature(cell_: Mapping[str, object]) -> str:
    """Canonical scientific identity, independent of registry metadata.

    Historical V14 endpoint cells sometimes left ``teacher_fraction`` at the
    cell-factory default 0.5 even when ``rollout_source`` was exactly student or
    teacher.  The V14 engine treats the endpoint source as authoritative, so we
    normalize this inert field to compare those rows with the explicit V14.2
    endpoint specification.
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


def _find_block(v14, name: str):
    hits = [b for b in v14.BLOCKS if str(b.name) == name]
    if len(hits) != 1:
        raise RuntimeError(f"Expected exactly one V14 block named {name}, found {len(hits)}")
    return hits[0]


def build_main_grid(v14) -> Tuple[MainGridBlock, Dict[str, str], object]:
    """Build the exact paper grid and the E76-source -> V14.2 cell map."""
    # Fail loudly if the imported V14 driver no longer matches the table.
    if len(tuple(v14.EMA_FINE)) != 18:
        raise AssertionError(f"Expected 18 EMA values, got {len(tuple(v14.EMA_FINE))}")
    if len(tuple(v14.LAMBDA_FINE)) != 7:
        raise AssertionError(f"Expected 7 lambda values, got {len(tuple(v14.LAMBDA_FINE))}")
    if tuple(float(x) for x in v14.CONTEXT_MAIN) != TARGET_KAPPAS:
        raise AssertionError(f"V14 CONTEXT_MAIN drifted: {v14.CONTEXT_MAIN}")
    if not _close(float(v14.BASE_ADAM_LR), TARGET_LR):
        raise AssertionError(f"V14 base Adam LR drifted: {v14.BASE_ADAM_LR}")

    cells: List[Dict[str, object]] = []
    for rho in TARGET_RHOS:
        for lam in v14.LAMBDA_FINE:
            for kappa in TARGET_KAPPAS:
                for rollout in ("student", "teacher"):
                    teacher_fraction = 0.0 if rollout == "student" else 1.0
                    for alpha in v14.EMA_FINE:
                        for kl in ("forward", "reverse"):
                            name = (
                                f"e85_p{_token(rho)}_l{_token(lam)}_k{_token(kappa)}_"
                                f"{rollout}_a{_token(alpha)}_{kl}"
                            )
                            cells.append(v14.cell(
                                name,
                                TARGET_BLOCK,
                                rho_phi=rho,
                                lam=lam,
                                c=kappa,
                                kl=kl,
                                rollout=rollout,
                                teacher_fraction=teacher_fraction,
                                alpha=alpha,
                                lr=TARGET_LR,
                                estimator="exact",
                                train_steps=BASE_STEPS,
                                record_history=False,
                                variant="canonical_96_seed_main_grid",
                            ))

    source = _find_block(v14, SOURCE_BLOCK)
    seed_start = int(source.seed_start)
    block = MainGridBlock(
        name=TARGET_BLOCK,
        cells=tuple(cells),
        seed_start=seed_start,
        n_seeds=TARGET_N_SEEDS,
        seed_family=f"v14_2_canonical_e76_{seed_start}_{seed_start + TARGET_N_SEEDS - 1}",
    )

    target_by_sig = {_scientific_signature(c): str(c["name"]) for c in block.cells}
    if len(target_by_sig) != len(block.cells):
        raise AssertionError("Duplicate scientific cell in V14.2 target grid")

    source_to_target: Dict[str, str] = {}
    for source_cell in source.cells:
        target_name = target_by_sig.get(_scientific_signature(source_cell))
        if target_name is not None:
            source_to_target[str(source_cell["name"])] = target_name

    return block, source_to_target, source


# -----------------------------------------------------------------------------
# Static validation and accounting
# -----------------------------------------------------------------------------

def registry_summary(v14, block: MainGridBlock, source_to_target: Mapping[str, str], source) -> Dict[str, object]:
    nominal_source_seeds = min(int(source.n_seeds), TARGET_N_SEEDS)
    nominal_reuse_rows = len(source_to_target) * nominal_source_seeds
    return {
        "version": "14.2",
        "block": block.name,
        "seed_family": block.seed_family,
        "canonical_seed_start": block.seed_start,
        "canonical_seed_end": block.seed_end,
        "target_cells": len(block.cells),
        "target_seeds_per_cell": block.n_seeds,
        "target_rows": block.n_runs,
        "axes": {
            "rollout_source": ["student", "teacher"],
            "ema_alpha": [float(x) for x in v14.EMA_FINE],
            "kl_direction": ["reverse", "forward"],
            "lambda": [float(x) for x in v14.LAMBDA_FINE],
            "kappa": list(TARGET_KAPPAS),
            "rho_phi": list(TARGET_RHOS),
            "learning_rate": TARGET_LR,
        },
        "reuse_anchor": {
            "source_block": SOURCE_BLOCK,
            "source_seed_start": int(source.seed_start),
            "source_n_seeds": int(source.n_seeds),
            "matching_scientific_cells": len(source_to_target),
            "nominal_reusable_rows": nominal_reuse_rows,
            "nominal_new_rows": block.n_runs - nominal_reuse_rows,
        },
    }


def run_static_self_tests(v14, block: MainGridBlock, source_to_target: Mapping[str, str], source) -> Dict[str, object]:
    if len(block.cells) != EXPECTED_TARGET_CELLS:
        raise AssertionError(f"Target cell count {len(block.cells)} != {EXPECTED_TARGET_CELLS}")
    if block.n_runs != EXPECTED_TARGET_ROWS:
        raise AssertionError(f"Target row count {block.n_runs} != {EXPECTED_TARGET_ROWS}")
    if block.n_seeds != 96:
        raise AssertionError("Paper main grid must use 96 paired task seeds")
    if int(source.n_seeds) != EXPECTED_E76_REUSE_SEEDS:
        raise AssertionError(
            f"Expected E76 to have {EXPECTED_E76_REUSE_SEEDS} seeds, got {source.n_seeds}"
        )
    if len(source_to_target) != EXPECTED_E76_MATCHING_CELLS:
        raise AssertionError(
            f"Expected {EXPECTED_E76_MATCHING_CELLS} reusable E76 cells, "
            f"found {len(source_to_target)}"
        )

    signatures = set()
    names = set()
    axis_values = {
        "rho": set(), "lambda": set(), "kappa": set(), "alpha": set(),
        "kl": set(), "rollout": set(), "lr": set(),
    }
    for cell in block.cells:
        name = str(cell["name"])
        if name in names:
            raise AssertionError(f"Duplicate cell name {name}")
        names.add(name)
        sig = _scientific_signature(cell)
        if sig in signatures:
            raise AssertionError(f"Duplicate scientific signature {name}")
        signatures.add(sig)

        core = cell["core"]
        axis_values["rho"].add(float(core["feature_overlap"]))
        axis_values["lambda"].add(float(core["initial_policy_concentration"]))
        axis_values["kappa"].add(float(core["context_strength"]))
        axis_values["alpha"].add(float(cell["ema_alpha"]))
        axis_values["kl"].add(str(cell["kl_direction"]))
        axis_values["rollout"].add(str(cell["rollout_source"]))
        axis_values["lr"].add(float(cell["learning_rate"]))

        if cell["estimator"] != "exact":
            raise AssertionError(f"Non-exact estimator leaked into {name}")
        if int(cell["train_steps"]) != BASE_STEPS:
            raise AssertionError(f"Unexpected train_steps in {name}")
        s = cell["structure"]
        if (int(s["horizon"]), int(s["vocab_size"]), int(s["feature_dim"])) != (4, 8, 64):
            raise AssertionError(f"Structural drift in {name}: {s}")
        ab = cell["ablation"]
        if not _close(float(ab["readout_compatibility"]), 0.0):
            raise AssertionError("rho_W must remain zero")
        if not _close(float(ab["support_placement_bias"]), 0.0):
            raise AssertionError("beta must remain zero")

    if axis_values["rho"] != set(TARGET_RHOS):
        raise AssertionError(f"rho grid mismatch: {axis_values['rho']}")
    if axis_values["lambda"] != set(float(x) for x in v14.LAMBDA_FINE):
        raise AssertionError("lambda grid mismatch")
    if axis_values["kappa"] != set(TARGET_KAPPAS):
        raise AssertionError("kappa grid mismatch")
    if axis_values["alpha"] != set(float(x) for x in v14.EMA_FINE):
        raise AssertionError("EMA grid mismatch")
    if axis_values["kl"] != {"forward", "reverse"}:
        raise AssertionError("KL grid mismatch")
    if axis_values["rollout"] != {"student", "teacher"}:
        raise AssertionError("rollout grid mismatch")
    if axis_values["lr"] != {TARGET_LR}:
        raise AssertionError("LR grid mismatch")

    nominal_new = block.n_runs - len(source_to_target) * int(source.n_seeds)
    if nominal_new != EXPECTED_NEW_ROWS_IF_E76_COMPLETE:
        raise AssertionError(
            f"Nominal new-row count {nominal_new} != {EXPECTED_NEW_ROWS_IF_E76_COMPLETE}"
        )

    return {
        "status": "PASS",
        **registry_summary(v14, block, source_to_target, source),
        "expected_e76_reusable_rows_if_complete": EXPECTED_E76_REUSABLE_ROWS,
        "expected_new_rows_if_e76_complete": EXPECTED_NEW_ROWS_IF_E76_COMPLETE,
    }


# -----------------------------------------------------------------------------
# File helpers / shard validation
# -----------------------------------------------------------------------------

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _csv_row_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("rb") as f:
        return max(0, sum(1 for _ in f) - 1)


def _merge_csv_files(paths: Sequence[Path], out: Path) -> None:
    paths = [p for p in paths if p.exists()]
    if not paths:
        if out.exists():
            out.unlink()
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    header: Optional[bytes] = None
    with tmp.open("wb") as w:
        for p in paths:
            with p.open("rb") as r:
                this_header = r.readline()
                if header is None:
                    header = this_header
                    w.write(this_header)
                elif this_header != header:
                    raise ValueError(f"CSV header mismatch while merging {p}")
                shutil.copyfileobj(r, w, length=1 << 20)
    os.replace(tmp, out)


def write_block_spec(path: Path, block: MainGridBlock, cells: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "v14_2_registry_block": block.name,
        "seed_family": block.seed_family,
        "purpose": "canonical fully-crossed 96-seed paper grid",
        "cells": list(cells),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _validate_shard(
    shard_dir: Path,
    *,
    expected_cells: int,
    expected_seed_start: int,
    expected_n_seeds: int,
) -> Tuple[bool, str]:
    runs = shard_dir / "runs.csv"
    manifest = shard_dir / "manifest.json"
    if not runs.exists() or not manifest.exists():
        return False, "missing runs.csv or manifest.json"
    expected_rows = expected_cells * expected_n_seeds
    if _csv_row_count(runs) != expected_rows:
        return False, f"row count {_csv_row_count(runs)} != {expected_rows}"
    try:
        key = pd.read_csv(runs, usecols=["cell_name", "task_seed"])
    except Exception as exc:
        return False, f"cannot read shard keys: {exc}"
    if key.duplicated(["cell_name", "task_seed"]).any():
        return False, "duplicate (cell_name, task_seed) rows"
    expected_seeds = list(range(expected_seed_start, expected_seed_start + expected_n_seeds))
    seeds = sorted(pd.to_numeric(key["task_seed"], errors="raise").astype(int).unique().tolist())
    if seeds != expected_seeds:
        return False, f"seed set mismatch: observed {seeds[:3]}... expected {expected_seeds[:3]}..."
    if key["cell_name"].nunique() != expected_cells:
        return False, f"cell count {key['cell_name'].nunique()} != {expected_cells}"
    return True, "complete"


def _make_shards(seed_start: int, n_seeds: int, shard_size: int, reuse_boundary: int) -> List[Tuple[int, int]]:
    """Build contiguous shards, never crossing the old-E76 reuse boundary."""
    shard_size = max(1, int(shard_size))
    end_exclusive = seed_start + n_seeds
    boundaries = {seed_start, end_exclusive}
    if seed_start < reuse_boundary < end_exclusive:
        boundaries.add(reuse_boundary)
    ordered_boundaries = sorted(boundaries)
    shards: List[Tuple[int, int]] = []
    for lo, hi in zip(ordered_boundaries[:-1], ordered_boundaries[1:]):
        s = lo
        while s < hi:
            n = min(shard_size, hi - s)
            shards.append((s, n))
            s += n
    return shards


def _claim_shard(path: Path) -> bool:
    """Atomically claim a shard, automatically clearing dead local claims.

    The experiments run on one multi-GPU host with a shared local filesystem.
    If a process is killed (OOM, timeout, shell interruption), its claim file
    would otherwise block recovery forever.  We therefore inspect the recorded
    PID and remove the claim only when that PID provably no longer exists.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(2):
        try:
            fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                pid = int(payload.get("pid", -1))
            except Exception:
                return False
            if pid <= 0:
                return False
            try:
                os.kill(pid, 0)
                return False  # live owner
            except ProcessLookupError:
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
                continue
            except PermissionError:
                return False
        else:
            try:
                os.write(fd, json.dumps({"pid": os.getpid(), "created_unix": time.time()}).encode("utf-8"))
            finally:
                os.close(fd)
            return True
    return False


def _release_claim(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _run_subprocess_to_log(cmd: Sequence[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(list(cmd), stdout=log, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, list(cmd))


# -----------------------------------------------------------------------------
# E76 external reuse cache
# -----------------------------------------------------------------------------

def _source_runs_path(output_root: Path, source_block_name: str, override: Optional[Path]) -> Path:
    return override if override is not None else output_root / source_block_name / "runs.csv"


def _target_cell_lookup(block: MainGridBlock) -> Dict[str, Mapping[str, object]]:
    return {str(c["name"]): c for c in block.cells}


def _normalize_reused_rows(
    df: pd.DataFrame,
    source_to_target: Mapping[str, str],
    target_cells: Mapping[str, Mapping[str, object]],
) -> pd.DataFrame:
    """Rewrite registry-only metadata so reused rows belong to canonical E85 cells."""
    out = df.copy()
    out["cell_name"] = out["cell_name"].astype(str).map(source_to_target)
    out = out[out["cell_name"].notna()].copy()

    if "analysis_block" in out.columns:
        out["analysis_block"] = TARGET_BLOCK
    if "variant" in out.columns:
        out["variant"] = "canonical_96_seed_main_grid"
    if "record_history" in out.columns:
        out["record_history"] = False

    # Historical endpoint cells may store teacher_fraction=0.5.  Normalize the
    # inert metadata to the explicit target value so configuration summaries are
    # invariant across reused and newly computed seeds.
    if "teacher_fraction" in out.columns:
        rollout_to_fraction = {
            name: (0.0 if str(cell["rollout_source"]) == "student" else 1.0)
            for name, cell in target_cells.items()
        }
        out["teacher_fraction"] = out["cell_name"].map(rollout_to_fraction).astype(float)
    return out


def prepare_external_reuse_cache(
    *,
    block: MainGridBlock,
    source_to_target: Mapping[str, str],
    source,
    output_root: Path,
    source_runs_override: Optional[Path],
    shards: Sequence[Tuple[int, int]],
    block_dir: Path,
    force: bool = False,
) -> Dict[str, object]:
    """Stream matching E76 rows into target-seed-shard cache files once."""
    cache_dir = block_dir / "external_reuse_e76"
    manifest_path = cache_dir / "manifest.json"
    source_path = _source_runs_path(output_root, SOURCE_BLOCK, source_runs_override)
    target_cells = _target_cell_lookup(block)

    if not source_path.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "status": "source_missing",
            "source_path": str(source_path),
            "reusable_rows": 0,
            "reusable_cells": 0,
            "note": "V14.2 will compute the complete canonical grid from scratch.",
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        return manifest

    stat = source_path.stat()
    fingerprint = {
        "source_path": str(source_path.resolve()),
        "source_size": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
        "source_mapping_cells": len(source_to_target),
        "canonical_seed_start": block.seed_start,
        "canonical_seed_end": block.seed_end,
        "source_seed_start": int(source.seed_start),
        "source_n_seeds": int(source.n_seeds),
        "shards": [[int(s), int(n)] for s, n in shards],
    }
    if not force and manifest_path.exists():
        try:
            old = json.loads(manifest_path.read_text(encoding="utf-8"))
            cache_files_ok = True
            audit_path = cache_dir / "reuse_by_shard.csv"
            if old.get("fingerprint") == fingerprint and audit_path.exists():
                audit = pd.read_csv(audit_path)
                for row in audit.itertuples(index=False):
                    if int(row.rows) > 0:
                        expected = cache_dir / f"seeds_{int(row.seed_start)}_{int(row.seed_end)}.csv"
                        if not expected.exists():
                            cache_files_ok = False
                            break
                if cache_files_ok:
                    return old
        except Exception:
            pass

    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Temporary cache files are created by appending filtered chunks.  Only
    # target scientific cells and canonical E76-overlap seeds are retained.
    source_names = set(source_to_target)
    canonical_seed_set = set(range(block.seed_start, block.seed_end + 1))
    source_seed_set = set(range(int(source.seed_start), int(source.seed_start) + int(source.n_seeds)))
    allowed_seeds = canonical_seed_set & source_seed_set

    shard_for_seed: Dict[int, Path] = {}
    for start, n in shards:
        path = cache_dir / f"seeds_{start}_{start+n-1}.csv"
        for seed in range(start, start + n):
            shard_for_seed[seed] = path

    wrote_header: Dict[Path, bool] = {}
    total = 0
    chunksize = 50_000
    for chunk in pd.read_csv(source_path, chunksize=chunksize):
        if not {"cell_name", "task_seed"}.issubset(chunk.columns):
            raise RuntimeError(f"{source_path} lacks cell_name/task_seed")
        seed_num = pd.to_numeric(chunk["task_seed"], errors="coerce")
        mask = chunk["cell_name"].astype(str).isin(source_names) & seed_num.isin(allowed_seeds)
        if not mask.any():
            continue
        part = chunk.loc[mask].copy()
        part["task_seed"] = pd.to_numeric(part["task_seed"], errors="raise").astype(int)
        part = _normalize_reused_rows(part, source_to_target, target_cells)
        for seed_group, rows in part.groupby("task_seed", sort=False):
            path = shard_for_seed.get(int(seed_group))
            if path is None:
                continue
            rows.to_csv(path, mode="a", header=not wrote_header.get(path, False), index=False)
            wrote_header[path] = True
            total += len(rows)

    # Audit key uniqueness and summarize cache coverage.
    all_keys: List[pd.DataFrame] = []
    shard_rows: List[Dict[str, object]] = []
    for start, n in shards:
        p = cache_dir / f"seeds_{start}_{start+n-1}.csv"
        if p.exists():
            keys = pd.read_csv(p, usecols=["cell_name", "task_seed"])
            if keys.duplicated(["cell_name", "task_seed"]).any():
                raise RuntimeError(f"Duplicate E76 reuse keys in {p}")
            all_keys.append(keys)
            shard_rows.append({
                "seed_start": start,
                "seed_end": start+n-1,
                "rows": len(keys),
                "cells": int(keys["cell_name"].nunique()),
            })
        else:
            shard_rows.append({
                "seed_start": start,
                "seed_end": start+n-1,
                "rows": 0,
                "cells": 0,
            })

    if all_keys:
        keys = pd.concat(all_keys, ignore_index=True)
        reusable_cells = int(keys["cell_name"].nunique())
        reusable_rows = int(len(keys))
        seed_count = int(keys["task_seed"].nunique())
    else:
        reusable_cells = reusable_rows = seed_count = 0

    audit = pd.DataFrame(shard_rows)
    audit.to_csv(cache_dir / "reuse_by_shard.csv", index=False)
    manifest = {
        "status": "ready",
        "fingerprint": fingerprint,
        "reusable_rows": reusable_rows,
        "reusable_cells": reusable_cells,
        "reusable_seed_count": seed_count,
        "nominal_expected_rows_if_source_complete": EXPECTED_E76_REUSABLE_ROWS,
        "source_complete_for_nominal_reuse": reusable_rows == EXPECTED_E76_REUSABLE_ROWS,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def preflight_audit(
    *,
    v14,
    block: MainGridBlock,
    source_to_target: Mapping[str, str],
    source,
    output_root: Path,
    source_runs_override: Optional[Path],
    seed_shard_size: int,
) -> Dict[str, object]:
    block_dir = output_root / block.name
    reuse_boundary = int(source.seed_start) + int(source.n_seeds)
    shards = _make_shards(block.seed_start, block.n_seeds, seed_shard_size, reuse_boundary)
    cache = prepare_external_reuse_cache(
        block=block,
        source_to_target=source_to_target,
        source=source,
        output_root=output_root,
        source_runs_override=source_runs_override,
        shards=shards,
        block_dir=block_dir,
    )
    reusable = int(cache.get("reusable_rows", 0))
    report = {
        **registry_summary(v14, block, source_to_target, source),
        "actual_reusable_rows_found": reusable,
        "actual_rows_to_compute_before_v14_2_recovery": block.n_runs - reusable,
        "source_complete_for_nominal_reuse": bool(cache.get("source_complete_for_nominal_reuse", False)),
        "n_seed_shards": len(shards),
        "seed_shards": [
            {"seed_start": s, "seed_end": s+n-1, "n_seeds": n} for s, n in shards
        ],
        "reuse_cache": cache,
    }
    block_dir.mkdir(parents=True, exist_ok=True)
    (block_dir / "v14_2_preflight_audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report


# -----------------------------------------------------------------------------
# Partial V14.2 recovery
# -----------------------------------------------------------------------------

def _read_csv_best_effort(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        return pd.read_csv(path, on_bad_lines="skip")
    except Exception as exc:
        print(f"[recovery] could not read {path}: {exc}; ignoring", flush=True)
        return None


def _partial_current_sources(block_dir: Path, shard_name: str) -> List[Tuple[str, pd.DataFrame]]:
    out: List[Tuple[str, pd.DataFrame]] = []
    shard_dir = block_dir / "shards" / shard_name
    if shard_dir.exists():
        d = _read_csv_best_effort(shard_dir / "runs.csv")
        if d is not None and len(d):
            out.append(("v14_2_existing_shard", d))
    work_root = block_dir / "shards" / ".recovery_work" / shard_name
    if work_root.exists():
        for attempt in sorted(p for p in work_root.glob("attempt_*") if p.is_dir()):
            d = _read_csv_best_effort(attempt / "runs.csv")
            if d is not None and len(d):
                out.append((f"v14_2_attempt:{attempt.name}", d))
    return out


def _external_source_for_shard(block_dir: Path, shard_name: str) -> Optional[pd.DataFrame]:
    path = block_dir / "external_reuse_e76" / f"{shard_name}.csv"
    return _read_csv_best_effort(path)


def _collect_complete_cells_for_shard(
    *,
    cells: Sequence[Mapping[str, object]],
    expected_seeds: Sequence[int],
    current_sources: Sequence[Tuple[str, pd.DataFrame]],
    external: Optional[pd.DataFrame],
) -> Tuple[pd.DataFrame, List[str], pd.DataFrame]:
    """Recover cells complete for every seed in this shard.

    V14.2 rows are preferred over external E76 rows when both are present.
    Partial cells are intentionally not reused because the engine accepts a
    cell x contiguous-seed rectangle, not arbitrary missing seed points.
    """
    target_names = [str(c["name"]) for c in cells]
    target_set = set(target_names)
    expected_set = set(int(s) for s in expected_seeds)
    order = {name: i for i, name in enumerate(target_names)}

    frames: List[pd.DataFrame] = []
    provenance_parts: List[pd.DataFrame] = []
    priority = 0
    for label, d in current_sources:
        if not {"cell_name", "task_seed"}.issubset(d.columns):
            continue
        x = d[d["cell_name"].astype(str).isin(target_set)].copy()
        x["task_seed"] = pd.to_numeric(x["task_seed"], errors="coerce")
        x = x[x["task_seed"].isin(expected_set)].copy()
        if not len(x):
            continue
        x["task_seed"] = x["task_seed"].astype(int)
        x["__priority"] = priority
        x["__source"] = label
        frames.append(x)
        priority += 1
    if external is not None and len(external):
        x = external[external["cell_name"].astype(str).isin(target_set)].copy()
        x["task_seed"] = pd.to_numeric(x["task_seed"], errors="coerce")
        x = x[x["task_seed"].isin(expected_set)].copy()
        if len(x):
            x["task_seed"] = x["task_seed"].astype(int)
            x["__priority"] = 10_000
            x["__source"] = f"reused:{SOURCE_BLOCK}"
            frames.append(x)

    if not frames:
        return pd.DataFrame(), [], pd.DataFrame(columns=["cell_name", "task_seed", "source"])

    all_rows = pd.concat(frames, ignore_index=True, sort=False)
    all_rows.sort_values(["__priority"], inplace=True, kind="stable")
    all_rows = all_rows.drop_duplicates(["cell_name", "task_seed"], keep="first")

    complete_names: List[str] = []
    for name, g in all_rows.groupby("cell_name", sort=False):
        seeds = set(pd.to_numeric(g["task_seed"], errors="raise").astype(int).tolist())
        if seeds == expected_set and len(g) == len(expected_seeds):
            complete_names.append(str(name))
    complete_names.sort(key=lambda x: order[x])
    complete_set = set(complete_names)

    selected = all_rows[all_rows["cell_name"].astype(str).isin(complete_set)].copy()
    provenance = selected[["cell_name", "task_seed", "__source"]].rename(columns={"__source": "source"})
    selected.drop(columns=["__priority", "__source"], inplace=True, errors="ignore")
    selected["__cell_order"] = selected["cell_name"].astype(str).map(order)
    selected.sort_values(["__cell_order", "task_seed"], inplace=True, kind="stable")
    selected.drop(columns=["__cell_order"], inplace=True)
    selected.reset_index(drop=True, inplace=True)
    return selected, complete_names, provenance


def _next_attempt_dir(block_dir: Path, shard_name: str) -> Path:
    root = block_dir / "shards" / ".recovery_work" / shard_name
    root.mkdir(parents=True, exist_ok=True)
    existing: List[int] = []
    for p in root.glob("attempt_*"):
        try:
            existing.append(int(p.name.split("_", 1)[1]))
        except Exception:
            pass
    idx = max(existing, default=0) + 1
    return root / f"attempt_{idx:04d}"


def _validate_attempt(path: Path, cell_names: Sequence[str], seed_start: int, n_seeds: int) -> Tuple[bool, str]:
    runs = path / "runs.csv"
    if not runs.exists():
        return False, "missing runs.csv"
    expected_rows = len(cell_names) * n_seeds
    if _csv_row_count(runs) != expected_rows:
        return False, f"row count {_csv_row_count(runs)} != {expected_rows}"
    key = pd.read_csv(runs, usecols=["cell_name", "task_seed"])
    if key.duplicated(["cell_name", "task_seed"]).any():
        return False, "duplicate keys"
    if set(key["cell_name"].astype(str)) != set(cell_names):
        return False, "cell set mismatch"
    expected_seeds = set(range(seed_start, seed_start + n_seeds))
    if set(pd.to_numeric(key["task_seed"], errors="raise").astype(int)) != expected_seeds:
        return False, "seed set mismatch"
    return True, "complete"


def _atomic_finalize_shard(
    *,
    shard_dir: Path,
    spec_path: Path,
    runs: pd.DataFrame,
    provenance: pd.DataFrame,
    manifest: Mapping[str, object],
) -> None:
    shard_dir.mkdir(parents=True, exist_ok=True)
    # Completion marker (manifest) is replaced last.
    for final_name, df in (("runs.csv", runs), ("provenance.csv", provenance)):
        tmp = shard_dir / f"{final_name}.tmp"
        df.to_csv(tmp, index=False)
        os.replace(tmp, shard_dir / final_name)
    shutil.copy2(spec_path, shard_dir / "experiment_spec.json")
    tmp_manifest = shard_dir / "manifest.json.tmp"
    tmp_manifest.write_text(json.dumps(dict(manifest), indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp_manifest, shard_dir / "manifest.json")


def _execute_one_shard(
    *,
    block: MainGridBlock,
    cells: Sequence[Mapping[str, object]],
    block_dir: Path,
    spec_path: Path,
    engine: Path,
    dtype: str,
    device: str,
    shard_index: int,
    n_shards: int,
    seed_start: int,
    n_seeds: int,
    recover_partial: bool,
) -> Dict[str, object]:
    shard_name = f"seeds_{seed_start}_{seed_start+n_seeds-1}"
    shard_dir = block_dir / "shards" / shard_name
    ok, _ = _validate_shard(
        shard_dir,
        expected_cells=len(cells),
        expected_seed_start=seed_start,
        expected_n_seeds=n_seeds,
    )
    if ok:
        return {"status": "already_complete", "device": device, "shard_index": shard_index}

    claim = block_dir / "shards" / ".claims" / f"{shard_name}.claim"
    if not _claim_shard(claim):
        return {"status": "claimed_elsewhere", "device": device, "shard_index": shard_index}

    try:
        expected_seeds = list(range(seed_start, seed_start + n_seeds))
        current = _partial_current_sources(block_dir, shard_name) if recover_partial else []
        external = _external_source_for_shard(block_dir, shard_name)
        recovered, recovered_names, recovered_prov = _collect_complete_cells_for_shard(
            cells=cells,
            expected_seeds=expected_seeds,
            current_sources=current,
            external=external,
        )
        recovered_set = set(recovered_names)
        missing = [c for c in cells if str(c["name"]) not in recovered_set]

        print(
            f"[{block.name}] {device}: shard {shard_index}/{n_shards} "
            f"seeds={seed_start}:{seed_start+n_seeds-1} "
            f"recovered={len(recovered_names)}/{len(cells)} cells; "
            f"compute={len(missing)}",
            flush=True,
        )

        computed = pd.DataFrame()
        computed_prov = pd.DataFrame(columns=["cell_name", "task_seed", "source"])
        attempt_dir: Optional[Path] = None
        attempt_manifest = None

        if missing:
            attempt_dir = _next_attempt_dir(block_dir, shard_name)
            attempt_dir.mkdir(parents=True, exist_ok=True)
            missing_spec = attempt_dir.parent / f"{attempt_dir.name}_spec.json"
            write_block_spec(missing_spec, block, missing)
            shutil.copy2(missing_spec, attempt_dir / "experiment_spec.json")

            cmd = [
                sys.executable, str(engine),
                "--spec", str(missing_spec),
                "--output-dir", str(attempt_dir),
                "--device", device,
                "--dtype", dtype,
                "--n-task-seeds", str(n_seeds),
                "--seed-start", str(seed_start),
                "--train-steps", str(BASE_STEPS),
                "--record-every", "10",
                "--sampled-trajectories-per-step", "256",
                "--paired-optimization-rng",
            ]
            log_path = attempt_dir.parent / f"{attempt_dir.name}.log"
            _run_subprocess_to_log(cmd, log_path)
            ok_attempt, reason = _validate_attempt(
                attempt_dir,
                [str(c["name"]) for c in missing],
                seed_start,
                n_seeds,
            )
            if not ok_attempt:
                raise RuntimeError(f"Engine attempt did not validate: {attempt_dir}: {reason}")
            computed = pd.read_csv(attempt_dir / "runs.csv")
            computed_prov = computed[["cell_name", "task_seed"]].copy()
            computed_prov["source"] = "computed:v14_2"
            mp = attempt_dir / "manifest.json"
            if mp.exists():
                try:
                    attempt_manifest = json.loads(mp.read_text(encoding="utf-8"))
                except Exception:
                    attempt_manifest = None

        parts = [x for x in (recovered, computed) if x is not None and len(x)]
        if not parts:
            raise RuntimeError(f"No rows available to finalize shard {shard_name}")
        final = pd.concat(parts, ignore_index=True, sort=False)
        if final.duplicated(["cell_name", "task_seed"]).any():
            raise RuntimeError(f"Duplicate keys while finalizing {shard_name}")

        expected_rows = len(cells) * n_seeds
        if len(final) != expected_rows:
            raise RuntimeError(f"Final shard rows {len(final)} != expected {expected_rows}")
        if final["cell_name"].nunique() != len(cells):
            raise RuntimeError("Final shard cell count mismatch")
        if set(pd.to_numeric(final["task_seed"], errors="raise").astype(int)) != set(expected_seeds):
            raise RuntimeError("Final shard seed set mismatch")

        order = {str(c["name"]): i for i, c in enumerate(cells)}
        final["__cell_order"] = final["cell_name"].astype(str).map(order)
        final.sort_values(["__cell_order", "task_seed"], inplace=True, kind="stable")
        final.drop(columns=["__cell_order"], inplace=True)
        final.reset_index(drop=True, inplace=True)

        provenance = pd.concat([recovered_prov, computed_prov], ignore_index=True, sort=False)
        provenance.drop_duplicates(["cell_name", "task_seed"], keep="first", inplace=True)
        provenance["__cell_order"] = provenance["cell_name"].astype(str).map(order)
        provenance.sort_values(["__cell_order", "task_seed"], inplace=True, kind="stable")
        provenance.drop(columns=["__cell_order"], inplace=True)
        provenance.reset_index(drop=True, inplace=True)
        if len(provenance) != expected_rows:
            raise RuntimeError("Provenance row count mismatch")

        manifest = {
            "v14_2_canonical_grid_shard": True,
            "created_unix": time.time(),
            "block": block.name,
            "seed_family": block.seed_family,
            "seed_start": seed_start,
            "seed_end": seed_start+n_seeds-1,
            "n_task_seeds": n_seeds,
            "n_cells": len(cells),
            "expected_rows": expected_rows,
            "recovered_cells": len(recovered_names),
            "computed_cells_this_attempt": len(missing),
            "reused_e76_rows": int((provenance["source"] == f"reused:{SOURCE_BLOCK}").sum()),
            "computed_rows": int((provenance["source"] == "computed:v14_2").sum()),
            "device_for_computed_rows": device if missing else None,
            "attempt_dir": str(attempt_dir) if attempt_dir else None,
            "attempt_engine_manifest": attempt_manifest,
            "full_spec_sha256": _sha256(spec_path),
        }
        _atomic_finalize_shard(
            shard_dir=shard_dir,
            spec_path=spec_path,
            runs=final,
            provenance=provenance,
            manifest=manifest,
        )
        ok, reason = _validate_shard(
            shard_dir,
            expected_cells=len(cells),
            expected_seed_start=seed_start,
            expected_n_seeds=n_seeds,
        )
        if not ok:
            raise RuntimeError(f"Final shard validation failed: {reason}")
        return {
            "status": "complete",
            "device": device,
            "shard_index": shard_index,
            "recovered_cells": len(recovered_names),
            "computed_cells": len(missing),
        }
    finally:
        _release_claim(claim)


def smoke_subset(v14, block: MainGridBlock) -> List[Dict[str, object]]:
    """Small but nontrivial subset spanning source-reusable and new EMA cells."""
    wanted = []
    alphas = {0.0, 0.000625, 0.0025, 0.16}
    for c in block.cells:
        core = c["core"]
        if (
            float(core["feature_overlap"]) in (0.0, 1.0)
            and _close(float(core["initial_policy_concentration"]), 2.5)
            and _close(float(core["context_strength"]), 0.9)
            and float(c["ema_alpha"]) in alphas
        ):
            wanted.append(c)
    return wanted[:32]


def _parse_shard_indices(spec: Optional[str], n_shards: int) -> Optional[List[int]]:
    if spec is None:
        return None
    out: List[int] = []
    seen = set()
    for raw in spec.split(","):
        raw = raw.strip()
        if not raw:
            continue
        i = int(raw)
        if i < 1 or i > n_shards:
            raise ValueError(f"shard index {i} outside 1..{n_shards}")
        if i not in seen:
            out.append(i)
            seen.add(i)
    return out


def execute_block(
    *,
    v14,
    block: MainGridBlock,
    source_to_target: Mapping[str, str],
    source,
    output_root: Path,
    source_runs_override: Optional[Path],
    engine: Path,
    device: str,
    devices: Optional[Sequence[str]],
    dtype: str,
    seed_shard_size: int,
    smoke: bool,
    overwrite: bool,
    recover_partial: bool,
    shard_order: str,
    shard_indices: Optional[str],
    max_shards: Optional[int],
    defer_finalize: bool,
) -> None:
    cells = smoke_subset(v14, block) if smoke else list(block.cells)
    n_seeds = min(4, block.n_seeds) if smoke else block.n_seeds
    block_dir = output_root / block.name
    if overwrite and block_dir.exists():
        shutil.rmtree(block_dir)
    block_dir.mkdir(parents=True, exist_ok=True)
    spec_path = output_root / "specs_v14_2" / f"{block.name}.json"
    write_block_spec(spec_path, block, cells)
    shutil.copy2(spec_path, block_dir / "experiment_spec.json")

    reuse_boundary = int(source.seed_start) + int(source.n_seeds)
    shards = _make_shards(block.seed_start, n_seeds, seed_shard_size, reuse_boundary)
    indexed = list(enumerate(shards, start=1))

    # Rebuild/check E76 reuse cache before workers launch.
    prepare_external_reuse_cache(
        block=MainGridBlock(block.name, tuple(cells), block.seed_start, n_seeds, block.seed_family),
        source_to_target={k: v for k, v in source_to_target.items() if v in {str(c['name']) for c in cells}},
        source=source,
        output_root=output_root,
        source_runs_override=source_runs_override,
        shards=shards,
        block_dir=block_dir,
    )

    explicit = _parse_shard_indices(shard_indices, len(shards))
    if explicit is not None:
        by_idx = dict(indexed)
        schedule = [(i, by_idx[i]) for i in explicit]
    else:
        schedule = list(indexed)
        if shard_order == "reverse":
            schedule.reverse()

    pending: List[Tuple[int, Tuple[int, int]]] = []
    for idx, (start, n) in schedule:
        d = block_dir / "shards" / f"seeds_{start}_{start+n-1}"
        ok, _ = _validate_shard(
            d, expected_cells=len(cells), expected_seed_start=start, expected_n_seeds=n
        )
        if not ok:
            pending.append((idx, (start, n)))
    if max_shards is not None:
        pending = pending[:max_shards]

    t0 = time.time()
    devs = [x.strip() for x in (devices or []) if x.strip()]
    if not devs:
        for idx, (start, n) in pending:
            result = _execute_one_shard(
                block=block, cells=cells, block_dir=block_dir, spec_path=spec_path,
                engine=engine, dtype=dtype, device=device,
                shard_index=idx, n_shards=len(shards), seed_start=start, n_seeds=n,
                recover_partial=recover_partial,
            )
            print(f"[{block.name}] {device}: shard {idx} status={result['status']}", flush=True)
    else:
        if devs == ["auto"]:
            raise ValueError("--devices auto is ambiguous; provide explicit cuda devices")
        queue: Queue = Queue()
        for item in pending:
            queue.put(item)

        def worker(slot: int, dev: str):
            results = []
            while True:
                try:
                    idx, (start, n) = queue.get_nowait()
                except Empty:
                    break
                try:
                    print(
                        f"[{block.name}] worker{slot}:{dev} claiming shard {idx}/{len(shards)} "
                        f"seeds={start}:{start+n-1}", flush=True
                    )
                    results.append(_execute_one_shard(
                        block=block, cells=cells, block_dir=block_dir, spec_path=spec_path,
                        engine=engine, dtype=dtype, device=dev,
                        shard_index=idx, n_shards=len(shards), seed_start=start, n_seeds=n,
                        recover_partial=recover_partial,
                    ))
                finally:
                    queue.task_done()
            return results

        failures: List[BaseException] = []
        with ThreadPoolExecutor(max_workers=len(devs)) as pool:
            futures = {pool.submit(worker, i, d): (i, d) for i, d in enumerate(devs, start=1)}
            for fut in as_completed(futures):
                i, d = futures[fut]
                try:
                    fut.result()
                except BaseException as exc:
                    failures.append(exc)
                    print(f"[{block.name}] worker{i}:{d} FAILED: {exc}", flush=True)
        if failures:
            raise RuntimeError(f"{len(failures)} worker(s) failed; first error: {failures[0]}")

    complete_dirs: List[Path] = []
    incomplete: List[int] = []
    for idx, (start, n) in indexed:
        d = block_dir / "shards" / f"seeds_{start}_{start+n-1}"
        ok, _ = _validate_shard(
            d, expected_cells=len(cells), expected_seed_start=start, expected_n_seeds=n
        )
        if ok:
            complete_dirs.append(d)
        else:
            incomplete.append(idx)
    if incomplete:
        print(
            f"[{block.name}] PARTIAL: {len(complete_dirs)}/{len(shards)} shards complete; "
            f"remaining={incomplete}", flush=True
        )
        return
    if defer_finalize:
        print(f"[{block.name}] all shards complete; finalization deferred", flush=True)
        return

    _merge_csv_files([d / "runs.csv" for d in complete_dirs], block_dir / "runs.csv")
    # Provenance has a stable, deliberately small schema.
    prov = pd.concat([pd.read_csv(d / "provenance.csv") for d in complete_dirs], ignore_index=True)
    prov.to_csv(block_dir / "seed_provenance.csv.gz", index=False, compression="gzip")

    expected_total = len(cells) * n_seeds
    observed = _csv_row_count(block_dir / "runs.csv")
    if observed != expected_total:
        raise RuntimeError(f"Canonical runs.csv {observed} rows != {expected_total}")

    provenance_counts = prov["source"].value_counts(dropna=False).to_dict()
    manifest = {
        "version": "14.2",
        "block": block.name,
        "seed_family": block.seed_family,
        "canonical_seed_start": block.seed_start,
        "canonical_seed_end": block.seed_start + n_seeds - 1,
        "n_cells": len(cells),
        "n_task_seeds": n_seeds,
        "expected_rows": expected_total,
        "observed_rows": observed,
        "seed_shard_size_requested": seed_shard_size,
        "actual_shards": [{"start": s, "n": n} for s, n in shards],
        "provenance_counts": provenance_counts,
        "smoke": smoke,
        "elapsed_seconds": time.time() - t0,
        "exploration_sha256": _sha256(Path(__file__)),
    }
    (block_dir / "v14_2_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"[{block.name}] COMPLETE: {observed:,}/{expected_total:,} rows", flush=True)


# -----------------------------------------------------------------------------
# Memory-bounded aggregation
# -----------------------------------------------------------------------------

def _find_small_complete_shard(block_dir: Path, expected_cells: int) -> Optional[Path]:
    root = block_dir / "shards"
    if not root.exists():
        return None
    candidates = []
    for d in root.glob("seeds_*_*"):
        if not d.is_dir() or not (d / "runs.csv").exists() or not (d / "manifest.json").exists():
            continue
        rows = _csv_row_count(d / "runs.csv")
        if rows > 0 and rows % expected_cells == 0:
            candidates.append((rows, d / "runs.csv"))
    return min(candidates, key=lambda x: x[0])[1] if candidates else None


def _metric_summary_batched(path: Path, metric_cols: Sequence[str], batch_size: int = 12) -> pd.DataFrame:
    metrics = list(metric_cols)
    if not metrics:
        return pd.DataFrame(columns=["cell_name"])
    pieces = []
    for lo in range(0, len(metrics), batch_size):
        batch = metrics[lo:lo+batch_size]
        print(f"[aggregate] metrics {lo+1}-{lo+len(batch)} / {len(metrics)}", flush=True)
        d = pd.read_csv(path, usecols=["cell_name", *batch])
        g = d.groupby("cell_name", sort=False, dropna=False)[batch]
        part = g.agg(["mean", "median", "std", "count"]).reset_index()
        part.columns = [
            "cell_name" if a == "cell_name" else f"{a}__{b}"
            for a, b in part.columns.to_flat_index()
        ]
        pieces.append(part)
    out = pieces[0]
    for part in pieces[1:]:
        out = out.merge(part, on="cell_name", how="outer", validate="one_to_one")
    ordered = ["cell_name"] + [
        f"{m}__{stat}" for stat in ("mean", "median", "std", "count") for m in metrics
    ]
    return out[ordered]


def _condition_id_summary(path: Path, chunksize: int = 200_000) -> Optional[pd.DataFrame]:
    header = pd.read_csv(path, nrows=0).columns
    if "condition_id" not in header:
        return None
    parts = []
    for chunk in pd.read_csv(path, usecols=["cell_name", "condition_id"], chunksize=chunksize):
        parts.append(chunk.drop_duplicates(["cell_name", "condition_id"]))
    unique = pd.concat(parts, ignore_index=True).drop_duplicates(["cell_name", "condition_id"])
    out = unique.groupby("cell_name", sort=False)["condition_id"].nunique().rename("condition_id_nunique").reset_index()
    out["recovered_or_multibatch"] = out["condition_id_nunique"].fillna(0).astype(int) > 1
    return out


def _seed_coverage_table(runs_path: Path, block: MainGridBlock) -> pd.DataFrame:
    key = pd.read_csv(runs_path, usecols=["cell_name", "task_seed"])
    key["task_seed"] = pd.to_numeric(key["task_seed"], errors="raise").astype(int)
    if key.duplicated(["cell_name", "task_seed"]).any():
        raise RuntimeError("Canonical runs.csv contains duplicate (cell_name, task_seed) keys")
    expected_set = set(range(block.seed_start, block.seed_end + 1))
    records = []
    for name, g in key.groupby("cell_name", sort=False):
        seeds = set(g["task_seed"].tolist())
        records.append({
            "cell_name": name,
            "n_rows": len(g),
            "n_unique_seeds": len(seeds),
            "seed_min": min(seeds) if seeds else np.nan,
            "seed_max": max(seeds) if seeds else np.nan,
            "exact_canonical_seed_set": seeds == expected_set,
            "missing_seed_count": len(expected_set - seeds),
            "unexpected_seed_count": len(seeds - expected_set),
        })
    return pd.DataFrame(records)


def _axis_coverage_table(v14, block: MainGridBlock) -> pd.DataFrame:
    records = []
    for c in block.cells:
        core = c["core"]
        records.append({
            "cell_name": c["name"],
            "rho_phi": core["feature_overlap"],
            "lambda": core["initial_policy_concentration"],
            "kappa": core["context_strength"],
            "rollout_source": c["rollout_source"],
            "ema_alpha": c["ema_alpha"],
            "kl_direction": c["kl_direction"],
            "learning_rate": c["learning_rate"],
        })
    d = pd.DataFrame(records)
    axes = ["rho_phi", "lambda", "kappa", "rollout_source", "ema_alpha", "kl_direction", "learning_rate"]
    # One target cell per exact axis combination.
    out = d.groupby(axes, dropna=False, sort=False).size().rename("n_target_cells").reset_index()
    return out


def _summary_ci(df: pd.DataFrame, group_cols: Sequence[str], value_cols: Sequence[str]) -> pd.DataFrame:
    records = []
    for key, sub in df.groupby(list(group_cols), dropna=False, sort=False):
        key_tuple = key if isinstance(key, tuple) else (key,)
        rec = {c: v for c, v in zip(group_cols, key_tuple)}
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


def _paired_kl_outputs(v14, runs_path: Path, agg: Path) -> None:
    header = pd.read_csv(runs_path, nrows=0).columns
    wanted = [
        "cell_name", "task_seed",
        "v14_core_feature_overlap", "v14_core_initial_policy_concentration", "v14_core_context_strength",
        "kl_direction", "rollout_source", "ema_alpha", "learning_rate",
        "new_match_gain", "old_match_forgetting",
    ]
    missing = [c for c in wanted if c not in header]
    if missing:
        print(f"[aggregate] paired KL export skipped; missing columns: {missing}", flush=True)
        return
    d = pd.read_csv(runs_path, usecols=wanted)
    d = v14.short_names(d)
    if "kappa" not in d.columns:
        d["kappa"] = d["c"]
    idx = ["task_seed", "rho_phi", "lambda", "kappa", "rollout_source", "ema_alpha", "learning_rate"]
    metrics = ["new_match_gain", "old_match_forgetting"]
    f = d[d["kl_direction"] == "forward"][idx + metrics].copy()
    r = d[d["kl_direction"] == "reverse"][idx + metrics].copy()
    paired = f.merge(r, on=idx, how="inner", validate="one_to_one", suffixes=("__forward", "__reverse"))
    out = paired[idx].copy()
    out["forward_minus_reverse_acquisition"] = paired["new_match_gain__forward"] - paired["new_match_gain__reverse"]
    out["forward_minus_reverse_forgetting"] = paired["old_match_forgetting__forward"] - paired["old_match_forgetting__reverse"]
    out["forward_minus_reverse_retention"] = -out["forward_minus_reverse_forgetting"]
    out.to_csv(agg / "main_grid_paired_kl_seed.csv.gz", index=False, compression="gzip")
    summary = _summary_ci(
        out,
        ["rho_phi", "lambda", "kappa", "rollout_source", "ema_alpha", "learning_rate"],
        ["forward_minus_reverse_acquisition", "forward_minus_reverse_forgetting", "forward_minus_reverse_retention"],
    )
    summary.to_csv(agg / "main_grid_paired_kl_summary.csv", index=False)


def aggregate(v14, block: MainGridBlock, output_root: Path, agg: Path) -> None:
    block_dir = output_root / block.name
    runs = block_dir / "runs.csv"
    if not runs.exists():
        raise FileNotFoundError(runs)
    observed = _csv_row_count(runs)
    if observed != block.n_runs:
        raise RuntimeError(f"Refusing aggregation: rows={observed}, expected={block.n_runs}")

    agg.mkdir(parents=True, exist_ok=True)
    coverage = _seed_coverage_table(runs, block)
    coverage.to_csv(agg / "main_grid_seed_coverage.csv", index=False)
    if len(coverage) != len(block.cells) or not coverage["exact_canonical_seed_set"].all():
        bad = int((~coverage["exact_canonical_seed_set"]).sum())
        raise RuntimeError(f"Refusing aggregation: {bad} cells lack the exact canonical 96-seed set")

    axis = _axis_coverage_table(v14, block)
    axis.to_csv(agg / "main_grid_axis_coverage.csv", index=False)
    if len(axis) != EXPECTED_TARGET_CELLS or not (axis["n_target_cells"] == 1).all():
        raise RuntimeError("Main-grid axis coverage is not exactly one cell per factorial combination")

    shard_schema = _find_small_complete_shard(block_dir, len(block.cells))
    if shard_schema is None:
        shard_schema = runs
        schema_df = pd.read_csv(runs, nrows=min(observed, len(block.cells) * 2))
    else:
        schema_df = pd.read_csv(shard_schema)
    metrics = v14.infer_metric_columns(schema_df)
    cfg = v14.cell_configuration_table(schema_df, metrics)
    if len(cfg) != len(block.cells):
        raise RuntimeError(f"Configuration table has {len(cfg)} cells, expected {len(block.cells)}")
    prov = _condition_id_summary(runs)
    cfg = cfg.drop(columns=["condition_id_nunique", "recovered_or_multibatch"], errors="ignore")
    if prov is not None:
        cfg = cfg.merge(prov, on="cell_name", how="left", validate="one_to_one")
    cfg["registry_block"] = block.name
    cfg.to_csv(agg / "cell_configurations.csv.gz", index=False, compression="gzip")

    batch_size = max(1, int(os.environ.get("V14_AGG_METRIC_BATCH", "12")))
    summary = _metric_summary_batched(runs, metrics, batch_size=batch_size)
    summary["registry_block"] = block.name
    summary.to_csv(agg / "cell_summary.csv.gz", index=False, compression="gzip")

    status = pd.DataFrame([{
        "block": block.name,
        "status": "complete",
        "expected_rows": block.n_runs,
        "observed_rows": observed,
        "expected_cells": len(block.cells),
        "observed_cells": len(cfg),
        "expected_seeds_per_cell": block.n_seeds,
        "all_cells_exact_seed_set": True,
    }])
    status.to_csv(agg / "completion_status.csv", index=False)

    provenance_path = block_dir / "seed_provenance.csv.gz"
    if provenance_path.exists():
        provenance = pd.read_csv(provenance_path)
        provenance.groupby("source", dropna=False).size().rename("rows").reset_index().to_csv(
            agg / "main_grid_provenance_summary.csv", index=False
        )
    _paired_kl_outputs(v14, runs, agg)

    manifest = {
        "version": "14.2",
        "block": block.name,
        "complete": True,
        "target_cells": len(block.cells),
        "target_rows": block.n_runs,
        "canonical_seed_start": block.seed_start,
        "canonical_seed_end": block.seed_end,
        "seed_family": block.seed_family,
        "aggregation_note": (
            "This is the canonical fully-crossed paper grid. Every cell has exactly the same "
            "96 task seeds; reused E76 rows are raw evidence, not re-estimated summaries."
        ),
        "exploration_sha256": _sha256(Path(__file__)),
    }
    (agg / "aggregation_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    (agg / "aggregate_status.md").write_text(
        "# V14.2 canonical main-grid status\n\n"
        f"- cells: {len(block.cells):,}/{EXPECTED_TARGET_CELLS:,}\n"
        f"- rows: {observed:,}/{EXPECTED_TARGET_ROWS:,}\n"
        f"- exact paired seeds per cell: {block.n_seeds}\n"
        f"- seed range: {block.seed_start}--{block.seed_end}\n"
        "- status: COMPLETE\n",
        encoding="utf-8",
    )


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
    both.drop_duplicates(keys, keep="last", inplace=True)
    both.to_csv(existing, index=False, compression=compression)


def update_global_aggregate(v14_2_agg: Path, global_agg: Path) -> None:
    manifest_path = v14_2_agg / "aggregation_manifest.json"
    if not manifest_path.exists():
        raise RuntimeError("Run --aggregate successfully before --update-global-aggregate")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not bool(manifest.get("complete", False)):
        raise RuntimeError("Refusing global update from incomplete V14.2 aggregate")
    global_agg.mkdir(parents=True, exist_ok=True)

    _merge_keyed_csv(
        global_agg / "cell_configurations.csv.gz",
        v14_2_agg / "cell_configurations.csv.gz",
        ["registry_block", "cell_name"],
        compression="gzip",
    )
    _merge_keyed_csv(
        global_agg / "cell_summary.csv.gz",
        v14_2_agg / "cell_summary.csv.gz",
        ["registry_block", "cell_name"],
        compression="gzip",
    )
    _merge_keyed_csv(
        global_agg / "completion_status.csv",
        v14_2_agg / "completion_status.csv",
        ["block"],
    )

    specialized = {
        "main_grid_seed_coverage.csv": "v14_2_main_grid_seed_coverage.csv",
        "main_grid_axis_coverage.csv": "v14_2_main_grid_axis_coverage.csv",
        "main_grid_provenance_summary.csv": "v14_2_main_grid_provenance_summary.csv",
        "main_grid_paired_kl_seed.csv.gz": "v14_2_main_grid_paired_kl_seed.csv.gz",
        "main_grid_paired_kl_summary.csv": "v14_2_main_grid_paired_kl_summary.csv",
        "aggregation_manifest.json": "v14_2_aggregation_manifest.json",
    }
    for src_name, dst_name in specialized.items():
        src = v14_2_agg / src_name
        if src.exists():
            shutil.copy2(src, global_agg / dst_name)

    update_manifest = {
        "updated_unix": time.time(),
        "v14_2_aggregate": str(v14_2_agg),
        "global_aggregate": str(global_agg),
        "policy": (
            "Preserve V14/V14.1 evidence; add V14.2 as an explicit canonical 96-seed main-grid "
            "registry block and copy V14.2-specific audits under v14_2_* names."
        ),
        "v14_2_manifest_sha256": _sha256(manifest_path),
    }
    (global_agg / "v14_2_global_update_manifest.json").write_text(
        json.dumps(update_manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

    status_path = global_agg / "aggregate_status.md"
    text = status_path.read_text(encoding="utf-8") if status_path.exists() else "# V14 aggregate status\n"
    begin = "\n<!-- V14.2 MAIN GRID BEGIN -->\n"
    end = "<!-- V14.2 MAIN GRID END -->\n"
    if begin in text and end in text:
        text = text.split(begin, 1)[0] + text.split(end, 1)[1]
    ext = (v14_2_agg / "aggregate_status.md").read_text(encoding="utf-8")
    status_path.write_text(text.rstrip() + begin + ext.rstrip() + "\n" + end, encoding="utf-8")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--v14-exploration", type=Path, default=DEFAULT_V14_EXPLORATION)
    p.add_argument("--engine", type=Path, default=DEFAULT_ENGINE)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--source-runs", type=Path, default=None,
                   help="Optional override for V14 E76 runs.csv; default is <output-root>/V14_E76_public_interactions/runs.csv")
    p.add_argument("--aggregate-dir", type=Path, default=None,
                   help="Default: <output-root>/aggregate_v14_2")
    p.add_argument("--global-aggregate-dir", type=Path, default=None,
                   help="Default: <output-root>/aggregate")
    p.add_argument("--device", default="auto")
    p.add_argument("--devices", default=None,
                   help="Comma-separated devices for dynamic shard scheduling, e.g. cuda:0,cuda:1,cuda:2")
    p.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    p.add_argument("--seed-shard-size", type=int, default=16)
    p.add_argument("--recover-partial", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--overwrite", action="store_true",
                   help="Delete only the V14.2 target block before running; source V14 evidence is never deleted")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--shard-order", choices=("forward", "reverse"), default="forward")
    p.add_argument("--shard-indices", default=None,
                   help="Optional 1-based comma-separated shard indices")
    p.add_argument("--max-shards", type=int, default=None)
    p.add_argument("--defer-finalize", action="store_true")

    p.add_argument("--self-test", action="store_true")
    p.add_argument("--engine-self-test", action="store_true")
    p.add_argument("--list", action="store_true")
    p.add_argument("--preflight", action="store_true")
    p.add_argument("--run", default=None, help="Use 'all', 'E85', or exact block name")
    p.add_argument("--aggregate", action="store_true")
    p.add_argument("--update-global-aggregate", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    v14 = load_v14_exploration(args.v14_exploration)
    block, source_to_target, source = build_main_grid(v14)
    agg = args.aggregate_dir or (args.output_root / "aggregate_v14_2")
    global_agg = args.global_aggregate_dir or (args.output_root / "aggregate")

    if args.self_test:
        print(json.dumps(run_static_self_tests(v14, block, source_to_target, source), indent=2, sort_keys=True))
        return
    if args.engine_self_test:
        cmd = [sys.executable, str(args.engine), "--self-test"]
        print("$", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
        return
    if args.list:
        print(json.dumps(registry_summary(v14, block, source_to_target, source), indent=2, sort_keys=True))
        return
    if args.preflight:
        print(json.dumps(preflight_audit(
            v14=v14,
            block=block,
            source_to_target=source_to_target,
            source=source,
            output_root=args.output_root,
            source_runs_override=args.source_runs,
            seed_shard_size=args.seed_shard_size,
        ), indent=2, sort_keys=True))
        return
    if args.run:
        if args.run not in ("all", "E85", TARGET_BLOCK):
            raise ValueError(f"Unknown --run {args.run!r}")
        execute_block(
            v14=v14,
            block=block,
            source_to_target=source_to_target,
            source=source,
            output_root=args.output_root,
            source_runs_override=args.source_runs,
            engine=args.engine,
            device=args.device,
            devices=(args.devices.split(",") if args.devices else None),
            dtype=args.dtype,
            seed_shard_size=args.seed_shard_size,
            smoke=args.smoke,
            overwrite=args.overwrite,
            recover_partial=args.recover_partial,
            shard_order=args.shard_order,
            shard_indices=args.shard_indices,
            max_shards=args.max_shards,
            defer_finalize=args.defer_finalize,
        )
    if args.aggregate:
        aggregate(v14, block, args.output_root, agg)
        print(f"V14.2 aggregate written to {agg}", flush=True)
    if args.update_global_aggregate:
        update_global_aggregate(agg, global_agg)
        print(f"Global V14 aggregate updated in {global_agg}", flush=True)
    if not any((args.run, args.aggregate, args.update_global_aggregate)):
        raise SystemExit(
            "Choose --self-test, --engine-self-test, --list, --preflight, --run ..., "
            "--aggregate, or --update-global-aggregate"
        )


if __name__ == "__main__":
    main()
