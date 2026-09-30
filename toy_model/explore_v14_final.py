#!/usr/bin/env python3
"""Final v14 preregistered exploration, aggregation, and draft-plot driver.

This suite is designed to close the toy-model program for the ICLR paper.  It
repeats the three paper axes on the final v14 core, adds high-resolution causal
experiments for the unresolved v13.2 questions, and keeps historical controls
only as explicitly labelled appendix ablations.

Paper-facing core
=================
Task/model controls:
  rho_phi : feature overlap / interference
  lambda  : initial-policy concentration
  c       : privileged-context behavioral displacement

Method choices:
  trajectory policy (student <-> teacher mixture)
  KL direction (forward/reverse)
  EMA teacher update coefficient alpha

Fixed core construction:
  rho_W=0, beta=0, projected reachable two-mode target (.30,.15), no target
  heterogeneity, no context noise, full off-path context, no private features,
  no anchoring/holdout/corridor, H=4,K=8,D=64.  Adam is the practice-facing
  optimizer; matched-scale SGD is an appendix mechanistic control.

Scientific standards
====================
* Exact occupancy is primary; sampled occupancy is a numerical robustness test.
* Main plots use 96 paired task seeds.  Appendix mechanisms use 64 and purely
  technical checks use 48.
* Alpha and learning-rate grids are log-spaced where a timescale/scale is the
  scientific quantity.  Teacher fraction and c use linear grids.  rho_phi uses
  an angle-spaced cosine grid so equal increments correspond to equal rotations
  in representation geometry.
* No single hidden score selects conclusions.  Acquisition, forgetting,
  target-path learning, teacher utility, gradient geometry, and drawdown are
  retained separately.
* "Best EMA" paper comparisons are cross-fitted across task seeds: one fold
  selects alpha and the other evaluates it, then roles are swapped.  This avoids
  selecting and testing an optimum on the same seeds.
* Matched-learning KL comparisons use per-seed Pareto frontiers over the dense
  LR sweep rather than a single cherry-picked LR.
* All high-level aggregates are generated without materializing a giant
  all_runs.csv.  Per-block runs.csv files remain canonical raw evidence.

Blocks
======
E70  final-core three-axis reproduction / interaction anchor
E71  high-resolution trajectory surface, including optimal-EMA trajectory plots
E72  dense KL x LR x alpha frontiers + local gradient geometry
E73  dense c x alpha coupling phase diagram for both KLs and rollout endpoints
E74  iso-JSD causal mediation of initial-policy concentration
E75  source-independent state-dependent teacher-competence intervention
E76  rho_phi x lambda x c interactions
E77  Gaussian context-quality and off-path availability ablations
E78  rho_W, beta, and target-shape appendix causal interventions
E79  near-zero-support / novel-branch coverage stress
E80  endpoint-compatible context + alpha->1 utility-erosion stress
E81  matched-scale Adam/SGD optimizer interaction
E82  structural (H/K/D), duration, and exact/sampled estimator robustness

Primary run budget is intentionally large: this is the final evidence suite, not
screening.  Use --list to inspect exact counts before launch.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ENGINE = "toy_model/contextual_sd_toy_v14.py"

BASE_H = 4
BASE_K = 8
BASE_D = 64
BASE_STEPS = 200
BASE_ADAM_LR = 1e-3
BASE_SGD_LR = 5e-2

N_MAIN = 96
N_APPENDIX = 64
N_TECH = 48

# Equal spacing in log(alpha) after the exact frozen boundary alpha=0.
EMA_FINE = (
    0.0,
    0.000625, 0.000883883476, 0.00125, 0.001767766953,
    0.0025, 0.003535533906, 0.005, 0.007071067812,
    0.01, 0.014142135624, 0.02, 0.028284271247,
    0.04, 0.056568542495, 0.08, 0.113137084990, 0.16,
)
EMA_MATCHED = (0.0, 0.00125, 0.0025, 0.005, 0.01, 0.02, 0.04)
EMA_CAUSAL = (0.0, 0.00125, 0.0025, 0.005, 0.01)
EMA_ANCHORS = (0.0, 0.0025, 0.02)
EMA_STRESS = EMA_FINE + (0.32, 0.64, 1.0)

# Equal log2 half-steps around the historical reference lambda=2.5.
LAMBDA_FINE = (
    0.625, 0.883883476, 1.25, 1.767766953,
    2.5, 3.535533906, 5.0,
)
LAMBDA_CAUSAL = (0.625, 1.25, 2.5, 3.535533906, 5.0)
LAMBDA_APPENDIX = (1.0, 2.5, 5.0)

CONTEXT_FINE = tuple(round(x / 10.0, 10) for x in range(0, 11))
CONTEXT_MAIN = (0.30, 0.60, 0.90)
TEACHER_FRACTIONS = tuple(round(x / 10.0, 10) for x in range(0, 11))
TEACHER_FRACTIONS_APPENDIX = (0.0, 0.25, 0.50, 0.75, 1.0)

# Equal angle increments 0,15,...,90 degrees in representation geometry.
RHO_PHI_ANGLE = tuple(round(math.cos(math.radians(a)), 12) for a in (90, 75, 60, 45, 30, 15, 0))
RHO_PHI_MAIN = (round(math.cos(math.radians(75)), 12), 0.5, round(math.cos(math.radians(30)), 12))

LR_FINE = (
    0.00025, 0.000353553391, 0.0005, 0.000707106781,
    0.001, 0.001414213562, 0.002, 0.002828427125, 0.004,
)

COMPETENCE_BIAS = (-1.0, -0.75, -0.50, -0.25, 0.0, 0.25, 0.50, 0.75, 1.0)
NOISE_LEVELS = (0.0, 0.125, 0.25, 0.50, 1.0, 2.0)
OFFPATH_LEVELS = (0.0, 0.10, 0.25, 0.50, 0.75, 1.0)
READOUT_LEVELS = (-0.90, -math.sqrt(0.5), -0.50, 0.0, 0.50, math.sqrt(0.5), 0.90)
BETA_LEVELS = (0.0, 0.125, 0.25, 0.50, 1.0, 2.0, 4.0)
ISO_JSD_TARGETS = (0.005, 0.015, 0.030, 0.050)


@dataclass(frozen=True)
class TargetShape:
    label: str
    m1: float
    m2: float

    @property
    def entropy(self) -> float:
        rest = (1.0 - self.m1 - self.m2) / (BASE_K - 2)
        p = np.array([self.m1, self.m2] + [rest] * (BASE_K - 2), dtype=float)
        return float(-(p * np.log(p)).sum())


TARGET_SHAPES = (
    TargetShape("soft", 0.20, 0.15),
    TargetShape("mild", 0.24, 0.12),
    TargetShape("historical", 0.30, 0.15),
    TargetShape("concentrated", 0.40, 0.18),
    TargetShape("sharp", 0.50, 0.20),
    TargetShape("entropy_match_skew", 0.31, 0.12),
    TargetShape("entropy_match_symmetric", 0.24, 0.24),
)


@dataclass
class Block:
    name: str
    question: str
    prediction: str
    tier: str
    n_seeds: int
    seed_start: int
    cells: List[Dict[str, object]]
    sampled_trajectories: int = 256
    default_steps: int = BASE_STEPS

    @property
    def n_runs(self) -> int:
        return len(self.cells) * self.n_seeds


BLOCKS: List[Block] = []


def core(rho_phi: float = 0.50, lam: float = 2.50, c: float = 0.60) -> Dict[str, float]:
    return {
        "feature_overlap": float(rho_phi),
        "initial_policy_concentration": float(lam),
        "context_strength": float(c),
    }


def ablation(**updates: object) -> Dict[str, object]:
    out: Dict[str, object] = {
        "readout_compatibility": 0.0,
        "support_placement_bias": 0.0,
        "target_primary_mass": 0.30,
        "target_secondary_mass": 0.15,
        "endpoint_compatible_context": False,
        "feature_noise_scale": 0.0,
        "behavior_noise_scale": 0.0,
        "offpath_context_multiplier": 1.0,
        "state_competence_bias": 0.0,
        "target_initial_jsd": None,
    }
    out.update(updates)
    return out


def cell(name: str, block: str, *,
         rho_phi: float = 0.50, lam: float = 2.50, c: float = 0.60,
         ab: Optional[Mapping[str, object]] = None,
         horizon: int = BASE_H, vocab_size: int = BASE_K, feature_dim: int = BASE_D,
         kl: str = "reverse", rollout: str = "student", teacher_fraction: float = 0.5,
         alpha: float = 0.0025, lr: float = BASE_ADAM_LR,
         optimizer: str = "adam", estimator: str = "exact",
         train_steps: int = BASE_STEPS, record_history: bool = False,
         variant: Optional[str] = None) -> Dict[str, object]:
    return {
        "name": name,
        "analysis_block": block,
        "variant": variant or name,
        "regime_name": "v14_final",
        "core": core(rho_phi, lam, c),
        "ablation": dict(ablation() if ab is None else ab),
        "structure": {
            "horizon": int(horizon),
            "vocab_size": int(vocab_size),
            "feature_dim": int(feature_dim),
        },
        "kl_direction": kl,
        "rollout_source": rollout,
        "teacher_fraction": float(teacher_fraction),
        "ema_alpha": float(alpha),
        "learning_rate": float(lr),
        "optimizer": optimizer,
        "estimator": estimator,
        "train_steps": int(train_steps),
        "record_history": bool(record_history),
    }


def rollout_name(fraction: float) -> str:
    if fraction == 0.0:
        return "student"
    if fraction == 1.0:
        return "teacher"
    return "mixture"


def register(block: Block) -> None:
    if any(b.name == block.name for b in BLOCKS):
        raise ValueError(f"duplicate block {block.name}")
    BLOCKS.append(block)


# -----------------------------------------------------------------------------
# Registry
# -----------------------------------------------------------------------------


# E70: broad final-core anchor.  Independent replication of the v13.2 core on
# the *new physically clean v14 interface* before any new causal claim is used.
e70: List[Dict[str, object]] = []
for lam in LAMBDA_FINE:
    for cval in CONTEXT_MAIN:
        for rp in RHO_PHI_MAIN:
            for kl in ("forward", "reverse"):
                for ro in ("student", "teacher"):
                    for a in EMA_ANCHORS:
                        e70.append(cell(
                            f"e70_l{lam:g}_c{cval:g}_p{rp:g}_{kl}_{ro}_a{a:g}",
                            "V14_E70_core_reproduction", rho_phi=rp, lam=lam, c=cval,
                            kl=kl, rollout=ro, alpha=a,
                        ))
register(Block(
    "V14_E70_core_reproduction",
    "Does the physically clean v14 core reproduce the complete v13.2 three-axis behavior over lambda, c and rho_phi?",
    "All v13.2 paper-facing qualitative mechanisms should survive with rho_W=beta=0 and fixed target masses; absolute endpoints may move only through the declared final rebaseline.",
    "main", N_MAIN, 140_000, e70,
))


# E71: trajectory paper surface.  Fine alpha grid is crossed rather than using
# a historically chosen optimum, enabling cross-fitted optimal-EMA plots.
e71: List[Dict[str, object]] = []
for lam in LAMBDA_FINE:
    for cval in CONTEXT_MAIN:
        for kl in ("forward", "reverse"):
            for frac in TEACHER_FRACTIONS:
                ro = rollout_name(frac)
                for a in EMA_FINE:
                    hist = (lam == 2.5 and cval == 0.60 and kl == "reverse" and frac in (0.0, 0.5, 1.0) and a in EMA_ANCHORS)
                    e71.append(cell(
                        f"e71_l{lam:g}_c{cval:g}_{kl}_t{frac:g}_a{a:g}",
                        "V14_E71_trajectory_surface", lam=lam, c=cval, kl=kl,
                        rollout=ro, teacher_fraction=frac, alpha=a,
                        record_history=hist,
                    ))
register(Block(
    "V14_E71_trajectory_surface",
    "How does the student-to-teacher trajectory frontier evolve with initial-policy concentration and context strength after coupling is optimized rather than fixed?",
    "Teacher advantage should grow with lambda in Adam; state selection should remain conditional rather than universally ordered.  Cross-fitted alpha selection must not create a spurious endpoint ranking.",
    "main", N_MAIN, 141_000, e71,
))


# E72: dense KL/LR frontier.  This is the exact toy analogue of the paper's
# matched-learning LLM comparison and local prefix-gradient bridge.
e72: List[Dict[str, object]] = []
for lam in LAMBDA_FINE:
    for cval in CONTEXT_MAIN:
        for ro in ("student", "teacher"):
            for a in EMA_MATCHED:
                for lr in LR_FINE:
                    for kl in ("forward", "reverse"):
                        e72.append(cell(
                            f"e72_l{lam:g}_c{cval:g}_{ro}_a{a:g}_lr{lr:.9g}_{kl}",
                            "V14_E72_kl_matched_learning", lam=lam, c=cval,
                            kl=kl, rollout=ro, alpha=a, lr=lr,
                        ))
register(Block(
    "V14_E72_kl_matched_learning",
    "At matched achieved acquisition, does reverse KL retain better, and does the F/R gap track measured local distribution/gradient geometry?",
    "Increasing lambda should enlarge JSD/gradient disagreement and the forward-learning/reverse-retention separation.  Pareto-frontier matching across LR must preserve a reverse retention advantage in the controlled regime.",
    "main", N_MAIN, 142_000, e72,
))


# E73: coupling phase diagram.  c is linear; alpha is log-spaced.
e73: List[Dict[str, object]] = []
for lam in LAMBDA_FINE:
    for cval in CONTEXT_FINE:
        for kl in ("forward", "reverse"):
            for ro in ("student", "teacher"):
                for a in EMA_FINE:
                    hist = (lam == 2.5 and cval in (0.20, 0.60, 1.0) and kl == "reverse" and ro == "student")
                    e73.append(cell(
                        f"e73_l{lam:g}_c{cval:g}_{kl}_{ro}_a{a:g}",
                        "V14_E73_coupling_phase", lam=lam, c=cval, kl=kl,
                        rollout=ro, alpha=a, record_history=hist,
                    ))
register(Block(
    "V14_E73_coupling_phase",
    "What is the final high-resolution teacher-timescale phase diagram, and how do c and lambda move the useful EMA window?",
    "A finite alpha>0 acquisition optimum should persist; stronger c should move the useful teacher more slowly, while fast tracking should erode teacher utility and eventually acquisition.",
    "main", N_MAIN, 143_000, e73,
))


# E74: explicit per-seed iso-JSD intervention.  This is stronger than merely
# post-hoc correlating lambda with JSD.
e74: List[Dict[str, object]] = []
for lam in LAMBDA_FINE:
    for jsd in ISO_JSD_TARGETS:
        ab = ablation(target_initial_jsd=jsd)
        for kl in ("forward", "reverse"):
            for ro in ("student", "teacher"):
                for a in (0.0, 0.0025, 0.005):
                    e74.append(cell(
                        f"e74_l{lam:g}_j{jsd:g}_{kl}_{ro}_a{a:g}",
                        "V14_E74_iso_jsd", lam=lam, c=0.60, ab=ab,
                        kl=kl, rollout=ro, alpha=a,
                    ))
register(Block(
    "V14_E74_iso_jsd",
    "Is lambda's effect mediated by the initial student/teacher policy distance, or does concentration matter after initial JSD is held fixed per seed?",
    "If F/R contrasts collapse across lambda at fixed JSD, distributional mismatch is the mediator; a residual lambda trend would identify a concentration/curvature effect beyond distance.",
    "causal", N_MAIN, 144_000, e74,
))


# E75: state-dependent teacher competence.  Positive bias selectively weakens
# privileged correction on student-heavy states, negative is the symmetric
# teacher-heavy control.  No intervention conditions on the selected rollout.
e75: List[Dict[str, object]] = []
for lam in LAMBDA_CAUSAL:
    for bias in COMPETENCE_BIAS:
        ab = ablation(state_competence_bias=bias)
        for kl in ("forward", "reverse"):
            for frac in TEACHER_FRACTIONS:
                for a in EMA_CAUSAL:
                    e75.append(cell(
                        f"e75_l{lam:g}_b{bias:+g}_{kl}_t{frac:g}_a{a:g}",
                        "V14_E75_state_competence", lam=lam, ab=ab, kl=kl,
                        rollout=rollout_name(frac), teacher_fraction=frac, alpha=a,
                    ))
register(Block(
    "V14_E75_state_competence",
    "Can state-dependent contextual-teacher competence reconcile deployment alignment with teacher-rollout rescue without a source-dependent hack?",
    "Weakening context specifically on student-heavy states should reduce student-trajectory transfer and may produce the missing easy/hard rollout reversal; the symmetric teacher-heavy control is required to establish causal direction.",
    "causal", N_MAIN, 145_000, e75,
))


# E76: final interaction audit for the three public task controls.
e76: List[Dict[str, object]] = []
for rp in RHO_PHI_ANGLE:
    for lam in LAMBDA_FINE:
        for cval in CONTEXT_MAIN:
            for kl in ("forward", "reverse"):
                for ro in ("student", "teacher"):
                    for a in EMA_CAUSAL:
                        e76.append(cell(
                            f"e76_p{rp:g}_l{lam:g}_c{cval:g}_{kl}_{ro}_a{a:g}",
                            "V14_E76_public_interactions", rho_phi=rp, lam=lam, c=cval,
                            kl=kl, rollout=ro, alpha=a,
                        ))
register(Block(
    "V14_E76_public_interactions",
    "Do rho_phi, lambda and c retain their dominant roles when crossed, or do interactions overturn the paper's causal factorization?",
    "rho_phi should predominantly scale forgetting/interference, lambda KL/rollout sensitivity, and c teacher timescale.  Secondary interactions may alter magnitudes but should not require a fourth core task parameter.",
    "causal", N_APPENDIX, 146_000, e76,
))


# E77a: two orthogonalized Gaussian context-corruption interventions.
e77a: List[Dict[str, object]] = []
for noise_type in ("feature", "behavior"):
    for scale in NOISE_LEVELS:
        if noise_type == "behavior" and scale == 0.0:
            continue  # shared zero-noise baseline is declared once in the feature branch
        for cval in CONTEXT_MAIN:
            for a in EMA_FINE:
                ab = ablation(**({"feature_noise_scale": scale} if noise_type == "feature" else {"behavior_noise_scale": scale}))
                e77a.append(cell(
                    f"e77a_{noise_type}_n{scale:g}_c{cval:g}_a{a:g}",
                    "V14_E77a_context_noise", c=cval, ab=ab,
                    kl="reverse", rollout="student", alpha=a,
                ))
register(Block(
    "V14_E77a_context_noise",
    "Does context quality/noise alter learnability without replacing c as the principal teacher-timescale controller?",
    "Noise should reduce teacher utility and attainable acquisition; the c-ordering of the stable EMA window should remain more systematic than generic corruption level.",
    "appendix", N_APPENDIX, 147_000, e77a,
))

# E77b: clean off-target-prefix availability / coverage-transfer ablation.
e77b: List[Dict[str, object]] = []
for off in OFFPATH_LEVELS:
    for lam in LAMBDA_APPENDIX:
        ab = ablation(offpath_context_multiplier=off)
        for frac in TEACHER_FRACTIONS:
            for a in EMA_CAUSAL:
                e77b.append(cell(
                    f"e77b_o{off:g}_l{lam:g}_t{frac:g}_a{a:g}",
                    "V14_E77b_offpath_availability", lam=lam, ab=ab,
                    kl="reverse", rollout=rollout_name(frac),
                    teacher_fraction=frac, alpha=a,
                ))
register(Block(
    "V14_E77b_offpath_availability",
    "How does reduced privileged usefulness away from target-consistent prefixes change trajectory transfer and the EMA frontier?",
    "Lower off-path availability should reduce broad transfer from poorly aligned trajectories; local target-path success need not imply deployment acquisition.",
    "appendix", N_APPENDIX, 148_000, e77b,
))


# E78: the three construction mechanisms deliberately excluded from the core.
# A single shared neutral baseline is declared per lambda/method cell, then each
# intervention contributes only non-neutral values.  This avoids duplicating an
# identical scientific cell under three different labels.
e78: List[Dict[str, object]] = []
for lam in LAMBDA_APPENDIX:
    for kl in ("forward", "reverse"):
        for ro in ("student", "teacher"):
            for a in EMA_CAUSAL:
                e78.append(cell(
                    f"e78_baseline_l{lam:g}_{kl}_{ro}_a{a:g}",
                    "V14_E78_core_excluded_ablation", lam=lam,
                    kl=kl, rollout=ro, alpha=a,
                    variant=f"kind=baseline|value=0|lambda={lam:g}",
                ))
for rw in READOUT_LEVELS:
    if abs(rw) < 1e-15:
        continue
    for lam in LAMBDA_APPENDIX:
        ab = ablation(readout_compatibility=rw)
        for kl in ("forward", "reverse"):
            for ro in ("student", "teacher"):
                for a in EMA_CAUSAL:
                    e78.append(cell(
                        f"e78_rhoW{rw:+.6g}_l{lam:g}_{kl}_{ro}_a{a:g}",
                        "V14_E78_core_excluded_ablation", lam=lam, ab=ab,
                        kl=kl, rollout=ro, alpha=a,
                        variant=f"kind=rhoW|value={rw:.12g}|lambda={lam:g}",
                    ))
for beta in BETA_LEVELS:
    if beta == 0.0:
        continue
    for lam in LAMBDA_APPENDIX:
        ab = ablation(support_placement_bias=beta)
        for kl in ("forward", "reverse"):
            for ro in ("student", "teacher"):
                for a in EMA_CAUSAL:
                    e78.append(cell(
                        f"e78_beta{beta:g}_l{lam:g}_{kl}_{ro}_a{a:g}",
                        "V14_E78_core_excluded_ablation", lam=lam, ab=ab,
                        kl=kl, rollout=ro, alpha=a,
                        variant=f"kind=beta|value={beta:g}|lambda={lam:g}",
                    ))
for shape in TARGET_SHAPES:
    if shape.label == "historical":
        continue
    for lam in LAMBDA_APPENDIX:
        ab = ablation(target_primary_mass=shape.m1, target_secondary_mass=shape.m2)
        for kl in ("forward", "reverse"):
            for ro in ("student", "teacher"):
                for a in EMA_CAUSAL:
                    e78.append(cell(
                        f"e78_shape_{shape.label}_l{lam:g}_{kl}_{ro}_a{a:g}",
                        "V14_E78_core_excluded_ablation", lam=lam, ab=ab,
                        kl=kl, rollout=ro, alpha=a,
                        variant=f"kind=target_shape|shape={shape.label}|entropy={shape.entropy:.9g}|lambda={lam:g}",
                    ))
register(Block(
    "V14_E78_core_excluded_ablation",
    "Do rho_W, beta or target shape provide a distinct causal explanation that would justify re-entering the core?",
    "They may modulate magnitude, but none should be necessary for the central Adam KL/coupling/rollout behavior.  Entropy-matched target skews should remain nearly equivalent.",
    "appendix", N_APPENDIX, 149_000, e78,
))


# E79: support novelty / previously unvisited behavior.  This is not claimed to
# be semantic representation learning; it tests coverage of low-p0 branches.
e79: List[Dict[str, object]] = []
for beta in (0.0, 0.25, 0.50, 1.0, 2.0, 4.0, 8.0):
    for rp in (0.0, RHO_PHI_MAIN[0], 0.50):
        for lam in LAMBDA_APPENDIX:
            ab = ablation(support_placement_bias=beta)
            for frac in TEACHER_FRACTIONS_APPENDIX:
                for a in (0.0, 0.0025, 0.01):
                    e79.append(cell(
                        f"e79_b{beta:g}_p{rp:g}_l{lam:g}_t{frac:g}_a{a:g}",
                        "V14_E79_novel_support", rho_phi=rp, lam=lam, ab=ab,
                        kl="reverse", rollout=rollout_name(frac),
                        teacher_fraction=frac, alpha=a,
                    ))
register(Block(
    "V14_E79_novel_support",
    "What happens when the desired behavior is concentrated in states/actions with extremely weak initial support?",
    "Low-support targets should expose coverage/trajectory effects without using exact zero probabilities.  Results are a support-novelty stress, not a claim of semantic knowledge acquisition.",
    "appendix", N_APPENDIX, 150_000, e79,
))


# E80: endpoint-compatible and alpha->1 closure controls.
e80: List[Dict[str, object]] = []
for compat in (False, True):
    for cval in CONTEXT_MAIN:
        ab = ablation(endpoint_compatible_context=compat)
        for kl in ("forward", "reverse"):
            for a in EMA_STRESS:
                e80.append(cell(
                    f"e80_compat{int(compat)}_c{cval:g}_{kl}_a{a:g}",
                    "V14_E80_endpoint_high_alpha", c=cval, ab=ab,
                    kl=kl, rollout="student", alpha=a,
                    record_history=(a in (0.0, 0.0025, 0.02, 0.16, 1.0)),
                ))
# Duration dependence only for stress points, excluding 200 because above has it.
for compat in (False, True):
    ab = ablation(endpoint_compatible_context=compat)
    for a in (0.02, 0.08, 0.32, 1.0):
        for steps in (100, 400, 800):
            e80.append(cell(
                f"e80_duration_compat{int(compat)}_a{a:g}_s{steps}",
                "V14_E80_endpoint_high_alpha", ab=ab,
                kl="reverse", rollout="student", alpha=a,
                train_steps=steps, record_history=True,
                variant=f"duration|compat={int(compat)}|alpha={a:g}|steps={steps}",
            ))
register(Block(
    "V14_E80_endpoint_high_alpha",
    "Does the finite EMA window and high-alpha utility erosion survive a jointly internalizable context in the final model, including exact synchronization and longer horizons in optimization time?",
    "The useful finite window should survive endpoint compatibility; alpha->1 should yield teacher-utility erosion/saturation rather than a claimed positive-gain runaway.",
    "appendix", N_APPENDIX, 151_000, e80,
))


# E81: optimizer is an appendix interaction, never selected by preferred sign.
e81: List[Dict[str, object]] = []
for opt, lr in (("adam", BASE_ADAM_LR), ("sgd", BASE_SGD_LR)):
    for lam in LAMBDA_APPENDIX:
        for cval in CONTEXT_MAIN:
            for kl in ("forward", "reverse"):
                for frac in TEACHER_FRACTIONS:
                    for a in EMA_CAUSAL:
                        e81.append(cell(
                            f"e81_{opt}_l{lam:g}_c{cval:g}_{kl}_t{frac:g}_a{a:g}",
                            "V14_E81_optimizer", lam=lam, c=cval, kl=kl,
                            rollout=rollout_name(frac), teacher_fraction=frac,
                            alpha=a, optimizer=opt, lr=lr,
                        ))
register(Block(
    "V14_E81_optimizer",
    "Does matched-scale optimizer preconditioning alter rollout/KL ordering while leaving the central EMA mechanism intact?",
    "Adam should remain practice-facing; SGD may recover student-acquisition/teacher-retention rollout ordering.  Optimizer-dependent signs must be reported as interactions, not used to select the core.",
    "appendix", N_APPENDIX, 152_000, e81,
))


# E82a/b: structural and optimization-duration robustness.
e82a: List[Dict[str, object]] = []
for h in (3, 4, 5):
    for lam in LAMBDA_APPENDIX:
        for kl in ("forward", "reverse"):
            for ro in ("student", "teacher"):
                for a in EMA_CAUSAL:
                    e82a.append(cell(
                        f"e82a_h{h}_l{lam:g}_{kl}_{ro}_a{a:g}",
                        "V14_E82a_horizon", horizon=h, lam=lam,
                        kl=kl, rollout=ro, alpha=a,
                    ))
register(Block(
    "V14_E82a_horizon", "Are final mechanisms robust to H=3,4,5?",
    "Qualitative axis behavior should survive sequence depth; numerical optima are not claimed to transfer to LLMs.",
    "technical", N_TECH, 153_000, e82a,
))

e82b: List[Dict[str, object]] = []
for steps in (100, 200, 400, 800):
    for lam in LAMBDA_APPENDIX:
        for kl in ("forward", "reverse"):
            for ro in ("student", "teacher"):
                for a in EMA_CAUSAL:
                    e82b.append(cell(
                        f"e82b_s{steps}_l{lam:g}_{kl}_{ro}_a{a:g}",
                        "V14_E82b_duration", lam=lam, kl=kl, rollout=ro,
                        alpha=a, train_steps=steps,
                    ))
register(Block(
    "V14_E82b_duration", "Are conclusions an artifact of the 200-step horizon?",
    "Signs and ordering should persist across 100/200/400/800 steps even if the finite-grid optimum drifts modestly.",
    "technical", N_TECH, 154_000, e82b,
))

# E82c: estimator replication in separate blocks so sample count is a runner
# parameter rather than a hidden per-cell field.
def estimator_cells(block_name: str, estimator: str) -> List[Dict[str, object]]:
    return [
        cell(
            f"{block_name}_{kl}_{ro}_a{a:g}", block_name,
            kl=kl, rollout=ro, alpha=a, estimator=estimator,
        )
        for kl in ("forward", "reverse")
        for ro in ("student", "teacher")
        for a in EMA_ANCHORS
    ]

register(Block(
    "V14_E82c_exact", "Exact occupancy numerical reference.",
    "Sampled estimators should converge toward this reference without changing the main qualitative contrasts.",
    "technical", N_TECH, 155_000, estimator_cells("V14_E82c_exact", "exact"), sampled_trajectories=256,
))
for suffix, ntraj, seed in (("sample64", 64, 156_000), ("sample256", 256, 157_000), ("sample1024", 1024, 158_000)):
    name = f"V14_E82c_{suffix}"
    register(Block(
        name, f"Sampled occupancy robustness with {ntraj} trajectories/step.",
        "Estimator bias/variance should shrink with sample count; exact occupancy remains primary.",
        "technical", N_TECH, seed, estimator_cells(name, "sampled"), sampled_trajectories=ntraj,
    ))

# E82d/e: structural robustness in branching factor K and feature dimension D.
# These are deliberately one-factor-at-a-time checks around H=4,K=8,D=64.
# K is an effective local branching factor, not a tokenizer vocabulary.  D is
# an effective fixed-feature/readout capacity check; because the projected target
# is regenerated in each hypothesis class, it must not be interpreted as a pure
# causal LLM model-size experiment.
STRUCTURAL_K = (6, 8, 12)
STRUCTURAL_D = (32, 64, 128)

e82d: List[Dict[str, object]] = []
for k in STRUCTURAL_K:
    for lam in LAMBDA_APPENDIX:
        for kl in ("forward", "reverse"):
            for ro in ("student", "teacher"):
                for a in EMA_ANCHORS:
                    e82d.append(cell(
                        f"e82d_k{k}_l{lam:g}_{kl}_{ro}_a{a:g}",
                        "V14_E82d_vocab", vocab_size=k, lam=lam,
                        kl=kl, rollout=ro, alpha=a,
                    ))
register(Block(
    "V14_E82d_vocab",
    "Are the final three-axis mechanisms robust to the effective local branching factor K?",
    "Headline KL/rollout/EMA signs should survive K=6,8,12.  K is a structural branching-factor robustness check, not literal tokenizer vocabulary.",
    "technical", N_TECH, 159_000, e82d,
))

e82e: List[Dict[str, object]] = []
for dval in STRUCTURAL_D:
    for lam in LAMBDA_APPENDIX:
        for kl in ("forward", "reverse"):
            for ro in ("student", "teacher"):
                for a in EMA_ANCHORS:
                    e82e.append(cell(
                        f"e82e_d{dval}_l{lam:g}_{kl}_{ro}_a{a:g}",
                        "V14_E82e_feature_dim", feature_dim=dval, lam=lam,
                        kl=kl, rollout=ro, alpha=a,
                    ))
register(Block(
    "V14_E82e_feature_dim",
    "Are the final mechanisms robust to fixed-feature/readout dimension D?",
    "Headline signs should survive D=32,64,128.  This tests effective linear capacity/structural robustness; it is not a pure model-scale causal claim because the projected reachable target is regenerated at each D.",
    "technical", N_TECH, 160_000, e82e,
))


# -----------------------------------------------------------------------------
# Registry validation
# -----------------------------------------------------------------------------


def _canonical_cell(cell_: Mapping[str, object]) -> str:
    return json.dumps(cell_, sort_keys=True, separators=(",", ":"), default=str)


def registry_summary() -> Dict[str, object]:
    by_tier: Dict[str, Dict[str, int]] = {}
    for b in BLOCKS:
        t = by_tier.setdefault(b.tier, {"blocks": 0, "cells": 0, "runs": 0})
        t["blocks"] += 1
        t["cells"] += len(b.cells)
        t["runs"] += b.n_runs
    return {
        "n_blocks": len(BLOCKS),
        "n_cells": sum(len(b.cells) for b in BLOCKS),
        "n_per_seed_runs": sum(b.n_runs for b in BLOCKS),
        "by_tier": by_tier,
    }


def run_static_self_tests() -> Dict[str, object]:
    names: set[str] = set()
    scientific: set[str] = set()
    seed_intervals: List[Tuple[int, int, str]] = []
    for b in BLOCKS:
        if not b.cells:
            raise AssertionError(f"empty block: {b.name}")
        lo, hi = b.seed_start, b.seed_start + b.n_seeds - 1
        seed_intervals.append((lo, hi, b.name))
        local_names: set[str] = set()
        for c in b.cells:
            n = str(c["name"])
            if n in names or n in local_names:
                raise AssertionError(f"duplicate cell name: {n}")
            names.add(n); local_names.add(n)
            # Scientific uniqueness ignores name/variant/recording flags.
            cc = dict(c)
            for k in ("name", "variant", "record_history"):
                cc.pop(k, None)
            key = _canonical_cell(cc)
            if key in scientific:
                raise AssertionError(f"duplicate scientific cell: {n}")
            scientific.add(key)
            core_ = c["core"]
            ab_ = c["ablation"]
            structure_ = c["structure"]
            if set(core_) != {"feature_overlap", "initial_policy_concentration", "context_strength"}:
                raise AssertionError(f"historical field leaked into v14 core: {core_}")
            if set(structure_) != {"horizon", "vocab_size", "feature_dim"}:
                raise AssertionError(f"unexpected v14 structural surface: {structure_}")
            if int(structure_["horizon"]) < 2 or int(structure_["vocab_size"]) < 4 or int(structure_["feature_dim"]) < 2:
                raise AssertionError(f"invalid structural cell: {structure_}")
            forbidden = {"target_complexity", "context_capability", "context_noise_std", "context_anchoring", "prompt_holdout_fraction", "state_private_fraction", "teacher_construction"}
            if forbidden & set(c):
                raise AssertionError(f"historical field leaked into cell surface: {forbidden & set(c)}")
            if float(ab_["feature_noise_scale"]) > 0 and float(ab_["behavior_noise_scale"]) > 0:
                raise AssertionError("two Gaussian noise interventions enabled together")
    # Distinct scientific blocks must use independent task-seed ranges.
    for i, (lo1, hi1, n1) in enumerate(seed_intervals):
        for lo2, hi2, n2 in seed_intervals[i+1:]:
            if not (hi1 < lo2 or hi2 < lo1):
                raise AssertionError(f"seed ranges overlap: {n1} and {n2}")

    if not np.all(np.diff(np.log(np.asarray(EMA_FINE[1:]))) > 0):
        raise AssertionError("EMA_FINE positive values are not strictly log ordered")
    ratios = np.asarray(EMA_FINE[2:]) / np.asarray(EMA_FINE[1:-1])
    if np.max(np.abs(ratios - math.sqrt(2))) > 5e-4:
        raise AssertionError("EMA_FINE is not sqrt(2)-spaced")
    lratios = np.asarray(LR_FINE[1:]) / np.asarray(LR_FINE[:-1])
    if np.max(np.abs(lratios - math.sqrt(2))) > 5e-6:
        raise AssertionError("LR_FINE is not sqrt(2)-spaced")
    # Lambda is sqrt(2)-spaced around 2.5.
    lamratios = np.asarray(LAMBDA_FINE[1:]) / np.asarray(LAMBDA_FINE[:-1])
    if np.max(np.abs(lamratios - math.sqrt(2))) > 5e-6:
        raise AssertionError("LAMBDA_FINE is not sqrt(2)-spaced")
    if 0.5 not in RHO_PHI_ANGLE:
        raise AssertionError("angle-spaced rho grid must contain reference rho_phi=.5")

    # Required closure experiments.
    required = {
        "V14_E74_iso_jsd", "V14_E75_state_competence",
        "V14_E80_endpoint_high_alpha", "V14_E81_optimizer",
        "V14_E82c_exact", "V14_E82c_sample1024",
        "V14_E82d_vocab", "V14_E82e_feature_dim",
    }
    if not required.issubset({b.name for b in BLOCKS}):
        raise AssertionError(f"missing closure blocks: {required - {b.name for b in BLOCKS}}")

    # Structural robustness must contain the declared one-factor grids.
    e82d_b = next(b for b in BLOCKS if b.name == "V14_E82d_vocab")
    if sorted({int(c["structure"]["vocab_size"]) for c in e82d_b.cells}) != list(STRUCTURAL_K):
        raise AssertionError("E82d K grid incomplete")
    if any(int(c["structure"]["horizon"]) != BASE_H or int(c["structure"]["feature_dim"]) != BASE_D for c in e82d_b.cells):
        raise AssertionError("E82d must vary only K")
    e82e_b = next(b for b in BLOCKS if b.name == "V14_E82e_feature_dim")
    if sorted({int(c["structure"]["feature_dim"]) for c in e82e_b.cells}) != list(STRUCTURAL_D):
        raise AssertionError("E82e D grid incomplete")
    if any(int(c["structure"]["horizon"]) != BASE_H or int(c["structure"]["vocab_size"]) != BASE_K for c in e82e_b.cells):
        raise AssertionError("E82e must vary only D")

    # E75 must contain symmetric competence controls and zero.
    e75b = next(b for b in BLOCKS if b.name == "V14_E75_state_competence")
    biases = sorted({float(c["ablation"]["state_competence_bias"]) for c in e75b.cells})
    if biases != sorted(COMPETENCE_BIAS):
        raise AssertionError("E75 competence grid incomplete")

    # Synthetic aggregation regression test: realized seed-dependent numeric
    # diagnostics must be treated as metrics, never first-row configuration.
    syn = pd.DataFrame({
        "cell_name": ["x", "x", "y", "y"],
        "task_seed": [1, 2, 1, 2],
        "learning_rate": [1e-3] * 4,
        "new_match_gain": [0.1, 0.2, 0.3, 0.4],
        "target_entropy": [1.1, 1.2, 1.3, 1.4],
        "readout_similarity_realized": [0.0, 0.1, 0.2, 0.3],
    })
    metrics = infer_metric_columns(syn)
    for col in ("new_match_gain", "target_entropy", "readout_similarity_realized"):
        if col not in metrics:
            raise AssertionError(f"aggregation regression: {col} not inferred as metric")
    if "train_steps" not in public_columns():
        raise AssertionError("aggregation regression: train_steps must be loaded for duration robustness")

    return {"status": "PASS", **registry_summary(), "n_unique_cells": len(names)}


# -----------------------------------------------------------------------------
# Execution
# -----------------------------------------------------------------------------


def selected_blocks(selector: str) -> List[Block]:
    if selector == "all":
        return list(BLOCKS)
    if selector == "paper":
        return [b for b in BLOCKS if b.tier in {"main", "causal"}]
    if selector in {"main", "causal", "appendix", "technical"}:
        return [b for b in BLOCKS if b.tier == selector]
    exact = [b for b in BLOCKS if b.name == selector]
    if exact:
        return exact
    raise ValueError(f"unknown --run {selector!r}")


def smoke_subset(block: Block) -> List[Dict[str, object]]:
    """Return a tiny but analysis-aware subset for end-to-end plumbing tests.

    Smoke runs are not inferential experiments.  Where an aggregate requires a
    paired motif (notably E72), include a coherent miniature motif so the smoke
    run exercises the corresponding aggregation path rather than merely random
    registry positions.
    """
    if len(block.cells) <= 8:
        return list(block.cells)

    if block.name == "V14_E72_kl_matched_learning":
        chosen: List[Dict[str, object]] = []
        # Geometry bridge: four lambda values, paired F/R, at the registered
        # base LR and one common (c, rollout, alpha) operating point.
        target_lams = (0.625, 1.25, 2.5, 5.0)
        for lam in target_lams:
            for kl in ("forward", "reverse"):
                hits = [c for c in block.cells if
                        math.isclose(float(c["core"]["initial_policy_concentration"]), lam, rel_tol=0, abs_tol=1e-9) and
                        math.isclose(float(c["core"]["context_strength"]), 0.6, rel_tol=0, abs_tol=1e-12) and
                        c["rollout_source"] == "student" and
                        math.isclose(float(c["ema_alpha"]), 0.0025, rel_tol=0, abs_tol=1e-12) and
                        math.isclose(float(c["learning_rate"]), BASE_ADAM_LR, rel_tol=0, abs_tol=1e-12) and
                        c["kl_direction"] == kl]
                if hits:
                    chosen.append(hits[0])
        # Matched-learning plumbing: add low/high LR endpoints for the reference
        # lambda, reusing the base-LR pair above.
        for lr in (LR_FINE[0], LR_FINE[-1]):
            for kl in ("forward", "reverse"):
                hits = [c for c in block.cells if
                        math.isclose(float(c["core"]["initial_policy_concentration"]), 2.5, rel_tol=0, abs_tol=1e-9) and
                        math.isclose(float(c["core"]["context_strength"]), 0.6, rel_tol=0, abs_tol=1e-12) and
                        c["rollout_source"] == "student" and
                        math.isclose(float(c["ema_alpha"]), 0.0025, rel_tol=0, abs_tol=1e-12) and
                        math.isclose(float(c["learning_rate"]), float(lr), rel_tol=0, abs_tol=1e-12) and
                        c["kl_direction"] == kl]
                if hits:
                    chosen.append(hits[0])
        # De-duplicate while preserving registry order.
        wanted = {c["name"] for c in chosen}
        out = [c for c in block.cells if c["name"] in wanted]
        if len(out) >= 8:
            return out

    # Generic coverage for blocks whose aggregators do not require a special
    # paired smoke motif.
    idx = {0, 1, len(block.cells)//3, len(block.cells)//2, 2*len(block.cells)//3, len(block.cells)-2, len(block.cells)-1}
    # Include at least one exact-sync / iso-JSD / competence cell if present.
    for i, c in enumerate(block.cells):
        ab = c["ablation"]
        if float(c["ema_alpha"]) == 1.0 or ab.get("target_initial_jsd") is not None or float(ab.get("state_competence_bias", 0.0)) != 0.0:
            idx.add(i)
            if len(idx) >= 10:
                break
    return [block.cells[i] for i in sorted(idx)]


def write_block_spec(path: Path, block: Block, cells: Sequence[Mapping[str, object]]) -> None:
    payload = {
        "v14_registry_block": block.name,
        "question": block.question,
        "prediction": block.prediction,
        "tier": block.tier,
        "cells": list(cells),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def execute_block(block: Block, output_root: Path, *, engine: str,
                  device: str, overwrite: bool, smoke: bool,
                  seed_batch_size: int, dtype: str) -> None:
    out = output_root / block.name
    out.parent.mkdir(parents=True, exist_ok=True)
    spec_path = output_root / "specs" / f"{block.name}.json"
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    cells = smoke_subset(block) if smoke else block.cells
    write_block_spec(spec_path, block, cells)
    n_seeds = min(4, block.n_seeds) if smoke else block.n_seeds
    cmd = [
        sys.executable, engine,
        "--spec", str(spec_path),
        "--output-dir", str(out),
        "--device", device,
        "--dtype", dtype,
        "--n-task-seeds", str(n_seeds),
        "--seed-start", str(block.seed_start),
        "--train-steps", str(block.default_steps),
        "--record-every", "10",
        "--sampled-trajectories-per-step", str(block.sampled_trajectories),
        "--paired-optimization-rng",
    ]
    if seed_batch_size > 0:
        cmd += ["--seed-batch-size", str(seed_batch_size)]
    if overwrite:
        cmd.append("--overwrite")
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


# -----------------------------------------------------------------------------
# Aggregation primitives
# -----------------------------------------------------------------------------


IDENTITY_COLUMNS = {
    "task_seed", "cell_name", "condition_id", "analysis_block", "variant",
}
CONTROL_COLUMNS = {
    "ema_alpha", "learning_rate", "teacher_fraction", "train_steps", "optimizer",
    "estimator", "kl_direction", "rollout_source", "horizon", "vocab_size",
    "feature_dim", "record_history", "run_training",
}
METRIC_PREFIXES = (
    "initial_", "final_", "new_", "old_", "context_", "support_", "target_",
    "readout_", "primary_", "terminal_", "geometry_", "teacher_",
)


def infer_metric_columns(df: pd.DataFrame) -> List[str]:
    """Robustly distinguish seed-varying diagnostics from cell configuration.

    Numeric columns that vary within a cell are metrics unless they are explicit
    controls.  This retains the strict guard that caught the v13.2 aggregation
    bug without silently assigning one random seed's realized diagnostic to a
    whole cell.
    """
    metrics: set[str] = set()
    for col in df.columns:
        if col in IDENTITY_COLUMNS or col in CONTROL_COLUMNS:
            continue
        if col.startswith(METRIC_PREFIXES) and pd.api.types.is_numeric_dtype(df[col]):
            metrics.add(col)
    if "cell_name" in df.columns:
        numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in IDENTITY_COLUMNS and c not in CONTROL_COLUMNS]
        for col in numeric:
            nunique = df.groupby("cell_name", sort=False)[col].nunique(dropna=False)
            if (nunique > 1).any():
                metrics.add(col)
    return sorted(metrics)


# Execution/batch provenance columns are intentionally allowed to vary inside a
# scientific cell.  In particular ``condition_id`` is hashed from both the
# scientific metadata *and the exact task-seed batch*.  Recovery runs may split
# a 96-seed cell into two 48-seed engine invocations, so the same cell then has
# multiple condition IDs even though its scientific configuration is identical.
# These fields must never be mistaken for paper-facing configuration columns.
EXECUTION_PROVENANCE_COLUMNS = {"condition_id"}


def cell_configuration_table(df: pd.DataFrame, metric_cols: Sequence[str]) -> pd.DataFrame:
    excluded = set(metric_cols) | {"task_seed"} | EXECUTION_PROVENANCE_COLUMNS
    config_cols = [c for c in df.columns if c not in excluded]
    if "cell_name" not in config_cols:
        raise ValueError("runs table lacks cell_name")
    varying = []
    for col in config_cols:
        if col == "cell_name":
            continue
        if (df.groupby("cell_name", sort=False)[col].nunique(dropna=False) > 1).any():
            varying.append(col)
    if varying:
        raise ValueError(f"Configuration columns unexpectedly vary within a cell: {', '.join(varying)}")
    out = df[config_cols].groupby("cell_name", sort=False, dropna=False).first().reset_index()

    # Retain an auditable summary of execution provenance without pretending it
    # is a scientific configuration.  A value >1 is expected for cells recovered
    # in multiple seed chunks and has no effect on the scientific aggregation.
    if "condition_id" in df.columns:
        prov = (df.groupby("cell_name", sort=False, dropna=False)["condition_id"]
                  .nunique(dropna=False)
                  .rename("condition_id_nunique")
                  .reset_index())
        out = out.merge(prov, on="cell_name", how="left", validate="one_to_one")
        out["recovered_or_multibatch"] = out["condition_id_nunique"].fillna(0).astype(int) > 1
    return out


def cell_summary_table(df: pd.DataFrame, metric_cols: Sequence[str]) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    g = df.groupby("cell_name", sort=False, dropna=False)
    for stat, func in (
        ("mean", "mean"), ("median", "median"), ("std", "std"), ("count", "count")
    ):
        part = getattr(g[list(metric_cols)], func)().reset_index()
        part = part.rename(columns={c: f"{c}__{stat}" for c in metric_cols})
        rows.append(part)
    out = rows[0]
    for part in rows[1:]:
        out = out.merge(part, on="cell_name", how="outer", validate="one_to_one")
    return out


def _normal_p_two_sided(mean: float, se: float) -> float:
    if not np.isfinite(mean) or not np.isfinite(se) or se <= 0:
        return float("nan")
    z = abs(mean / se)
    return float(math.erfc(z / math.sqrt(2.0)))


def _bh_adjust(pvals: Sequence[float]) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan)
    valid = np.isfinite(p)
    vals = p[valid]
    if len(vals) == 0:
        return out
    order = np.argsort(vals)
    ranked = vals[order]
    m = len(vals)
    q = ranked * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    inv = np.empty_like(order)
    inv[order] = np.arange(m)
    out[valid] = q[inv]
    return out


def summarize_seed_values(df: pd.DataFrame, group_cols: Sequence[str], value_col: str,
                          expected_sign: Optional[int] = None) -> pd.DataFrame:
    """Summarize paired per-seed values, including sparse smoke-test inputs.

    Full scientific runs always contain the requested grouping/value columns.
    Smoke subsets intentionally do not guarantee complete paired factorials, so
    derived contrast tables can be empty.  Returning a schema-valid empty frame
    keeps aggregation a plumbing test without fabricating inferential results.
    """
    stat_cols = [
        "n", "mean", "median", "std", "se", "ci95_low",
        "ci95_high", "positive_fraction", "negative_fraction",
        "p_two_sided_normal",
    ]
    if expected_sign in (-1, 1):
        stat_cols.append("expected_sign_fraction")
    empty_cols = list(group_cols) + stat_cols + ["q_bh"]
    required = list(group_cols) + [value_col]
    if df.empty or any(c not in df.columns for c in required):
        return pd.DataFrame(columns=empty_cols)

    records: List[Dict[str, object]] = []
    for key, sub in df.groupby(list(group_cols), dropna=False, sort=False):
        vals = pd.to_numeric(sub[value_col], errors="coerce").to_numpy(float)
        vals = vals[np.isfinite(vals)]
        n = len(vals)
        mean = float(np.mean(vals)) if n else float("nan")
        sd = float(np.std(vals, ddof=1)) if n > 1 else float("nan")
        se = sd / math.sqrt(n) if n > 1 else float("nan")
        rec = {c: v for c, v in zip(group_cols, key if isinstance(key, tuple) else (key,))}
        rec.update({
            "n": n, "mean": mean, "median": float(np.median(vals)) if n else float("nan"),
            "std": sd, "se": se,
            "ci95_low": mean - 1.96 * se if np.isfinite(se) else float("nan"),
            "ci95_high": mean + 1.96 * se if np.isfinite(se) else float("nan"),
            "positive_fraction": float(np.mean(vals > 0)) if n else float("nan"),
            "negative_fraction": float(np.mean(vals < 0)) if n else float("nan"),
            "p_two_sided_normal": _normal_p_two_sided(mean, se),
        })
        if expected_sign in (-1, 1):
            rec["expected_sign_fraction"] = float(np.mean(expected_sign * vals > 0)) if n else float("nan")
        records.append(rec)
    out = pd.DataFrame(records)
    if not out.empty:
        out["q_bh"] = _bh_adjust(out["p_two_sided_normal"].to_numpy(float))
    return out


def load_block(output_root: Path, block_name: str, usecols: Optional[Sequence[str]] = None) -> pd.DataFrame:
    path = output_root / block_name / "runs.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    if usecols is None:
        return pd.read_csv(path)
    header = pd.read_csv(path, nrows=0).columns
    cols = [c for c in usecols if c in header]
    missing = [c for c in usecols if c not in header]
    if missing:
        raise ValueError(f"{block_name} missing required columns: {missing}")
    return pd.read_csv(path, usecols=cols)


def public_columns() -> List[str]:
    return [
        "analysis_block", "variant", "cell_name", "task_seed",
        "v14_core_feature_overlap", "v14_core_initial_policy_concentration", "v14_core_context_strength",
        "v14_ablation_readout_compatibility", "v14_ablation_support_placement_bias",
        "v14_ablation_target_primary_mass", "v14_ablation_target_secondary_mass",
        "v14_ablation_endpoint_compatible_context", "v14_ablation_feature_noise_scale",
        "v14_ablation_behavior_noise_scale", "v14_ablation_offpath_context_multiplier",
        "v14_ablation_state_competence_bias", "v14_ablation_target_initial_jsd",
        "v14_structure_horizon", "v14_structure_vocab_size", "v14_structure_feature_dim",
        "v14_structure_nonterminal_states", "v14_structure_readout_parameters",
        "kl_direction", "rollout_source", "teacher_fraction",
        "ema_alpha", "learning_rate", "optimizer", "estimator", "train_steps",
    ]


OUTCOMES = [
    "new_match_gain", "old_match_forgetting", "new_target_path_match_gain",
    "new_support_success_gain", "new_match_drawdown_from_peak",
    "context_utility_drawdown_from_peak",
    "final_context_context_teacher_utility_student_occupancy",
    "initial_context_geometry_jsd_student_occupancy",
    "initial_context_geometry_tv_student_occupancy",
    "initial_context_geometry_forward_reverse_logit_grad_cosine_student_occupancy",
    "initial_context_geometry_log_forward_reverse_logit_grad_norm_ratio_student_occupancy",
    "initial_context_geometry_teacher_top1_student_support_student_occupancy",
    "initial_context_geometry_target_top1_student_support_student_occupancy",
    "initial_context_v14_context_strength_realized",
    "initial_context_v14_target_jsd_reachable",
    "initial_context_v14_target_jsd_achieved",
    "initial_context_v14_competence_reliability_student_occupancy",
    "initial_context_v14_competence_reliability_teacher_occupancy",
]


def short_names(df: pd.DataFrame) -> pd.DataFrame:
    ren = {
        "v14_core_feature_overlap": "rho_phi",
        "v14_core_initial_policy_concentration": "lambda",
        "v14_core_context_strength": "c",
        "v14_ablation_readout_compatibility": "rho_W",
        "v14_ablation_support_placement_bias": "beta",
        "v14_ablation_state_competence_bias": "competence_bias",
        "v14_ablation_offpath_context_multiplier": "offpath_multiplier",
        "v14_ablation_feature_noise_scale": "feature_noise",
        "v14_ablation_behavior_noise_scale": "behavior_noise",
        "v14_ablation_target_initial_jsd": "target_initial_jsd",
        "v14_ablation_endpoint_compatible_context": "endpoint_compatible",
        "v14_structure_horizon": "H",
        "v14_structure_vocab_size": "K",
        "v14_structure_feature_dim": "D",
        "v14_structure_nonterminal_states": "n_prefix_states",
        "v14_structure_readout_parameters": "n_readout_parameters",
    }
    return df.rename(columns=ren)


# -----------------------------------------------------------------------------
# Cross-fitted alpha selection
# -----------------------------------------------------------------------------


def crossfit_alpha_rows(df: pd.DataFrame, group_cols: Sequence[str], *,
                        alpha_col: str = "ema_alpha", objective: str = "new_match_gain") -> pd.DataFrame:
    """Two-fold cross-fitted discrete alpha selection, evaluated per seed."""
    records: List[pd.DataFrame] = []
    work = df.copy()
    work["_fold"] = pd.to_numeric(work["task_seed"]).astype(int) % 2
    for fold in (0, 1):
        train = work[work["_fold"] != fold]
        test = work[work["_fold"] == fold]
        means = train.groupby(list(group_cols) + [alpha_col], dropna=False)[objective].mean().reset_index()
        means = means.sort_values(list(group_cols) + [objective, alpha_col], ascending=[True]*len(group_cols)+[False, True])
        chosen = means.groupby(list(group_cols), dropna=False, sort=False).first().reset_index()
        chosen = chosen[list(group_cols) + [alpha_col]].rename(columns={alpha_col: "selected_alpha"})
        eval_ = test.merge(chosen, on=list(group_cols), how="inner", validate="many_to_one")
        eval_ = eval_[np.isclose(eval_[alpha_col].astype(float), eval_["selected_alpha"].astype(float), rtol=0, atol=1e-12)].copy()
        eval_["selection_fold"] = 1 - fold
        eval_["evaluation_fold"] = fold
        records.append(eval_)
    return pd.concat(records, ignore_index=True) if records else pd.DataFrame()


# -----------------------------------------------------------------------------
# Specialized aggregates
# -----------------------------------------------------------------------------


def aggregate_trajectory(output_root: Path, agg: Path) -> None:
    cols = public_columns() + OUTCOMES
    df = short_names(load_block(output_root, "V14_E71_trajectory_surface", cols))
    curve_cols = ["lambda", "c", "kl_direction", "teacher_fraction", "ema_alpha"]
    curves = df.groupby(curve_cols, dropna=False)[[c for c in OUTCOMES if c in df]].agg(["mean", "std", "count"])
    curves.columns = [f"{a}__{b}" for a, b in curves.columns]
    curves.reset_index().to_csv(agg / "trajectory_paper_curves.csv", index=False)

    # Cross-fit alpha separately for each trajectory fraction.
    cf = crossfit_alpha_rows(df, ["lambda", "c", "kl_direction", "teacher_fraction"])
    keep = ["task_seed", "lambda", "c", "kl_direction", "teacher_fraction", "selected_alpha"] + [c for c in OUTCOMES if c in cf]
    cf[keep].to_csv(agg / "trajectory_crossfit_optimal_ema_seed.csv.gz", index=False, compression="gzip")

    # Endpoint comparison with source-specific cross-fitted optima.
    end = cf[cf["teacher_fraction"].isin([0.0, 1.0])]
    metrics = ["new_match_gain", "old_match_forgetting", "new_target_path_match_gain", "final_context_context_teacher_utility_student_occupancy"]
    contrast_frames = []
    idx = ["task_seed", "lambda", "c", "kl_direction"]
    for m in metrics:
        piv = end.pivot_table(index=idx, columns="teacher_fraction", values=m, aggfunc="first")
        if 0.0 in piv.columns and 1.0 in piv.columns:
            d = (piv[1.0] - piv[0.0]).rename(f"teacher_minus_student__{m}").reset_index()
            contrast_frames.append(d)
    cont = contrast_frames[0]
    for x in contrast_frames[1:]:
        cont = cont.merge(x, on=idx, how="outer", validate="one_to_one")
    cont.to_csv(agg / "trajectory_crossfit_endpoint_contrasts_seed.csv.gz", index=False, compression="gzip")
    summaries = []
    for m in metrics:
        col = f"teacher_minus_student__{m}"
        if col in cont:
            s = summarize_seed_values(cont, ["lambda", "c", "kl_direction"], col)
            s["metric"] = m
            summaries.append(s)
    pd.concat(summaries, ignore_index=True).to_csv(agg / "trajectory_crossfit_endpoint_summary.csv", index=False)

    # Common-alpha endpoint comparison: select alpha on the average endpoint
    # acquisition in the other fold, then evaluate both sources at that alpha.
    epraw = df[df["teacher_fraction"].isin([0.0, 1.0])].copy()
    epraw["_endpoint_mean_objective"] = epraw.groupby(["task_seed", "lambda", "c", "kl_direction", "ema_alpha"])["new_match_gain"].transform("mean")
    common = crossfit_alpha_rows(
        epraw.drop_duplicates(["task_seed", "lambda", "c", "kl_direction", "ema_alpha"]),
        ["lambda", "c", "kl_direction"], objective="_endpoint_mean_objective"
    )[["task_seed", "lambda", "c", "kl_direction", "selected_alpha"]]
    eval_ep = epraw.merge(common, on=["task_seed", "lambda", "c", "kl_direction"], how="inner")
    eval_ep = eval_ep[np.isclose(eval_ep["ema_alpha"], eval_ep["selected_alpha"], atol=1e-12)]
    common_frames = []
    for m in metrics:
        piv = eval_ep.pivot_table(index=idx, columns="teacher_fraction", values=m, aggfunc="first")
        if 0.0 in piv.columns and 1.0 in piv.columns:
            common_frames.append((piv[1.0]-piv[0.0]).rename(f"teacher_minus_student__{m}").reset_index())
    common_cont = common_frames[0]
    for x in common_frames[1:]:
        common_cont = common_cont.merge(x, on=idx, how="outer")
    common_cont.to_csv(agg / "trajectory_common_alpha_endpoint_contrasts_seed.csv.gz", index=False, compression="gzip")

    # Raw per-seed linear slopes over teacher fraction for each alpha.
    slope_records = []
    for keys, sub in df.groupby(["task_seed", "lambda", "c", "kl_direction", "ema_alpha"], sort=False):
        x = sub["teacher_fraction"].to_numpy(float)
        if len(np.unique(x)) < 3:
            continue
        for m in metrics[:3]:
            y = sub[m].to_numpy(float)
            if np.isfinite(y).sum() != len(y):
                continue
            coef = np.polyfit(x, y, deg=2)
            slope_records.append({
                "task_seed": keys[0], "lambda": keys[1], "c": keys[2],
                "kl_direction": keys[3], "ema_alpha": keys[4], "metric": m,
                "linear_term": float(coef[1]), "quadratic_term": float(coef[0]),
                "endpoint_delta": float(y[np.argmax(x)] - y[np.argmin(x)]),
            })
    slopes = pd.DataFrame(slope_records)
    slopes.to_csv(agg / "trajectory_seed_shape_coefficients.csv.gz", index=False, compression="gzip")
    if not slopes.empty:
        summarize_seed_values(slopes, ["lambda", "c", "kl_direction", "ema_alpha", "metric"], "linear_term").to_csv(
            agg / "trajectory_slope_summary.csv", index=False
        )


def _pareto_frontier(acq: np.ndarray, forget: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(acq) & np.isfinite(forget)
    acq, forget = acq[mask], forget[mask]
    if len(acq) == 0:
        return np.array([]), np.array([])
    order = np.argsort(-acq)  # high acquisition to low
    best = np.inf
    keep: List[int] = []
    for idx in order:
        if forget[idx] < best - 1e-12:
            keep.append(idx)
            best = forget[idx]
    keep = sorted(keep, key=lambda i: acq[i])
    x = acq[keep]; y = forget[keep]
    # collapse duplicate x with minimum forgetting
    ux = [] ; uy = []
    for val in np.unique(x):
        ux.append(val); uy.append(np.min(y[np.isclose(x, val, atol=1e-12)]))
    return np.asarray(ux), np.asarray(uy)


def matched_frontier_seed(df: pd.DataFrame, group_cols: Sequence[str], label: str) -> pd.DataFrame:
    """Per-seed matched-acquisition comparison using a lower forgetting envelope.

    For target acquisition x, each KL is allowed to choose any registered LR
    (and whatever alpha rows are present in ``df``) that achieves acquisition
    >= x; the retained value is the minimum forgetting among those choices.
    This is a conservative Pareto comparison and remains defined when several
    LRs have equal forgetting or the LR->acquisition curve is non-monotonic.
    """
    recs: List[Dict[str, object]] = []
    for key, sub in df.groupby(["task_seed"] + list(group_cols), dropna=False, sort=False):
        f = sub[sub["kl_direction"] == "forward"]
        r = sub[sub["kl_direction"] == "reverse"]
        af = f["new_match_gain"].to_numpy(float); ar = r["new_match_gain"].to_numpy(float)
        ff = f["old_match_forgetting"].to_numpy(float); fr = r["old_match_forgetting"].to_numpy(float)
        mf = np.isfinite(af) & np.isfinite(ff); mr = np.isfinite(ar) & np.isfinite(fr)
        af, ff, ar, fr = af[mf], ff[mf], ar[mr], fr[mr]
        if len(af) == 0 or len(ar) == 0:
            continue
        lo = max(float(af.min()), float(ar.min()))
        hi = min(float(af.max()), float(ar.max()))
        if hi < lo - 1e-12:
            continue
        for q in (0.0, 0.25, 0.50, 0.75, 1.0):
            x = lo + q * max(0.0, hi - lo)
            elig_f = ff[af >= x - 1e-12]
            elig_r = fr[ar >= x - 1e-12]
            if len(elig_f) == 0 or len(elig_r) == 0:
                continue
            forget_f = float(np.min(elig_f)); forget_r = float(np.min(elig_r))
            vals = key if isinstance(key, tuple) else (key,)
            row = {"task_seed": vals[0]}
            for c, v in zip(group_cols, vals[1:]):
                row[c] = v
            row.update({
                "matched_quantile": q, "matched_acquisition": x,
                "forward_forgetting": forget_f, "reverse_forgetting": forget_r,
                "forward_minus_reverse_forgetting": forget_f - forget_r,
                "frontier_label": label,
            })
            recs.append(row)
    return pd.DataFrame(recs)


def aggregate_kl(output_root: Path, agg: Path) -> None:
    cols = public_columns() + OUTCOMES
    df = short_names(load_block(output_root, "V14_E72_kl_matched_learning", cols))
    curve = df.groupby(["lambda", "c", "rollout_source", "ema_alpha", "learning_rate", "kl_direction"], dropna=False)[
        ["new_match_gain", "old_match_forgetting", "new_match_drawdown_from_peak"]
    ].agg(["mean", "std", "count"])
    curve.columns = [f"{a}__{b}" for a,b in curve.columns]
    curve.reset_index().to_csv(agg / "kl_lr_alpha_curves.csv", index=False)

    raw_match = matched_frontier_seed(df, ["lambda", "c", "rollout_source", "ema_alpha"], "fixed_alpha")
    raw_match.to_csv(agg / "kl_matched_learning_seed.csv.gz", index=False, compression="gzip")
    if not raw_match.empty:
        summarize_seed_values(raw_match, ["lambda", "c", "rollout_source", "ema_alpha", "matched_quantile"], "forward_minus_reverse_forgetting", expected_sign=1).to_csv(
            agg / "kl_matched_learning_summary.csv", index=False
        )

    # Cross-fit alpha independently for each (KL, LR) curve, then construct the
    # held-out matched-learning Pareto frontier across LR.
    cf = crossfit_alpha_rows(df, ["lambda", "c", "rollout_source", "kl_direction", "learning_rate"])
    cf.to_csv(agg / "kl_crossfit_optimal_alpha_rows.csv.gz", index=False, compression="gzip")
    cf_match = matched_frontier_seed(cf, ["lambda", "c", "rollout_source"], "crossfit_optimal_alpha")
    cf_match.to_csv(agg / "kl_matched_learning_crossfit_seed.csv.gz", index=False, compression="gzip")
    if not cf_match.empty:
        summarize_seed_values(cf_match, ["lambda", "c", "rollout_source", "matched_quantile"], "forward_minus_reverse_forgetting", expected_sign=1).to_csv(
            agg / "kl_matched_learning_crossfit_summary.csv", index=False
        )

    # Fixed-LR paired KL contrasts + geometry bridge.
    base = df[np.isclose(df["learning_rate"], BASE_ADAM_LR, atol=1e-12)].copy()
    idx = ["task_seed", "lambda", "c", "rollout_source", "ema_alpha"]
    f = base[base["kl_direction"] == "forward"].set_index(idx)
    r = base[base["kl_direction"] == "reverse"].set_index(idx)
    common = f.index.intersection(r.index)
    bridge_rows = []
    geom_cols = [
        "initial_context_geometry_jsd_student_occupancy",
        "initial_context_geometry_tv_student_occupancy",
        "initial_context_geometry_forward_reverse_logit_grad_cosine_student_occupancy",
        "initial_context_geometry_log_forward_reverse_logit_grad_norm_ratio_student_occupancy",
        "initial_context_geometry_teacher_top1_student_support_student_occupancy",
    ]
    for ix in common:
        rf = f.loc[ix]; rr = r.loc[ix]
        row = {c: v for c,v in zip(idx, ix)}
        row["forward_minus_reverse_acquisition"] = float(rf["new_match_gain"] - rr["new_match_gain"])
        row["forward_minus_reverse_forgetting"] = float(rf["old_match_forgetting"] - rr["old_match_forgetting"])
        for g in geom_cols:
            row[g] = float(rr[g])
        row["initial_gradient_disagreement_one_minus_cosine"] = 1.0 - row["initial_context_geometry_forward_reverse_logit_grad_cosine_student_occupancy"]
        bridge_rows.append(row)
    bridge_schema = idx + [
        "forward_minus_reverse_acquisition",
        "forward_minus_reverse_forgetting",
        *geom_cols,
        "initial_gradient_disagreement_one_minus_cosine",
    ]
    bridge = pd.DataFrame(bridge_rows, columns=bridge_schema)
    bridge.to_csv(agg / "geometry_bridge_per_seed.csv.gz", index=False, compression="gzip")

    corr_records = []
    xcols = ["initial_context_geometry_jsd_student_occupancy", "initial_gradient_disagreement_one_minus_cosine", "initial_context_geometry_log_forward_reverse_logit_grad_norm_ratio_student_occupancy", "initial_context_geometry_teacher_top1_student_support_student_occupancy"]
    # Smoke subsets are intentionally sparse and may contain no complete
    # forward/reverse pair at BASE_ADAM_LR.  In that case the bridge is not
    # inferentially defined; emit schema-valid empty files rather than failing.
    if not bridge.empty:
        for (seed, cval, ro, a), sub in bridge.groupby(["task_seed", "c", "rollout_source", "ema_alpha"], sort=False):
            if len(sub) < 4:
                continue
            for xcol in xcols:
                for ycol in ("forward_minus_reverse_acquisition", "forward_minus_reverse_forgetting"):
                    x = sub[xcol].to_numpy(float); y = sub[ycol].to_numpy(float)
                    if np.std(x) <= 1e-14 or np.std(y) <= 1e-14:
                        corr = float("nan")
                    else:
                        corr = float(np.corrcoef(x, y)[0,1])
                    corr_records.append({"task_seed": seed, "c": cval, "rollout_source": ro, "ema_alpha": a, "x_metric": xcol, "y_metric": ycol, "pearson_across_lambda": corr})
    corr_schema = ["task_seed", "c", "rollout_source", "ema_alpha", "x_metric", "y_metric", "pearson_across_lambda"]
    corr = pd.DataFrame(corr_records, columns=corr_schema)
    corr.to_csv(agg / "geometry_bridge_correlations_seed.csv.gz", index=False, compression="gzip")
    if not corr.empty:
        summarize_seed_values(corr, ["c", "rollout_source", "ema_alpha", "x_metric", "y_metric"], "pearson_across_lambda").to_csv(
            agg / "geometry_bridge_summary.csv", index=False
        )
    else:
        pd.DataFrame(columns=["c", "rollout_source", "ema_alpha", "x_metric", "y_metric", "n", "mean", "median", "std", "se", "ci95_low", "ci95_high", "positive_fraction", "negative_fraction", "p_two_sided_normal", "q_bh"]).to_csv(
            agg / "geometry_bridge_summary.csv", index=False
        )


def aggregate_ema(output_root: Path, agg: Path) -> None:
    cols = public_columns() + OUTCOMES
    df = short_names(load_block(output_root, "V14_E73_coupling_phase", cols))
    curve = df.groupby(["lambda", "c", "kl_direction", "rollout_source", "ema_alpha"], dropna=False)[
        ["new_match_gain", "old_match_forgetting", "final_context_context_teacher_utility_student_occupancy", "context_utility_drawdown_from_peak"]
    ].agg(["mean", "std", "count"])
    curve.columns = [f"{a}__{b}" for a,b in curve.columns]
    curve.reset_index().to_csv(agg / "ema_paper_curves.csv", index=False)

    cf = crossfit_alpha_rows(df, ["lambda", "c", "kl_direction", "rollout_source"])
    cf[["task_seed", "lambda", "c", "kl_direction", "rollout_source", "selected_alpha"] + [m for m in OUTCOMES if m in cf]].to_csv(
        agg / "ema_crossfit_optimal_rows.csv.gz", index=False, compression="gzip"
    )
    # Mean-grid optimum is useful for the visual phase diagram, while cross-fit
    # rows are used for inference.
    means = df.groupby(["lambda", "c", "kl_direction", "rollout_source", "ema_alpha"], dropna=False)["new_match_gain"].mean().reset_index()
    means = means.sort_values(["lambda", "c", "kl_direction", "rollout_source", "new_match_gain", "ema_alpha"], ascending=[True,True,True,True,False,True])
    opt = means.groupby(["lambda", "c", "kl_direction", "rollout_source"], sort=False).first().reset_index().rename(columns={"ema_alpha":"mean_grid_optimal_alpha", "new_match_gain":"mean_grid_optimal_acquisition"})
    opt.to_csv(agg / "ema_mean_grid_optima.csv", index=False)

    # Cross-fitted alpha distribution / evaluation metrics.
    rows = []
    for m in ("new_match_gain", "old_match_forgetting", "final_context_context_teacher_utility_student_occupancy"):
        s = summarize_seed_values(cf, ["lambda", "c", "kl_direction", "rollout_source", "selected_alpha"], m)
        s["metric"] = m
        rows.append(s)
    pd.concat(rows, ignore_index=True).to_csv(agg / "ema_crossfit_optimal_summary.csv", index=False)

    # Practical lever comparison: reverse frozen -> reverse optimal EMA, then
    # reverse optimal -> forward optimal.  Alpha is cross-fitted separately.
    idx = ["task_seed", "lambda", "c", "rollout_source"]
    rev = cf[cf["kl_direction"] == "reverse"].set_index(idx)
    fwd = cf[cf["kl_direction"] == "forward"].set_index(idx)
    frozen = df[(df["kl_direction"] == "reverse") & np.isclose(df["ema_alpha"], 0.0)].set_index(idx)
    common = rev.index.intersection(fwd.index).intersection(frozen.index)
    recs = []
    for ix in common:
        r = rev.loc[ix]; f = fwd.loc[ix]; z = frozen.loc[ix]
        rec = {c:v for c,v in zip(idx, ix)}
        rec.update({
            "ema_acquisition_gain": float(r["new_match_gain"] - z["new_match_gain"]),
            "ema_forgetting_cost": float(r["old_match_forgetting"] - z["old_match_forgetting"]),
            "forward_after_ema_acquisition_gain": float(f["new_match_gain"] - r["new_match_gain"]),
            "forward_after_ema_forgetting_cost": float(f["old_match_forgetting"] - r["old_match_forgetting"]),
            "reverse_selected_alpha": float(r["selected_alpha"]),
            "forward_selected_alpha": float(f["selected_alpha"]),
        })
        recs.append(rec)
    guide = pd.DataFrame(recs)
    guide.to_csv(agg / "practical_lever_comparison_seed.csv.gz", index=False, compression="gzip")
    sums = []
    for m in ("ema_acquisition_gain", "ema_forgetting_cost", "forward_after_ema_acquisition_gain", "forward_after_ema_forgetting_cost"):
        s = summarize_seed_values(guide, ["lambda", "c", "rollout_source"], m)
        s["metric"] = m; sums.append(s)
    pd.concat(sums, ignore_index=True).to_csv(agg / "practical_lever_comparison_summary.csv", index=False)


def aggregate_iso_jsd(output_root: Path, agg: Path) -> None:
    cols = public_columns() + OUTCOMES
    df = short_names(load_block(output_root, "V14_E74_iso_jsd", cols))
    idx = ["task_seed", "lambda", "target_initial_jsd", "rollout_source", "ema_alpha"]
    f = df[df["kl_direction"] == "forward"].set_index(idx)
    r = df[df["kl_direction"] == "reverse"].set_index(idx)
    recs = []
    for ix in f.index.intersection(r.index):
        a, b = f.loc[ix], r.loc[ix]
        row = {c:v for c,v in zip(idx, ix)}
        row.update({
            "forward_minus_reverse_acquisition": float(a["new_match_gain"] - b["new_match_gain"]),
            "forward_minus_reverse_forgetting": float(a["old_match_forgetting"] - b["old_match_forgetting"]),
            "achieved_jsd": float(b["initial_context_v14_target_jsd_achieved"]),
            "jsd_reachable": float(b["initial_context_v14_target_jsd_reachable"]),
            "realized_context_strength": float(b["initial_context_v14_context_strength_realized"]),
        })
        recs.append(row)
    out = pd.DataFrame(recs)
    out.to_csv(agg / "iso_jsd_seed_contrasts.csv.gz", index=False, compression="gzip")
    if not out.empty:
        summaries = []
        for m in ("forward_minus_reverse_acquisition", "forward_minus_reverse_forgetting"):
            s = summarize_seed_values(out[out["jsd_reachable"] > 0.5], ["lambda", "target_initial_jsd", "rollout_source", "ema_alpha"], m)
            s["metric"] = m; summaries.append(s)
        pd.concat(summaries, ignore_index=True).to_csv(agg / "iso_jsd_summary.csv", index=False)
        reach = out.groupby(["lambda", "target_initial_jsd"], dropna=False)["jsd_reachable"].mean().reset_index(name="reachable_fraction")
        reach.to_csv(agg / "iso_jsd_reachability.csv", index=False)


def aggregate_competence(output_root: Path, agg: Path) -> None:
    cols = public_columns() + OUTCOMES
    df = short_names(load_block(output_root, "V14_E75_state_competence", cols))
    curve = df.groupby(["lambda", "competence_bias", "kl_direction", "teacher_fraction", "ema_alpha"], dropna=False)[
        ["new_match_gain", "old_match_forgetting", "new_target_path_match_gain", "initial_context_v14_competence_reliability_student_occupancy", "initial_context_v14_competence_reliability_teacher_occupancy"]
    ].agg(["mean", "std", "count"])
    curve.columns = [f"{a}__{b}" for a,b in curve.columns]
    curve.reset_index().to_csv(agg / "state_competence_paper_curves.csv", index=False)

    cf = crossfit_alpha_rows(df, ["lambda", "competence_bias", "kl_direction", "teacher_fraction"])
    cf.to_csv(agg / "state_competence_crossfit_rows.csv.gz", index=False, compression="gzip")
    end = cf[cf["teacher_fraction"].isin([0.0,1.0])]
    idx = ["task_seed", "lambda", "competence_bias", "kl_direction"]
    recs = []
    for ix, sub in end.groupby(idx, sort=False):
        s = sub[sub["teacher_fraction"] == 0.0]
        t = sub[sub["teacher_fraction"] == 1.0]
        if len(s) != 1 or len(t) != 1:
            continue
        row = {c:v for c,v in zip(idx, ix)}
        for m in ("new_match_gain", "old_match_forgetting", "new_target_path_match_gain"):
            row[f"teacher_minus_student__{m}"] = float(t.iloc[0][m] - s.iloc[0][m])
        recs.append(row)
    con = pd.DataFrame(recs)
    con.to_csv(agg / "state_competence_endpoint_contrasts_seed.csv.gz", index=False, compression="gzip")
    sums = []
    for m in ("new_match_gain", "old_match_forgetting", "new_target_path_match_gain"):
        col = f"teacher_minus_student__{m}"
        s = summarize_seed_values(con, ["lambda", "competence_bias", "kl_direction"], col)
        s["metric"] = m; sums.append(s)
    pd.concat(sums, ignore_index=True).to_csv(agg / "state_competence_endpoint_summary.csv", index=False)


def paired_method_contrasts(df: pd.DataFrame, group_cols: Sequence[str],
                            contrast_col: str, a_value: object, b_value: object,
                            metrics: Sequence[str], label: str) -> pd.DataFrame:
    idx = ["task_seed"] + list(group_cols)
    a = df[df[contrast_col] == a_value].set_index(idx)
    b = df[df[contrast_col] == b_value].set_index(idx)
    recs = []
    for ix in a.index.intersection(b.index):
        ra, rb = a.loc[ix], b.loc[ix]
        row = {c:v for c,v in zip(idx, ix)}
        for m in metrics:
            row[f"{label}__{m}"] = float(ra[m] - rb[m])
        recs.append(row)
    return pd.DataFrame(recs)


def aggregate_public_interactions(output_root: Path, agg: Path) -> None:
    cols = public_columns() + OUTCOMES
    df = short_names(load_block(output_root, "V14_E76_public_interactions", cols))
    # F-R and T-S contrasts at each public-control combination.
    kl = paired_method_contrasts(df, ["rho_phi","lambda","c","rollout_source","ema_alpha"], "kl_direction", "forward", "reverse", ["new_match_gain","old_match_forgetting"], "forward_minus_reverse")
    ro = paired_method_contrasts(df, ["rho_phi","lambda","c","kl_direction","ema_alpha"], "rollout_source", "teacher", "student", ["new_match_gain","old_match_forgetting"], "teacher_minus_student")
    kl.to_csv(agg / "public_interaction_kl_seed.csv.gz", index=False, compression="gzip")
    ro.to_csv(agg / "public_interaction_rollout_seed.csv.gz", index=False, compression="gzip")
    # Per-seed slopes over rho_phi quantify whether it remains predominantly an
    # interference axis after crossing lambda and c.
    recs = []
    for keys, sub in df.groupby(["task_seed","lambda","c","kl_direction","rollout_source","ema_alpha"], sort=False):
        x = sub["rho_phi"].to_numpy(float)
        for m in ("new_match_gain","old_match_forgetting"):
            y = sub[m].to_numpy(float)
            if len(np.unique(x)) >= 4 and np.isfinite(y).all():
                slope = float(np.polyfit(x,y,1)[0])
                recs.append({"task_seed":keys[0],"lambda":keys[1],"c":keys[2],"kl_direction":keys[3],"rollout_source":keys[4],"ema_alpha":keys[5],"metric":m,"rho_phi_slope":slope})
    slopes = pd.DataFrame(recs)
    slopes.to_csv(agg / "feature_overlap_interaction_slopes_seed.csv.gz", index=False, compression="gzip")
    if not slopes.empty:
        summarize_seed_values(slopes, ["lambda","c","kl_direction","rollout_source","ema_alpha","metric"], "rho_phi_slope").to_csv(agg / "feature_overlap_interaction_summary.csv", index=False)
    else:
        pd.DataFrame(columns=["lambda","c","kl_direction","rollout_source","ema_alpha","metric","n","mean"]).to_csv(agg / "feature_overlap_interaction_summary.csv", index=False)


def aggregate_appendix(output_root: Path, agg: Path) -> None:
    # E77a context noise: grid optima by corruption type/scale/c.
    cols = public_columns() + OUTCOMES
    noise = short_names(load_block(output_root, "V14_E77a_context_noise", cols))
    noise["noise_type"] = np.where(noise["feature_noise"] > 0, "feature", np.where(noise["behavior_noise"] > 0, "behavior", "zero"))
    noise["noise_scale"] = np.maximum(noise["feature_noise"], noise["behavior_noise"])
    nmean = noise.groupby(["noise_type","noise_scale","c","ema_alpha"])["new_match_gain"].mean().reset_index()
    nopt = nmean.sort_values(["noise_type","noise_scale","c","new_match_gain","ema_alpha"], ascending=[True,True,True,False,True]).groupby(["noise_type","noise_scale","c"], sort=False).first().reset_index().rename(columns={"ema_alpha":"optimal_alpha"})
    nopt.to_csv(agg / "context_noise_ema_optima.csv", index=False)

    # E77b offpath: cross-fitted alpha, endpoint trajectory contrasts.
    off = short_names(load_block(output_root, "V14_E77b_offpath_availability", cols))
    offcf = crossfit_alpha_rows(off, ["offpath_multiplier","lambda","teacher_fraction","kl_direction"] if "kl_direction" in off else ["offpath_multiplier","lambda","teacher_fraction"])
    offcf.to_csv(agg / "offpath_crossfit_rows.csv.gz", index=False, compression="gzip")

    # E78 excluded core mechanisms: retain a compact cell-level summary plus
    # paired method contrasts.  Variant carries the intervention kind/label.
    ex = short_names(load_block(output_root, "V14_E78_core_excluded_ablation", cols))
    ex.groupby(["variant","lambda","kl_direction","rollout_source","ema_alpha"], dropna=False)[["new_match_gain","old_match_forgetting","initial_context_geometry_jsd_student_occupancy","initial_context_geometry_teacher_top1_student_support_student_occupancy"]].mean().reset_index().to_csv(agg / "excluded_core_ablation_summary.csv", index=False)

    # E79 novel support.
    nov = short_names(load_block(output_root, "V14_E79_novel_support", cols))
    nov.groupby(["beta","rho_phi","lambda","teacher_fraction","ema_alpha"], dropna=False)[["new_match_gain","old_match_forgetting","initial_context_geometry_target_top1_student_support_student_occupancy","new_target_path_match_gain"]].agg(["mean","std","count"]).to_csv(agg / "novel_support_summary.csv")

    # E80 endpoint/high-alpha.
    hi = short_names(load_block(output_root, "V14_E80_endpoint_high_alpha", cols))
    hi.groupby(["endpoint_compatible","c","kl_direction","ema_alpha"], dropna=False)[["new_match_gain","old_match_forgetting","final_context_context_teacher_utility_student_occupancy","context_utility_drawdown_from_peak"]].agg(["mean","std","count"]).to_csv(agg / "endpoint_high_alpha_summary.csv")

    # E81 optimizer interaction.
    op = short_names(load_block(output_root, "V14_E81_optimizer", cols))
    op.groupby(["optimizer","lambda","c","kl_direction","teacher_fraction","ema_alpha"], dropna=False)[["new_match_gain","old_match_forgetting","new_target_path_match_gain"]].agg(["mean","std","count"]).to_csv(agg / "optimizer_paper_curves.csv")


def aggregate_technical(output_root: Path, agg: Path) -> None:
    cols = public_columns() + OUTCOMES
    structural_frames = []
    structural_specs = (
        ("V14_E82a_horizon", "H", "horizon_robustness.csv"),
        ("V14_E82d_vocab", "K", "vocab_robustness.csv"),
        ("V14_E82e_feature_dim", "D", "feature_dim_robustness.csv"),
    )
    for block, factor, outname in structural_specs:
        df = short_names(load_block(output_root, block, cols))
        group = [factor, "lambda", "kl_direction", "rollout_source", "ema_alpha"]
        summary = df.groupby(group, dropna=False)[
            ["new_match_gain", "old_match_forgetting", "new_target_path_match_gain",
             "initial_context_geometry_jsd_student_occupancy"]
        ].agg(["mean","std","count"])
        summary.to_csv(agg / outname)
        long = df[["task_seed", factor, "lambda", "kl_direction", "rollout_source", "ema_alpha",
                   "new_match_gain", "old_match_forgetting", "new_target_path_match_gain",
                   "initial_context_geometry_jsd_student_occupancy"]].copy()
        long["structural_factor"] = factor
        long["structural_value"] = pd.to_numeric(long[factor], errors="coerce")
        structural_frames.append(long.drop(columns=[factor]))
    if structural_frames:
        pd.concat(structural_frames, ignore_index=True).to_csv(
            agg / "structural_robustness_per_seed.csv.gz", index=False, compression="gzip"
        )

    df = short_names(load_block(output_root, "V14_E82b_duration", cols))
    group = ["lambda","kl_direction","rollout_source","ema_alpha","train_steps"]
    df.groupby(group, dropna=False)[["new_match_gain","old_match_forgetting"]].agg(["mean","std","count"]).to_csv(agg / "duration_robustness.csv")

    frames = []
    for name in ("V14_E82c_exact","V14_E82c_sample64","V14_E82c_sample256","V14_E82c_sample1024"):
        df = short_names(load_block(output_root, name, cols))
        df["estimator_block"] = name
        frames.append(df)
    est = pd.concat(frames, ignore_index=True)
    est.groupby(["estimator_block","kl_direction","rollout_source","ema_alpha"])[["new_match_gain","old_match_forgetting"]].agg(["mean","std","count"]).to_csv(agg / "estimator_robustness.csv")


def aggregate_cells(output_root: Path, agg: Path) -> None:
    configs_all = []
    summaries_all = []
    completion = []
    for b in BLOCKS:
        path = output_root / b.name / "runs.csv"
        if not path.exists():
            completion.append({"block":b.name,"status":"missing","expected_rows":b.n_runs,"observed_rows":0})
            continue
        df = pd.read_csv(path)
        metrics = infer_metric_columns(df)
        cfg = cell_configuration_table(df, metrics)
        sm = cell_summary_table(df, metrics)
        cfg["registry_block"] = b.name
        sm["registry_block"] = b.name
        configs_all.append(cfg); summaries_all.append(sm)
        completion.append({"block":b.name,"status":"complete" if len(df)==b.n_runs else "row_count_mismatch","expected_rows":b.n_runs,"observed_rows":len(df),"expected_cells":len(b.cells),"observed_cells":df["cell_name"].nunique()})
    if configs_all:
        pd.concat(configs_all, ignore_index=True).to_csv(agg / "cell_configurations.csv.gz", index=False, compression="gzip")
        pd.concat(summaries_all, ignore_index=True).to_csv(agg / "cell_summary.csv.gz", index=False, compression="gzip")
    status = pd.DataFrame(completion)
    status.to_csv(agg / "completion_status.csv", index=False)
    lines = ["# V14 aggregate status", "", f"Registry: {registry_summary()}", ""]
    for r in completion:
        lines.append(f"- {r['block']}: {r['status']} ({r['observed_rows']}/{r['expected_rows']} rows)")
    (agg / "aggregate_status.md").write_text("\n".join(lines)+"\n", encoding="utf-8")




def _synthetic_row(seed: int, block: str, *, lam: float = 2.5, cval: float = 0.6,
                   rho: float = 0.5, kl: str = "reverse", rollout: str = "student",
                   frac: float = 0.0, alpha: float = 0.0025, lr: float = 0.001,
                   competence: float = 0.0, target_jsd: Optional[float] = None) -> Dict[str, object]:
    # Deterministic smooth synthetic outcomes used only to regression-test
    # aggregation plumbing; they are not scientific model outputs.
    seed_jitter = (seed - 2.5) * 1e-3
    kl_f = 1.0 if kl == "forward" else 0.0
    ro_t = frac if rollout == "mixture" else (1.0 if rollout == "teacher" else 0.0)
    acq = 0.10 + 0.02 * math.log(max(lam, 1e-6)) + 5.0 * lr + 0.8 * alpha + 0.002 * kl_f + 0.006 * ro_t + 0.004 * competence + seed_jitter
    forgetting = 0.03 + 0.012 * math.log(max(lam, 1e-6)) + 0.3 * alpha + 0.008 * kl_f + 0.002 * ro_t + seed_jitter
    cos = 0.95 - 0.03 * math.log(max(lam, 1e-6))
    jsd = float(target_jsd) if target_jsd is not None else 0.02 + 0.02 * math.log(max(lam, 1e-6)) + 0.01 * cval
    return {
        "analysis_block": block, "variant": "synthetic", "cell_name": f"synthetic_{block}_{seed}_{lam}_{cval}_{rho}_{kl}_{rollout}_{frac}_{alpha}_{lr}_{competence}_{target_jsd}",
        "task_seed": seed, "condition_id": f"id{seed}",
        "v14_core_feature_overlap": rho, "v14_core_initial_policy_concentration": lam, "v14_core_context_strength": cval,
        "v14_ablation_readout_compatibility": 0.0, "v14_ablation_support_placement_bias": 0.0,
        "v14_ablation_target_primary_mass": 0.30, "v14_ablation_target_secondary_mass": 0.15,
        "v14_ablation_endpoint_compatible_context": False, "v14_ablation_feature_noise_scale": 0.0,
        "v14_ablation_behavior_noise_scale": 0.0, "v14_ablation_offpath_context_multiplier": 1.0,
        "v14_ablation_state_competence_bias": competence, "v14_ablation_target_initial_jsd": target_jsd,
        "v14_structure_horizon": 4, "v14_structure_vocab_size": 8, "v14_structure_feature_dim": 64,
        "v14_structure_nonterminal_states": 585, "v14_structure_readout_parameters": 512,
        "kl_direction": kl, "rollout_source": rollout,
        "teacher_fraction": frac, "ema_alpha": alpha, "learning_rate": lr, "optimizer": "adam", "estimator": "exact",
        "train_steps": 200,
        "new_match_gain": acq, "old_match_forgetting": forgetting,
        "new_target_path_match_gain": acq + 0.01 * ro_t, "new_support_success_gain": acq * 0.3,
        "new_match_drawdown_from_peak": max(0.0, alpha - 0.02), "context_utility_drawdown_from_peak": 0.2 * alpha,
        "final_context_context_teacher_utility_student_occupancy": 0.8 - 0.5 * alpha,
        "initial_context_geometry_jsd_student_occupancy": jsd,
        "initial_context_geometry_tv_student_occupancy": math.sqrt(max(jsd, 0.0)),
        "initial_context_geometry_forward_reverse_logit_grad_cosine_student_occupancy": cos,
        "initial_context_geometry_log_forward_reverse_logit_grad_norm_ratio_student_occupancy": 0.2 + 0.4 * math.log(max(lam,1e-6)),
        "initial_context_geometry_teacher_top1_student_support_student_occupancy": 0.15,
        "initial_context_geometry_target_top1_student_support_student_occupancy": 0.13,
        "initial_context_v14_context_strength_realized": cval,
        "initial_context_v14_target_jsd_reachable": 1.0,
        "initial_context_v14_target_jsd_achieved": jsd,
        "initial_context_v14_competence_reliability_student_occupancy": 1.0 - 0.1 * max(competence,0),
        "initial_context_v14_competence_reliability_teacher_occupancy": 1.0,
    }


def run_aggregation_self_test() -> Dict[str, object]:
    import tempfile
    with tempfile.TemporaryDirectory(prefix="v14_agg_test_") as td:
        root = Path(td)
        agg = root / "aggregate"; agg.mkdir()
        seeds = (1,2,3,4)

        # Regression test for recovery/multi-batch provenance: condition_id is
        # batch-dependent metadata and may legitimately differ across seeds of
        # one scientific cell.  The configuration table must ignore it while
        # preserving the number of distinct execution IDs for auditability.
        provenance_rows = []
        for seed in seeds:
            r = _synthetic_row(seed, "V14_TEST_multibatch", lam=2.5)
            r["cell_name"] = "same_scientific_cell"
            r["condition_id"] = "batch_A" if seed <= 2 else "batch_B"
            provenance_rows.append(r)
        provenance_df = pd.DataFrame(provenance_rows)
        provenance_metrics = infer_metric_columns(provenance_df)
        provenance_cfg = cell_configuration_table(provenance_df, provenance_metrics)
        assert len(provenance_cfg) == 1
        assert int(provenance_cfg.loc[0, "condition_id_nunique"]) == 2
        assert bool(provenance_cfg.loc[0, "recovered_or_multibatch"])

        rows = []
        for seed in seeds:
            for lam in (0.8,1.0,2.5,4.0):
                for kl in ("forward","reverse"):
                    for frac in (0.0,0.5,1.0):
                        for a in (0.0,0.0025):
                            rows.append(_synthetic_row(seed,"V14_E71_trajectory_surface",lam=lam,kl=kl,rollout=rollout_name(frac),frac=frac,alpha=a))
        d = root / "V14_E71_trajectory_surface"; d.mkdir(); pd.DataFrame(rows).to_csv(d/"runs.csv",index=False)
        aggregate_trajectory(root,agg)

        rows=[]
        for seed in seeds:
            for lam in (0.8,1.0,2.5,4.0):
                for kl in ("forward","reverse"):
                    for a in (0.0,0.0025):
                        for lr in (0.0005,0.001,0.002):
                            rows.append(_synthetic_row(seed,"V14_E72_kl_matched_learning",lam=lam,kl=kl,rollout="student",frac=0.0,alpha=a,lr=lr))
        d=root/"V14_E72_kl_matched_learning"; d.mkdir(); pd.DataFrame(rows).to_csv(d/"runs.csv",index=False)
        aggregate_kl(root,agg)

        rows=[]
        for seed in seeds:
            for lam in (0.8,1.0,2.5,4.0):
                for cval in (0.3,0.6):
                    for kl in ("forward","reverse"):
                        for ro,frac in (("student",0.0),("teacher",1.0)):
                            for a in (0.0,0.0025):
                                rows.append(_synthetic_row(seed,"V14_E73_coupling_phase",lam=lam,cval=cval,kl=kl,rollout=ro,frac=frac,alpha=a))
        d=root/"V14_E73_coupling_phase"; d.mkdir(); pd.DataFrame(rows).to_csv(d/"runs.csv",index=False)
        aggregate_ema(root,agg)

        rows=[]
        for seed in seeds:
            for lam in (0.8,1.0,2.5,4.0):
                for jsd in (0.005,0.015):
                    for kl in ("forward","reverse"):
                        rows.append(_synthetic_row(seed,"V14_E74_iso_jsd",lam=lam,kl=kl,rollout="student",frac=0,alpha=0,target_jsd=jsd))
        d=root/"V14_E74_iso_jsd"; d.mkdir(); pd.DataFrame(rows).to_csv(d/"runs.csv",index=False)
        aggregate_iso_jsd(root,agg)

        rows=[]
        for seed in seeds:
            for lam in (0.8,1.0,2.5,4.0):
                for bias in (-0.5,0.0,0.5):
                    for frac in (0.0,1.0):
                        for a in (0.0,0.0025):
                            rows.append(_synthetic_row(seed,"V14_E75_state_competence",lam=lam,kl="reverse",rollout=rollout_name(frac),frac=frac,alpha=a,competence=bias))
        d=root/"V14_E75_state_competence"; d.mkdir(); pd.DataFrame(rows).to_csv(d/"runs.csv",index=False)
        aggregate_competence(root,agg)

        rows=[]
        for seed in seeds:
            for rho in (0.0,0.25,0.5,1.0):
                for kl in ("forward","reverse"):
                    for ro,frac in (("student",0.0),("teacher",1.0)):
                        rows.append(_synthetic_row(seed,"V14_E76_public_interactions",rho=rho,kl=kl,rollout=ro,frac=frac,alpha=0.0))
        d=root/"V14_E76_public_interactions"; d.mkdir(); pd.DataFrame(rows).to_csv(d/"runs.csv",index=False)
        aggregate_public_interactions(root,agg)

        # Sparse-smoke regression: E72 may contain no complete F/R pair at the
        # base LR.  This must produce schema-valid empty geometry outputs rather
        # than raising a KeyError.
        sparse_root = root / "sparse_smoke"
        sparse_agg = sparse_root / "aggregate"; sparse_agg.mkdir(parents=True)
        sparse_rows = []
        for seed in seeds:
            for kl in ("forward", "reverse"):
                sparse_rows.append(_synthetic_row(seed, "V14_E72_kl_matched_learning", lam=1.0, kl=kl, rollout="student", frac=0.0, alpha=0.0, lr=0.00025))
        sd = sparse_root / "V14_E72_kl_matched_learning"; sd.mkdir(parents=True)
        pd.DataFrame(sparse_rows).to_csv(sd / "runs.csv", index=False)
        aggregate_kl(sparse_root, sparse_agg)
        if not (sparse_agg / "geometry_bridge_summary.csv").exists():
            raise AssertionError("sparse E72 aggregation did not emit geometry bridge schema")

        required = (
            "trajectory_crossfit_endpoint_summary.csv", "kl_matched_learning_summary.csv",
            "geometry_bridge_summary.csv", "ema_mean_grid_optima.csv",
            "practical_lever_comparison_summary.csv", "iso_jsd_summary.csv",
            "state_competence_endpoint_summary.csv", "feature_overlap_interaction_summary.csv",
        )
        missing=[x for x in required if not (agg/x).exists()]
        if missing:
            raise AssertionError(f"synthetic aggregation missing outputs: {missing}")
        return {"status":"PASS","files_checked":len(required),"generated_files":len(list(agg.iterdir()))}

# -----------------------------------------------------------------------------
# Draft plots (generated only after aggregate)
# -----------------------------------------------------------------------------


def generate_draft_plots(agg: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"Skipping plots: matplotlib unavailable ({exc})")
        return
    figs = agg / "figures"
    figs.mkdir(exist_ok=True)

    p = agg / "trajectory_crossfit_endpoint_summary.csv"
    if p.exists():
        d = pd.read_csv(p)
        d = d[(d["metric"] == "new_match_gain") & (d["kl_direction"] == "reverse")]
        plt.figure(figsize=(7.0,4.5))
        for cval, sub in d.groupby("c"):
            sub = sub.sort_values("lambda")
            plt.plot(sub["lambda"], sub["mean"], marker="o", label=f"c={cval:g}")
        plt.axhline(0, linewidth=1)
        plt.xscale("log")
        plt.xlabel("initial-policy concentration λ")
        plt.ylabel("teacher − student acquisition (cross-fitted α*)")
        plt.legend()
        plt.tight_layout(); plt.savefig(figs / "draft_axis1_trajectory.png", dpi=180); plt.close()

    p = agg / "geometry_bridge_per_seed.csv.gz"
    if p.exists():
        d = pd.read_csv(p)
        d = d[np.isclose(d["c"],0.6) & np.isclose(d["ema_alpha"],0.0025) & (d["rollout_source"]=="student")]
        g = d.groupby("lambda")[["initial_gradient_disagreement_one_minus_cosine","forward_minus_reverse_acquisition","forward_minus_reverse_forgetting"]].mean().reset_index()
        plt.figure(figsize=(7.0,4.5))
        plt.plot(g["initial_gradient_disagreement_one_minus_cosine"], g["forward_minus_reverse_acquisition"], marker="o")
        plt.xlabel("initial F/R gradient disagreement (1 − cosine)")
        plt.ylabel("forward − reverse acquisition")
        plt.tight_layout(); plt.savefig(figs / "draft_axis2_geometry.png", dpi=180); plt.close()

    p = agg / "ema_mean_grid_optima.csv"
    if p.exists():
        d = pd.read_csv(p)
        d = d[(d["kl_direction"]=="reverse") & (d["rollout_source"]=="student")]
        plt.figure(figsize=(7.0,4.5))
        for lam, sub in d.groupby("lambda"):
            sub = sub.sort_values("c")
            plt.plot(sub["c"], sub["mean_grid_optimal_alpha"], marker="o", label=f"λ={lam:g}")
        plt.yscale("symlog", linthresh=5e-4)
        plt.xlabel("context strength c")
        plt.ylabel("observed optimal EMA α")
        plt.legend(ncol=2)
        plt.tight_layout(); plt.savefig(figs / "draft_axis3_ema_timescale.png", dpi=180); plt.close()

    p = agg / "state_competence_endpoint_summary.csv"
    if p.exists():
        d = pd.read_csv(p)
        d = d[(d["metric"]=="new_match_gain") & (d["kl_direction"]=="reverse")]
        plt.figure(figsize=(7.0,4.5))
        for lam, sub in d.groupby("lambda"):
            sub = sub.sort_values("competence_bias")
            plt.plot(sub["competence_bias"], sub["mean"], marker="o", label=f"λ={lam:g}")
        plt.axhline(0, linewidth=1)
        plt.xlabel("state-dependent competence bias")
        plt.ylabel("teacher − student acquisition (cross-fitted α*)")
        plt.legend(ncol=2)
        plt.tight_layout(); plt.savefig(figs / "draft_rollout_competence.png", dpi=180); plt.close()

    p = agg / "practical_lever_comparison_summary.csv"
    if p.exists():
        d = pd.read_csv(p)
        # one reference c/rollout curve, two interventions in acquisition/forgetting plane
        d = d[np.isclose(d["c"],0.6) & (d["rollout_source"]=="student")]
        piv = d.pivot_table(index="lambda", columns="metric", values="mean", aggfunc="first").reset_index()
        if {"ema_acquisition_gain","ema_forgetting_cost","forward_after_ema_acquisition_gain","forward_after_ema_forgetting_cost"}.issubset(piv.columns):
            plt.figure(figsize=(6.2,5.0))
            plt.scatter(piv["ema_forgetting_cost"], piv["ema_acquisition_gain"], label="frozen → optimal EMA")
            plt.scatter(piv["forward_after_ema_forgetting_cost"], piv["forward_after_ema_acquisition_gain"], label="reverse → forward after EMA")
            for _, r in piv.iterrows():
                plt.annotate(f"λ={r['lambda']:g}", (r["ema_forgetting_cost"],r["ema_acquisition_gain"]), fontsize=8)
            plt.xlabel("additional forgetting")
            plt.ylabel("additional acquisition")
            plt.legend()
            plt.tight_layout(); plt.savefig(figs / "draft_practical_levers.png", dpi=180); plt.close()


def aggregate(output_root: Path, *, plot: bool = True) -> None:
    agg = output_root / "aggregate"
    agg.mkdir(parents=True, exist_ok=True)
    aggregate_cells(output_root, agg)
    # Specialized analyses are only attempted when their raw blocks exist.
    funcs = [
        ("V14_E71_trajectory_surface", aggregate_trajectory),
        ("V14_E72_kl_matched_learning", aggregate_kl),
        ("V14_E73_coupling_phase", aggregate_ema),
        ("V14_E74_iso_jsd", aggregate_iso_jsd),
        ("V14_E75_state_competence", aggregate_competence),
        ("V14_E76_public_interactions", aggregate_public_interactions),
    ]
    for block, func in funcs:
        if (output_root / block / "runs.csv").exists():
            print(f"Aggregating {block}...")
            func(output_root, agg)
    if all((output_root / name / "runs.csv").exists() for name in (
        "V14_E77a_context_noise","V14_E77b_offpath_availability","V14_E78_core_excluded_ablation","V14_E79_novel_support","V14_E80_endpoint_high_alpha","V14_E81_optimizer"
    )):
        aggregate_appendix(output_root, agg)
    if all((output_root / name / "runs.csv").exists() for name in (
        "V14_E82a_horizon","V14_E82b_duration","V14_E82c_exact","V14_E82c_sample64","V14_E82c_sample256","V14_E82c_sample1024",
        "V14_E82d_vocab","V14_E82e_feature_dim"
    )):
        aggregate_technical(output_root, agg)
    if plot:
        generate_draft_plots(agg)

    manifest = {
        "registry": registry_summary(),
        "exploration_sha256": _sha256(Path(__file__)),
        "aggregation_note": "No giant all_runs.csv is created. Per-block runs.csv are canonical raw evidence.",
        "crossfit_note": "Optimal-alpha inferential tables use two-fold task-seed cross-fitting.",
        "matched_learning_note": "KL matched-learning tables use per-seed nondominated acquisition/forgetting LR frontiers.",
    }
    (agg / "aggregation_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--engine", default=ENGINE)
    p.add_argument("--output-root", type=Path, default=Path("results/v14_final"))
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", choices=("float32","float64"), default="float32")
    p.add_argument("--seed-batch-size", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--engine-self-test", action="store_true")
    p.add_argument("--aggregation-self-test", action="store_true")
    p.add_argument("--list", action="store_true")
    p.add_argument("--run", default=None, help="all, paper (=main+causal), main, causal, appendix, technical, or exact block name")
    p.add_argument("--aggregate", action="store_true")
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        print(json.dumps(run_static_self_tests(), indent=2, sort_keys=True))
        return
    if args.engine_self_test:
        cmd = [sys.executable, args.engine, "--self-test"]
        print("$", " ".join(cmd)); subprocess.run(cmd, check=True); return
    if args.aggregation_self_test:
        print(json.dumps(run_aggregation_self_test(), indent=2, sort_keys=True)); return
    if args.list:
        print(json.dumps(registry_summary(), indent=2, sort_keys=True))
        for b in BLOCKS:
            print(f"{b.name:34s} tier={b.tier:9s} cells={len(b.cells):6d} seeds={b.n_seeds:3d} runs={b.n_runs:9d}  {b.question}")
        return
    # Be forgiving about the common ``--run aggregate`` spelling while
    # keeping ``--aggregate`` as the documented interface.
    if args.run == "aggregate":
        print("Note: '--run aggregate' is accepted as an alias for '--aggregate'.", flush=True)
        args.aggregate = True
        args.run = None
    if args.run:
        for b in selected_blocks(args.run):
            execute_block(
                b, args.output_root, engine=args.engine,
                device=args.device, overwrite=args.overwrite, smoke=args.smoke,
                seed_batch_size=args.seed_batch_size, dtype=args.dtype,
            )
    if args.aggregate:
        aggregate(args.output_root, plot=not args.no_plots)
    if not args.run and not args.aggregate:
        raise SystemExit("Choose --self-test, --engine-self-test, --aggregation-self-test, --list, --run ..., or --aggregate")


if __name__ == "__main__":
    main()
