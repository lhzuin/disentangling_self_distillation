#!/usr/bin/env python3
"""Final v14 mechanism-isolation toy model for contextualized self-distillation.

V14 is a clean paper-facing model, not another historical compatibility surface.
It deliberately exposes only the quantities that survived the v13.2 model-
selection tournament:

  * feature_overlap rho_phi: shared representation / interference,
  * initial_policy_concentration lambda: inverse-temperature-like confidence
    of the initial policy,
  * context_strength c: same-weight behavioral displacement induced by
    privileged information,
  * KL direction, rollout policy, and EMA teacher update rate alpha.

The core task fixes the remaining construction choices: an orthogonal new
readout (rho_W=0), no support-placement bias (beta=0), a reachable projected
2-mode target with masses (.30,.15), no target heterogeneity, no context noise,
full context availability at every prefix, no private features/anchoring/
prompt holdout, and H=4,K=8,D=64.

Scientifically useful controls are implemented as *default-off ablations* and
kept outside CoreTaskSpec.  They are applied after the core construction so the
main model cannot silently depend on them:

  * readout compatibility rho_W,
  * support-placement beta,
  * target-mass / concentration controls,
  * target-compatible privileged features (endpoint-null causal control),
  * fixed Gaussian feature-space or behavior-space context corruption,
  * off-target-prefix context availability,
  * a source-independent state-dependent teacher-competence intervention,
  * per-seed calibration to a requested initial student/teacher JSD (iso-distance
    causal test of lambda).

Implementation strategy
=======================
V14 embeds its validated task/optimizer/rollout/KL/metric backend directly in
this file.  The public API does not expose the historical nuisance controls;
the corresponding internal fields are fixed constants that preserve the
validated task generator and optimizer semantics.  The contextual teacher is
implemented here as a weight-dependent input feature intervention.

The file contains extensive self-tests for:
  * core specification hygiene and default-off ablations,
  * exact c=0 identity,
  * context calibration,
  * endpoint-null target-compatible construction,
  * exact alpha=1 synchronization,
  * deterministic paired Gaussian perturbations,
  * analytic forward/reverse logit-gradient formulae,
  * JSD calibration monotonicity/accuracy.

Run contract
============
The exploration driver writes JSON specs consumed here.  Each block produces:
  runs.csv       one row per task seed and cell
  history.csv    only for cells explicitly marked record_history=true
  manifest.json  source hashes, environment, CLI and exact spec
  experiment_spec.json

Typical use:
  python toy_model/contextual_sd_toy_v14.py --self-test

  python toy_model/contextual_sd_toy_v14.py --spec block.json --output-dir results/v14/block
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import shutil
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor
Prefix = Tuple[int, ...]

V14_HORIZON = 4
V14_VOCAB_SIZE = 8
V14_FEATURE_DIM = 64
V14_TARGET_PRIMARY = 0.30
V14_TARGET_SECONDARY = 0.15
V14_CORE_READOUT_COMPATIBILITY = 0.0
V14_CORE_SUPPORT_PLACEMENT_BIAS = 0.0
V14_PINV_RTOL = 1e-7
V14_VERSION = "14.0.1"


# -----------------------------------------------------------------------------
# Embedded frozen backend
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class RegimeSpec:
    """Task properties; each field has an operational, separately logged meaning."""

    name: str
    horizon: int
    feature_similarity: float
    readout_similarity: float
    surprise_strength: float
    target_complexity: float
    context_strength: float
    offpath_context_multiplier: float
    context_capability: float = 0.90
    context_noise_std: float = 0.15
    primary_mass: float = 0.30
    secondary_mass: float = 0.15
    # --- v5 additions; the defaults below reproduce v4 exactly -------------
    prior_sharpness: float = 1.0
    state_private_fraction: float = 0.0
    target_realization: str = "prescribed"
    target_temperature: float = 1.0
    support_width: int = 2
    # --- v8 additions; the defaults below reproduce v7 exactly --------------
    context_anchoring: float = 0.0
    prompt_holdout_fraction: float = 0.0
    teacher_construction: str = "capped_mixture"

    def validate(self, vocab_size: int) -> None:
        if self.horizon < 2:
            raise ValueError("horizon must be at least 2")
        for name, value in (
            ("feature_similarity", self.feature_similarity),
            ("readout_similarity", self.readout_similarity),
        ):
            if not -1.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [-1, 1]")
        if self.surprise_strength < 0 or self.target_complexity < 0:
            raise ValueError("surprise_strength and target_complexity must be nonnegative")
        if not 0.0 <= self.context_strength <= 1.0:
            raise ValueError("context_strength must lie in [0, 1]")
        if not 0.0 <= self.offpath_context_multiplier <= 1.0:
            raise ValueError("offpath_context_multiplier must lie in [0, 1]")
        if not 0.0 < self.context_capability < 1.0:
            raise ValueError("context_capability must lie strictly between 0 and 1")
        if self.context_noise_std < 0.0:
            raise ValueError("context_noise_std must be nonnegative")
        if not (self.primary_mass >= self.secondary_mass > 0):
            raise ValueError("target masses must satisfy primary_mass >= secondary_mass > 0")
        if self.primary_mass + self.secondary_mass >= 1.0:
            raise ValueError("the two target modes must leave positive residual mass")
        if vocab_size < 4:
            raise ValueError("vocab_size must be at least 4 for meaningful multimodality")
        if self.prior_sharpness <= 0:
            raise ValueError("prior_sharpness must be positive")
        if not 0.0 <= self.state_private_fraction < 1.0:
            raise ValueError("state_private_fraction must lie in [0, 1)")
        if self.target_realization not in {"prescribed", "projected"}:
            raise ValueError("target_realization must be 'prescribed' or 'projected'")
        if self.target_temperature <= 0:
            raise ValueError("target_temperature must be positive")
        if self.support_width not in {1, 2}:
            raise ValueError("support_width must be 1 or 2")
        if not 0.0 <= self.context_anchoring <= 1.0:
            raise ValueError("context_anchoring must lie in [0, 1]")
        if not 0.0 <= self.prompt_holdout_fraction < 1.0:
            raise ValueError("prompt_holdout_fraction must lie in [0, 1)")
        if self.teacher_construction not in {"capped_mixture", "hard_context"}:
            raise ValueError("teacher_construction must be 'capped_mixture' or 'hard_context'")


@dataclass(frozen=True)
class Condition:
    analysis_block: str
    variant: str
    regime: RegimeSpec
    kl_direction: str
    rollout_source: str
    ema_alpha: float
    learning_rate: float
    optimizer: str = "adam"
    estimator: str = "exact"
    teacher_fraction: float = 0.5

    def validate(self) -> None:
        if self.kl_direction not in {"forward", "reverse"}:
            raise ValueError(f"invalid KL direction: {self.kl_direction}")
        if self.rollout_source not in {"student", "teacher", "mixture"}:
            raise ValueError(f"invalid rollout source: {self.rollout_source}")
        if self.optimizer not in {"adam", "sgd", "normalized_sgd"}:
            raise ValueError(f"invalid optimizer: {self.optimizer}")
        if self.estimator not in {"exact", "sampled"}:
            raise ValueError(f"invalid estimator: {self.estimator}")
        if not 0.0 <= self.ema_alpha < 1.0:
            raise ValueError("ema_alpha must lie in [0, 1)")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0.0 <= self.teacher_fraction <= 1.0:
            raise ValueError("teacher_fraction must lie in [0, 1]")


@dataclass(frozen=True)
class RunConfig:
    vocab_size: int = 8
    feature_dim: int = 64
    train_steps: int = 200
    record_every: int = 20
    prompts_per_step: int = 0
    n_task_seeds: int = 24
    seed_start: int = 1000
    optimization_seed: int = 904_021
    sampled_trajectories_per_step: int = 256
    saved_trajectories_per_policy: int = 4
    trajectory_seeds_per_condition: int = 3
    old_temperature: float = 1.0
    weight_decay: float = 0.0
    gradient_clip_norm: float = 10.0
    dtype: str = "float32"

    def validate(self) -> None:
        if self.vocab_size < 4:
            raise ValueError("vocab_size must be at least 4")
        if self.feature_dim < 4:
            raise ValueError("feature_dim must be at least 4")
        if self.train_steps < 1 or self.record_every < 1:
            raise ValueError("train_steps and record_every must be positive")
        if self.n_task_seeds < 2:
            raise ValueError("at least two paired task seeds are required")
        if self.sampled_trajectories_per_step < 1:
            raise ValueError("sampled_trajectories_per_step must be positive")
        if self.old_temperature <= 0:
            raise ValueError("old_temperature must be positive")


STANDARD_MODERATE = RegimeSpec(
    name="standard_moderate",
    horizon=4,
    feature_similarity=0.55,
    readout_similarity=-0.35,
    surprise_strength=0.50,
    target_complexity=0.30,
    context_strength=0.6,
    offpath_context_multiplier=1.0,
    prior_sharpness=2.5,
    target_realization="projected",
)
def enumerate_prefixes(horizon: int, vocab_size: int) -> Tuple[List[Prefix], Dict[Prefix, int]]:
    prefixes: List[Prefix] = [()]
    frontier: List[Prefix] = [()]
    for _depth in range(1, horizon):
        frontier = [prefix + (token,) for prefix in frontier for token in range(vocab_size)]
        prefixes.extend(frontier)
    return prefixes, {prefix: i for i, prefix in enumerate(prefixes)}


def tree_tensors(
    prefixes: Sequence[Prefix], index: Mapping[Prefix, int], horizon: int, vocab_size: int
) -> Tuple[List[Tensor], Tensor]:
    by_depth = [
        torch.tensor([i for i, prefix in enumerate(prefixes) if len(prefix) == depth], dtype=torch.long)
        for depth in range(horizon)
    ]
    children = torch.full((len(prefixes), vocab_size), -1, dtype=torch.long)
    for i, prefix in enumerate(prefixes):
        if len(prefix) < horizon - 1:
            for token in range(vocab_size):
                children[i, token] = index[prefix + (token,)]
    return by_depth, children


def _unit_rows(x: Tensor, eps: float = 1e-12) -> Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


def _center_readout(w: Tensor) -> Tensor:
    """Remove softmax-invariant common-token directions."""

    return w - w.mean(dim=0, keepdim=True)


def _orthogonal_like(base: Tensor, raw: Tensor, eps: float = 1e-12) -> Tensor:
    base_flat = base.reshape(-1)
    raw_flat = raw.reshape(-1)
    residual = raw_flat - torch.dot(raw_flat, base_flat) / torch.dot(base_flat, base_flat).clamp_min(eps) * base_flat
    if residual.norm() < 1e-8:
        residual = torch.roll(base_flat, shifts=1)
        residual = residual - torch.dot(residual, base_flat) / torch.dot(base_flat, base_flat).clamp_min(eps) * base_flat
    residual = residual * (base_flat.norm() / residual.norm().clamp_min(eps))
    return residual.reshape_as(base)


@dataclass
class ProblemBatch:
    phi_old: Tensor
    phi_new: Tensor
    w_old: Tensor
    w_new_latent: Tensor
    q_old: Tensor
    q_new_target: Tensor
    q_context_target: Tensor
    target_support: Tensor
    old_support: Tensor
    on_target_tree: Tensor
    state_is_trainable: Tensor
    state_prompt_id: Tensor
    task_diagnostics: Dict[str, Tensor]


@dataclass
class PreparedProblem:
    prefixes: List[Prefix]
    index: Dict[Prefix, int]
    by_depth: List[Tensor]
    children: Tensor
    problem: ProblemBatch


def construct_problem_batch(
    regime: RegimeSpec,
    cfg: RunConfig,
    task_seeds: Sequence[int],
    prefixes: Sequence[Prefix],
    index: Mapping[Prefix, int],
    device: torch.device,
    dtype: torch.dtype,
) -> ProblemBatch:
    """Construct paired random task instances with exact geometry.

    The target distribution is explicit rather than an accidental softmax of
    Gaussian logits.  A centered latent readout supplies a controlled conflict
    score; a prior-support term controls surprise; and an independent state-wise
    residual controls representational difficulty.  The two highest resulting
    scores receive fixed target masses, guaranteeing multimodality.
    """

    n_states = len(prefixes)
    k = cfg.vocab_size
    d = cfg.feature_dim
    batches: Dict[str, List[Tensor]] = {
        name: []
        for name in (
            "phi_old",
            "phi_new",
            "w_old",
            "w_new",
            "q_old",
            "q_target",
            "q_context_target",
            "target_support",
            "old_support",
            "on_tree",
            "state_is_trainable",
            "state_prompt_id",
        )
    }
    diagnostics: Dict[str, List[Tensor]] = {}

    for seed in task_seeds:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))

        phi_old = _unit_rows(torch.randn(n_states, d, generator=generator, dtype=torch.float64)) * math.sqrt(d)
        raw_new = torch.randn(n_states, d, generator=generator, dtype=torch.float64)
        projection = (raw_new * phi_old).sum(dim=1, keepdim=True) / (phi_old.square().sum(dim=1, keepdim=True))
        phi_perp = raw_new - projection * phi_old
        phi_perp = _unit_rows(phi_perp) * math.sqrt(d)
        rho_f = regime.feature_similarity
        phi_new = rho_f * phi_old + math.sqrt(max(0.0, 1.0 - rho_f**2)) * phi_perp

        # P2: state-private feature directions.  These are identical for the
        # old and the new task, so the readout conflict rho_r acts on them at
        # full strength, while their *privacy* means a state's contribution can
        # only be learned where that state actually receives gradient weight.
        # feature_similarity therefore continues to describe the shared block,
        # which is the block the Hiratani/Lee geometry refers to.
        lam = regime.state_private_fraction
        if lam > 0.0:
            private = torch.eye(n_states, dtype=torch.float64) * math.sqrt(n_states)
            phi_old = torch.cat(
                [math.sqrt(1.0 - lam) * phi_old, math.sqrt(lam) * private], dim=1
            )
            phi_new = torch.cat(
                [math.sqrt(1.0 - lam) * phi_new, math.sqrt(lam) * private], dim=1
            )

        w_old = _center_readout(torch.randn(k, d, generator=generator, dtype=torch.float64) / math.sqrt(d))
        raw_readout = _center_readout(torch.randn(k, d, generator=generator, dtype=torch.float64) / math.sqrt(d))
        w_perp = _center_readout(_orthogonal_like(w_old, raw_readout))
        # Re-orthogonalize after centering to remove numerical leakage.
        w_perp = _orthogonal_like(w_old, w_perp)
        rho_r = regime.readout_similarity
        w_new = _center_readout(rho_r * w_old + math.sqrt(max(0.0, 1.0 - rho_r**2)) * w_perp)

        # P3: entrenchment.  Scaling the *initial readout* sharpens the prior
        # policy and lowers student mass on the eventual teacher modes without
        # rescaling the student's logits during training, so the support gap
        # and the optimisation step size stay independent factors.
        if regime.prior_sharpness != 1.0:
            w_old = w_old * regime.prior_sharpness
            w_new = w_new * regime.prior_sharpness

        # P2 (cont.): the private block starts at zero, so q_old, the initial
        # student policy, and every v4 diagnostic are unchanged at lam = 0 and
        # unchanged at initialisation for any lam.
        if lam > 0.0:
            pad = torch.zeros(k, n_states, dtype=torch.float64)
            w_old = torch.cat([w_old, pad], dim=1)
            w_new = torch.cat([w_new, pad.clone()], dim=1)

        old_logits = phi_old @ w_old.T / cfg.old_temperature
        q_old = F.softmax(old_logits, dim=-1)
        initial_new_logits = phi_new @ w_old.T / cfg.old_temperature
        p_initial_new = F.softmax(initial_new_logits, dim=-1)

        latent_scores = phi_new @ w_new.T
        prior_surprisal = -torch.log(p_initial_new.clamp_min(torch.finfo(torch.float64).tiny))
        complexity_noise = torch.randn(n_states, k, generator=generator, dtype=torch.float64)
        complexity_noise = complexity_noise - complexity_noise.mean(dim=-1, keepdim=True)
        complexity_noise = complexity_noise / complexity_noise.std(dim=-1, keepdim=True).clamp_min(1e-12)
        target_scores = (
            latent_scores
            + regime.surprise_strength * prior_surprisal
            + regime.target_complexity * complexity_noise
        )
        modes = torch.topk(target_scores, k=2, dim=-1).indices
        q_target = torch.full(
            (n_states, k),
            (1.0 - regime.primary_mass - regime.secondary_mass) / (k - 2),
            dtype=torch.float64,
        )
        q_target.scatter_(1, modes[:, :1], regime.primary_mass)
        q_target.scatter_(1, modes[:, 1:2], regime.secondary_mass)

        # P1: realizability.  In v4 the target was prescribed state by state
        # and was not in the family softmax(phi_new @ W.T); the realized
        # relative fit residual was 0.75-0.93, so no setting of W could reach
        # it and both divergences converged to nearly the same projection.
        # "projected" replaces the target with its least-squares image inside
        # the expressible family, so perfect acquisition becomes attainable and
        # any residual forward/reverse difference is an optimisation-path
        # effect rather than an artefact of a common unreachable residual.
        prescribed_centered = torch.log(q_target)
        prescribed_centered = prescribed_centered - prescribed_centered.mean(dim=-1, keepdim=True)
        fitted_readout_target = torch.linalg.lstsq(phi_new, prescribed_centered).solution
        if regime.target_realization == "projected":
            q_target = F.softmax(
                (phi_new @ fitted_readout_target) / regime.target_temperature, dim=-1
            )
            modes = torch.topk(q_target, k=2, dim=-1).indices

        # Context is informative but not assumed perfect.  Noise is sampled
        # once for each task/state and then held fixed throughout training; it
        # therefore represents systematic contextual imperfection rather than
        # per-step optimization noise.
        context_noise = torch.randn(n_states, k, generator=generator, dtype=torch.float64)
        context_noise = context_noise - context_noise.mean(dim=-1, keepdim=True)
        context_noise = context_noise / context_noise.std(dim=-1, keepdim=True).clamp_min(1e-12)
        context_logits = torch.log(q_target) + regime.context_noise_std * context_noise
        if regime.teacher_construction == "hard_context":
            # P4: declared hard-contradiction endpoint.  The contextual target
            # removes the entrenched top-1 answer instead of merely
            # down-weighting it, so the teacher can express a target with
            # essentially no mass on the prior answer.
            entrenched = torch.topk(p_initial_new, k=1, dim=-1).indices
            context_logits = context_logits.scatter(
                1, entrenched, torch.full_like(entrenched, -30.0, dtype=context_logits.dtype)
            )
        q_context_target = F.softmax(context_logits, dim=-1)
        # Q3: the on-tree set gates the off-path context multiplier.  With
        # width 2 the second mode is frequently a token the student already
        # favours, which widens the tree and weakens the intended barrier; a
        # width of 1 makes it a single-path barrier.
        target_support = torch.zeros(n_states, k, dtype=torch.bool)
        if regime.support_width >= 2:
            target_support.scatter_(1, modes, True)
        else:
            target_support.scatter_(1, modes[:, :1], True)

        old_modes = torch.topk(q_old, k=2, dim=-1).indices
        old_support = torch.zeros(n_states, k, dtype=torch.bool)
        old_support.scatter_(1, old_modes, True)

        on_tree = torch.ones(n_states, dtype=torch.bool)
        for state_idx, prefix in enumerate(prefixes):
            ancestor: Prefix = ()
            for token in prefix:
                ancestor_idx = index[ancestor]
                if not bool(target_support[ancestor_idx, token]):
                    on_tree[state_idx] = False
                    break
                ancestor = ancestor + (token,)

        # Q4: verify the support barrier instead of assuming it.  Occupancy
        # sums to one at every depth, so averaging on-tree mass over all states
        # is dominated by the root and the shallow states, which are always
        # on-tree; that measurement suggested a ratio near 1.8x when the true
        # terminal-depth ratio is several orders of magnitude.  Measure at the
        # terminal depth only.
        # R2: every state belongs to the depth-1 subtree ("prompt") it descends
        # from; the root belongs to none and is always trainable.  Held-out
        # prompts never receive gradient, so acquisition on them measures
        # generalisation rather than fitting.
        prompt_of = torch.tensor(
            [(-1 if len(pfx) == 0 else pfx[0]) for pfx in prefixes], dtype=torch.long
        )
        n_prompts = k
        n_holdout = int(round(regime.prompt_holdout_fraction * n_prompts))
        if n_holdout >= n_prompts:
            raise ValueError("prompt_holdout_fraction leaves no trainable prompts")
        is_holdout_prompt = torch.zeros(n_prompts, dtype=torch.bool)
        if n_holdout > 0:
            # Guarded: drawing from ``generator`` unconditionally would advance
            # the RNG stream and change every subsequent draw, so a v8 run with
            # prompt_holdout_fraction = 0 would stop being numerically identical
            # to the corresponding v7 run.
            is_holdout_prompt[torch.randperm(n_prompts, generator=generator)[:n_holdout]] = True
        state_is_holdout = torch.where(
            prompt_of >= 0, is_holdout_prompt[prompt_of.clamp_min(0)],
            torch.zeros_like(prompt_of, dtype=torch.bool),
        )
        state_is_trainable = ~state_is_holdout

        diag_by_depth, diag_children = tree_tensors(prefixes, index, regime.horizon, k)

        def _terminal_on_tree_mass(policy: Tensor) -> Tensor:
            occ = torch.zeros(n_states, dtype=torch.float64)
            occ[0] = 1.0
            for depth in range(regime.horizon - 1):
                ids = diag_by_depth[depth]
                mass = occ[ids].unsqueeze(-1) * policy[ids, :]
                occ = occ.index_add(
                    0, diag_children[ids].reshape(-1), mass.reshape(-1)
                )
            terminal = diag_by_depth[regime.horizon - 1]
            total = occ[terminal].sum().clamp_min(1e-300)
            return occ[terminal][on_tree[terminal]].sum() / total

        # The initial teacher reproduces PrefixTreeDistillation.teacher_components
        # at initialisation, where W_teacher == w_old so the base equals the
        # initial student policy on the new task.
        diag_multiplier = torch.where(
            on_tree,
            torch.ones(n_states, dtype=torch.float64),
            torch.full((n_states,), regime.offpath_context_multiplier, dtype=torch.float64),
        ).unsqueeze(-1)
        diag_rho = (
            regime.context_strength * regime.context_capability * diag_multiplier
        )
        q_teacher_initial = (
            1.0 - diag_rho
        ) * p_initial_new + diag_rho * q_context_target

        student_on_tree_mass = _terminal_on_tree_mass(p_initial_new)
        teacher_on_tree_mass = _terminal_on_tree_mass(q_teacher_initial)
        on_tree_mass_ratio = teacher_on_tree_mass / student_on_tree_mass.clamp_min(1e-300)

        feature_cos = F.cosine_similarity(phi_old, phi_new, dim=-1).mean()
        readout_cos = F.cosine_similarity(w_old.reshape(1, -1), w_new.reshape(1, -1), dim=-1).squeeze(0)
        mode_conflict = 1.0 - (modes[:, 0] == old_modes[:, 0]).to(torch.float64).mean()
        q_entropy = -(q_target * torch.log(q_target)).sum(dim=-1).mean()
        mode_support_mean = p_initial_new.gather(1, modes).mean()
        primary_support_mean = p_initial_new.gather(1, modes[:, :1]).mean()
        primary_support_q10 = torch.quantile(p_initial_new.gather(1, modes[:, :1]).squeeze(1), 0.10)
        support_below_1e2 = (p_initial_new.gather(1, modes[:, :1]) < 1e-2).to(torch.float64).mean()
        support_below_1e4 = (p_initial_new.gather(1, modes[:, :1]) < 1e-4).to(torch.float64).mean()
        target_centered_logits = torch.log(q_target) - torch.log(q_target).mean(dim=-1, keepdim=True)
        fitted_readout = torch.linalg.lstsq(phi_new, target_centered_logits).solution
        target_fit_residual = (
            (phi_new @ fitted_readout - target_centered_logits).norm()
            / target_centered_logits.norm().clamp_min(1e-12)
        )
        context_target_tv = 0.5 * (q_context_target - q_target).abs().sum(dim=-1).mean()
        # P5: is the target reachable at all?  n_effective_parameters counts
        # the free readout entries; n_target_constraints counts the
        # softmax-identifiable target logits.  capacity_ratio < 1 means the
        # model is structurally unable to represent its own target, which
        # suppresses every method difference simultaneously.
        # With a private block, feature_similarity_realized measures the *total*
        # cosine, which is (1 - lam) * rho_f + lam because the private
        # directions are shared by both tasks.  The shared block is the one the
        # Hiratani/Lee geometry refers to, so it is reported separately.
        shared_feature_cos = (
            F.cosine_similarity(phi_old[:, :d], phi_new[:, :d], dim=-1).mean()
        )
        n_parameters = torch.tensor(
            float(w_old.shape[0] * w_old.shape[1]), dtype=torch.float64
        )
        n_constraints = torch.tensor(float(n_states * (k - 1)), dtype=torch.float64)
        capacity_ratio = n_parameters / n_constraints.clamp_min(1.0)

        for name, value in {
            "feature_similarity_realized": feature_cos,
            "readout_similarity_realized": readout_cos,
            "primary_mode_conflict_rate": mode_conflict,
            "target_entropy": q_entropy,
            "initial_mode_support_mean": mode_support_mean,
            "initial_primary_support_mean": primary_support_mean,
            "initial_primary_support_q10": primary_support_q10,
            "initial_primary_support_below_1e2": support_below_1e2,
            "initial_primary_support_below_1e4": support_below_1e4,
            "target_tree_state_fraction": on_tree.to(torch.float64).mean(),
            "trainable_state_fraction": state_is_trainable.to(torch.float64).mean(),
            "student_terminal_on_tree_mass": student_on_tree_mass,
            "teacher_terminal_on_tree_mass": teacher_on_tree_mass,
            "terminal_on_tree_mass_ratio": on_tree_mass_ratio,
            "target_linear_fit_relative_error": target_fit_residual,
            "context_target_tv": context_target_tv,
            "feature_similarity_shared_block": shared_feature_cos,
            "n_effective_parameters": n_parameters,
            "n_target_constraints": n_constraints,
            "capacity_ratio": capacity_ratio,
        }.items():
            diagnostics.setdefault(name, []).append(value)

        batches["phi_old"].append(phi_old)
        batches["phi_new"].append(phi_new)
        batches["w_old"].append(w_old)
        batches["w_new"].append(w_new)
        batches["q_old"].append(q_old)
        batches["q_target"].append(q_target)
        batches["q_context_target"].append(q_context_target)
        batches["target_support"].append(target_support)
        batches["old_support"].append(old_support)
        batches["on_tree"].append(on_tree)
        batches["state_is_trainable"].append(state_is_trainable)
        batches["state_prompt_id"].append(prompt_of)

    def stack(name: str) -> Tensor:
        return torch.stack(batches[name]).to(device=device)

    return ProblemBatch(
        phi_old=stack("phi_old").to(dtype=dtype),
        phi_new=stack("phi_new").to(dtype=dtype),
        w_old=stack("w_old").to(dtype=dtype),
        w_new_latent=stack("w_new").to(dtype=dtype),
        q_old=stack("q_old").to(dtype=dtype),
        q_new_target=stack("q_target").to(dtype=dtype),
        q_context_target=stack("q_context_target").to(dtype=dtype),
        target_support=stack("target_support").bool(),
        old_support=stack("old_support").bool(),
        on_target_tree=stack("on_tree").bool(),
        state_is_trainable=stack("state_is_trainable").bool(),
        state_prompt_id=stack("state_prompt_id").long(),
        task_diagnostics={
            name: torch.stack(values).to(device=device, dtype=dtype)
            for name, values in diagnostics.items()
        },
    )


def prepare_problem(
    regime: RegimeSpec,
    cfg: RunConfig,
    task_seeds: Sequence[int],
    device: torch.device,
    dtype: torch.dtype,
) -> PreparedProblem:
    prefixes, index = enumerate_prefixes(regime.horizon, cfg.vocab_size)
    if regime.state_private_fraction > 0.0 and len(prefixes) > 2048:
        # The private block is one column per state, so its cost is quadratic
        # in the number of prefixes.  Locality regimes are meant to run on
        # small trees; fail early rather than exhausting memory mid-suite.
        raise ValueError(
            f"state_private_fraction > 0 requires at most 2048 prefixes, "
            f"but regime '{regime.name}' has {len(prefixes)} "
            f"(horizon={regime.horizon}, vocab={cfg.vocab_size}). "
            f"Reduce horizon or vocab_size for locality regimes."
        )
    by_depth, children = tree_tensors(
        prefixes, index, regime.horizon, cfg.vocab_size
    )
    problem = construct_problem_batch(
        regime, cfg, task_seeds, prefixes, index, device, dtype
    )
    return PreparedProblem(prefixes, index, by_depth, children, problem)


def problem_key(regime: RegimeSpec, cfg: RunConfig, task_seeds: Sequence[int]) -> Tuple[object, ...]:
    return (
        *asdict(regime).values(),
        cfg.vocab_size,
        cfg.feature_dim,
        cfg.old_temperature,
        *task_seeds,
    )


# ---------------------------------------------------------------------------
# Batched PyTorch model
# ---------------------------------------------------------------------------


class PrefixTreeDistillation(nn.Module):
    """A batch of paired task instances trained as independent replicas."""

    def __init__(
        self,
        problem: ProblemBatch,
        condition: Condition,
        cfg: RunConfig,
        by_depth: Sequence[Tensor],
        children: Tensor,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.condition = condition
        self.cfg = cfg
        self.batch_size = problem.w_old.shape[0]
        self.horizon = condition.regime.horizon
        self.device = device

        for name in (
            "phi_old",
            "phi_new",
            "w_old",
            "w_new_latent",
            "q_old",
            "q_new_target",
            "q_context_target",
            "target_support",
            "old_support",
            "on_target_tree",
            "state_is_trainable",
            "state_prompt_id",
        ):
            self.register_buffer(name, getattr(problem, name))
        self.task_diagnostics = problem.task_diagnostics
        self.by_depth = [indices.to(device=device) for indices in by_depth]
        # ``nn.Module`` already defines a children() method, so using
        # ``children`` as a buffer name raises a KeyError in register_buffer.
        self.register_buffer("tree_children", children.to(device=device))

        self.W = nn.Parameter(problem.w_old.clone())
        self.register_buffer("W_initial", problem.w_old.clone())
        self.register_buffer("W_teacher", problem.w_old.clone())

        # R1: the contextual shift as it would be calibrated at initialisation,
        # i.e. read off the base teacher before any weights have moved.  Only
        # consulted when regime.context_anchoring > 0.
        with torch.no_grad():
            base_logits_0 = (
                torch.einsum("bsd,bkd->bsk", self.phi_new, self.W_teacher)
                / self.cfg.old_temperature
            )
            tiny_0 = torch.finfo(base_logits_0.dtype).tiny
            z_star_0 = torch.log(self.q_context_target.clamp_min(tiny_0))
            self.register_buffer("context_fixed_shift", z_star_0 - base_logits_0)

        # R2: how many prompts are available to sample from each step.
        self.n_prompts = int(self.state_prompt_id.max().item()) + 1

        if condition.optimizer == "adam":
            self.optimizer = torch.optim.Adam(
                [self.W], lr=condition.learning_rate, weight_decay=cfg.weight_decay
            )
        elif condition.optimizer == "sgd":
            self.optimizer = torch.optim.SGD(
                [self.W], lr=condition.learning_rate, weight_decay=cfg.weight_decay
            )
        else:
            self.optimizer = None

    def logits(self, task: str) -> Tensor:
        if task == "new":
            phi = self.phi_new
        elif task == "old":
            phi = self.phi_old
        else:
            raise ValueError(f"unknown task: {task}")
        return torch.einsum("bsd,bkd->bsk", phi, self.W) / self.cfg.old_temperature

    def student_log_policy(self, task: str) -> Tensor:
        return F.log_softmax(self.logits(task), dim=-1)

    @torch.no_grad()
    def teacher_components(self) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return base, contextual endpoint, effective weight, and teacher.

        The teacher is an arithmetic mixture in probability space:

            q_t = (1-rho_s) p_ema,t + rho_s q_context,
            rho_s = context_strength * context_capability * path_multiplier.

        Because context_capability is strictly below one, even full context
        retains a nonzero direct dependence on the frozen/EMA weight teacher.
        This avoids the degeneracy of logit interpolation, where context=1
        deleted the EMA contribution, and it does not preserve artificial
        low-support barriers through a geometric mean.
        """
        base_logits = (
            torch.einsum("bsd,bkd->bsk", self.phi_new, self.W_teacher)
            / self.cfg.old_temperature
        )
        base_probs = F.softmax(base_logits, dim=-1)
        regime = self.condition.regime
        context_multiplier = torch.where(
            self.on_target_tree,
            torch.ones_like(self.on_target_tree, dtype=base_logits.dtype),
            torch.full_like(
                self.on_target_tree,
                regime.offpath_context_multiplier,
                dtype=base_logits.dtype,
            ),
        )
        rho = (
            regime.context_strength
            * regime.context_capability
            * context_multiplier.unsqueeze(-1)
        )
        anchoring = regime.context_anchoring
        if anchoring > 0.0:
            # R1: the contextual shift is read with the CURRENT weights.  With
            # anchoring = 1 it is calibrated once, at initialisation, and then
            # rides on a moving base, so a drifting teacher progressively
            # mis-applies its own context.  anchoring = 0 falls through to the
            # v7 probability mixture below and is bit-for-bit identical to it.
            tiny = torch.finfo(base_logits.dtype).tiny
            z_star = torch.log(self.q_context_target.clamp_min(tiny))
            delta = (1.0 - anchoring) * (z_star - base_logits) + anchoring * self.context_fixed_shift
            teacher_probs = F.softmax(base_logits + rho * delta, dim=-1)
            return base_probs, self.q_context_target, rho, teacher_probs

        teacher_probs = (1.0 - rho) * base_probs + rho * self.q_context_target
        return base_probs, self.q_context_target, rho, teacher_probs

    @torch.no_grad()
    def teacher_log_policy(self) -> Tensor:
        teacher_probs = self.teacher_components()[-1]
        tiny = torch.finfo(teacher_probs.dtype).tiny
        return teacher_probs.clamp_min(tiny).log()

    def occupancy(self, policy: Tensor) -> Tensor:
        """Exact state occupancy; each depth has unit mass for each replica."""

        b, n_states, _k = policy.shape
        occ = torch.zeros(b, n_states, device=policy.device, dtype=policy.dtype)
        occ[:, 0] = 1.0
        for depth in range(self.horizon - 1):
            indices = self.by_depth[depth]
            mass = occ[:, indices, None] * policy[:, indices, :]
            child_indices = self.tree_children[indices].reshape(1, -1).expand(b, -1)
            occ.scatter_add_(1, child_indices, mass.reshape(b, -1))
        return occ

    @torch.no_grad()
    def sampled_occupancy(
        self, p: Tensor, q: Tensor, generator: Optional[torch.Generator]
    ) -> Tensor:
        """Monte Carlo occupancy from complete student/teacher trajectories."""

        b, n_states, k = p.shape
        n = self.cfg.sampled_trajectories_per_step
        states = torch.zeros(b, n, dtype=torch.long, device=self.device)
        counts = torch.zeros(b, n_states, dtype=p.dtype, device=self.device)
        batch_index = torch.arange(b, device=self.device)[:, None].expand(b, n)
        if self.condition.rollout_source == "mixture":
            choose_teacher = torch.rand(b, n, generator=generator, device=self.device) < self.condition.teacher_fraction
        else:
            choose_teacher = None

        for depth in range(self.horizon):
            counts.scatter_add_(1, states, torch.ones_like(states, dtype=p.dtype))
            p_here = p[batch_index, states]
            q_here = q[batch_index, states]
            if self.condition.rollout_source == "student":
                probs = p_here
            elif self.condition.rollout_source == "teacher":
                probs = q_here
            else:
                probs = torch.where(choose_teacher[..., None], q_here, p_here)
            actions = torch.multinomial(probs.reshape(-1, k), 1, generator=generator).reshape(b, n)
            if depth < self.horizon - 1:
                states = self.tree_children[states, actions]
        return counts / float(n)

    @torch.no_grad()
    def rollout_weights(
        self, p: Tensor, q: Tensor, generator: Optional[torch.Generator]
    ) -> Tensor:
        if self.condition.estimator == "sampled":
            occ = self.sampled_occupancy(p, q, generator)
        else:
            d_p = self.occupancy(p)
            d_q = self.occupancy(q)
            if self.condition.rollout_source == "student":
                occ = d_p
            elif self.condition.rollout_source == "teacher":
                occ = d_q
            else:
                beta = self.condition.teacher_fraction
                occ = (1.0 - beta) * d_p + beta * d_q
        return occ / float(self.horizon)

    def gradient_geometry(self) -> Dict[str, Tensor]:
        """Exact local gradient diagnostics for both KLs and rollout sources.

        These counterfactual gradients are evaluated at the same parameters.
        They reveal whether two nominal methods actually provide distinct
        optimization signals before final metrics are compared.
        """

        student_log = self.student_log_policy("new")
        p = student_log.exp()
        teacher_log = self.teacher_log_policy()
        q = teacher_log.exp()
        weights = {
            "student": self.occupancy(p.detach()) / float(self.horizon),
            "teacher": self.occupancy(q.detach()) / float(self.horizon),
        }
        token_losses = {
            "forward": (q * (teacher_log - student_log)).sum(dim=-1),
            "reverse": (p * (student_log - teacher_log)).sum(dim=-1),
        }
        gradients: Dict[Tuple[str, str], Tensor] = {}
        for direction in ("forward", "reverse"):
            for source in ("student", "teacher"):
                per_seed = (weights[source] * token_losses[direction]).sum(dim=-1)
                gradient = torch.autograd.grad(
                    per_seed.sum(), self.W, retain_graph=True, create_graph=False
                )[0]
                gradients[(direction, source)] = gradient.detach().reshape(self.batch_size, -1)

        def cosine(a: Tensor, b: Tensor) -> Tensor:
            return F.cosine_similarity(a, b, dim=-1, eps=1e-12)

        current_source = self.condition.rollout_source
        current_direction = self.condition.kl_direction
        if current_source == "mixture":
            beta = self.condition.teacher_fraction
            forward = (1.0 - beta) * gradients[("forward", "student")] + beta * gradients[("forward", "teacher")]
            reverse = (1.0 - beta) * gradients[("reverse", "student")] + beta * gradients[("reverse", "teacher")]
        else:
            forward = gradients[("forward", current_source)]
            reverse = gradients[("reverse", current_source)]
        student = gradients[(current_direction, "student")]
        teacher = gradients[(current_direction, "teacher")]
        return {
            "forward_reverse_gradient_cosine_current_rollout": cosine(forward, reverse),
            "student_teacher_gradient_cosine_current_kl": cosine(student, teacher),
            "forward_gradient_norm_current_rollout": forward.norm(dim=-1),
            "reverse_gradient_norm_current_rollout": reverse.norm(dim=-1),
            "student_rollout_gradient_norm_current_kl": student.norm(dim=-1),
            "teacher_rollout_gradient_norm_current_kl": teacher.norm(dim=-1),
        }

    def heldout_match(self, values: Tensor, occupancy: Tensor) -> Tensor:
        """Occupancy-weighted mean of ``values`` over held-out prompts only.

        Renormalised by the held-out occupancy mass so the result is comparable
        to the trained-prompt measure.  Returns NaN for replicas in which no
        prompt is held out, so that a v7 run never reports a spurious value.
        """

        held = (~self.state_is_trainable).to(occupancy.dtype)
        mass = (occupancy * held).sum(dim=-1)
        total = (occupancy * held * values).sum(dim=-1)
        result = total / mass.clamp_min(1e-30)
        return torch.where(mass > 0, result, torch.full_like(result, float("nan")))

    def apply_prompt_mask(
        self, weights: Tensor, generator: Optional[torch.Generator]
    ) -> Tensor:
        """Zero the loss weight of held-out and unsampled prompts.

        Returns ``weights`` unchanged when no prompt is held out and no prompt
        minibatching is requested, so the v7 loss is recovered bit-for-bit.
        """

        per_step = self.cfg.prompts_per_step
        all_trainable = bool(self.state_is_trainable.all().item())
        if per_step <= 0 and all_trainable:
            return weights

        mask = self.state_is_trainable.to(weights.dtype)
        if per_step > 0:
            # Sample a fresh minibatch of prompts per replica per step.  A
            # prompt that is not sampled contributes exactly zero this step,
            # which is what a finite prompt batch does and what exact tree
            # enumeration does not.
            trainable_prompts = torch.ones(
                self.batch_size, self.n_prompts, dtype=torch.bool, device=weights.device
            )
            held = self.state_prompt_id.clone()
            held[~self.state_is_trainable] = -1
            for prompt in range(self.n_prompts):
                is_held = (
                    (self.state_prompt_id == prompt) & (~self.state_is_trainable)
                ).any(dim=-1)
                trainable_prompts[:, prompt] &= ~is_held
            scores = torch.rand(
                self.batch_size, self.n_prompts, generator=generator,
                device=weights.device, dtype=weights.dtype,
            )
            scores = scores.masked_fill(~trainable_prompts, -1.0)
            keep = min(per_step, int(trainable_prompts.sum(dim=-1).min().item()))
            chosen = torch.zeros_like(trainable_prompts)
            top = scores.topk(keep, dim=-1).indices
            chosen.scatter_(1, top, True)
            state_chosen = torch.gather(
                chosen, 1, self.state_prompt_id.clamp_min(0)
            )
            # The root belongs to no prompt and is always trained.
            state_chosen = state_chosen | (self.state_prompt_id < 0)
            mask = mask * state_chosen.to(weights.dtype)
        return weights * mask

    def train_step(self, generator: Optional[torch.Generator]) -> Dict[str, Tensor]:
        if self.optimizer is not None:
            self.optimizer.zero_grad(set_to_none=True)
        elif self.W.grad is not None:
            self.W.grad.zero_()

        student_log_probs = self.student_log_policy("new")
        p = student_log_probs.exp()
        teacher_log_probs = self.teacher_log_policy()
        q = teacher_log_probs.exp()
        weights = self.rollout_weights(p.detach(), q.detach(), generator)

        if self.condition.kl_direction == "forward":
            token_kl = (q * (teacher_log_probs - student_log_probs)).sum(dim=-1)
        elif self.condition.kl_direction == "reverse":
            token_kl = (p * (student_log_probs - teacher_log_probs)).sum(dim=-1)
        else:  # defensive; Condition.validate already rejects this
            raise ValueError(f"unknown KL direction: {self.condition.kl_direction}")
        # R2: finite prompts.  Held-out subtrees get exactly zero weight, and
        # with prompts_per_step > 0 only a sampled minibatch of the remaining
        # subtrees does -- so a state the run never samples contributes nothing,
        # rather than a very small amount that Adam can still act on.
        weights = self.apply_prompt_mask(weights, generator)
        loss_per_seed = (weights * token_kl).sum(dim=-1)
        loss = loss_per_seed.sum()
        loss.backward()

        grad = self.W.grad
        if grad is None:
            raise RuntimeError("missing gradient")
        grad_norm = grad.reshape(self.batch_size, -1).norm(dim=-1).detach()

        if self.condition.optimizer == "normalized_sgd":
            with torch.no_grad():
                normalized = grad / grad_norm.clamp_min(1e-12)[:, None, None]
                if self.cfg.weight_decay:
                    normalized = normalized + self.cfg.weight_decay * self.W
                self.W.add_(normalized, alpha=-self.condition.learning_rate)
        else:
            if self.cfg.gradient_clip_norm > 0:
                # Clip each independent replica, not the aggregate batch tensor.
                scale = (self.cfg.gradient_clip_norm / grad_norm.clamp_min(1e-12)).clamp(max=1.0)
                self.W.grad.mul_(scale[:, None, None])
            assert self.optimizer is not None
            self.optimizer.step()

        alpha = self.condition.ema_alpha
        if alpha > 0:
            with torch.no_grad():
                self.W_teacher.lerp_(self.W.detach(), alpha)

        return {
            "train_loss": loss_per_seed.detach(),
            "gradient_norm_unclipped": grad_norm,
        }

    def support_success(self, policy: Tensor, support: Tensor) -> Tensor:
        """Probability that every generated token remains in the valid support tree."""

        b, n_states, _k = policy.shape
        alive = torch.zeros(b, n_states, device=policy.device, dtype=policy.dtype)
        alive[:, 0] = 1.0
        for depth in range(self.horizon):
            indices = self.by_depth[depth]
            allowed_mass = alive[:, indices, None] * policy[:, indices, :] * support[:, indices, :]
            if depth == self.horizon - 1:
                return allowed_mass.sum(dim=(1, 2))
            child_indices = self.tree_children[indices].reshape(1, -1).expand(b, -1)
            alive_next = torch.zeros_like(alive)
            alive_next.scatter_add_(1, child_indices, allowed_mass.reshape(b, -1))
            alive = alive_next
        raise AssertionError("unreachable")

    def top1_path_success(self, policy: Tensor, target: Tensor) -> Tensor:
        b = policy.shape[0]
        states = torch.zeros(b, dtype=torch.long, device=self.device)
        probability = torch.ones(b, dtype=policy.dtype, device=self.device)
        batch_index = torch.arange(b, device=self.device)
        for depth in range(self.horizon):
            actions = target[batch_index, states].argmax(dim=-1)
            probability = probability * policy[batch_index, states, actions]
            if depth < self.horizon - 1:
                states = self.tree_children[states, actions]
        return probability

    @staticmethod
    def weighted_mean(values: Tensor, occupancy: Tensor, horizon: int) -> Tensor:
        return (occupancy / float(horizon) * values).sum(dim=-1)

    @torch.no_grad()
    def metrics(self) -> Dict[str, Tensor]:
        student_new_log = self.student_log_policy("new")
        p_new = student_new_log.exp()
        p_old = self.student_log_policy("old").exp()
        q_base, q_context, context_weight, q_teacher = self.teacher_components()
        teacher_log = q_teacher.clamp_min(torch.finfo(q_teacher.dtype).tiny).log()
        d_new = self.occupancy(p_new)
        d_teacher = self.occupancy(q_teacher)
        d_old_student = self.occupancy(p_old)
        d_new_target = self.occupancy(self.q_new_target)
        d_old_ref = self.occupancy(self.q_old)

        tv_new = 0.5 * (p_new - self.q_new_target).abs().sum(dim=-1)
        tv_old = 0.5 * (p_old - self.q_old).abs().sum(dim=-1)
        tv_teacher = 0.5 * (q_teacher - self.q_new_target).abs().sum(dim=-1)
        tv_base_target = 0.5 * (q_base - self.q_new_target).abs().sum(dim=-1)
        tv_context_target = 0.5 * (q_context - self.q_new_target).abs().sum(dim=-1)
        tv_teacher_base = 0.5 * (q_teacher - q_base).abs().sum(dim=-1)
        tv_student_teacher = 0.5 * (p_new - q_teacher).abs().sum(dim=-1)
        forward_kl_new = (
            self.q_new_target
            * (torch.log(self.q_new_target) - student_new_log)
        ).sum(dim=-1)
        forward_kl_teacher = (q_teacher * (teacher_log - student_new_log)).sum(dim=-1)
        reverse_kl_teacher = (p_new * (student_new_log - teacher_log)).sum(dim=-1)
        rollout_occupancy_tv = (
            0.5 * (d_new - d_teacher).abs().sum(dim=-1) / float(self.horizon)
        )

        return {
            "new_match_student_occupancy": self.weighted_mean(1.0 - tv_new, d_new, self.horizon),
            "new_match_target_occupancy": self.weighted_mean(1.0 - tv_new, d_new_target, self.horizon),
            # R2: the same acquisition measure restricted to prompts that never
            # received gradient, renormalised over that subset so it is on the
            # same scale as the trained measure.  NaN when nothing is held out,
            # rather than silently equal to the trained value.
            "new_match_heldout_prompts": self.heldout_match(1.0 - tv_new, d_new_target),
            # S1: the same held-out restriction under the STUDENT's own
            # occupancy.  The metric above scores held-out prompts under the
            # target occupancy, which is the distribution teacher rollouts train
            # on, so it cannot isolate an on-policy advantage.  This one is the
            # deployment analogue: unseen prompts, along trajectories the model
            # actually generates.
            "new_match_heldout_deployment": self.heldout_match(1.0 - tv_new, d_new),
            "new_forward_kl_target_occupancy": self.weighted_mean(forward_kl_new, d_new_target, self.horizon),
            "new_support_success": self.support_success(p_new, self.target_support),
            "new_top1_path_success": self.top1_path_success(p_new, self.q_new_target),
            "old_match_reference_occupancy": self.weighted_mean(1.0 - tv_old, d_old_ref, self.horizon),
            "old_match_student_occupancy": self.weighted_mean(1.0 - tv_old, d_old_student, self.horizon),
            "old_support_success": self.support_success(p_old, self.old_support),
            "old_top1_path_success": self.top1_path_success(p_old, self.q_old),
            "teacher_utility_student_occupancy": self.weighted_mean(1.0 - tv_teacher, d_new, self.horizon),
            "teacher_utility_target_occupancy": self.weighted_mean(1.0 - tv_teacher, d_new_target, self.horizon),
            "teacher_base_utility_student_occupancy": self.weighted_mean(1.0 - tv_base_target, d_new, self.horizon),
            "context_endpoint_utility_student_occupancy": self.weighted_mean(1.0 - tv_context_target, d_new, self.horizon),
            "teacher_context_shift_student_occupancy": self.weighted_mean(tv_teacher_base, d_new, self.horizon),
            "student_teacher_tv_student_occupancy": self.weighted_mean(tv_student_teacher, d_new, self.horizon),
            "student_teacher_forward_kl_student_occupancy": self.weighted_mean(forward_kl_teacher, d_new, self.horizon),
            "student_teacher_reverse_kl_student_occupancy": self.weighted_mean(reverse_kl_teacher, d_new, self.horizon),
            "student_teacher_rollout_occupancy_tv": rollout_occupancy_tv,
            "effective_context_weight_student_occupancy": self.weighted_mean(context_weight.squeeze(-1), d_new, self.horizon),
            "teacher_drift": (self.W_teacher - self.W_initial).reshape(self.batch_size, -1).norm(dim=-1),
            "student_drift": (self.W - self.W_initial).reshape(self.batch_size, -1).norm(dim=-1),
        }

    @torch.no_grad()
    def sample_sequences(
        self, policy: Tensor, n: int, generator: Optional[torch.Generator]
    ) -> List[List[str]]:
        b, _s, k = policy.shape
        states = torch.zeros(b, n, dtype=torch.long, device=self.device)
        batch_index = torch.arange(b, device=self.device)[:, None].expand(b, n)
        tokens: List[Tensor] = []
        for depth in range(self.horizon):
            probs = policy[batch_index, states]
            actions = torch.multinomial(probs.reshape(-1, k), 1, generator=generator).reshape(b, n)
            tokens.append(actions)
            if depth < self.horizon - 1:
                states = self.tree_children[states, actions]
        stacked = torch.stack(tokens, dim=-1).cpu().tolist()
        return [[" ".join(map(str, sequence)) for sequence in replica] for replica in stacked]


# ---------------------------------------------------------------------------
# Validation, execution, tabulation, and paired inference
# ---------------------------------------------------------------------------

def stable_hash(parts: Iterable[object]) -> str:
    text = "|".join(str(part) for part in parts)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def seeded_generator(device: torch.device, seed: int) -> Optional[torch.Generator]:
    """Create a device-compatible RNG; MPS currently uses its global RNG."""

    if device.type == "mps":
        torch.manual_seed(seed)
        return None
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


def condition_metadata(condition: Condition, cfg: RunConfig) -> Dict[str, object]:
    r = condition.regime
    return {
        "teacher_construction": r.teacher_construction,
        "prior_sharpness": r.prior_sharpness,
        "state_private_fraction": r.state_private_fraction,
        "target_realization": r.target_realization,
        "target_temperature": r.target_temperature,
        "support_width": r.support_width,
        "context_anchoring": r.context_anchoring,
        "prompt_holdout_fraction": r.prompt_holdout_fraction,
        "prompts_per_step": cfg.prompts_per_step,
        "analysis_block": condition.analysis_block,
        "variant": condition.variant,
        "regime": r.name,
        "horizon": r.horizon,
        "feature_similarity": r.feature_similarity,
        "readout_similarity": r.readout_similarity,
        "surprise_strength": r.surprise_strength,
        "target_complexity": r.target_complexity,
        "context_strength": r.context_strength,
        "offpath_context_multiplier": r.offpath_context_multiplier,
        "context_capability": r.context_capability,
        "context_noise_std": r.context_noise_std,
        "primary_mass": r.primary_mass,
        "secondary_mass": r.secondary_mass,
        "kl_direction": condition.kl_direction,
        "rollout_source": condition.rollout_source,
        "teacher_fraction": condition.teacher_fraction,
        "ema_alpha": condition.ema_alpha,
        "learning_rate": condition.learning_rate,
        "optimizer": condition.optimizer,
        "estimator": condition.estimator,
    }


def tensors_to_rows(
    values: Mapping[str, Tensor], task_seeds: Sequence[int], base: Mapping[str, object]
) -> List[Dict[str, object]]:
    cpu_values = {name: tensor.detach().cpu().numpy() for name, tensor in values.items()}
    rows: List[Dict[str, object]] = []
    for i, seed in enumerate(task_seeds):
        row = dict(base)
        row["task_seed"] = int(seed)
        for name, array in cpu_values.items():
            row[name] = float(array[i])
        rows.append(row)
    return rows


class _EmbeddedBackend:
    """Namespace exposing the validated backend used by the v14 model."""

    STANDARD_MODERATE = STANDARD_MODERATE
    RegimeSpec = RegimeSpec
    Condition = Condition
    RunConfig = RunConfig
    PrefixTreeDistillation = PrefixTreeDistillation
    prepare_problem = staticmethod(prepare_problem)
    problem_key = staticmethod(problem_key)
    condition_metadata = staticmethod(condition_metadata)
    stable_hash = staticmethod(stable_hash)
    seeded_generator = staticmethod(seeded_generator)
    tensors_to_rows = staticmethod(tensors_to_rows)


# -----------------------------------------------------------------------------
# Backend access
# -----------------------------------------------------------------------------


def load_backend():
    """Return v14's self-contained, validated backend namespace."""
    base = _EmbeddedBackend
    _allow_exact_sync(base)
    return base


def _allow_exact_sync(base) -> None:
    """Allow the mathematically valid alpha=1 stress point, changing nothing else."""
    original = base.Condition.validate
    if getattr(original, "_v14_exact_sync_patch", False):
        return

    def validate(self):
        if float(self.ema_alpha) == 1.0:
            probe = replace(self, ema_alpha=float(np.nextafter(1.0, 0.0)))
            return original(probe)
        return original(self)

    validate._v14_exact_sync_patch = True  # type: ignore[attr-defined]
    base.Condition.validate = validate


# -----------------------------------------------------------------------------
# Clean public specifications
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class CoreTaskSpec:
    """The complete public task/model parameterization of v14."""

    feature_overlap: float = 0.50
    initial_policy_concentration: float = 2.50
    context_strength: float = 0.60

    def validate(self) -> None:
        if not -1.0 <= float(self.feature_overlap) <= 1.0:
            raise ValueError("feature_overlap must lie in [-1,1]")
        if float(self.initial_policy_concentration) <= 0.0:
            raise ValueError("initial_policy_concentration must be positive")
        if not 0.0 <= float(self.context_strength) <= 1.0:
            raise ValueError("context_strength must lie in [0,1]")


@dataclass(frozen=True)
class StructuralSpec:
    """Technical structure, never a paper-facing task axis.

    H controls autoregressive depth, K the effective local branching factor,
    and D the fixed feature/readout dimension.  The default H=4,K=8,D=64 is
    the paper reference; non-default values are used only in explicitly
    labelled structural-robustness experiments.
    """

    horizon: int = V14_HORIZON
    vocab_size: int = V14_VOCAB_SIZE
    feature_dim: int = V14_FEATURE_DIM

    def validate(self) -> None:
        if self.horizon < 2:
            raise ValueError("horizon must be at least 2")
        if self.vocab_size < 4:
            raise ValueError("vocab_size must be at least 4")
        if self.feature_dim < 2:
            raise ValueError("feature_dim must be at least 2")


@dataclass(frozen=True)
class AblationSpec:
    """Default-off causal interventions; none is part of the core model.

    ``state_competence_bias`` is source-independent.  A positive value weakens
    the privileged feature on states that the *initial student* visits more than
    the initial contextual teacher; a negative value performs the symmetric
    control on teacher-heavy states.  The state mask is computed once at t=0 and
    then frozen.  Thus the intervention never conditions on the rollout source
    used by the training cell.

    ``target_initial_jsd`` requests a per-seed value of c calibrated so that the
    initial student/contextual-teacher JSD under the initial student occupancy
    equals the target.  It is an explicit causal control for asking whether
    lambda has an effect beyond the policy-space distance it induces.  When set,
    CoreTaskSpec.context_strength is recorded but not used to set the signal.
    """

    readout_compatibility: float = V14_CORE_READOUT_COMPATIBILITY
    support_placement_bias: float = V14_CORE_SUPPORT_PLACEMENT_BIAS
    target_primary_mass: float = V14_TARGET_PRIMARY
    target_secondary_mass: float = V14_TARGET_SECONDARY

    endpoint_compatible_context: bool = False
    feature_noise_scale: float = 0.0
    behavior_noise_scale: float = 0.0
    offpath_context_multiplier: float = 1.0
    state_competence_bias: float = 0.0
    target_initial_jsd: Optional[float] = None

    def validate(self, vocab_size: int = V14_VOCAB_SIZE) -> None:
        if not -1.0 <= float(self.readout_compatibility) <= 1.0:
            raise ValueError("readout_compatibility must lie in [-1,1]")
        if float(self.support_placement_bias) < 0.0:
            raise ValueError("support_placement_bias must be nonnegative")
        m1, m2 = float(self.target_primary_mass), float(self.target_secondary_mass)
        if not (m1 >= m2 > 0.0 and m1 + m2 < 1.0):
            raise ValueError("target masses must satisfy m1>=m2>0 and m1+m2<1")
        if vocab_size < 4:
            raise ValueError("vocab_size must be at least 4")
        rest_mass = (1.0 - m1 - m2) / float(vocab_size - 2)
        if rest_mass > m2 + 1e-12:
            raise ValueError(
                "target_primary_mass/target_secondary_mass no longer define two "
                "elevated modes at this vocab_size; increase vocab_size or adjust "
                "the explicitly labelled target-shape ablation"
            )
        if self.feature_noise_scale < 0.0 or self.behavior_noise_scale < 0.0:
            raise ValueError("Gaussian noise scales must be nonnegative")
        if self.feature_noise_scale > 0.0 and self.behavior_noise_scale > 0.0:
            raise ValueError(
                "feature- and behavior-space Gaussian noise are separate causal "
                "interventions; do not enable both in the same cell"
            )
        if not 0.0 <= float(self.offpath_context_multiplier) <= 1.0:
            raise ValueError("offpath_context_multiplier must lie in [0,1]")
        if not -1.0 <= float(self.state_competence_bias) <= 1.0:
            raise ValueError("state_competence_bias must lie in [-1,1]")
        if self.target_initial_jsd is not None and float(self.target_initial_jsd) < 0.0:
            raise ValueError("target_initial_jsd must be nonnegative")

    @property
    def is_core(self) -> bool:
        return (
            self.readout_compatibility == V14_CORE_READOUT_COMPATIBILITY
            and self.support_placement_bias == V14_CORE_SUPPORT_PLACEMENT_BIAS
            and self.target_primary_mass == V14_TARGET_PRIMARY
            and self.target_secondary_mass == V14_TARGET_SECONDARY
            and not self.endpoint_compatible_context
            and self.feature_noise_scale == 0.0
            and self.behavior_noise_scale == 0.0
            and self.offpath_context_multiplier == 1.0
            and self.state_competence_bias == 0.0
            and self.target_initial_jsd is None
        )


# -----------------------------------------------------------------------------
# Parsing and embedded-backend mapping
# -----------------------------------------------------------------------------


def _strict_dataclass(cls, data: Mapping[str, object]):
    allowed = set(cls.__dataclass_fields__)
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} fields: {sorted(unknown)}")
    out = cls(**data)
    return out


def core_from_cell(cell: Mapping[str, object]) -> CoreTaskSpec:
    spec = _strict_dataclass(CoreTaskSpec, dict(cell.get("core", {})))
    spec.validate()
    return spec


def structure_from_cell(cell: Mapping[str, object]) -> StructuralSpec:
    spec = _strict_dataclass(StructuralSpec, dict(cell.get("structure", {})))
    spec.validate()
    return spec


def ablation_from_cell(cell: Mapping[str, object], vocab_size: int) -> AblationSpec:
    spec = _strict_dataclass(AblationSpec, dict(cell.get("ablation", {})))
    spec.validate(vocab_size)
    return spec


def _regime_from_specs(base, cell: Mapping[str, object], core: CoreTaskSpec,
                       structural: StructuralSpec, ablation: AblationSpec):
    """Map v14 onto the embedded backend while fixing every historical control."""
    # Historical fields appear only here.  They are private compatibility
    # constants and never appear as v14 scientific controls.
    overrides = {
        "horizon": int(structural.horizon),
        "feature_similarity": float(core.feature_overlap),
        "readout_similarity": float(ablation.readout_compatibility),
        "surprise_strength": float(ablation.support_placement_bias),
        "target_complexity": 0.0,
        "context_strength": 0.60,       # ignored by the v14 teacher
        "context_capability": 0.90,     # ignored by the v14 teacher
        "context_noise_std": 0.0,
        "offpath_context_multiplier": 1.0,  # v14 applies its own clean ablation
        "prior_sharpness": float(core.initial_policy_concentration),
        "state_private_fraction": 0.0,
        "target_realization": "projected",
        "target_temperature": 1.0,
        "support_width": 2,
        "context_anchoring": 0.0,
        "prompt_holdout_fraction": 0.0,
        "teacher_construction": "capped_mixture",
        "primary_mass": float(ablation.target_primary_mass),
        "secondary_mass": float(ablation.target_secondary_mass),
    }
    regime = replace(
        base.STANDARD_MODERATE,
        name=str(cell.get("regime_name", "v14_core")),
        **overrides,
    )
    regime.validate(int(structural.vocab_size))
    return regime


def _condition_from_cell(base, cell: Mapping[str, object], regime):
    return base.Condition(
        analysis_block=str(cell.get("analysis_block", "v14")),
        variant=str(cell.get("variant", cell.get("name", "declared"))),
        regime=regime,
        kl_direction=str(cell.get("kl_direction", "reverse")),
        rollout_source=str(cell.get("rollout_source", "student")),
        ema_alpha=float(cell.get("ema_alpha", 0.0025)),
        learning_rate=float(cell.get("learning_rate", 0.001)),
        optimizer=str(cell.get("optimizer", "adam")),
        estimator=str(cell.get("estimator", "exact")),
        teacher_fraction=float(cell.get("teacher_fraction", 0.5)),
    )


# -----------------------------------------------------------------------------
# Numerical helpers
# -----------------------------------------------------------------------------


def _center_last(x: Tensor) -> Tensor:
    return x - x.mean(dim=-1, keepdim=True)


def _per_seed_rms(x: Tensor) -> Tensor:
    return x.reshape(x.shape[0], -1).square().mean(dim=-1).sqrt()


def _weighted_mean(values: Tensor, occupancy: Tensor, horizon: int) -> Tensor:
    return (values * occupancy / float(horizon)).sum(dim=-1)


def _safe_cosine(a: Tensor, b: Tensor) -> Tensor:
    return F.cosine_similarity(a, b, dim=-1, eps=1e-12)


def _conditional_weighted_mean(values: Tensor, occupancy: Tensor, valid: Tensor) -> Tensor:
    w = occupancy * valid.to(dtype=occupancy.dtype)
    den = w.sum(dim=-1)
    num = (values * w).sum(dim=-1)
    nan = torch.full_like(den, float("nan"))
    return torch.where(den > 1e-20, num / den.clamp_min(1e-30), nan)


def _top_state_signal_fraction(contribution: Tensor, fraction: float) -> Tensor:
    n_states = int(contribution.shape[-1])
    k = max(1, int(math.ceil(fraction * n_states)))
    top = torch.topk(contribution, k=k, dim=-1, largest=True, sorted=False).values.sum(dim=-1)
    total = contribution.sum(dim=-1)
    return torch.where(total > 1e-30, top / total.clamp_min(1e-30), torch.zeros_like(total))


def _orthogonalize_and_scale(noise: Tensor, signal: Tensor, scale: float) -> Tensor:
    """Orthogonalize a perturbation to signal per seed and set relative RMS."""
    if scale == 0.0:
        return torch.zeros_like(noise)
    b = noise.shape[0]
    n = noise.reshape(b, -1)
    s = signal.reshape(b, -1)
    s2 = s.square().sum(dim=-1, keepdim=True)
    proj = (n * s).sum(dim=-1, keepdim=True) / s2.clamp_min(1e-30)
    n = n - proj * s
    n_rms = n.square().mean(dim=-1, keepdim=True).sqrt()
    s_rms = s.square().mean(dim=-1, keepdim=True).sqrt()
    target = float(scale) * s_rms
    n = n * target / n_rms.clamp_min(1e-30)
    return n.reshape_as(noise)


def _deterministic_gaussian(shape_tail: Sequence[int], task_seeds: Sequence[int],
                            device: torch.device, dtype: torch.dtype,
                            offset: int) -> Tensor:
    rows: List[Tensor] = []
    for seed in task_seeds:
        gen = torch.Generator(device=device)
        gen.manual_seed(int(seed) + int(offset))
        rows.append(torch.randn(tuple(shape_tail), generator=gen, device=device, dtype=dtype))
    return torch.stack(rows, dim=0)


def _jsd_state(p: Tensor, q: Tensor) -> Tensor:
    tiny = torch.finfo(p.dtype).tiny
    m = 0.5 * (p + q)
    return 0.5 * (
        (p * (torch.log(p.clamp_min(tiny)) - torch.log(m.clamp_min(tiny)))).sum(dim=-1)
        + (q * (torch.log(q.clamp_min(tiny)) - torch.log(m.clamp_min(tiny)))).sum(dim=-1)
    )


# -----------------------------------------------------------------------------
# Final weight-dependent contextual teacher
# -----------------------------------------------------------------------------


class V14ContextModel:
    """Clean v14 contextual teacher wrapped around the embedded learner."""

    def __init__(self, base, prepared, condition, cfg, device: torch.device,
                 core: CoreTaskSpec, ablation: AblationSpec,
                 task_seeds: Sequence[int]) -> None:
        self.base = base
        self.prepared = prepared
        self.condition = condition
        self.cfg = cfg
        self.device = device
        self.core = core
        self.ablation = ablation
        self.task_seeds = tuple(int(x) for x in task_seeds)

        self.inner = base.PrefixTreeDistillation(
            prepared.problem, condition, cfg, prepared.by_depth, prepared.children, device
        )
        self._legacy_teacher_components = self.inner.teacher_components
        self._calibration: Dict[str, Tensor] = {}
        self._build_context_features()
        # V14 changes only the contextual teacher; the base learner's training,
        # KL, rollout, optimizer, occupancy, and EMA code remain untouched.
        self.inner.teacher_components = self.teacher_components  # type: ignore[method-assign]

    @torch.no_grad()
    def _solve_relative_feature(self, delta: Tensor, W: Tensor) -> Tensor:
        relative_readout = W[:, :-1, :] - W[:, -1:, :]
        pinv = torch.linalg.pinv(relative_readout, rtol=V14_PINV_RTOL)
        rel = delta[..., :-1] - delta[..., -1:]
        rhs = float(self.cfg.old_temperature) * rel
        return torch.einsum("bsm,bdm->bsd", rhs, pinv)

    @torch.no_grad()
    def _solve_endpoint_compatible_feature(self, delta: Tensor, W0: Tensor) -> Tensor:
        """Minimum-norm feature matching the initial correction and null at W*."""
        inner = self.inner
        tiny = torch.finfo(inner.q_new_target.dtype).tiny
        T = float(self.cfg.old_temperature)
        target_logits = T * _center_last(torch.log(inner.q_new_target.clamp_min(tiny)))
        solution = torch.linalg.lstsq(inner.phi_new, target_logits).solution
        W_star = solution.transpose(-1, -2).contiguous()
        R0 = W0[:, :-1, :] - W0[:, -1:, :]
        Rstar = W_star[:, :-1, :] - W_star[:, -1:, :]
        A = torch.cat([R0, Rstar], dim=1)
        pinv = torch.linalg.pinv(A, rtol=V14_PINV_RTOL)
        rel = delta[..., :-1] - delta[..., -1:]
        rhs0 = T * rel
        rhs = torch.cat([rhs0, torch.zeros_like(rhs0)], dim=-1)
        psi = torch.einsum("bsm,bdm->bsd", rhs, pinv)
        endpoint = torch.einsum("bsd,bmd->bsm", psi, Rstar) / T
        desired_rel = rel
        num = endpoint.reshape(endpoint.shape[0], -1).norm(dim=-1)
        den = desired_rel.reshape(desired_rel.shape[0], -1).norm(dim=-1).clamp_min(1e-30)
        self._endpoint_relative_error = num / den
        return psi

    @torch.no_grad()
    def _calibrate_c_for_jsd(self, p0: Tensor, q_target: Tensor,
                             d_student: Tensor, requested: float) -> Tensor:
        """Per-seed bisection for student-occupancy-weighted initial JSD."""
        b = p0.shape[0]
        lo = torch.zeros(b, device=p0.device, dtype=p0.dtype)
        hi = torch.ones_like(lo)
        target = torch.full_like(lo, float(requested))

        def score(cvec: Tensor) -> Tensor:
            q = (1.0 - cvec[:, None, None]) * p0 + cvec[:, None, None] * q_target
            jsd = _jsd_state(p0, q)
            return _weighted_mean(jsd, d_student, self.inner.horizon)

        max_score = score(hi)
        reachable = max_score + 1e-12 >= target
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            val = score(mid)
            go_right = val < target
            lo = torch.where(go_right, mid, lo)
            hi = torch.where(go_right, hi, mid)
        c = 0.5 * (lo + hi)
        c = torch.where(reachable, c, torch.ones_like(c))
        achieved = score(c)
        self._jsd_calibration_target = target
        self._jsd_calibration_achieved = achieved
        self._jsd_calibration_reachable = reachable.to(dtype=p0.dtype)
        return c

    @torch.no_grad()
    def _build_context_features(self) -> None:
        inner = self.inner
        dtype = inner.W_teacher.dtype
        tiny = torch.finfo(dtype).tiny
        T = float(self.cfg.old_temperature)
        W0 = inner.W_teacher.detach()

        base_logits = torch.einsum("bsd,bkd->bsk", inner.phi_new, W0) / T
        p0 = F.softmax(base_logits, dim=-1)
        d_student0 = inner.occupancy(p0)

        if self.ablation.target_initial_jsd is None:
            cvec = torch.full(
                (inner.batch_size,), float(self.core.context_strength),
                device=base_logits.device, dtype=dtype,
            )
            self._jsd_calibration_target = torch.full_like(cvec, float("nan"))
            self._jsd_calibration_achieved = torch.full_like(cvec, float("nan"))
            self._jsd_calibration_reachable = torch.ones_like(cvec)
        else:
            cvec = self._calibrate_c_for_jsd(
                p0, inner.q_new_target, d_student0,
                float(self.ablation.target_initial_jsd),
            )

        q_ctx0 = (1.0 - cvec[:, None, None]) * p0 + cvec[:, None, None] * inner.q_new_target
        q_ctx0 = q_ctx0 / q_ctx0.sum(dim=-1, keepdim=True).clamp_min(tiny)
        desired_signal = _center_last(
            torch.log(q_ctx0.clamp_min(tiny)) - torch.log(p0.clamp_min(tiny))
        )

        self._endpoint_relative_error = torch.full_like(cvec, float("nan"))
        if self.ablation.endpoint_compatible_context:
            psi_signal = self._solve_endpoint_compatible_feature(desired_signal, W0)
        else:
            psi_signal = self._solve_relative_feature(desired_signal, W0)

        # Optional Gaussian context corruption.  Both interventions are fixed
        # in input space after construction and are exactly zero in the core.
        psi_noise = torch.zeros_like(psi_signal)
        if self.ablation.feature_noise_scale > 0.0:
            raw = _deterministic_gaussian(
                psi_signal.shape[1:], self.task_seeds, psi_signal.device,
                psi_signal.dtype, offset=14_000_003,
            )
            psi_noise = _orthogonalize_and_scale(
                raw, psi_signal, float(self.ablation.feature_noise_scale)
            )
        elif self.ablation.behavior_noise_scale > 0.0:
            raw_delta = _deterministic_gaussian(
                desired_signal.shape[1:], self.task_seeds, desired_signal.device,
                desired_signal.dtype, offset=14_000_019,
            )
            raw_delta = _center_last(raw_delta)
            delta_noise = _orthogonalize_and_scale(
                raw_delta, desired_signal, float(self.ablation.behavior_noise_scale)
            )
            psi_noise = self._solve_relative_feature(delta_noise, W0)

        # Initial teacher occupancy used to define the source-independent
        # competence mask.  This mask is frozen and never sees the training
        # rollout source.
        provisional_logits = torch.einsum(
            "bsd,bkd->bsk", inner.phi_new + psi_signal, W0
        ) / T
        provisional_q = F.softmax(provisional_logits, dim=-1)
        d_teacher0 = inner.occupancy(provisional_q)
        occ_gap = (d_student0 - d_teacher0) / (d_student0 + d_teacher0 + 1e-12)
        bias = float(self.ablation.state_competence_bias)
        if bias >= 0.0:
            reliability = 1.0 - bias * torch.relu(occ_gap)
        else:
            reliability = 1.0 - (-bias) * torch.relu(-occ_gap)
        reliability = reliability.clamp(0.0, 1.0)

        offpath = torch.where(
            inner.on_target_tree,
            torch.ones_like(inner.on_target_tree, dtype=dtype),
            torch.full_like(
                inner.on_target_tree,
                float(self.ablation.offpath_context_multiplier), dtype=dtype,
            ),
        )
        state_scale = reliability * offpath
        psi_signal_eff = state_scale.unsqueeze(-1) * psi_signal
        psi_noise_eff = state_scale.unsqueeze(-1) * psi_noise
        psi_total = psi_signal_eff + psi_noise_eff

        self.psi_signal = psi_signal_eff
        self.psi_noise = psi_noise_eff
        self.psi_total = psi_total
        self.base_logits_initial = base_logits.detach().clone()
        self.context_signal_delta_initial = _center_last(
            torch.einsum("bsd,bkd->bsk", psi_signal_eff, W0) / T
        ).detach().clone()
        self.context_noise_delta_initial = _center_last(
            torch.einsum("bsd,bkd->bsk", psi_noise_eff, W0) / T
        ).detach().clone()
        self.context_delta_initial = (
            self.context_signal_delta_initial + self.context_noise_delta_initial
        ).detach().clone()
        self.context_strength_realized = cvec.detach().clone()
        self.state_competence_reliability = reliability.detach().clone()
        self.state_occupancy_gap_initial = occ_gap.detach().clone()
        self.initial_student_occupancy = d_student0.detach().clone()
        self.initial_teacher_occupancy_provisional = d_teacher0.detach().clone()
        self.nominal_context_weight = (
            cvec[:, None, None] * state_scale.unsqueeze(-1)
        ).detach().clone()

        # The calibration target is the actual post-ablation same-weight teacher.
        ctx_probs = F.softmax(self._context_logits_from(W0), dim=-1)
        self.initial_context_teacher = ctx_probs.detach().clone()
        tv = 0.5 * (ctx_probs - q_ctx0).abs().sum(dim=-1)
        phi_rms = _per_seed_rms(inner.phi_new)
        signal_rms = _per_seed_rms(psi_signal_eff)
        noise_rms = _per_seed_rms(psi_noise_eff)
        total_rms = _per_seed_rms(psi_total)
        joint, base_resid = self._joint_internalization_residuals()

        relative_readout = W0[:, :-1, :] - W0[:, -1:, :]
        singular = torch.linalg.svdvals(relative_readout)
        smax = singular.max(dim=-1).values
        threshold = V14_PINV_RTOL * smax[:, None]
        active = singular > threshold
        smin = torch.where(active, singular, torch.full_like(singular, float("inf"))).min(dim=-1).values

        self._calibration.update({
            "context_calibration_tv_mean": tv.mean(dim=-1),
            "context_calibration_tv_max": tv.max(dim=-1).values,
            "context_readout_active_rank": active.sum(dim=-1).to(dtype),
            "context_readout_condition_number": smax / smin.clamp_min(1e-30),
            "context_feature_signal_rms": signal_rms,
            "context_feature_noise_rms": noise_rms,
            "context_feature_total_rms": total_rms,
            "context_feature_signal_to_phi_rms": signal_rms / phi_rms.clamp_min(1e-30),
            "context_feature_noise_to_phi_rms": noise_rms / phi_rms.clamp_min(1e-30),
            "context_feature_total_to_phi_rms": total_rms / phi_rms.clamp_min(1e-30),
            "joint_internalization_fit_relative_error": joint,
            "base_internalization_fit_relative_error": base_resid,
            "target_compatible_endpoint_constraint_relative_error": self._endpoint_relative_error,
            "v14_realized_context_strength": cvec,
            "v14_target_jsd_reachable": self._jsd_calibration_reachable,
            "v14_target_jsd_achieved": self._jsd_calibration_achieved,
        })

    @torch.no_grad()
    def _joint_internalization_residuals(self) -> Tuple[Tensor, Tensor]:
        inner = self.inner
        tiny = torch.finfo(inner.q_new_target.dtype).tiny
        target = float(self.cfg.old_temperature) * _center_last(
            torch.log(inner.q_new_target.clamp_min(tiny))
        )
        phi = inner.phi_new
        ctx_phi = phi + self.psi_total

        def residual(X: Tensor, Y: Tensor) -> Tensor:
            solution = torch.linalg.lstsq(X, Y).solution
            pred = X @ solution
            num = (pred - Y).reshape(Y.shape[0], -1).norm(dim=-1)
            den = Y.reshape(Y.shape[0], -1).norm(dim=-1).clamp_min(1e-30)
            return num / den

        base = residual(phi, target)
        joint = residual(torch.cat([phi, ctx_phi], dim=1), torch.cat([target, target], dim=1))
        return joint, base

    def _context_logits_from(self, W_teacher: Tensor) -> Tensor:
        return torch.einsum(
            "bsd,bkd->bsk", self.inner.phi_new + self.psi_total, W_teacher
        ) / float(self.cfg.old_temperature)

    @torch.no_grad()
    def teacher_components(self) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        base_logits = torch.einsum(
            "bsd,bkd->bsk", self.inner.phi_new, self.inner.W_teacher
        ) / float(self.cfg.old_temperature)
        base_probs = F.softmax(base_logits, dim=-1)
        ctx_probs = F.softmax(self._context_logits_from(self.inner.W_teacher), dim=-1)
        return base_probs, ctx_probs, self.nominal_context_weight, ctx_probs

    @torch.no_grad()
    def _distribution_geometry_metrics(self) -> Dict[str, Tensor]:
        inner = self.inner
        dtype = inner.W.dtype
        tiny = torch.finfo(dtype).tiny
        log_p = inner.student_log_policy("new")
        _, _, _, q = self.teacher_components()
        log_q = torch.log(q.clamp_min(tiny))
        p = log_p.exp()
        q_target = inner.q_new_target
        d_student = inner.occupancy(p)
        h = inner.horizon
        log_p_safe = torch.log(p.clamp_min(tiny))

        fkl = (q * (log_q - log_p_safe)).sum(dim=-1)
        rkl = (p * (log_p_safe - log_q)).sum(dim=-1)
        tv = 0.5 * (p - q).abs().sum(dim=-1)
        jsd = _jsd_state(p, q)
        entropy_p = -(p * log_p_safe).sum(dim=-1)
        entropy_q = -(q * log_q).sum(dim=-1)
        log_t = torch.log(q_target.clamp_min(tiny))
        entropy_t = -(q_target * log_t).sum(dim=-1)

        teacher_top = q.argmax(dim=-1, keepdim=True)
        target_top = q_target.argmax(dim=-1, keepdim=True)
        support_teacher = p.gather(-1, teacher_top).squeeze(-1)
        support_target = p.gather(-1, target_top).squeeze(-1)

        g_f = p - q
        lr = log_p_safe - log_q
        g_r = p * (lr - (p * lr).sum(dim=-1, keepdim=True))
        n_f = torch.linalg.vector_norm(g_f, dim=-1)
        n_r = torch.linalg.vector_norm(g_r, dim=-1)
        denom = n_f * n_r
        valid = denom > 1e-20
        cosine = (g_f * g_r).sum(dim=-1) / denom.clamp_min(1e-30)
        log_ratio = torch.log(n_f.clamp_min(1e-30) / n_r.clamp_min(1e-30))
        jsd_contrib = d_student * jsd / float(h)

        return {
            "geometry_forward_kl_student_occupancy": _weighted_mean(fkl, d_student, h),
            "geometry_reverse_kl_student_occupancy": _weighted_mean(rkl, d_student, h),
            "geometry_jsd_student_occupancy": _weighted_mean(jsd, d_student, h),
            "geometry_tv_student_occupancy": _weighted_mean(tv, d_student, h),
            "geometry_student_entropy_student_occupancy": _weighted_mean(entropy_p, d_student, h),
            "geometry_teacher_entropy_student_occupancy": _weighted_mean(entropy_q, d_student, h),
            "geometry_target_entropy_student_occupancy": _weighted_mean(entropy_t, d_student, h),
            "geometry_teacher_top1_student_support_student_occupancy": _weighted_mean(support_teacher, d_student, h),
            "geometry_target_top1_student_support_student_occupancy": _weighted_mean(support_target, d_student, h),
            "geometry_forward_logit_grad_norm_student_occupancy": _weighted_mean(n_f, d_student, h),
            "geometry_reverse_logit_grad_norm_student_occupancy": _weighted_mean(n_r, d_student, h),
            "geometry_forward_reverse_logit_grad_cosine_student_occupancy": _conditional_weighted_mean(cosine, d_student, valid),
            "geometry_log_forward_reverse_logit_grad_norm_ratio_student_occupancy": _conditional_weighted_mean(log_ratio, d_student, valid),
            "geometry_valid_gradient_comparison_occupancy_fraction": _weighted_mean(valid.to(dtype), d_student, h),
            "geometry_jsd_top10_state_signal_fraction": _top_state_signal_fraction(jsd_contrib, 0.10),
            "geometry_jsd_top25_state_signal_fraction": _top_state_signal_fraction(jsd_contrib, 0.25),
        }

    @torch.no_grad()
    def context_metrics(self) -> Dict[str, Tensor]:
        inner = self.inner
        tiny = torch.finfo(inner.W_teacher.dtype).tiny
        T = float(self.cfg.old_temperature)
        base_logits = torch.einsum("bsd,bkd->bsk", inner.phi_new, inner.W_teacher) / T
        signal_logits = torch.einsum(
            "bsd,bkd->bsk", inner.phi_new + self.psi_signal, inner.W_teacher
        ) / T
        ctx_logits = self._context_logits_from(inner.W_teacher)
        base_probs = F.softmax(base_logits, dim=-1)
        ctx_probs = F.softmax(ctx_logits, dim=-1)
        delta = _center_last(ctx_logits - base_logits)
        signal_delta = _center_last(signal_logits - base_logits)
        noise_delta = delta - signal_delta
        delta_change = _center_last(delta - self.context_delta_initial)
        base_drift = _center_last(base_logits - self.base_logits_initial)

        flat_base = base_drift.reshape(base_drift.shape[0], -1)
        flat_change = delta_change.reshape(delta_change.shape[0], -1)
        den = flat_base.square().sum(dim=-1)
        proj = (flat_base * flat_change).sum(dim=-1) / den.clamp_min(1e-30)
        proj = torch.where(den > 1e-20, proj, torch.zeros_like(proj))

        d_student = inner.occupancy(inner.student_log_policy("new").exp())
        d_teacher = inner.occupancy(ctx_probs)
        d_target = inner.occupancy(inner.q_new_target)
        kl_ctx_base = (ctx_probs * (
            torch.log(ctx_probs.clamp_min(tiny)) - torch.log(base_probs.clamp_min(tiny))
        )).sum(dim=-1)
        tv_ctx_base = 0.5 * (ctx_probs - base_probs).abs().sum(dim=-1)
        tv_ctx_target = 0.5 * (ctx_probs - inner.q_new_target).abs().sum(dim=-1)
        tv_base_target = 0.5 * (base_probs - inner.q_new_target).abs().sum(dim=-1)

        teacher_gap = inner.W.detach() - inner.W_teacher
        base_dir = _center_last(torch.einsum("bsd,bkd->bsk", inner.phi_new, teacher_gap) / T)
        signal_dir = _center_last(torch.einsum("bsd,bkd->bsk", self.psi_signal, teacher_gap) / T)
        noise_dir = _center_last(torch.einsum("bsd,bkd->bsk", self.psi_noise, teacher_gap) / T)
        extra_dir = signal_dir + noise_dir
        flat_dir = base_dir.reshape(base_dir.shape[0], -1)
        den_dir = flat_dir.square().sum(dim=-1)

        def projected(x: Tensor) -> Tensor:
            flat = x.reshape(x.shape[0], -1)
            out = (flat_dir * flat).sum(dim=-1) / den_dir.clamp_min(1e-30)
            return torch.where(den_dir > 1e-20, out, torch.zeros_like(out))

        reliability = self.state_competence_reliability
        values: Dict[str, Tensor] = {
            "context_same_weight_forward_kl_student_occupancy": _weighted_mean(kl_ctx_base, d_student, inner.horizon),
            "context_same_weight_forward_kl_target_occupancy": _weighted_mean(kl_ctx_base, d_target, inner.horizon),
            "context_same_weight_tv_student_occupancy": _weighted_mean(tv_ctx_base, d_student, inner.horizon),
            "context_teacher_utility_student_occupancy": _weighted_mean(1.0 - tv_ctx_target, d_student, inner.horizon),
            "context_base_utility_student_occupancy": _weighted_mean(1.0 - tv_base_target, d_student, inner.horizon),
            "context_utility_gain_student_occupancy": _weighted_mean(tv_base_target - tv_ctx_target, d_student, inner.horizon),
            "context_feedback_projection_from_init": proj,
            "context_feedback_gain_from_init": 1.0 + proj,
            "context_ema_direction_feedback_projection": projected(extra_dir),
            "context_ema_direction_feedback_gain": 1.0 + projected(extra_dir),
            "context_ema_direction_signal_projection": projected(signal_dir),
            "context_ema_direction_noise_projection": projected(noise_dir),
            "context_delta_cosine_to_initial": _safe_cosine(delta.reshape(delta.shape[0], -1), self.context_delta_initial.reshape(delta.shape[0], -1)),
            "context_delta_rms": _per_seed_rms(delta),
            "context_delta_change_rms": _per_seed_rms(delta_change),
            "context_signal_delta_rms": _per_seed_rms(signal_delta),
            "context_noise_delta_rms": _per_seed_rms(noise_delta),
            "teacher_parameter_drift_rms": _per_seed_rms(inner.W_teacher - inner.W_initial),
            "teacher_student_parameter_gap_rms": _per_seed_rms(inner.W_teacher - inner.W.detach()),
            "v14_context_strength_realized": self.context_strength_realized,
            "v14_competence_reliability_student_occupancy": _weighted_mean(reliability, d_student, inner.horizon),
            "v14_competence_reliability_teacher_occupancy": _weighted_mean(reliability, d_teacher, inner.horizon),
            "v14_competence_reliability_gap_teacher_minus_student": (
                _weighted_mean(reliability, d_teacher, inner.horizon)
                - _weighted_mean(reliability, d_student, inner.horizon)
            ),
            "v14_initial_student_teacher_occupancy_tv": 0.5 * (self.initial_student_occupancy - self.initial_teacher_occupancy_provisional).abs().sum(dim=-1) / float(inner.horizon),
        }
        values.update(self._calibration)
        values.update(self._distribution_geometry_metrics())
        return values


# -----------------------------------------------------------------------------
# Metadata and experiment runner
# -----------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _metadata(base, condition, cfg, core: CoreTaskSpec,
              structural: StructuralSpec, ablation: AblationSpec,
              cell: Mapping[str, object]) -> Dict[str, object]:
    meta = base.condition_metadata(condition, cfg)
    # Paper-facing names are explicit; historical aliases remain in the backend
    # metadata only for compatibility and can be dropped by the aggregator.
    meta.update({f"v14_core_{k}": v for k, v in asdict(core).items()})
    meta.update({f"v14_structure_{k}": v for k, v in asdict(structural).items()})
    meta.update({f"v14_ablation_{k}": v for k, v in asdict(ablation).items()})
    meta["v14_version"] = V14_VERSION
    meta["v14_core_only"] = bool(ablation.is_core)
    meta["cell_name"] = str(cell.get("name", condition.variant))
    meta["record_history"] = bool(cell.get("record_history", False))
    meta["run_training"] = bool(cell.get("run_training", True))
    meta["train_steps"] = int(cell.get("train_steps", cfg.train_steps))
    meta["record_every"] = int(cell.get("record_every", cfg.record_every))
    meta["sampled_trajectories_per_step"] = int(cfg.sampled_trajectories_per_step)
    meta["v14_structure_nonterminal_states"] = int(sum(int(cfg.vocab_size) ** t for t in range(int(structural.horizon))))
    meta["v14_structure_readout_parameters"] = int(cfg.vocab_size) * int(cfg.feature_dim)
    alpha = float(condition.ema_alpha)
    if alpha == 0.0:
        half_life = math.inf
    elif alpha == 1.0:
        half_life = 0.0
    else:
        half_life = math.log(0.5) / math.log(1.0 - alpha)
    meta["ema_half_life_steps"] = half_life
    meta["teacher_weights_exactly_synchronized_between_steps"] = alpha == 1.0
    return meta


def _append_rows(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    if rows:
        pd.DataFrame(rows).to_csv(path, mode="a", header=not path.exists(), index=False)


def run_cell(base, cell: Mapping[str, object], cfg, task_seeds: Sequence[int],
             device: torch.device, dtype: torch.dtype, output_dir: Path,
             problem_cache: MutableMapping[Tuple[object, ...], Any],
             paired_optimization_rng: bool) -> None:
    core = core_from_cell(cell)
    structural = structure_from_cell(cell)

    # H lives in the regime, whereas K and D live in RunConfig in the frozen
    # backend.  Construct a cell-local immutable config so structural robustness
    # can vary K/D without leaking them into the paper-facing task controls.
    cell_cfg = replace(
        cfg,
        vocab_size=int(structural.vocab_size),
        feature_dim=int(structural.feature_dim),
    )
    cell_cfg.validate()
    ablation = ablation_from_cell(cell, cell_cfg.vocab_size)
    regime = _regime_from_specs(base, cell, core, structural, ablation)
    condition = _condition_from_cell(base, cell, regime)
    condition.validate()

    # Explicitly include K/D in our outer cache key even if a future frozen
    # backend changes its own problem_key implementation.
    key = (
        "v14", int(structural.vocab_size), int(structural.feature_dim),
        base.problem_key(regime, cell_cfg, task_seeds),
    )
    if key not in problem_cache:
        problem_cache[key] = base.prepare_problem(regime, cell_cfg, task_seeds, device, dtype)
    prepared = problem_cache[key]

    model = V14ContextModel(
        base, prepared, condition, cell_cfg, device, core, ablation, task_seeds
    )
    meta = _metadata(base, condition, cell_cfg, core, structural, ablation, cell)
    condition_id = base.stable_hash([
        *meta.values(), *task_seeds, cell_cfg.feature_dim, cell_cfg.vocab_size,
    ])
    meta["condition_id"] = condition_id

    initial_base = model.inner.metrics()
    initial_context = model.context_metrics()
    initial_grad = model.inner.gradient_geometry()

    run_training = bool(cell.get("run_training", True))
    steps = int(cell.get("train_steps", cell_cfg.train_steps)) if run_training else 0
    record_every = max(1, int(cell.get("record_every", cell_cfg.record_every)))
    record_history = bool(cell.get("record_history", False))

    if paired_optimization_rng:
        # Common random numbers are paired across KL/rollout/EMA/LR within a
        # scientifically identical task+ablation+optimizer/estimator block.
        rng_id = base.stable_hash([
            condition.analysis_block,
            asdict(core), asdict(structural), asdict(ablation),
            condition.optimizer, condition.estimator,
            *task_seeds, cell_cfg.feature_dim, cell_cfg.vocab_size,
        ])
    else:
        rng_id = condition_id
    generator = base.seeded_generator(
        device, cfg.optimization_seed + int(rng_id[:8], 16) % 1_000_000
    )

    history_rows: List[Dict[str, object]] = []
    match_trace = [initial_base["new_match_student_occupancy"].detach().clone()]
    support_trace = [initial_base["new_support_success"].detach().clone()]
    old_trace = [initial_base["old_match_reference_occupancy"].detach().clone()]
    utility_trace = [initial_context["context_teacher_utility_student_occupancy"].detach().clone()]

    for step in range(1, steps + 1):
        train_stats = model.inner.train_step(generator)
        should_record = step == 1 or step == steps or step % record_every == 0
        if should_record:
            bm = model.inner.metrics()
            cm = model.context_metrics()
            match_trace.append(bm["new_match_student_occupancy"].detach().clone())
            support_trace.append(bm["new_support_success"].detach().clone())
            old_trace.append(bm["old_match_reference_occupancy"].detach().clone())
            utility_trace.append(cm["context_teacher_utility_student_occupancy"].detach().clone())
            if record_history:
                history_rows.extend(base.tensors_to_rows(
                    {**bm, **cm, **train_stats}, task_seeds,
                    {**meta, "step": step},
                ))

    final_base = model.inner.metrics()
    final_context = model.context_metrics()
    final_grad = model.inner.gradient_geometry()

    run_values: Dict[str, Tensor] = {**prepared.problem.task_diagnostics}
    for prefix, values in (
        ("initial", initial_base), ("final", final_base),
        ("initial_context", initial_context), ("final_context", final_context),
        ("initial_param", initial_grad), ("final_param", final_grad),
    ):
        for name, value in values.items():
            run_values[f"{prefix}_{name}"] = value

    run_values.update(
        new_match_gain=final_base["new_match_student_occupancy"] - initial_base["new_match_student_occupancy"],
        new_target_path_match_gain=final_base["new_match_target_occupancy"] - initial_base["new_match_target_occupancy"],
        new_support_success_gain=final_base["new_support_success"] - initial_base["new_support_success"],
        old_match_forgetting=initial_base["old_match_reference_occupancy"] - final_base["old_match_reference_occupancy"],
    )
    if "new_match_heldout_deployment" in final_base:
        run_values["new_heldout_deployment_match_gain"] = (
            final_base["new_match_heldout_deployment"] - initial_base["new_match_heldout_deployment"]
        )

    match_stack = torch.stack(match_trace)
    support_stack = torch.stack(support_trace)
    old_stack = torch.stack(old_trace)
    utility_stack = torch.stack(utility_trace)
    match_peak = match_stack.max(dim=0).values
    support_peak = support_stack.max(dim=0).values
    utility_peak = utility_stack.max(dim=0).values
    run_values.update(
        new_match_peak=match_peak,
        new_match_drawdown_from_peak=match_peak - final_base["new_match_student_occupancy"],
        support_success_peak=support_peak,
        support_success_drawdown_from_peak=support_peak - final_base["new_support_success"],
        old_match_max_drawdown=initial_base["old_match_reference_occupancy"] - old_stack.min(dim=0).values,
        context_utility_peak=utility_peak,
        context_utility_drawdown_from_peak=utility_peak - final_context["context_teacher_utility_student_occupancy"],
    )

    _append_rows(base.tensors_to_rows(run_values, task_seeds, meta), output_dir / "runs.csv")
    if record_history:
        _append_rows(history_rows, output_dir / "history.csv")

    print(
        f"{condition.analysis_block} | {meta['cell_name']} | "
        f"lambda={core.initial_policy_concentration:g} c={core.context_strength:g} "
        f"rho_phi={core.feature_overlap:g} KL={condition.kl_direction} "
        f"rollout={condition.rollout_source} alpha={condition.ema_alpha:g} "
        f"opt={condition.optimizer} est={condition.estimator} "
        f"lr={condition.learning_rate:g} steps={steps}",
        flush=True,
    )


# -----------------------------------------------------------------------------
# Self-tests
# -----------------------------------------------------------------------------


def _gradient_formula_self_test() -> Dict[str, float]:
    torch.manual_seed(20260904)
    z = torch.randn(7, dtype=torch.float64, requires_grad=True)
    q = torch.softmax(torch.randn(7, dtype=torch.float64), dim=-1).detach()
    log_q = torch.log(q)
    p = torch.softmax(z, dim=-1)
    log_p = torch.log_softmax(z, dim=-1)
    forward = (q * (log_q - log_p)).sum()
    gf_auto = torch.autograd.grad(forward, z, retain_graph=True)[0]
    gf = p.detach() - q
    reverse = (p * (log_p - log_q)).sum()
    gr_auto = torch.autograd.grad(reverse, z)[0]
    lr = log_p.detach() - log_q
    gr = p.detach() * (lr - (p.detach() * lr).sum())
    return {
        "forward_logit_gradient_formula_max_abs_error": float((gf_auto - gf).abs().max()),
        "reverse_logit_gradient_formula_max_abs_error": float((gr_auto - gr).abs().max()),
    }


def run_self_tests(base) -> Dict[str, object]:
    device = torch.device("cpu")
    dtype = torch.float64
    cfg = base.RunConfig(
        vocab_size=6, feature_dim=64, train_steps=3, record_every=1,
        n_task_seeds=4, seed_start=141000,
        sampled_trajectories_per_step=32, dtype="float64",
    )
    seeds = list(range(cfg.seed_start, cfg.seed_start + cfg.n_task_seeds))

    def make(
        core: CoreTaskSpec,
        ab: AblationSpec,
        alpha: float = 0.0,
        structural: StructuralSpec = StructuralSpec(),
    ):
        structural.validate()
        local_cfg = replace(
            cfg,
            vocab_size=int(structural.vocab_size),
            feature_dim=int(structural.feature_dim),
        )
        local_cfg.validate()
        ab.validate(local_cfg.vocab_size)
        cell = {
            "name": "selftest", "analysis_block": "v14_selftest",
            "core": asdict(core), "ablation": asdict(ab),
            "structure": asdict(structural), "kl_direction": "reverse",
            "rollout_source": "student", "ema_alpha": alpha,
            "learning_rate": 0.001, "optimizer": "adam", "estimator": "exact",
        }
        regime = _regime_from_specs(base, cell, core, structural, ab)
        cond = _condition_from_cell(base, cell, regime)
        cond.validate()
        prepared = base.prepare_problem(regime, local_cfg, seeds, device, dtype)
        return V14ContextModel(base, prepared, cond, local_cfg, device, core, ab, seeds)

    core = CoreTaskSpec()
    ab = AblationSpec()
    model = make(core, ab)
    p0 = model.inner.student_log_policy("new").exp()
    _, _, _, q0 = model.teacher_components()
    expected = (1.0 - core.context_strength) * p0 + core.context_strength * model.inner.q_new_target
    expected = expected / expected.sum(dim=-1, keepdim=True)
    calibration = float((q0 - expected).detach().abs().max().cpu())

    zero = make(replace(core, context_strength=0.0), ab)
    pz = zero.inner.student_log_policy("new").exp()
    _, _, _, qz = zero.teacher_components()
    zero_diff = float((pz - qz).detach().abs().max().cpu())

    # Endpoint-compatible context must preserve the initial core teacher while
    # being approximately null at the fitted target readout.
    compat = make(core, replace(ab, endpoint_compatible_context=True))
    _, _, _, qc = compat.teacher_components()
    compat_initial = float((qc - expected).detach().abs().max().cpu())
    finite_endpoint = compat._endpoint_relative_error[torch.isfinite(compat._endpoint_relative_error)]
    if finite_endpoint.numel() == 0:
        raise AssertionError("endpoint-compatible self-test produced no finite endpoint residuals")
    endpoint_err = float(finite_endpoint.detach().max().cpu())

    # Noise is deterministic and zero is an exact no-op.
    noisy_a = make(core, replace(ab, feature_noise_scale=0.5))
    noisy_b = make(core, replace(ab, feature_noise_scale=0.5))
    noise_repeat = float((noisy_a.psi_noise - noisy_b.psi_noise).detach().abs().max().cpu())

    # JSD calibration accuracy for a deliberately small common target.
    jsd_model = make(core, replace(ab, target_initial_jsd=0.01))
    jsd_reach = float(jsd_model._jsd_calibration_reachable.detach().min().cpu())
    jsd_err = float((jsd_model._jsd_calibration_achieved - 0.01).detach().abs().max().cpu())

    # Exact synchronization remains an explicit stress point.
    sync = make(core, ab, alpha=1.0)
    sync.inner.train_step(None)
    sync_gap = float((sync.inner.W_teacher - sync.inner.W.detach()).detach().abs().max().cpu())

    # Structural plumbing: non-default H/K/D must reach the embedded backend
    # without becoming core task variables.  K=6 is the smallest value for
    # which the fixed (.30,.15) target still has two elevated modes.
    structural_probe = StructuralSpec(horizon=3, vocab_size=6, feature_dim=48)
    structural_model = make(core, ab, structural=structural_probe)
    structural_shape = tuple(int(x) for x in structural_model.inner.W.shape[-2:])
    structural_horizon = int(structural_model.inner.horizon)

    formula = _gradient_formula_self_test()
    results: Dict[str, object] = {
        "core_default_ablation_is_noop": ab.is_core,
        "core_initial_teacher_calibration_max_abs": calibration,
        "c0_teacher_vs_student_max_abs": zero_diff,
        "endpoint_compatible_initial_teacher_max_abs": compat_initial,
        "endpoint_compatible_endpoint_relative_error_max": endpoint_err,
        "feature_noise_repeatability_max_abs": noise_repeat,
        "iso_jsd_reachable_min": jsd_reach,
        "iso_jsd_calibration_max_abs_error": jsd_err,
        "alpha1_student_teacher_weight_gap": sync_gap,
        "structural_override_horizon": structural_horizon,
        "structural_override_vocab_size": structural_shape[0],
        "structural_override_feature_dim": structural_shape[1],
        **formula,
    }
    failures = []
    if not ab.is_core:
        failures.append("default ablation is not a core no-op")
    # Repeatability is a numerical property, not a bitwise-identity claim.
    # Independent LAPACK/SVD calls can differ by O(machine epsilon) even when
    # seeded inputs are identical, as observed in the user's 5.55e-17 failure.
    repeat_tol = 64.0 * torch.finfo(dtype).eps
    for name, value, tol in (
        ("core calibration", calibration, 5e-9),
        ("c=0 identity", zero_diff, 5e-10),
        ("endpoint initial match", compat_initial, 5e-8),
        ("endpoint null", endpoint_err, 5e-6),
        ("noise repeatability", noise_repeat, repeat_tol),
        ("iso-JSD calibration", jsd_err, 1e-8),
        ("alpha=1 sync", sync_gap, 0.0),
        ("forward gradient formula", formula["forward_logit_gradient_formula_max_abs_error"], 5e-12),
        ("reverse gradient formula", formula["reverse_logit_gradient_formula_max_abs_error"], 5e-12),
    ):
        if value > tol:
            failures.append(f"{name}: {value} > {tol}")
    if jsd_reach < 1.0:
        failures.append("iso-JSD target was not reachable for every self-test seed")
    if structural_horizon != 3 or structural_shape != (6, 48):
        failures.append(
            f"structural override plumbing failed: H={structural_horizon}, "
            f"K,D={structural_shape}, expected H=3,K=6,D=48"
        )
    if failures:
        raise AssertionError(f"v14 self-tests failed: {failures}; results={results}")
    return results


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        return device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Accepted and ignored so older v14 launch scripts remain command-line
    # compatible.  The model never reads or imports the supplied path.
    p.add_argument(
        "--base-script", dest="_legacy_base_script", type=Path, default=None,
        help=argparse.SUPPRESS,
    )
    p.add_argument("--spec", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=Path("results/v14"))
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    p.add_argument("--vocab-size", type=int, default=V14_VOCAB_SIZE)
    p.add_argument("--feature-dim", type=int, default=V14_FEATURE_DIM)
    p.add_argument("--n-task-seeds", type=int, default=96)
    p.add_argument("--seed-start", type=int, default=140000)
    p.add_argument("--seed-batch-size", type=int, default=0)
    p.add_argument("--train-steps", type=int, default=200)
    p.add_argument("--record-every", type=int, default=10)
    p.add_argument("--optimization-seed", type=int, default=904021)
    p.add_argument("--sampled-trajectories-per-step", type=int, default=256)
    p.add_argument("--prompts-per-step", type=int, default=0)
    p.add_argument("--gradient-clip-norm", type=float, default=10.0)
    p.add_argument("--paired-optimization-rng", action="store_true")
    p.add_argument("--self-test", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    base = load_backend()
    if args.self_test:
        print(json.dumps(run_self_tests(base), indent=2, sort_keys=True))
        return
    if args.spec is None:
        raise SystemExit("--spec is required unless --self-test is used")

    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    cells = spec.get("cells", [])
    if not cells:
        raise ValueError("spec must contain a non-empty 'cells' list")

    device = resolve_device(args.device)
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    cfg = base.RunConfig(
        vocab_size=args.vocab_size,
        feature_dim=args.feature_dim,
        train_steps=args.train_steps,
        record_every=args.record_every,
        prompts_per_step=args.prompts_per_step,
        n_task_seeds=args.n_task_seeds,
        seed_start=args.seed_start,
        optimization_seed=args.optimization_seed,
        sampled_trajectories_per_step=args.sampled_trajectories_per_step,
        gradient_clip_norm=args.gradient_clip_norm,
        dtype=args.dtype,
    )
    cfg.validate()
    all_seeds = list(range(args.seed_start, args.seed_start + args.n_task_seeds))
    batch_size = args.seed_batch_size if args.seed_batch_size > 0 else len(all_seeds)
    seed_batches = [all_seeds[i:i + batch_size] for i in range(0, len(all_seeds), batch_size)]

    if args.output_dir.exists() and args.overwrite:
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / "runs.csv").exists() and not args.overwrite:
        raise FileExistsError(f"{args.output_dir}/runs.csv exists; use --overwrite or a new directory")
    shutil.copy2(args.spec, args.output_dir / "experiment_spec.json")

    start = time.time()
    problem_cache: Dict[Tuple[object, ...], Any] = {}
    total = len(cells) * len(seed_batches)
    counter = 0
    for batch_idx, seeds in enumerate(seed_batches):
        for cell in cells:
            counter += 1
            print(f"[{counter}/{total}] seed_batch={batch_idx+1}/{len(seed_batches)}", end=" ", flush=True)
            run_cell(
                base, cell, cfg, seeds, device, dtype, args.output_dir,
                problem_cache, args.paired_optimization_rng,
            )

    manifest = {
        "v14_version": V14_VERSION,
        "created_unix": time.time(),
        "elapsed_seconds": time.time() - start,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "backend": "embedded",
        "v14_script_sha256": _sha256(Path(__file__)),
        "spec_sha256": _sha256(args.spec),
        "args": {
            **{k: v for k, v in vars(args).items() if not k.startswith("_")},
            "spec": str(args.spec),
            "output_dir": str(args.output_dir),
        },
        "n_cells": len(cells),
        "n_task_seeds": len(all_seeds),
        "task_seed_start": all_seeds[0],
        "task_seed_end": all_seeds[-1],
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
