# Contextual self-distillation toy model (v14)

This document describes the standalone mechanism-isolation model implemented in
[`toy_model/contextual_sd_toy_v14.py`](../toy_model/contextual_sd_toy_v14.py).
The model is a controlled, finite autoregressive system for studying how KL
direction, rollout policy, contextual information, and teacher synchronization
affect new-task acquisition and old-task retention.

Version 14 is self-contained. It does not import an earlier toy-model version or
require another source file at runtime.

## 1. Scope and research question

The toy model isolates four parts of contextual self-distillation:

1. a student and teacher sharing a low-dimensional softmax readout;
2. an old task and a context-defined new task sharing that readout;
3. a contextual feature that changes teacher behavior without directly changing
   the student input;
4. occupancy-weighted forward or reverse KL training under student, teacher, or
   mixed rollouts.

The central question is whether a student can acquire a reachable new policy
from its contextualized teacher while retaining the old policy, and how that
trade-off changes with representation overlap, initial policy concentration,
context strength, KL direction, rollout source, and EMA coupling.

This is a mechanism model rather than a numerical fit to language-model data.
Its parameters should be interpreted as controlled causal axes, not as estimates
of properties of a particular foundation model.

## 2. Model overview

The environment is a complete prefix tree with horizon `H` and vocabulary size
`K`. A state `s` is a token prefix of length `0` through `H - 1`. The number of
nonterminal states is

```text
N = 1 + K + K^2 + ... + K^(H-1).
```

The paper-reference structure is:

| Symbol | Meaning | Default |
|---|---|---:|
| `H` | autoregressive horizon | 4 |
| `K` | vocabulary size / local branching factor | 8 |
| `D` | feature and readout dimension | 64 |

Each task `x` has a state-feature matrix `phi_x(s)` and shares the trainable
readout `W`. The student policy is

```text
p_W(a | s, x) = softmax(phi_x(s) W^T / T)_a,
```

where `T=1` in the reference model. The same `W` is used for the old and new
tasks, so learning the new task can interfere with old-task behavior.

The readout is initialized from the old task:

```text
W_student(0) = W_teacher(0) = W_old.
```

The old reference policy is the policy induced by `phi_old` and `W_old`. Only
the new task supplies the training objective; the old task is evaluated to
measure forgetting.

## 3. Core scientific axes

Only three task-level variables are exposed by `CoreTaskSpec`.

| JSON field | Symbol | Default | Valid range | Interpretation |
|---|---:|---:|---:|---|
| `feature_overlap` | `rho_phi` | 0.50 | `[-1, 1]` | Alignment between old- and new-task state features |
| `initial_policy_concentration` | `lambda` | 2.50 | `> 0` | Scale applied to the initial readout; larger values make the initial policy more concentrated |
| `context_strength` | `c` | 0.60 | `[0, 1]` | Initial behavioral displacement from the uncontextualized policy toward the new target |

### Feature overlap

For every state, the new-task feature is constructed as

```text
phi_new = rho_phi * phi_old + sqrt(1 - rho_phi^2) * phi_perp,
```

where `phi_perp` is normalized and orthogonal to `phi_old`. This makes the
shared representation geometry explicit instead of relying on an accidental
correlation between random features.

### Initial policy concentration

`lambda` scales the initial old-task readout before training. It changes prior
confidence and the initial support assigned to the new target without changing
the optimizer learning rate.

### Context strength

At initialization, let `p0` be the uncontextualized student policy on the new
task and `q*` the reachable new-task target. The desired same-weight contextual
teacher is

```text
q_ctx,0 = (1 - c) * p0 + c * q*.
```

The implementation converts this desired behavioral shift into an input feature
intervention. Consequently, `c=0` is an identity control: the initial
contextual teacher and student policies coincide up to numerical precision.

## 4. Fixed core construction

The reference v14 task fixes the remaining construction choices:

- new readout compatibility is zero;
- support-placement bias is zero;
- target complexity/noise is zero;
- target construction uses two prototype modes with masses `0.30` and `0.15`;
- the target is projected into the representable linear-softmax policy family;
- context is available at every prefix;
- no state-private features, prompt holdout, or context anchoring are used;
- context corruption is disabled;
- target support width is two.

The prototype target assigns `0.30` and `0.15` to its two selected modes and
distributes the remaining mass uniformly. The model then least-squares fits its
centered log probabilities with `phi_new` and reconstructs a softmax policy.
This projection makes the training target reachable. It also means that the
final target probabilities are not required to remain exactly `0.30` and
`0.15`; those values define the pre-projection target shape.

With the default target masses, `K` must be at least 6 for the first two modes
to remain elevated relative to every residual token. Although the structural
validator accepts `K >= 4`, target-shape validation rejects smaller values
unless compatible masses are supplied as an explicit ablation.

## 5. Contextual teacher

The contextual signal is represented by a fixed per-state feature `psi`. At
teacher weights `W_teacher`, the teacher policy is

```text
q_t(a | s) = softmax((phi_new(s) + psi(s)) W_teacher(t)^T / T)_a.
```

At initialization, `psi` is solved with a pseudoinverse so that the relative
context logits reproduce `q_ctx,0`. The solver works with token logits relative
to the final token, which removes the softmax-invariant common-logit direction.

The important distinction is:

- the student sees `phi_new`;
- the contextual teacher sees `phi_new + psi`;
- both policies use related readout weights;
- `psi` is constructed once and remains fixed during training;
- teacher behavior can still change because `W_teacher` changes.

This is a weight-dependent input intervention, not a fixed probability-space
mixture applied at every training step. The reported `nominal_context_weight`
is diagnostic metadata and should not be interpreted as a persistent mixture
coefficient in the teacher policy.

## 6. Teacher synchronization

After each student update, the teacher readout is updated as

```text
W_teacher <- (1 - alpha) * W_teacher + alpha * W_student.
```

| `ema_alpha` | Meaning |
|---:|---|
| `0` | frozen teacher |
| between `0` and `1` | EMA teacher |
| `1` | exact synchronization after every student step |

For `0 < alpha < 1`, the recorded half-life in optimizer steps is

```text
log(0.5) / log(1 - alpha).
```

Teacher logits used for a training step are computed before that step's student
update; synchronization affects subsequent steps.

## 7. Training objective and rollout distribution

Let `p` be the student policy and `q` the contextual teacher policy. V14 supports
two token-level objectives:

```text
forward KL: KL(q || p) = sum_a q(a) [log q(a) - log p(a)]
reverse KL: KL(p || q) = sum_a p(a) [log p(a) - log q(a)].
```

Each token loss is weighted by a state occupancy. For any state-wise value
`f(s)`, the normalized tree expectation is

```text
E_d[f] = (1 / H) * sum_s d(s) f(s).
```

The rollout source selects `d`:

| `rollout_source` | Occupancy |
|---|---|
| `student` | occupancy induced by `p` |
| `teacher` | occupancy induced by `q` |
| `mixture` | `(1 - teacher_fraction) d_p + teacher_fraction d_q` for the exact estimator |

The default estimator enumerates the tree exactly. The `sampled` estimator uses
complete Monte Carlo trajectories. For sampled mixture rollouts, a source is
chosen once per trajectory rather than independently at every state.

Occupancy weights are detached from autograd. Training differentiates the
occupancy-weighted KL loss, not a policy-gradient objective through the rollout
distribution.

Supported optimizers are `adam`, `sgd`, and `normalized_sgd`. Gradient clipping
is applied independently to each task-seed replica for Adam and SGD.

## 8. Default-off ablations

`AblationSpec` contains causal interventions that are not part of the core
model. A row is marked `v14_core_only=true` only when every field has its core
default.

| Field | Default | Effect |
|---|---:|---|
| `readout_compatibility` | `0.0` | Correlation between the old readout and the latent new-task readout |
| `support_placement_bias` | `0.0` | Biases target modes toward tokens with low initial student probability |
| `target_primary_mass` | `0.30` | Primary prototype target mass |
| `target_secondary_mass` | `0.15` | Secondary prototype target mass |
| `endpoint_compatible_context` | `false` | Solves a context feature that matches the initial correction and is null at a fitted target readout |
| `feature_noise_scale` | `0.0` | Adds fixed Gaussian feature-space corruption with the requested RMS relative to the signal |
| `behavior_noise_scale` | `0.0` | Adds a fixed behavioral logit perturbation and maps it back into feature space |
| `offpath_context_multiplier` | `1.0` | Scales context outside the target-supported prefix tree |
| `state_competence_bias` | `0.0` | Weakens context on initially student-heavy or teacher-heavy states using a frozen, source-independent mask |
| `target_initial_jsd` | `null` | Calibrates `c` per seed to a requested initial student/teacher JSD |

Feature-space and behavior-space noise cannot be enabled in the same cell. Both
are deterministic for a fixed task seed and are orthogonalized against the
uncorrupted context signal before scaling.

When `target_initial_jsd` is set, `core.context_strength` is still recorded but
does not set the signal. A per-seed bisection chooses `c` in `[0,1]`. If the
requested JSD is unreachable, the model uses `c=1` and records
`v14_target_jsd_reachable=0`.

For `state_competence_bias`:

- a positive value weakens context on states favored by the initial student;
- a negative value weakens context on states favored by the initial teacher;
- the mask is computed at initialization and never depends on the cell's
  training rollout source.

## 9. Experiment specification

Experiments are declared as JSON with a non-empty `cells` list. The following is
a complete minimal example:

```json
{
  "cells": [
    {
      "name": "core_reverse_student",
      "analysis_block": "main",
      "core": {
        "feature_overlap": 0.5,
        "initial_policy_concentration": 2.5,
        "context_strength": 0.6
      },
      "structure": {
        "horizon": 4,
        "vocab_size": 8,
        "feature_dim": 64
      },
      "ablation": {},
      "kl_direction": "reverse",
      "rollout_source": "student",
      "ema_alpha": 0.0025,
      "learning_rate": 0.001,
      "optimizer": "adam",
      "estimator": "exact",
      "teacher_fraction": 0.5,
      "train_steps": 200,
      "record_every": 10,
      "record_history": true,
      "run_training": true
    }
  ]
}
```

Cell defaults are:

| Field | Default |
|---|---:|
| `name` | condition variant or `declared` |
| `analysis_block` | `v14` |
| `kl_direction` | `reverse` |
| `rollout_source` | `student` |
| `ema_alpha` | `0.0025` |
| `learning_rate` | `0.001` |
| `optimizer` | `adam` |
| `estimator` | `exact` |
| `teacher_fraction` | `0.5` |
| `train_steps` | command-line `--train-steps` |
| `record_every` | command-line `--record-every` |
| `record_history` | `false` |
| `run_training` | `true` |

Unknown fields inside `core`, `structure`, or `ablation` are rejected. Use the
cell's `structure` object to vary `H`, `K`, or `D`; it is the authoritative
structural configuration for that cell.

## 10. Running the model

The toy model directly requires PyTorch, NumPy, and pandas. The repository
dependency file provides these packages.

Run the built-in validation suite first:

```bash
python toy_model/contextual_sd_toy_v14.py --self-test
```

Run an experiment:

```bash
python toy_model/contextual_sd_toy_v14.py \
  --spec path/to/experiment.json \
  --output-dir results/toy_model/v14/example \
  --device auto \
  --dtype float32 \
  --n-task-seeds 96 \
  --seed-start 140000 \
  --train-steps 200 \
  --record-every 10 \
  --paired-optimization-rng
```

Important command-line options:

| Option | Default | Purpose |
|---|---:|---|
| `--spec` | none | JSON experiment specification; required except with `--self-test` |
| `--output-dir` | `results/v14` | Output directory |
| `--overwrite` | off | Remove and recreate the selected output directory |
| `--device` | `auto` | Explicit PyTorch device, or CUDA when available and CPU otherwise |
| `--dtype` | `float32` | Computation dtype (`float32` or `float64`) |
| `--n-task-seeds` | `96` | Number of independent paired task replicas |
| `--seed-start` | `140000` | First task seed; subsequent seeds are consecutive |
| `--seed-batch-size` | `0` | Replicas per batch; zero runs all seeds together |
| `--train-steps` | `200` | Default training steps for cells that do not override it |
| `--record-every` | `10` | Default metric-history interval |
| `--optimization-seed` | `904021` | Base seed for sampled rollouts and prompt sampling |
| `--sampled-trajectories-per-step` | `256` | Trajectories used by the sampled estimator |
| `--prompts-per-step` | `0` | Optional prompt minibatch size; zero uses all trainable prompts |
| `--gradient-clip-norm` | `10.0` | Per-replica gradient clipping threshold; nonpositive disables clipping |
| `--paired-optimization-rng` | off | Uses common random numbers across matched method cells |

The current v14 exploration and recovery drivers invoke this engine directly
and do not require an earlier toy-model version. The engine still accepts the
old `--base-script` argument silently for compatibility with external legacy
commands, but its value is ignored and no file is loaded.

`--overwrite` recursively removes the exact selected output directory before
the run. Use a dedicated result path.

## 11. Reproducibility and pairing

Task instances use consecutive seeds starting at `--seed-start`. Each task is
initially constructed with CPU `float64` random draws, then transferred to the
requested device and computation dtype.

Without `--paired-optimization-rng`, each condition receives an optimization RNG
derived from its condition identifier. With the flag enabled, common random
numbers are shared across cells that have the same:

- analysis block;
- core, structural, and ablation specifications;
- optimizer and estimator;
- task seeds and tensor dimensions.

The paired RNG intentionally excludes KL direction, rollout source, EMA rate,
and learning rate. This allows controlled method comparisons when the estimator
contains sampling noise. Exact-estimator cells do not normally consume this RNG
during occupancy computation.

Task-seed batching changes memory use without changing which task seeds are
included. Because the seed batch participates in condition identifiers and
optimization RNG derivation, changing the batch size can change sampled-
estimator trajectories. Keep the batch size fixed for strict sampled-run
comparisons.

Numerically identical results across hardware, PyTorch versions, or linear
algebra backends are not guaranteed, particularly for pseudoinverse, SVD, and
least-squares operations. The included tolerances test the intended numerical
properties rather than requiring universal bitwise identity.

## 12. Output contract

Each run directory contains:

| File | Contents |
|---|---|
| `runs.csv` | One summary row per task seed and cell |
| `history.csv` | Intermediate recorded steps for cells with `record_history=true`; omitted if no history is requested |
| `manifest.json` | Version, environment, device, CLI arguments, source/spec hashes, seed range, runtime, and embedded-backend marker |
| `experiment_spec.json` | Exact copy of the input specification |

`runs.csv` combines several namespaces:

- unprefixed task-construction diagnostics;
- `initial_*` and `final_*` learner metrics;
- `initial_context_*` and `final_context_*` context diagnostics;
- `initial_param_*` and `final_param_*` parameter-gradient diagnostics;
- acquisition, forgetting, peak, and drawdown summaries;
- `v14_core_*`, `v14_structure_*`, and `v14_ablation_*` metadata.

Historical backend metadata is retained for compatibility, but the `v14_*`
columns are the authoritative paper-facing parameter names.

## 13. Key metrics

Total variation at a state is

```text
TV(p, q) = 0.5 * sum_a |p(a) - q(a)|.
```

The most useful summary metrics are:

| Metric | Interpretation |
|---|---|
| `new_match_student_occupancy` | `1 - TV(student, new target)`, weighted by the student's new-task occupancy |
| `new_match_target_occupancy` | The same match weighted by target-policy occupancy |
| `new_match_gain` | Final minus initial student-occupancy new-task match |
| `new_support_success` | Probability that every generated token stays inside the two-mode target support tree |
| `old_match_reference_occupancy` | `1 - TV(student old-task policy, old reference)`, weighted by old-reference occupancy |
| `old_match_forgetting` | Initial minus final old-reference match; positive values mean forgetting |
| `teacher_utility_student_occupancy` | Teacher match to the new target along student-visited states |
| `student_teacher_rollout_occupancy_tv` | Difference between student and teacher state occupancies |
| `new_match_heldout_deployment` | Student-occupancy match restricted to held-out prompts |

The core model has no prompt holdout, so held-out metrics are `NaN` by design.
They remain in the output schema for compatibility with diagnostic variants.

Context diagnostics report:

- calibration error between the requested and realized initial contextual
  teacher;
- context feature and logit RMS values;
- context utility relative to the uncontextualized weight teacher;
- changes in the contextual logit displacement as teacher weights move;
- teacher/student parameter drift;
- forward KL, reverse KL, JSD, entropy, TV, and local logit-gradient geometry;
- signal concentration in the top 10% and 25% of states.

Parameter diagnostics compare counterfactual forward/reverse and
student-/teacher-rollout gradients at the same weights. They are diagnostic
measurements and do not add extra optimizer updates.

## 14. Interpreting results

Recommended comparisons use paired task seeds and vary one declared factor at a
time. In particular:

- compare KL directions at fixed rollout source, task, optimizer, and teacher
  dynamics;
- compare rollout sources at fixed KL direction;
- report new-task gain together with old-task forgetting;
- inspect teacher utility and student/teacher occupancy divergence before
  attributing a result to KL direction;
- verify calibration, target reachability, capacity ratio, and active readout
  rank before interpreting a failed learning condition;
- label every non-default `AblationSpec` field as an ablation rather than part
  of the reference model.

The toy model can distinguish optimization-path mechanisms under controlled
geometry. It does not establish that the same effect size will occur in a
large language model, nor does it model finite datasets, natural-language
semantics, or transformer representation learning.

## 15. Built-in validation

`--self-test` checks:

- default ablations are a core no-op;
- exact `c=0` student/teacher identity;
- contextual teacher calibration;
- endpoint-compatible construction;
- deterministic Gaussian perturbations;
- per-seed JSD calibration;
- exact `alpha=1` synchronization;
- non-default structural plumbing;
- analytic forward- and reverse-KL logit-gradient formulas.

Run this suite after changing task construction, context calibration, teacher
updates, KL losses, optimizer behavior, or structural configuration.
