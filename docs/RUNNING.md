# Running experiments and evaluations

This guide describes the main execution routes in the current codebase.

## Which entry point should be used?

| Goal | Entry point |
|---|---|
| Run the contextual self-distillation mechanism-isolation toy model | `toy_model/contextual_sd_toy_v14.py` (see [TOY_MODEL.md](TOY_MODEL.md)) |
| Train one dataset/phase directly | `main.py` |
| Run a complete experiment or sweep from configuration rows | `run_experiments.py` |
| Evaluate every checkpoint of one experiment | `run_all_checkpoint_evals.py` |
| Evaluate one model/checkpoint on one project dataset | `eval_lib.run_standalone_eval()` |
| Run external forgetting benchmarks | `scripts/lmeval.sh` / `scripts/run_lmeval.py` |
| Calibrate untouched-model baselines | `scripts/calibrate_baselines.py` |
| Analyze response lengths | `metrics_and_plots/analyze_response_lengths.py` |
| Analyze training-log statistics | `metrics_and_plots/analyze_training_log_stats.py` |
| Extract raw single-phase metric tables | `metrics_and_plots/build_single_phase_metrics_raw.py` |
| Rebuild paper figures and statistical tables | `figures/fig1.py`, `figures/fig2.py`, `figures/fig3.py`, `figures/stats_axes.py` |

## Direct single-phase training with `main.py`

`main.py` trains exactly one phase. It loads the requested student/reference model, resolves the dataset adapter and contextualization strategy, builds the training dataset, and launches `DistilTrainer`.

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --dataset_name math_contradiction \
  --model_name ministral-3-3b \
  --output_dir outputs/example/math_contradiction \
  --seed 31 \
  --learning_rate 2e-5 \
  --num_train_epochs 1 \
  --num_prompts_per_batch 32 \
  --alpha 0 \
  --generate_from_teacher true \
  --sync_ref_model true \
  --ref_model_mixup_alpha 0.02 \
  --optimal_policy_source teacher \
  --optim_loss jsd \
  --context_strategy dataset_default \
  --save_steps 20
```

Important training arguments:

- `--alpha 0`: forward KL.
- `--alpha 1`: reverse KL.
- `0 < --alpha < 1`: generalized Jensen-Shannon objective implemented by the trainer.
- `--generate_from_teacher false`: student rollout.
- `--generate_from_teacher true`: teacher rollout.
- `--sync_ref_model false`: fixed reference model.
- `--sync_ref_model true`: update the reference model every `ref_model_sync_steps`; `ref_model_mixup_alpha` controls the interpolation strength.
- `--optimal_policy_source teacher`: distill the teacher distribution.
- `--optimal_policy_source dataset --optim_loss cross_entropy`: dataset-target SFT path.
- `--context_strategy ...`: teacher contextualization policy.

### Spatial Contradiction example

Spatial Contradiction uses the same training and evaluation entry points as the other tasks:

```bash
python main.py \
  --dataset_name spatial_contradiction2 \
  --model_name qwen2.5-7b \
  --output_dir outputs/example_spatial/spatial_contradiction2 \
  --learning_rate 1e-5 \
  --num_train_epochs 1 \
  --context_strategy dataset_default
```

The adapter uses exact boxed-coordinate scoring and exposes a `test_data` split for dataset-specific audits. The paired ordinary-semantics view remains available as `spatial_standard2`.

### Multi-process/tensor-parallel training

When launching more than one training process, use `torchrun`:

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run \
  --nproc_per_node 2 \
  --master_port 29501 \
  main.py \
  --dataset_name tooluse \
  --model_name qwen2.5-7b \
  --output_dir outputs/example_tp/tooluse \
  --num_prompts_per_batch 32 \
  --vllm_tensor_parallel_size 2 \
  --learning_rate 1e-5
```

The code checks two invariants before training:

1. `WORLD_SIZE % vllm_tensor_parallel_size == 0`.
2. `num_prompts_per_batch % WORLD_SIZE == 0`.

These constraints keep vLLM sharding valid and make `num_prompts_per_batch` a world-size-independent global batch definition.

## CSV orchestration with `run_experiments.py`

`run_experiments.py` always reads an experiments CSV. It does not expose a separate “single experiment” CLI configuration. To run one experiment through the orchestrator, provide a CSV with exactly one enabled row.

### CSV schema

The required fields are:

```text
exp_name
alpha
generate_from_teacher
ref_model_mixup_alpha
learning_rate
sync_ref_model
optimal_policy_source
optim_loss
```

Common optional fields include:

```text
enabled
status
seed
phase_sequence
name_suffix
num_train_epochs
save_steps
context_strategy
feedback_model_source
model
notes
```

Example:

```csv
enabled,status,exp_name,seed,phase_sequence,name_suffix,alpha,generate_from_teacher,ref_model_mixup_alpha,learning_rate,sync_ref_model,optimal_policy_source,optim_loss,context_strategy,feedback_model_source,num_train_epochs,save_steps
1,,fwd_teacher_ema_default_ep1/lr2e-5,31,math_contradiction,math_contradiction_only,0,TRUE,0.02,2e-5,TRUE,teacher,jsd,dataset_default,,1,20
```

Use `lr<value>` as the final `exp_name` component. For example, the row above
creates `lr2e-5_s31_math_contradiction_only`. The CSV `learning_rate` value is
authoritative; `run_experiments.py` checks that it agrees with the directory
label. Legacy `vN` directories can still be analyzed, but no learning rate is
guessed from a version number. If such a legacy run has neither recorded
configuration nor a saved training command, its learning rate remains missing.

`phase_sequence` may contain a single dataset or a space/comma-separated sequence. For multi-phase runs, each phase starts from the latest usable checkpoint of the previous phase.

The optional `model` column overrides the command-level `--initial_model` for that row.

### Complete orchestrated run

```bash
cd <repo-root>
env \
  CUDA_VISIBLE_DEVICES=0 \
  VISIBLE_DEVICES=0 \
  WANDB_MODE=offline \
  HF_ALLOW_CODE_EVAL=1 \
  DUMP_TEACHER_PROMPTS=1 \
  python run_experiments.py \
    --experiments_csv /path/to/experiments.csv \
    --master_port 10143 \
    --run_lm_eval true \
    --run_cleanup true \
    --resume \
    --resume_partial_phase \
    --output_root outputs/ministral \
    --vllm_tensor_parallel_size 1 \
    --nproc_per_node 1 \
    --initial_model ministral-3-3b
```

For each enabled row the orchestrator can perform:

1. phase training;
2. checkpoint evaluation after each phase;
3. response-length analysis;
4. training-log statistics;
5. interim checkpoint cleanup while preserving the latest required checkpoints;
6. final lm-eval on the latest checkpoint of the final phase;
7. final cleanup.

### Resume and progress identity

The orchestrator stores global progress under `logs/experiment_progress.json` and phase/stage markers inside each experiment root. Progress identity includes the run name, a canonicalized model identity, and the output root. This prevents a completed Qwen2.5 run from causing a same-named Ministral run to be skipped.

`experiment_config.json` stores the selected `initial_model` and `model_identity`, which are also used by downstream evaluation and summary code.

Useful flags:

- `--resume`: skip completed phases/stages.
- `--resume_partial_phase`: resume an incomplete phase from its latest usable checkpoint.
- `--max_retries N`: retry failures; retries enable resume behavior.
- `--continue_on_error`: continue to subsequent CSV rows after a failed run.
- `--no_skip_done`: ignore completed-progress entries and revisit the row.
- `--reverse_experiments`: iterate enabled rows in reverse order, useful for two workers consuming opposite ends of the same CSV.

## Context strategy syntax

The contextualized loader supports:

```text
strategy
strategy(prompt_variation)
strategy(prompt_variation,stage1_variation)
strategy(*)
strategy(*,*)
strategy_a|strategy_b
```

Examples:

```text
gold_response_as_guidance(3)
self_feedback_concrete(2,4)
self_feedback_concrete(*,*)
gold_hint|self_feedback_concrete
```

Wildcards and strategy pools are resolved per dataset row with deterministic RNG derived from the training seed, dataset name, row index, and strategy specification.

Some strategies select non-default canonical training views. A strategy pool may only combine strategies that consume the same view. Source-only views reject dataset-target/SFT supervision when no gold completion exists.

## Evaluate all checkpoints in an experiment

```bash
CUDA_VISIBLE_DEVICES=0 python run_all_checkpoint_evals.py \
  --experiment_root outputs/<method>/<run> \
  --skip_existing
```

Useful options:

```text
--dataset_order ...
--eval_subset ...
--gpu_memory_utilization 0.6
--tensor_parallel_size 1
--temperature 0.0
--<dataset>_max_new_tokens N
```

The script loads each checkpoint once for all pending datasets, reuses the shared adapter/eval library, and writes model-aware baseline points into `training_eval_curve.json`.

## Evaluate one model/checkpoint on one project dataset

For a single evaluation without scanning an experiment tree, use the library API:

```bash
python - <<'PY'
from eval_lib import run_standalone_eval

summary = run_standalone_eval(
    dataset_name="math_contradiction",
    model_path="ministral-3-3b",
    output_dir="eval_outputs/ministral_math_contradiction",
    gpu_memory_utilization=0.6,
    tensor_parallel_size=1,
)
print(summary)
PY
```

The model may be a registry key, registered HF repository id, recognized local Trainer checkpoint, or a generic causal-LM path.

## lm-eval-harness

Run the project wrapper on a checkpoint directory:

```bash
VISIBLE_DEVICES=0 ./scripts/lmeval.sh \
  outputs/<method>/<run>/<phase>/checkpoint-<step>
```

Default tasks:

```text
hellaswag,mmlu,truthfulqa,winogrande,humaneval,ifeval
```

Relevant environment variables:

| Variable | Purpose |
|---|---|
| `VISIBLE_DEVICES` | GPUs exposed to lm-eval; falls back to `CUDA_VISIBLE_DEVICES`. |
| `GPU_MEMORY_UTILIZATION` | vLLM memory fraction. |
| `TASKS` | Comma-separated benchmark list. |
| `LM_EVAL_BATCH_SIZE` | lm-eval batch size; default `auto`. |
| `LM_EVAL_TENSOR_PARALLEL_SIZE` | vLLM TP size. |
| `LM_EVAL_MAX_MODEL_LEN` | Optional maximum model length. |
| `LM_EVAL_APPLY_CHAT_TEMPLATE` | Opt in to lm-eval chat templating; default is off. |
| `LM_EVAL_LIMIT` | Optional smoke-test limit. |

The runner records package versions, resolved model arguments, protocol id, task list, and checkpoint path. `scripts/summarize_lmeval.py` then writes a compact `lm_eval_summary.json` with model-specific baselines and deltas.

A configuration-only smoke test is available:

```bash
VISIBLE_DEVICES=0 ./scripts/lmeval.sh <checkpoint-dir> --dry-run
```

## Untouched-model baseline calibration

When adding a new base model, measure its project-dataset and lm-eval baselines before computing model-relative deltas:

```bash
python -m scripts.calibrate_baselines \
  --model <registry-key> \
  --datasets tooluse science math_contradiction spatial_contradiction2 spatial_standard2 \
  --seeds 31 37 717 \
  --output-root baseline_results
```

Each seed runs in an isolated process. The aggregate report contains means, sample standard deviations, per-seed values, environment provenance, and registry-ready dictionaries.

The Qwen2.5 registry contains the Spatial Contradiction v2 measurement (`spatial_contradiction2=0.015`). For other model/dataset combinations, leave the baseline absent until that model has been calibrated; baseline-dependent metrics will then fail clearly or remain unavailable instead of reusing Qwen2.5 values.

## Response-length and training-log analysis

These can run independently of the orchestrator:

```bash
python metrics_and_plots/analyze_response_lengths.py \
  --input outputs/<method>/<run> \
  --eval_subset math_contradiction \
  --length-mode hf_tokens_if_available
```

```bash
python metrics_and_plots/analyze_training_log_stats.py \
  --input outputs/<method>/<run> \
  --dataset_order math_contradiction
```

## Cross-experiment metric extraction

The included extractor scans a single outputs tree and writes per-seed, per-step, and per-checkpoint CSVs for single-phase runs:

```bash
python metrics_and_plots/build_single_phase_metrics_raw.py \
  --outputs-root outputs/ministral \
  --output-dir metrics_out/ministral/all_methods/complete \
  --model ministral-3-3b \
  --learning-metric accuracy_delta \
  --include-debug-columns
```

The three outputs are:

- `context_single_phase_all_per_seed.csv`
- `single_phase_training_dynamics_steps.csv`
- `single_phase_training_dynamics_checkpoints.csv`

Both `--model` and `--model-name-or-path` are accepted. Per-run `experiment_config.initial_model` metadata takes precedence; the CLI value is a fallback and consistency check.

The extractor does not aggregate sequential experiments or produce plots. Use separate output roots/directories for different base models and build paper-specific summaries from these raw tables.

## Paper figures and statistical tables

The paper-figure pipeline is separate from raw metric extraction. First use
`figures/collect_llm_experiments.py` to combine the per-seed metric CSVs into
`figures/data/runs.parquet`. Then run:

```bash
uv run --offline figures/fig1.py
uv run --offline figures/fig2.py
uv run --offline figures/fig3.py
uv run --offline figures/stats_axes.py
```

The figure scripts write main-paper PDFs under `figures/out/final/`, appendix
variants under `figures/out/final/appendix/`, and statistical CSVs under
`figures/out/stats/`. The source release does not bundle the required experiment
parquet or controlled-model aggregate CSVs. Their expected locations, collector
commands, exact output filenames, fixed analysis slices, and statistical tests
are documented in [figures/README.md](../figures/README.md).

## Cleanup utilities

The orchestrator uses `scripts/cleanup_checkpoints.sh` during managed runs. It can also be invoked directly; start with a dry run:

```bash
./scripts/cleanup_checkpoints.sh outputs/<method>/<run> --dry-run
```

Without `--dry-run`, the script removes recognized disposable Trainer checkpoint artifacts. Use `--keep-last-<phase>` when the newest checkpoint for a phase must remain loadable.
