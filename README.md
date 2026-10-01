# Disentangling Self-Distillation

This repository contains the code, data, analysis, and toy-model artifacts for
[*Disentangling Self-Distillation: Measuring and Modeling Acquisition and
Retention*](https://arxiv.org/abs/2609.39494).

The project is a research codebase for studying **on-policy self-distillation for continual adaptation of language models**. A student model is optimized against a teacher distribution derived from the same model family, while the teacher can receive privileged, instance-specific context such as reference responses, hints, feedback, rationales, source documents, or successful sibling trajectories.

The framework exposes the main self-distillation design axes independently: rollout source (student or teacher), divergence direction (forward KL, reverse KL, or intermediate generalized Jensen-Shannon objectives), teacher coupling (frozen or synchronized/EMA-style), supervision source (teacher distribution or dataset target), contextualization strategy, and prompt/template realization.

This repository is based on and extends Idan Shenfeld and collaborators' open-source [SDFT Self-Distillation project](https://github.com/idanshen/Self-Distillation), associated with the paper [*Self-Distillation Enables Continual Learning*](https://arxiv.org/abs/2601.19897). It is an independent research extension, not the official upstream distribution. Its current architecture includes a model registry, contextualization strategies, multi-dataset adapters, model-aware baselines, reproducible lm-eval integration, experiment orchestration, cross-experiment metric extraction, and a mechanism-isolation toy model.

## Highlights

- **On-policy self-distillation** with student- or teacher-generated rollouts.
- **Forward KL, reverse KL, and intermediate generalized Jensen-Shannon objectives** through the `alpha` parameter.
- **Frozen or synchronized reference models**, including EMA-style updates through `ref_model_mixup_alpha`.
- **Static and dynamic privileged-context strategies**, including reference examples/guidance, compact hints, source grounding, self-feedback, rationalization, sibling-response selection, and SDPO-like feedback.
- **Central model registry** for architecture-specific loading, chat templating, vLLM configuration, checkpoint recognition, weight synchronization, and model-specific evaluation baselines.
- **Central dataset adapter layer** for training views, evaluation prompting, scoring, generation limits, expert/reference metrics, and chance floors.
- **Sequential and single-phase experiment orchestration** with resumable, model-aware progress tracking.
- **Checkpoint evaluation and lm-eval-harness integration** with reproducibility metadata and model-aware baselines.
- **Cross-experiment raw metric extraction** for final accuracy, learning deltas, response length, training dynamics, checkpoint dynamics, and lm-eval results.
- **Standalone mechanism-isolation toy model** for controlled studies of KL direction, rollout policy, contextual information, and teacher synchronization.

## Repository lineage

The project builds directly on the SDFT implementation from [idanshen/Self-Distillation](https://github.com/idanshen/Self-Distillation), which implements on-policy self-distillation for continual learning. This repository provides a broader experimental framework around that training method, covering multiple model families, datasets, contextualization policies, evaluation protocols, analysis workflows, and controlled toy-model experiments.

When using this repository, cite *Disentangling Self-Distillation: Measuring and
Modeling Acquisition and Retention* and the upstream SDFT work as appropriate.
Full BibTeX metadata will be added when the arXiv identifier is assigned.

## Installation

Python 3.12 is recommended.

The repository contains **two dependency specifications**:

| File | Environment | Intended use |
|---|---|---|
| `requirements-vlm.txt` | `distillation-vlm` | **Current environment.** Multi-model training/evaluation, current TRL/vLLM/Transformers stack, and `lm_eval==0.4.12`. |
| `requirements.txt` | `distillation` | **Legacy environment.** Reproducing the original Qwen2.5-only setup. It is not the recommended environment for the current multi-model code. |

Clone or download this repository, then enter its root directory:

```bash
cd /path/to/disentangling-self-distillation
```

Then create the current environment:

```bash
python3.12 -m venv distillation-vlm
source distillation-vlm/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-vlm.txt
```

The current requirements pin the core stack used by this branch, including PyTorch 2.11, Transformers 5.5.1, vLLM 0.20.0, TRL 0.27.0, and `lm_eval[ifeval]==0.4.12`.

See [docs/INSTALLATION.md](docs/INSTALLATION.md) for environment validation, the legacy setup, and model-specific notes.

## Model support

Models are selected through `model_registry.py`. A model can be referenced by its registry key, Hugging Face repository id, or—when recognized—an HF Trainer checkpoint path.

| Registry key | Upstream model | Status in the current text-only pipeline |
|---|---|---|
| `qwen2.5-7b` | `Qwen/Qwen2.5-7B-Instruct` | **Validated; default model.** |
| `ministral-3-3b` | `mistralai/Ministral-3-3B-Instruct-2512-BF16` | **Validated for the current training and evaluation workflows.** |
| `qwen3.5-4b` | `Qwen/Qwen3.5-4B` | **Registered but not yet end-to-end validated.** |
| `ministral-3-3b-fp8` | `mistralai/Ministral-3-3B-Instruct-2512` | Optional FP8 registry entry; use only on compatible hardware and validate independently of the BF16 path. |

The registry also owns model-specific base accuracies and lm-eval baselines. Dataset and metrics code query these values through the shared adapter/registry APIs rather than duplicating model-specific constants.

## Datasets

The evaluation/training adapters currently cover:

| Adapter | Bundled-data provenance and license |
|---|---|
| `tooluse` | Prepared from [ToolAlpaca](https://github.com/tangqiaoyu/ToolAlpaca), Apache-2.0. |
| `science` | Chemistry L-3 originates from [SciKnowEval](https://huggingface.co/datasets/hicai-zju/SciKnowEval), whose original release is MIT-licensed. The prepared training snapshot distributed through `idanshen/Self-Distillation` has no explicit license for its modifications; it is not covered by this project's MIT grant. |
| `math_contradiction` | Project-original transformations and annotations are MIT-licensed; source questions derive from the [DeepMind Mathematics Dataset](https://github.com/google-deepmind/mathematics_dataset), Apache-2.0. |
| `spatial_contradiction2` | Project-original synthetic data, MIT. |
| `spatial_standard2` | Ordinary-world control paired with the project-original spatial data, MIT. |

Dataset locations, prompt construction, scoring, canonical training views, and default generation limits are defined in `dataset_adapters.py`.

The list above describes adapter support, not a guarantee that every prepared dataset is distributed with the source release. Prepared Hugging Face datasets are runtime artifacts under `data/`; before running an adapter, make sure its configured train/evaluation directories exist locally. Dataset construction and preparation utilities are provided under `data_utils/` where available. See [data/README.md](data/README.md) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the precise licensing boundary.

## Quick start

### 1. Train one phase directly

`main.py` is the direct entry point for **one training phase on one dataset**. It does not read the experiments CSV.

```bash
cd <repo-root>
CUDA_VISIBLE_DEVICES=0 python main.py \
  --dataset_name math_contradiction \
  --model_name ministral-3-3b \
  --output_dir outputs/example_run/math_contradiction \
  --learning_rate 2e-5 \
  --num_train_epochs 1 \
  --alpha 0 \
  --generate_from_teacher true \
  --sync_ref_model true \
  --ref_model_mixup_alpha 0.02 \
  --optimal_policy_source teacher \
  --optim_loss jsd \
  --context_strategy dataset_default \
  --save_steps 20
```

For multi-process or tensor-parallel training, launch `main.py` through `torchrun` and ensure that the training world size is divisible by `--vllm_tensor_parallel_size`. The global `--num_prompts_per_batch` must also be divisible by the world size.

### 2. Run an experiment sweep from CSV

`run_experiments.py` is the recommended orchestrator for complete experiment workflows. It is **CSV-driven**: a single experiment is represented by a CSV with one enabled row; a sweep contains multiple enabled rows.

Example row:

```csv
enabled,status,exp_name,seed,phase_sequence,name_suffix,alpha,generate_from_teacher,ref_model_mixup_alpha,learning_rate,sync_ref_model,optimal_policy_source,optim_loss,context_strategy,feedback_model_source,num_train_epochs,save_steps
1,,fwd_teacher_ema_default_ep1/lr2e-5,31,math_contradiction,math_contradiction_only,0,TRUE,0.02,2e-5,TRUE,teacher,jsd,dataset_default,,1,20
```

The final `exp_name` component records the learning rate explicitly. The
example produces a run directory named
`lr2e-5_s31_math_contradiction_only`. The `learning_rate` column remains the
authoritative training value, and the orchestrator rejects a conflicting
`lr...` label. Historical `vN` run directories remain readable, but their
learning rate is never inferred from the version label.

Run it with:

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

A CSV may also contain an optional `model` column; when present, it overrides `--initial_model` for that row.

The orchestrator records `experiment_config.json`, per-phase checkpoints and logs, checkpoint evaluations, lm-eval summaries, response-length analysis, training-log statistics, cleanup markers, and model-aware progress metadata.

See [docs/RUNNING.md](docs/RUNNING.md) for the full execution routes and resume semantics.

## Evaluation routes

### Evaluate all checkpoints in an experiment

```bash
CUDA_VISIBLE_DEVICES=0 python run_all_checkpoint_evals.py \
  --experiment_root outputs/<method>/<run> \
  --skip_existing
```

The evaluator reads the experiment's `initial_model` metadata, resolves model-specific baseline accuracies, evaluates checkpoints through `eval_lib.py`, and writes `training_eval_curve.json` plus PNG/PDF plots.

### Run lm-eval-harness on one checkpoint

```bash
VISIBLE_DEVICES=0 ./scripts/lmeval.sh \
  outputs/<method>/<run>/<phase>/checkpoint-<step>
```

The current lm-eval protocol uses `lm_eval==0.4.12` with the vLLM backend and the default task set:

`hellaswag,mmlu,truthfulqa,winogrande,humaneval,ifeval`

The runner writes raw lm-eval results, environment/protocol metadata, and a compact model-aware `lm_eval_summary.json`.

### Calibrate a new model's baselines

```bash
python -m scripts.calibrate_baselines \
  --model <registry-key-or-model-id> \
  --datasets tooluse science math_contradiction spatial_contradiction2 spatial_standard2 \
  --seeds 31 37 717
```

This uses the same project evaluation paths as checkpoint evaluation and lm-eval, then produces per-seed measurements and registry-ready aggregate values.

## Metrics extraction

The included cross-experiment extractor writes three raw CSV tables for single-phase runs: one row per seed, one row per training step, and one row per evaluated checkpoint. It is model-aware: pass `--model` (or `--model-name-or-path`) as a fallback and consistency check when a run does not record `initial_model`.

Example:

```bash
python metrics_and_plots/build_single_phase_metrics_raw.py \
  --outputs-root outputs/ministral \
  --output-dir metrics_out/ministral/all_methods/complete \
  --model ministral-3-3b \
  --learning-metric accuracy_delta \
  --include-debug-columns
```

The extractor intentionally does not generate publication plots or aggregate sequential experiments. Build those analyses from the emitted CSVs with the plotting/statistical workflow used for the corresponding paper artifact.

A single metrics invocation should aggregate runs that share the same base model. Use separate invocations/output directories when comparing experiments trained from different base models.

## Paper figures and statistical tables

The `figures/` directory contains the source for the three paper-figure families
and their matched-pair statistical tables. Its collector converts the raw metric
CSVs into one analysis parquet, while `fig1.py`, `fig2.py`, `fig3.py`, and
`stats_axes.py` generate the final and appendix artifacts under `figures/out/`.

The figure inputs are experiment artifacts and are not bundled with this source
tree. See [figures/README.md](figures/README.md) for the required inputs, complete
commands, output filenames, fixed analysis slices, and statistical protocol.

## Contextualization strategies

`--context_strategy` supports fixed strategies, prompt variations, stage-1 variations for two-stage strategies, random per-row variations, and per-row strategy pools:

```text
dataset_default
gold_hint(2)
self_feedback_concrete(1,2)
self_feedback_concrete(*,*)
gold_hint|self_feedback_concrete
```

Strategy selection and wildcard variation choices are deterministic for a fixed dataset row, seed, and strategy specification.

The strategy registry is in `contextualizer/contextualization_strategies.py`; prompt text is isolated in `contextualizer/contextualization_prompt_templates.py`.

## Project structure

```text
<repo-root>/
├── main.py                         # direct single-phase training
├── run_experiments.py              # CSV experiment orchestrator
├── distil_trainer.py               # self-distillation trainer
├── distil_config.py                # trainer configuration
├── model_registry.py               # model loading/runtime/baseline registry
├── dataset_adapters.py             # dataset training/eval/scoring abstraction
├── eval_lib.py                     # shared standalone evaluation functions
├── run_all_checkpoint_evals.py     # checkpoint sweep evaluation + curves
├── contextualizer/                 # context strategies, templates, runtime manager
├── metrics_and_plots/              # per-run analysis and cross-run metric extraction
├── figures/                        # paper figures, data collectors, and statistical tables
├── scripts/                        # lm-eval, baseline calibration, cleanup utilities
├── toy_model/                      # standalone contextual-SD mechanism model
├── data/                           # optional/local prepared dataset artifacts
├── data_utils/                     # dataset builders, verifiers, and preparation utilities
│   └── spatial_contradiction2/     # Spatial Contradiction construction + exact verifier
├── requirements-vlm.txt            # current environment
├── requirements.txt                # legacy Qwen2.5 environment
├── LICENSE                          # MIT license for original project contributions
├── THIRD_PARTY_NOTICES.md           # upstream provenance and license exceptions
├── LICENSES/                        # copies of applicable third-party licenses
└── docs/                           # detailed project documentation
```

Generated folders such as `outputs*`, `metrics_out`, `baseline_results`, `logs`, `wandb`, virtual environments, and caches are runtime artifacts rather than source modules.

## Further documentation

- [Installation and environments](docs/INSTALLATION.md)
- [Running training, evaluation, lm-eval, and metrics](docs/RUNNING.md)
- [Architecture and data flow](docs/ARCHITECTURE.md)
- [Extending the codebase](docs/EXTENDING.md)
- [Contextual self-distillation toy model](docs/TOY_MODEL.md)
- [Paper figures and statistical tables](figures/README.md)
- [Dataset provenance and licensing](data/README.md)
- [Third-party notices](THIRD_PARTY_NOTICES.md)

## Reproducibility notes

- Experiment roots contain `experiment_config.json`, which records the selected base model and training configuration.
- `run_experiments.py` uses model-aware progress identities so runs with the same experiment name but different base models are not treated as the same completed job.
- Dataset base accuracies and lm-eval baselines are model-specific and live in `model_registry.py`.
- `run_all_checkpoint_evals.py`, `scripts/summarize_lmeval.py`, and the metrics pipeline resolve baselines from the recorded model rather than assuming Qwen2.5.
- Baseline calibration is seed-replicated and records package/protocol provenance.

## License

Original code, documentation, and project-created data in this repository are
released under the [MIT License](LICENSE). Third-party and upstream-derived
material retains its original license or license status and is not relicensed by
the project MIT grant. In particular, ToolAlpaca and the DeepMind Mathematics
Dataset are Apache-2.0, the original SciKnowEval release is MIT, and the prepared
Chemistry L-3 training snapshot inherited from `idanshen/Self-Distillation` has
no explicit license for its modifications. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) before redistributing data or
upstream-derived code.

## Acknowledgements

This project is based on the [SDFT Self-Distillation implementation](https://github.com/idanshen/Self-Distillation) and its associated paper, [*Self-Distillation Enables Continual Learning*](https://arxiv.org/abs/2601.19897). It also builds on Hugging Face Transformers/TRL, vLLM, lm-evaluation-harness, and the original dataset/evaluation projects represented by the adapters in this repository.
