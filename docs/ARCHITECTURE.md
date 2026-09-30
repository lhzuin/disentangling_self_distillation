# Architecture

The codebase is organized around a small number of explicit registries and shared execution paths. Model-specific and dataset-specific behavior should be added to those abstractions rather than scattered across trainer, evaluator, and plotting scripts.

## High-level data flow

```text
CSV sweep                                direct phase
run_experiments.py                       main.py
        |                                   |
        +--------- torchrun/main.py <-------+
                            |
             +--------------+--------------+
             |              |              |
      model_registry.py  dataset_adapters.py  contextualizer/
             |              |              |
             +--------------+--------------+
                            |
                    DistilTrainer
                            |
             checkpoints + logs + metadata
                            |
             +--------------+---------------+
             |                              |
run_all_checkpoint_evals.py          scripts/lmeval.sh
       -> eval_lib.py                -> run_lmeval.py
       -> adapters                   -> summarize_lmeval.py
             |                              |
             +---------------+--------------+
                             |
                   metrics_and_plots/
```

## 1. Model registry

`model_registry.py` is the authoritative location for model-family behavior.

A `ModelSpec` describes:

- canonical registry key and Hugging Face repository id;
- loader family (`CAUSAL_LM`, `QWEN_MULTIMODAL`, `MISTRAL_COMMON`);
- tokenizer/evaluation padding policy;
- vLLM engine kwargs;
- checkpoint-specific vLLM kwargs;
- standalone-evaluation overrides;
- chat-template kwargs and text-vs-token-id templating mode;
- HF-to-vLLM parameter-name mapping for dynamic weight synchronization;
- optional tensor-parallel head-count checks;
- model-specific project-dataset baseline accuracies;
- model-specific lm-eval baseline percentages.

`resolve_model_spec()` resolves, in order:

1. a registered short key;
2. a registered Hugging Face repository id;
3. a recognized local Trainer checkpoint, using `config.json`/`model_type`;
4. otherwise, a generic causal-LM specification.

This keeps architecture-specific branches out of `main.py`, `eval_lib.py`, and metrics code.

### Chat-template policy

`ChatTemplateMode.TEXT` renders conversational messages to text before tokenization. `ChatTemplateMode.TOKEN_IDS` applies the chat template directly with `tokenize=True` and passes token ids to vLLM; this is used for tokenizer backends where a text round-trip is not appropriate.

Model-specific template options such as Qwen3.5's `enable_thinking=False` also live in the registry.

## 2. Dataset adapters

`dataset_adapters.py` owns dataset-specific semantics.

Each `DatasetAdapter` defines the relevant pieces of:

- canonical train/eval paths;
- training row formatting;
- optional named training views;
- evaluation prompt construction;
- response scoring;
- response-record serialization;
- default `max_new_tokens` / `max_model_len`;
- context-facing question/reference/source extraction;
- structural validation for generated context examples.

`DatasetSpec` stores dataset-level analysis metadata such as expert/reference accuracy, chance floor, and whether the adapter is training-only.

**Base-model accuracy is not duplicated here.** `get_dataset_metric_defaults("baseline", model_name_or_path=...)` resolves it from the selected `ModelSpec`. Omitting the model uses `DEFAULT_MODEL_KEY` (`qwen2.5-7b`).

The Spatial Contradiction adapter illustrates the separation between dataset semantics and model metadata. `SpatialContradiction2Adapter` owns prompt/reference handling and delegates exact `\boxed{(x, y)}` verification to `data_utils/spatial_contradiction2/spatial_contradiction_core.py`. Its Qwen2.5 base accuracy lives in `model_registry.py`, not in the adapter.

This separation is important:

```text
model_registry.py   -> properties of a model
  base accuracy
  lm-eval baseline
  loading/runtime policy

dataset_adapters.py -> properties of a task
  prompt/schema/scorer
  generation limits
  expert/reference metric
  chance floor
```

## 3. Contextualization layer

The `contextualizer/` package separates strategy semantics from prompt text and runtime generation.

### `contextualization_strategies_base.py`

Defines the strategy interface and common facilities:

- `ContextStrategy` for one-stage/static strategies;
- `TwoStageStrategy` for strategies that generate auxiliary feedback/rationale before constructing the final teacher prompt;
- dataset restrictions;
- canonical train-view selection;
- prompt/stage-1 variation validation;
- sibling-response requirements;
- stage-1 generation length policy.

### `contextualization_strategies.py`

Implements concrete strategies and registers them in `STRATEGY_REGISTRY`.

Strategies return chat messages, not tokenizer-specific strings. Dataset adapters then merge the contextualized user message with the task's original prompt shell where needed.

Static strategies can be computed in `datasets.Dataset.map`. Dynamic strategies declare what runtime information they require (student response, feedback generation, sibling selection) so `DistilTrainer` can construct the teacher prompt after generation.

### `contextualization_prompt_templates.py`

Contains prompt text only. Template variables use `string.Template` syntax such as `$question`, `$full_reference`, `$compact_answer`, `$feedback_text`, and `$sibling_text`.

### `contextualized_train_loader.py`

Resolves strategy expressions and canonical training views, applies deterministic per-row strategy/variation selection, and builds the dataset schema consumed by the trainer.

Supported selection syntax includes fixed variations, wildcard variations, and `|`-separated strategy pools. Random choices are deterministic for fixed seed/dataset/row/spec inputs.

### `contextualization_manager.py`

Provides the bridge between dataset mapping and runtime prompt construction. It serializes raw examples/references for dynamic strategies and exposes methods used by the trainer to build stage-1 and final teacher prompts.

## 4. Training flow

`main.py` is a thin phase-level launcher:

1. parse the model/dataset/training arguments;
2. resolve student, reference, and rollout model specifications;
3. validate world-size/tensor-parallel compatibility;
4. load model/tokenizer or processor through the registry;
5. build the contextualized training dataset;
6. construct `DistilConfig`;
7. run `DistilTrainer.train()`.

`DistilTrainer` is derived from TRL training infrastructure but implements the project's self-distillation path. It can generate rollouts with the student or teacher via colocated vLLM, construct dynamic contextualized teacher prompts, compute teacher/student token distributions, and optimize the selected distillation or dataset-target objective.

For `optim_loss="jsd"`:

- `alpha == 0` selects forward KL;
- `alpha == 1` selects reverse KL;
- intermediate alpha values use the implemented generalized Jensen-Shannon mixture objective.

The reference model can be fixed or synchronized with the student. Dynamic model-to-vLLM weight synchronization uses the mapping declared in the resolved `ModelSpec`.

## 5. Experiment orchestration

`run_experiments.py` reads enabled CSV rows and turns each row into one or more phase-level `main.py` launches.

It also manages:

- phase dependencies and latest-checkpoint handoff;
- resume and partial-phase resume;
- checkpoint evaluation;
- response-length/training-log analysis;
- interim and final cleanup;
- final lm-eval;
- atomic progress/history/failure files;
- optional concurrent workers;
- model-aware progress identity and experiment-root validation.

Every experiment root includes `experiment_config.json`. This is the authoritative downstream record of the initial base model and high-level run configuration.

## 6. Project-dataset evaluation

`eval_lib.py` is the shared implementation for standalone generation and scoring. It resolves the model specification, loads the appropriate tokenizer/processor and vLLM engine, prepares prompts according to `ChatTemplateMode`, generates outputs, delegates scoring to the dataset adapter, and writes result/response JSONs.

`run_all_checkpoint_evals.py` applies this path to every checkpoint in an experiment. It reads `experiment_config.json` to select the correct base-model baseline and adds a step-zero reference point only when that model has a registered measurement for the dataset.

## 7. lm-eval and baseline calibration

`scripts/run_lmeval.py` runs lm-evaluation-harness using the model registry to build vLLM model arguments. `scripts/summarize_lmeval.py` extracts the selected benchmark metrics and compares them with the model's registered lm-eval baselines.

`scripts/calibrate_baselines.py` evaluates an untouched registered model over multiple seeds and produces registry-ready project-dataset and lm-eval baseline dictionaries. Dataset and lm-eval workers run in isolated processes to avoid cross-run CUDA/vLLM state.

## 8. Metrics extraction

The analysis utilities live together under `metrics_and_plots/`.
`analyze_response_lengths.py` and `analyze_training_log_stats.py` produce
per-run diagnostics used by the experiment orchestrator.
`build_single_phase_metrics_raw.py` discovers single-phase experiment outputs
and writes three minimally processed analysis tables: per-seed final metrics,
per-step training dynamics, and per-checkpoint evaluation dynamics. It reuses
dependency-light parsing helpers from `metrics_utils.py` and experiment naming
metadata from `experiment_registry.py`.

Each run's recorded `experiment_config.initial_model` is authoritative. The `--model`/`--model-name-or-path` option provides a fallback and consistency check, allowing accuracy deltas and lm-eval deltas to use the corresponding `ModelSpec` baselines.

The public release intentionally leaves paper-specific plotting and sequential aggregation outside this extractor. Keep one base model per outputs tree when possible and build downstream statistical summaries from the emitted raw CSVs.

## 9. Paper-figure pipeline

The paper-specific layer lives under `figures/`. It is downstream of
`metrics_and_plots/`: `collect_llm_experiments.py` converts per-seed metric CSVs
and optional dynamics tables into `figures/data/runs.parquet`, and `load_runs.py`
is the shared reader used by every figure and statistical script. `fig1.py`,
`fig2.py`, and `fig3.py` generate the main and appendix PDFs; `stats_axes.py`
implements the matched-pair Wilcoxon analyses. Controlled toy-model panels read
the V14 aggregate CSVs directly rather than passing through the run parquet.

See [`figures/README.md`](../figures/README.md) for the frozen slices, required
external artifacts, output contract, and exact reproduction commands.

## 10. Mechanism-isolation toy model

`toy_model/contextual_sd_toy_v14.py` is a standalone research model rather than part of the language-model training runtime above. It embeds its complete task, rollout, optimization, teacher-update, and metric backend in one file and consumes declarative JSON experiment cells. This separation keeps controlled finite-tree mechanism experiments independent of model loading, datasets, vLLM, and the main trainer.

See [TOY_MODEL.md](TOY_MODEL.md) for the mathematical construction, configuration schema, output contract, and reproducibility guidance.

## 11. Extension principle

A new capability should normally enter the system through exactly one of these abstractions:

- new base model -> `model_registry.py`;
- new dataset -> `dataset_adapters.py` plus data preparation;
- new contextualization policy -> `contextualizer/contextualization_strategies.py` and optionally `contextualization_prompt_templates.py`;
- new extracted metric -> `metrics_and_plots/build_single_phase_metrics_raw.py` or a reusable helper in `metrics_utils.py`.

Avoid adding architecture checks, dataset-name conditionals, or strategy-specific prompt text directly inside generic trainer/evaluator/orchestration paths unless the abstraction itself cannot express the requirement.
