# Extending the codebase

This guide describes the preferred maintenance paths for adding datasets, models, contextualization strategies, and analysis outputs. The main design rule is to extend the existing registries/interfaces instead of adding one-off branches to generic code.

## Add a new dataset

Dataset-specific training, evaluation, and context semantics belong in `dataset_adapters.py`.

### 1. Prepare the dataset on disk

Choose stable train/eval directories under `data/`, for example:

```text
data/my_dataset/
├── train_data/
└── eval_data/
```

If the dataset needs more than one canonical training representation, expose them as named views rather than creating strategy-specific path logic elsewhere.

### 2. Implement a `DatasetAdapter`

At minimum implement:

```python
class MyDatasetAdapter(DatasetAdapter):
    name = "my_dataset"
    train_path = "data/my_dataset/train_data"
    eval_path = "data/my_dataset/eval_data"
    default_max_new_tokens = 1024
    default_max_model_len = None
    results_filename = "eval_my_dataset_results.json"
    responses_filename = "eval_my_dataset_responses.json"

    def format_train_example(self, example: dict) -> dict:
        ...

    def build_eval_output(self, eval_data, tokenizer) -> EvalOutput:
        ...

    def score_responses(self, responses, references) -> list[int]:
        ...

    def build_response_records(...):
        ...
```

The training formatter should return the fields required by the project training path, normally including `prompt`, `teacher_prompt`, and `target` for the dataset-default strategy.

### 3. Implement context-facing hooks when needed

Context strategies should not inspect dataset-specific schemas directly. Override adapter methods such as:

```text
get_prompt_messages()
get_question_text()
get_gold_answer_text()
get_gold_response_text()
get_reference_for_context()
get_source_context_text_for_context()
get_scoring_reference_for_context()
is_usable_context_response_example()
```

Only override what differs from the generic adapter behavior.

### 4. Register the dataset

Add a `DatasetSpec` entry:

```python
DATASET_SPECS["my_dataset"] = DatasetSpec(
    adapter=MyDatasetAdapter(),
    expert_accuracy=<reference accuracy or None>,
    floor_accuracy=<chance floor>,
)
```

Do **not** place base-model accuracies in `DatasetSpec`. Those values are model-specific and belong in each model's `dataset_base_accuracy` mapping in `model_registry.py`.

For closely related dataset variants, prefer subclassing an existing adapter when the prompt/scoring contract is identical. `SpatialStandard2Adapter`, for example, subclasses `SpatialContradiction2Adapter` and changes only the target/reference view and artifact names. This avoids duplicating evaluation and contextualization logic.

### 5. Add baseline measurements

For every base model that will be used in baseline-relative metrics, calibrate the untouched model and add:

```python
MODEL_REGISTRY["..."].dataset_base_accuracy["my_dataset"] = ...
```

Prefer the baseline calibration script rather than an ad-hoc evaluation command so the protocol and seed aggregation remain reproducible.

### 6. Test the adapter

Recommended checks:

- load train/eval splits;
- render several prompts;
- verify reference alignment;
- score known-correct and known-incorrect outputs;
- run a small standalone vLLM evaluation;
- test every context hook used by the intended strategies;
- verify response records contain enough information for rescoring/auditing;
- run a tiny train/eval smoke test before a full experiment.

## Add a new model

Model-specific loading/runtime behavior belongs in `model_registry.py`.

### 1. Determine the model family and loader requirements

Identify:

- HF repository id;
- Transformers model/processor classes;
- vLLM loader/tokenizer/config requirements;
- whether evaluation needs different kwargs for upstream vs Trainer checkpoints;
- chat-template requirements;
- whether chat templating should produce text or token ids;
- tensor-parallel head counts if known;
- parameter-name mapping required for HF -> vLLM weight synchronization.

### 2. Add or reuse a `ModelFamily`

Reuse `CAUSAL_LM`, `QWEN_MULTIMODAL`, or `MISTRAL_COMMON` when the loading contract matches. Add a new family only when the generic loader helpers cannot represent the architecture cleanly.

### 3. Add a `ModelSpec`

A typical entry looks like:

```python
"my-model": ModelSpec(
    key="my-model",
    hf_repo_id="org/model",
    family=ModelFamily.CAUSAL_LM,
    trust_remote_code=False,
    checkpoint_model_types=("my_model_type",),
    vllm_extra_kwargs={},
    vllm_eval_extra_kwargs={},
    chat_template_kwargs={},
    chat_template_mode=ChatTemplateMode.TEXT,
    num_attention_heads=None,
    num_key_value_heads=None,
    dataset_base_accuracy={},
    lm_eval_baselines_percent={},
)
```

Keep model-specific flags declarative in the registry. Do not add `if model == ...` branches in `main.py`, `eval_lib.py`, or metrics code when a `ModelSpec` field can express the requirement.

### 4. Support local Trainer checkpoints

Add the correct `checkpoint_model_types`. If Trainer checkpoints need a different vLLM serialization/backend from the upstream repository, use:

```text
checkpoint_vllm_extra_kwargs
checkpoint_vllm_eval_extra_kwargs
```

Test both the untouched upstream model and a saved Trainer checkpoint.

### 5. Check chat templating

If the model needs extra template flags, place them in `chat_template_kwargs`. If the tokenizer backend should not round-trip through rendered text, use `ChatTemplateMode.TOKEN_IDS` and verify that all evaluation/training prompt paths recover the original chat messages from the adapter.

### 6. Check dynamic weight synchronization

For colocated vLLM training, compare HF parameter names with vLLM parameter names. Encode systematic differences in `VllMWeightSyncSpec` rather than scattering string replacements through `DistilTrainer`.

### 7. Run staged validation

Recommended order:

1. registry resolution only;
2. tokenizer/chat-template smoke test;
3. untouched-model standalone project-dataset evaluation;
4. lm-eval dry run;
5. lm-eval real run;
6. short one-phase training;
7. checkpoint reload/evaluation;
8. dynamic-context strategy smoke test;
9. multi-process/TP test if required;
10. full experiment sweep.

### 8. Calibrate baselines

Run `scripts/calibrate_baselines.py` over multiple seeds, then populate both:

```text
dataset_base_accuracy
lm_eval_baselines_percent
```

Metrics should remain unavailable (`None`) for unmeasured model/dataset combinations rather than borrowing another model's baseline.

## Add a new contextualization strategy

Strategy semantics belong in `contextualizer/contextualization_strategies.py`; prompt wording belongs in `contextualizer/contextualization_prompt_templates.py`.

### 1. Decide whether the strategy is static or dynamic

Use `ContextStrategy` when the final teacher prompt can be built entirely from the dataset row/reference.

Use `TwoStageStrategy` when an auxiliary feedback/rationale/rewrite must be generated before constructing the final teacher prompt.

Set the relevant capability flags:

```text
requires_student_response
requires_feedback_generation
requires_sibling_selection
can_precompute
supported_datasets
train_view
supports_dataset_targets
```

### 2. Add prompt templates

For a simple template strategy, add a list to `contextualization_prompt_templates.py`:

```python
MY_STRATEGY_PROMPT_TEMPLATES = [
    """
$question

Useful context:
$full_reference

Answer the task.
""",
]
```

Available common variables include:

```text
$question
$compact_answer
$full_reference
$short_hint
$source_context
$student_response
$feedback_text
$sibling_text
```

For two-stage strategies, add both final and stage-1 template lists.

### 3. Implement the strategy class

Simple case:

```python
class MyStrategy(ContextStrategy):
    name = "my_strategy"
    can_precompute = True
    prompt_templates = MY_STRATEGY_PROMPT_TEMPLATES
```

Override `build_messages()` or `build_template_variables()` only when the generic template path is insufficient.

### 4. Register it

Add one entry to `STRATEGY_REGISTRY`:

```python
"my_strategy": MyStrategy(),
```

No additional parser branch is normally required: the loader and manager resolve strategy names through the registry.

### 5. Variations and pools

Template-list length determines the normal prompt-variation range. Two-stage strategies can also expose stage-1 variations. The loader already supports:

```text
my_strategy(2)
my_strategy(2,3)
my_strategy(*)
my_strategy(*,*)
my_strategy|another_strategy
```

If a variation is only valid for certain datasets, implement the existing validation hooks rather than checking dataset names in the loader.

### 6. Test strategy parity and runtime requirements

Test:

- fixed variation 1;
- every additional fixed variation;
- wildcard selection reproducibility;
- strategy-pool reproducibility;
- dataset restriction errors;
- static precomputation or dynamic metadata schema;
- preservation of system-message shells;
- stage-1 generation and final teacher prompt;
- sibling fallback behavior where applicable;
- a short trainer run.

## Add a new extracted metric

The included single-phase analysis layer lives under `metrics_and_plots/`.

General rules:

1. Prefer extending the row builders in `build_single_phase_metrics_raw.py` rather than reparsing experiment folders in a new script.
2. Keep metric formulas in shared helpers such as `metrics_utils.py` when they are reusable.
3. Keep experiment naming/alias rules in `experiment_registry.py`.
4. Preserve the extractor's single-phase-only contract unless sequential support is added and documented explicitly.
5. Resolve model-relative baselines through `model_registry.py` rather than embedding constants.
6. When a metric depends on the base model, use the run's recorded model identity and the CLI fallback consistently.
7. Add columns without silently changing the meaning of existing columns.
8. Document metric semantics in the relevant public guide and implementation docstrings when they affect interpretation of published tables.

## Maintenance checklist

Before merging an extension:

- run `python -m py_compile` on modified Python files;
- run focused unit/smoke tests for the new registry entry or adapter;
- validate CLI `--help` output;
- verify existing default-model commands still resolve to `DEFAULT_MODEL_KEY`;
- verify unknown/missing baselines remain missing instead of falling back to another model;
- verify experiment metadata records the selected model;
- verify generated result files remain compatible with the metrics pipeline;
- update README/docs only where public interfaces or supported capabilities changed.
