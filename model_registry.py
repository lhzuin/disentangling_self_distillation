"""Model registry for the contextualized self-distillation codebase.

The registry is the authoritative source for model-specific training,
evaluation, and baseline metadata. ``ModelSpec`` entries describe how a model
is loaded through Transformers, how vLLM should be configured for upstream and
Trainer-checkpoint evaluation, how chat templates are applied, how HF parameter
names map to vLLM during weight synchronization, and which project/lm-eval
baselines belong to that model.

``DEFAULT_MODEL_KEY`` is ``qwen2.5-7b``. Registered short keys, registered
Hugging Face repository ids, and recognized local Trainer checkpoints resolve
to the same model-family policy through ``resolve_model_spec``. Unknown
models/paths use the generic causal-LM fallback.

Current project validation status is documented in README.md. Qwen2.5 and the
BF16 Ministral path have been exercised in the current text-only workflows;
Qwen3.5 remains the primary registered model awaiting end-to-end validation.
Model-specific requirements and extension guidance are documented under
``docs/``.
"""


from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class ModelFamily(str, Enum):
    """How a model's weights/tokenizer must be loaded.

    CAUSAL_LM uses the standard AutoModelForCausalLM + AutoTokenizer path.
    It is also the fallback family for unregistered models and local checkpoints
    whose Hugging Face `model_type` is not explicitly registered.
    """

    CAUSAL_LM = "causal_lm"
    QWEN_MULTIMODAL = "qwen_multimodal"
    MISTRAL_COMMON = "mistral_common"


class ChatTemplateMode(str, Enum):
    """
    How conversational prompts should be prepared.

    TEXT:
        Render the chat template to text with ``tokenize=False`` before normal
        tokenizer processing.

    TOKEN_IDS:
        Apply the chat template directly with tokenize=True.
        Required by tokenizers such as MistralCommonBackend, where the
        text round-trip is explicitly discouraged.
    """

    TEXT = "text"
    TOKEN_IDS = "token_ids"


@dataclass(frozen=True)
class VllMWeightSyncSpec:
    """
    Declarative translation from Hugging Face parameter names to the names
    expected when dynamically reloading parameters into vLLM.

    Most models need no translation. Model-specific differences belong here,
    rather than as architecture checks inside DistilTrainer.
    """

    prefix_replacements: tuple[tuple[str, str], ...] = ()
    substring_replacements: tuple[tuple[str, str], ...] = ()
    skip_prefixes: tuple[str, ...] = ()

    def map_name(self, name: str) -> str | None:
        if any(name.startswith(prefix) for prefix in self.skip_prefixes):
            return None

        # Match vLLM WeightsMapper semantics: substring replacements first,
        # then prefix replacements.
        for old, new in self.substring_replacements:
            if old in name:
                name = name.replace(old, new, 1)

        for old, new in self.prefix_replacements:
            if name.startswith(old):
                name = new + name[len(old):]
                break

        return name


@dataclass(frozen=True)
class ModelSpec:
    """Metadata needed to load, serve, and score one base model."""

    key: str
    hf_repo_id: str
    family: ModelFamily = ModelFamily.CAUSAL_LM

    # Loading.
    trust_remote_code: bool = False
    tokenizer_padding_side_eval: str = "left"

    # Extra kwargs merged into `vllm.LLM(...)` (both standalone eval and the
    # DistilTrainer colocate engine). Empty for the default/causal_lm family.
    vllm_extra_kwargs: dict[str, Any] = field(default_factory=dict)

    # Hugging Face config `model_type` values that identify local Trainer
    # checkpoints belonging to this registered model family. Empty by default,
    # so unregistered/unrecognized local checkpoints use the generic fallback.
    checkpoint_model_types: tuple[str, ...] = ()

    # Optional vLLM kwargs used specifically when loading a local Hugging Face
    # Trainer checkpoint. None means reuse `vllm_extra_kwargs`.
    #
    # This lets an upstream repository and its Trainer checkpoints share the
    # same model/tokenizer family while using different serialization formats.
    checkpoint_vllm_extra_kwargs: Optional[dict[str, Any]] = None

    # Extra vLLM kwargs used only for standalone evaluation. Keep these
    # separate from training/runtime kwargs so evaluation-only optimizations
    # cannot silently change DistilTrainer behavior.
    vllm_eval_extra_kwargs: dict[str, Any] = field(default_factory=dict)

    # Additional vLLM kwargs used only when standalone evaluation loads a
    # recognized local HF Trainer checkpoint. These are layered on top of both
    # checkpoint loading kwargs and generic evaluation kwargs.
    checkpoint_vllm_eval_extra_kwargs: dict[str, Any] = field(
        default_factory=dict
    )

    # Extra kwargs merged into every `tokenizer.apply_chat_template(...)` /
    # `maybe_apply_chat_template(...)` call for this model (e.g. disabling
    # Qwen3.5's default "thinking" mode so scorers keep working).
    chat_template_kwargs: dict[str, Any] = field(default_factory=dict)

    # How conversational prompts should be prepared.
    # TEXT is the standard Qwen2.5/generic causal-LM prompt path.
    chat_template_mode: ChatTemplateMode = ChatTemplateMode.TEXT

    # HF -> vLLM parameter-name compatibility for dynamic weight sync.
    # Empty by default when no parameter-name translation is required.
    vllm_weight_sync: VllMWeightSyncSpec = field(
        default_factory=VllMWeightSyncSpec
    )

    # Architecture facts used only for an early, friendly tensor-parallel
    # sanity check. Left as None when not confirmed from the model's own
    # config.json; the TP check is skipped rather than guessing.
    num_attention_heads: Optional[int] = None
    num_key_value_heads: Optional[int] = None

    # Authoritative per-dataset base (pre-training) accuracies for this model.
    # Dataset adapters and analysis scripts resolve base-model baselines from this
    # mapping; they must not duplicate or silently substitute values from another
    # model.
    dataset_base_accuracy: dict[str, float] = field(default_factory=dict)

    # lm-eval-harness baselines (percent, 0-100) for this model, in the same
    # shape as summarize_lmeval.BASELINES_PERCENT.
    lm_eval_baselines_percent: dict[str, float] = field(default_factory=dict)

    # True once someone has actually run+verified the DistilTrainer training
    # loop (not just generation/eval) against this model.
    training_verified: bool = False
    notes: str = ""

    def requires_tp_group_size_check(self) -> bool:
        return self.num_attention_heads is not None


DEFAULT_MODEL_KEY = "qwen2.5-7b"

# Qwen2.5 lm-eval reference values used by model-aware summaries.
_QWEN25_7B_LM_EVAL_BASELINES_PERCENT: dict[str, float] = {
    "hellaswag": 80.47,
    "mmlu": 71.80,
    "truthfulqa_mc2": 64.80,
    "winogrande": 70.48,
    "ifeval": 57.12,
    "prior_task_avg": 68.93,
}

# Qwen2.5-7B-Instruct base-model accuracies. Model-specific dataset baselines
# are authoritative here and are consumed through dataset_adapters.py.
_QWEN25_7B_DATASET_BASE_ACCURACY: dict[str, float] = {
    "tooluse": 0.412,
    "science": 0.335,
    "math_contradiction": 0.1345,
    "spatial_contradiction2": 0.015,
    "spatial_standard2": 0.262,
}


_MINISTRAL3_3B_DATASET_BASE_ACCURACY: dict[str, float] = {
    "tooluse": 0.065292,
    "science": 0.363577,
    "math_contradiction": 0.124,
    "spatial_contradiction2": 0.008,
    "spatial_standard2": 0.196,
}


_MINISTRAL3_3B_LM_EVAL_BASELINES_PERCENT: dict[str, float] = {
    "hellaswag": 73.4515,
    "mmlu": 67.4263,
    "truthfulqa_mc2": 56.1476,
    "winogrande": 68.5083,
    "ifeval": 54.2206,
    "humaneval": 51.8293,
    "prior_task_avg": 63.9508,
}

MODEL_REGISTRY: dict[str, ModelSpec] = {
    "qwen2.5-7b": ModelSpec(
        key="qwen2.5-7b",
        hf_repo_id="Qwen/Qwen2.5-7B-Instruct",
        family=ModelFamily.CAUSAL_LM,
        trust_remote_code=False,
        num_attention_heads=28,
        num_key_value_heads=4,
        dataset_base_accuracy=dict(_QWEN25_7B_DATASET_BASE_ACCURACY),
        lm_eval_baselines_percent=dict(_QWEN25_7B_LM_EVAL_BASELINES_PERCENT),
        training_verified=True,
        notes="Default validated model for the project.",
    ),
    "qwen3.5-4b": ModelSpec(
        key="qwen3.5-4b",
        hf_repo_id="Qwen/Qwen3.5-4B",
        family=ModelFamily.QWEN_MULTIMODAL,
        trust_remote_code=True,
        vllm_extra_kwargs={
            # Skip vision-tower weights/profiling to save memory when only
            # text prompts are used, per the model's own vLLM recipe.
            "language_model_only": True,
        },
        checkpoint_model_types=("qwen3_5",),
        chat_template_kwargs={
            # Qwen3.5 thinks by default; the dataset adapters' scorers
            # (JSON/regex answer extraction) assume a direct response.
            "enable_thinking": False,
        },
        chat_template_mode=ChatTemplateMode.TEXT,
        vllm_weight_sync=VllMWeightSyncSpec(
            skip_prefixes=(
                "model.visual.",
                "visual.",
            ),
        ),
        num_attention_heads=16,
        num_key_value_heads=4,
        dataset_base_accuracy={},
        lm_eval_baselines_percent={},
        training_verified=False,
        notes=(
            "Native multimodal Qwen3.5 architecture. Supported by the pinned "
            "Transformers 5.5.x / vLLM 0.20.x stack. Training uses the full "
            "Transformers multimodal wrapper while colocated vLLM is configured "
            "with language_model_only=True for the current text-only experiments. "
            "Thinking is disabled through chat_template_kwargs. End-to-end project "
            "validation is still pending for this model."
        ),
    ),
    "ministral-3-3b": ModelSpec(
        key="ministral-3-3b",
        # BF16 checkpoint, used as the default/safe choice regardless of
        # which GPU generation this is run on. The sibling FP8 checkpoint
        # (registry key "ministral-3-3b-fp8", same underlying model) wants
        # an accelerated FP8 Triton kernel (w8a8_block_fp8_matmul_triton per
        # its model card) for matmuls; that requires Hopper/Ada tensor
        # cores. It does not run on Volta (V100). `transformers` alone falls
        # back to a BF16 dequant on load for the FP8 checkpoint, but vLLM's
        # own kernel dispatch is not guaranteed to. This BF16 variant avoids
        # that axis of risk entirely and matches how every other model in
        # this codebase is already trained (bf16 throughout) -- correct on
        # both V100 and H100, just without the FP8 checkpoint's claimed
        # (no-loss) memory/throughput advantage on Hopper/Ada. Use
        # "ministral-3-3b-fp8" explicitly to opt into that on suitable
        # hardware.
        hf_repo_id="mistralai/Ministral-3-3B-Instruct-2512-BF16",
        family=ModelFamily.MISTRAL_COMMON,
        trust_remote_code=False,
        vllm_extra_kwargs={
            "tokenizer_mode": "mistral",
            "config_format": "mistral",
            "load_format": "mistral",
        },
        # Hugging Face Trainer checkpoints keep the Mistral tokenizer family
        # but are stored as ordinary HF config/weight checkpoints rather than
        # Mistral-native consolidated checkpoints. vLLM's default config/weight
        # loaders are therefore correct; only the tokenizer mode must be forced.
        checkpoint_model_types=("mistral3",),
        checkpoint_vllm_extra_kwargs={
            "tokenizer_mode": "mistral",
        },
        # Standalone evaluations in this codebase are text-only. Avoid
        # initializing the unused Pixtral image processor / vision path when
        # evaluating either the upstream model or an HF Trainer checkpoint.
        vllm_eval_extra_kwargs={
            "language_model_only": True,
        },
        # vLLM 0.20's native Mistral3 loader does not reliably load the HF
        # weight layout written by Trainer checkpoints. For standalone
        # checkpoint evaluation only, use vLLM's Transformers modeling backend
        # so the checkpoint is interpreted by the same model implementation
        # that saved it. This evaluation override does not affect training or
        # upstream-model evaluation.
        checkpoint_vllm_eval_extra_kwargs={
            "model_impl": "transformers",
        },
        chat_template_kwargs={},
        chat_template_mode=ChatTemplateMode.TOKEN_IDS,
        vllm_weight_sync=VllMWeightSyncSpec(
            prefix_replacements=(
                ("model.language_model.", "language_model.model."),
                ("model.vision_tower.", "vision_encoder."),
                (
                    "model.multi_modal_projector.",
                    "vision_language_adapter.",
                ),
            ),
            substring_replacements=(
                (".linear_1.", ".w_in."),
                (".linear_2.", ".w_out."),
            ),
        ),
        num_attention_heads=None,  # not confirmed from config.json; skip TP head check
        num_key_value_heads=None,
        dataset_base_accuracy= dict(_MINISTRAL3_3B_DATASET_BASE_ACCURACY),
        lm_eval_baselines_percent=dict(_MINISTRAL3_3B_LM_EVAL_BASELINES_PERCENT),
        training_verified=True,
        notes=(
            "Native vision-language checkpoint with BF16 weights. Loads through "
            "Mistral3ForConditionalGeneration + MistralCommonBackend. "
            "MistralCommonBackend requires direct chat-template tokenization "
            "(tokenize=True), so DistilTrainer uses TOKEN_IDS chat-template mode. "
            "vLLM uses the registered Mistral tokenizer/config/load settings. "
            "The current text-only training and evaluation workflow has been validated."
        ),
    ),
    "ministral-3-3b-fp8": ModelSpec(
        key="ministral-3-3b-fp8",
        # The FP8-native release. Mistral's model card claims this is
        # "no-loss" and uses an accelerated FP8 Triton kernel
        # (w8a8_block_fp8_matmul_triton) for matmuls -- but that kernel
        # requires Hopper/Ada tensor cores (H100/L40/RTX 40-series) and does
        # NOT run on Volta (V100). Only use this key when running on
        # confirmed FP8-capable hardware; use "ministral-3-3b" (BF16)
        # otherwise, including on V100.
        hf_repo_id="mistralai/Ministral-3-3B-Instruct-2512",
        family=ModelFamily.MISTRAL_COMMON,
        trust_remote_code=False,
        vllm_extra_kwargs={
            "tokenizer_mode": "mistral",
            "config_format": "mistral",
            "load_format": "mistral",
        },
        chat_template_kwargs={},
        chat_template_mode=ChatTemplateMode.TOKEN_IDS,
        vllm_weight_sync=VllMWeightSyncSpec(
            prefix_replacements=(
                ("model.language_model.", "language_model.model."),
                ("model.vision_tower.", "vision_encoder."),
                (
                    "model.multi_modal_projector.",
                    "vision_language_adapter.",
                ),
            ),
            substring_replacements=(
                (".linear_1.", ".w_in."),
                (".linear_2.", ".w_out."),
            ),
        ),
        num_attention_heads=None,
        num_key_value_heads=None,
        dataset_base_accuracy={},
        lm_eval_baselines_percent={},
        training_verified=False,
        notes=(
            "Native vision-language FP8 checkpoint. Requires Hopper/Ada "
            "tensor cores (H100/L40/RTX 40-series) for its accelerated FP8 "
            "matmul kernel -- do not use on Volta (V100); use "
            "'ministral-3-3b' (BF16) there instead. Uses the same "
            "MistralCommonBackend TOKEN_IDS chat-template mode and HF-to-vLLM "
            "weight-name mapping as the BF16 sibling. Validate the FP8 runtime path "
            "independently on compatible hardware before using it for experiments."
        ),
    ),
}


def _resolve_local_checkpoint_spec(
    model_name_or_path: str,
) -> Optional[ModelSpec]:
    """
    Resolve a local Hugging Face Trainer checkpoint to a registered model
    family using its config.json `model_type`.

    Only explicitly registered `checkpoint_model_types` are recognized.
    Everything else uses ``resolve_model_spec``'s generic CAUSAL_LM fallback.
    """
    path = Path(model_name_or_path)

    if not path.is_dir():
        return None

    config_path = path / "config.json"
    if not config_path.is_file():
        return None

    try:
        with config_path.open("r", encoding="utf-8") as handle:
            config_data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None

    model_type = config_data.get("model_type")
    if not isinstance(model_type, str) or not model_type:
        return None

    matches = [
        spec
        for spec in MODEL_REGISTRY.values()
        if model_type in spec.checkpoint_model_types
    ]

    if not matches:
        return None

    if len(matches) > 1:
        raise ValueError(
            f"Local checkpoint {model_name_or_path!r} with model_type "
            f"{model_type!r} matches multiple registered model specs: "
            f"{[spec.key for spec in matches]}"
        )

    base_spec = matches[0]
    checkpoint_vllm_kwargs = (
        base_spec.vllm_extra_kwargs
        if base_spec.checkpoint_vllm_extra_kwargs is None
        else base_spec.checkpoint_vllm_extra_kwargs
    )

    checkpoint_eval_kwargs = dict(base_spec.vllm_eval_extra_kwargs)
    checkpoint_eval_kwargs.update(
        base_spec.checkpoint_vllm_eval_extra_kwargs
    )

    return replace(
        base_spec,
        hf_repo_id=str(path),
        vllm_extra_kwargs=dict(checkpoint_vllm_kwargs),
        vllm_eval_extra_kwargs=checkpoint_eval_kwargs,
    )


def resolve_model_spec(model_name_or_path: str) -> ModelSpec:
    """
    Return the ModelSpec for a CLI-facing model name/path.

    Resolution order:
      1. Registered short key.
      2. Registered Hugging Face repository id.
      3. Recognized local HF Trainer checkpoint, identified from
         config.json `model_type`.
      4. Generic CAUSAL_LM fallback.

    The final fallback provides the standard causal-LM loading policy for
    arbitrary unregistered models and unrecognized local checkpoints.
    """
    if model_name_or_path in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_name_or_path]

    for spec in MODEL_REGISTRY.values():
        if spec.hf_repo_id == model_name_or_path:
            return spec

    local_checkpoint_spec = _resolve_local_checkpoint_spec(model_name_or_path)
    if local_checkpoint_spec is not None:
        return local_checkpoint_spec

    return ModelSpec(
        key=model_name_or_path,
        hf_repo_id=model_name_or_path,
        family=ModelFamily.CAUSAL_LM,
    )


def default_model_spec() -> ModelSpec:
    return MODEL_REGISTRY[DEFAULT_MODEL_KEY]


def known_model_keys() -> tuple[str, ...]:
    return tuple(MODEL_REGISTRY.keys())


def _require_multimodal_qwen_classes():
    try:
        from transformers import AutoModelForMultimodalLM, AutoProcessor
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "Loading a qwen_multimodal model (e.g. Qwen/Qwen3.5-4B) requires "
            "a Transformers build that provides AutoModelForMultimodalLM. "
            "Install the pinned current environment from requirements-vlm.txt; "
            "see README.md 'Model support'."
        ) from exc
    return AutoModelForMultimodalLM, AutoProcessor


def _require_mistral_common_classes():
    try:
        from transformers import Mistral3ForConditionalGeneration, MistralCommonBackend
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "Loading a mistral_common model (e.g. Ministral-3-3B-Instruct) "
            "requires the Transformers/Mistral stack pinned in "
            "requirements-vlm.txt. See README.md 'Model support'."
        ) from exc
    return Mistral3ForConditionalGeneration, MistralCommonBackend


def load_model_and_tokenizer_hf(spec: ModelSpec, *, torch_dtype):
    """Load (model, tokenizer_or_processor) for training via `transformers`.

    For ``ModelFamily.CAUSAL_LM``, loading uses
    ``AutoModelForCausalLM.from_pretrained`` and ``AutoTokenizer.from_pretrained``.
    """

    if spec.family == ModelFamily.CAUSAL_LM:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model = AutoModelForCausalLM.from_pretrained(
            spec.hf_repo_id,
            torch_dtype=torch_dtype,
            trust_remote_code=spec.trust_remote_code,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            spec.hf_repo_id, trust_remote_code=spec.trust_remote_code
        )
        return model, tokenizer

    if spec.family == ModelFamily.QWEN_MULTIMODAL:
        logger.warning(
            "Loading %s via the experimental qwen_multimodal path. This has "
            "not been verified against DistilTrainer's training loop -- see "
            "model_registry.py module docstring.",
            spec.hf_repo_id,
        )
        auto_model_cls, auto_processor_cls = _require_multimodal_qwen_classes()
        model = auto_model_cls.from_pretrained(spec.hf_repo_id, torch_dtype=torch_dtype)
        processor = auto_processor_cls.from_pretrained(spec.hf_repo_id)
        return model, processor

    if spec.family == ModelFamily.MISTRAL_COMMON:
        model_cls, tokenizer_cls = _require_mistral_common_classes()
        model = model_cls.from_pretrained(spec.hf_repo_id, torch_dtype=torch_dtype)
        tokenizer = tokenizer_cls.from_pretrained(spec.hf_repo_id)
        return model, tokenizer

    raise ValueError(f"Unhandled model family: {spec.family}")  # pragma: no cover


def load_tokenizer_for_eval(spec: ModelSpec):
    """Load the tokenizer/processor used to build vLLM prompt strings.

    For the CAUSAL_LM family, evaluation uses ``AutoTokenizer`` with the
    registry-defined evaluation padding side. Other families use their registered
    processor/tokenizer backend.
    """

    if spec.family == ModelFamily.CAUSAL_LM:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(
            spec.hf_repo_id, padding_side=spec.tokenizer_padding_side_eval
        )

    if spec.family == ModelFamily.QWEN_MULTIMODAL:
        _, auto_processor_cls = _require_multimodal_qwen_classes()
        return auto_processor_cls.from_pretrained(spec.hf_repo_id)

    if spec.family == ModelFamily.MISTRAL_COMMON:
        _, tokenizer_cls = _require_mistral_common_classes()
        return tokenizer_cls.from_pretrained(spec.hf_repo_id)

    raise ValueError(f"Unhandled model family: {spec.family}")  # pragma: no cover


def vllm_engine_kwargs(spec: ModelSpec) -> dict[str, Any]:
    """Extra kwargs shared by generic/training vLLM engine construction."""

    return dict(spec.vllm_extra_kwargs)


def vllm_eval_engine_kwargs(spec: ModelSpec) -> dict[str, Any]:
    """Return vLLM kwargs for standalone evaluation.

    Evaluation-only overrides are layered on top of the model/checkpoint
    loading kwargs without changing DistilTrainer behavior.
    """
    kwargs = vllm_engine_kwargs(spec)
    kwargs.update(spec.vllm_eval_extra_kwargs)
    return kwargs


def check_tensor_parallel_compatibility(spec: ModelSpec, tensor_parallel_size: int) -> None:
    """Fail fast with a clear message if TP size can't evenly shard heads.

    Silently skipped when the registry doesn't have confirmed head counts
    for this model (e.g. Ministral, pending config.json confirmation) rather
    than guessing and potentially blocking a valid configuration.
    """

    if tensor_parallel_size <= 1 or not spec.requires_tp_group_size_check():
        return

    for label, heads in (
        ("num_attention_heads", spec.num_attention_heads),
        ("num_key_value_heads", spec.num_key_value_heads),
    ):
        if heads is not None and heads % tensor_parallel_size != 0:
            raise ValueError(
                f"vllm_tensor_parallel_size={tensor_parallel_size} does not evenly "
                f"divide {spec.key}'s {label} ({heads}). Choose a tensor-parallel "
                f"size that divides every attention head count."
            )
