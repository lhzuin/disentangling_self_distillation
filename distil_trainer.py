# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# distill_trainer_context.py
"""Self-distillation trainer built on Hugging Face/TRL training infrastructure.

The trainer supports student- or teacher-generated rollouts through vLLM,
contextualized teacher prompts, fixed or synchronized reference models,
model-registry-aware chat templating and weight synchronization, teacher-distribution
distillation, and dataset-target cross-entropy. ``alpha`` selects forward KL
(``0``), reverse KL (``1``), or the implemented intermediate generalized
Jensen-Shannon objective.
"""

import inspect
import math
import os
from collections import defaultdict, deque
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from typing import Any, Callable, Optional, Union

import datasets
import torch
import torch.utils.data
import transformers
from accelerate import logging
from accelerate.utils import broadcast_object_list, gather, gather_object, is_peft_model, set_seed
from datasets import Dataset, IterableDataset
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import DataLoader, Sampler
from transformers import (
    AutoConfig,
    AutoProcessor,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    TrainerCallback,
    is_wandb_available,
)
from transformers.trainer_utils import seed_worker
from transformers.utils import is_datasets_available, is_flash_attn_2_available, is_peft_available, is_rich_available

from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template, prepare_multimodal_messages
from trl.extras.profiling import profiling_context, profiling_decorator
from trl.import_utils import is_liger_kernel_available, is_vllm_available
from trl.models import prepare_deepspeed, prepare_fsdp, unwrap_model_for_generation #prepare_peft_model, unwrap_model_for_generation
from trl.models.utils import _ForwardRedirection
from trl.trainer.base_trainer import BaseTrainer
from distil_config import DistilConfig
from accelerate.state import AcceleratorState
from trl.trainer.utils import (
    RepeatSampler,
    disable_dropout_in_model,
    ensure_master_addr_port,
    entropy_from_logits,
    identity,
    nanmax,
    nanmin,
    nanstd,
    pad,
    print_prompt_completions_sample,
    selective_log_softmax,
    shuffle_sequence_dict,
    split_pixel_values_by_grid,
    split_tensor_dict,
    unsplit_pixel_values_by_grid,
)
from torch.nn.functional import log_softmax, kl_div

import json
from contextualizer.contextualization_manager import ContextualizationManager, json_loads_safe
from dataset_adapters import get_dataset_adapter
from model_registry import (
    ChatTemplateMode,
    ModelSpec,
    resolve_model_spec,
)


if is_peft_available():
    from peft import PeftConfig, PeftModel

if is_vllm_available():
    from vllm import LLM, SamplingParams

if is_wandb_available():
    import wandb


logger = logging.get_logger(__name__)


class MemoryEfficientSyncRefModelCallback(TrainerCallback):
    """
    Memory-efficient callback to synchronize the model with a reference model.
    
    Unlike the default SyncRefModelCallback, this version iterates through parameters
    one at a time instead of gathering all parameters at once. This reduces peak memory
    usage from O(full_model_size) to O(single_param_size), making it feasible to sync
    large models with DeepSpeed ZeRO-3.
    """

    def __init__(
        self,
        ref_model: Union[PreTrainedModel, nn.Module],
        accelerator: Optional[Any],
    ):
        self.accelerator = accelerator
        self.ref_model = ref_model

    @staticmethod
    def _sync_param(model_param, ref_param, alpha):
        """Sync a single parameter: ref = alpha * model + (1 - alpha) * ref"""
        ref_param.data.mul_(1.0 - alpha).add_(model_param.data, alpha=alpha)

    @staticmethod
    def sync_target_model_memory_efficient(model, target_model, alpha):
        """
        Sync target_model to track model, gathering one parameter at a time.
        
        This is O(1) in memory overhead instead of O(N) where N is model size.
        """
        deepspeed_plugin = AcceleratorState().deepspeed_plugin
        is_zero3 = deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3
        
        if is_zero3:
            import deepspeed
            
            # Iterate through parameters one at a time
            for (name, model_param), (_, ref_param) in zip(
                model.named_parameters(), target_model.named_parameters()
            ):
                # Gather only this pair of parameters
                with deepspeed.zero.GatheredParameters(
                    [model_param, ref_param], modifier_rank=0
                ):
                    if deepspeed.comm.get_rank() == 0:
                        MemoryEfficientSyncRefModelCallback._sync_param(
                            model_param, ref_param, alpha
                        )
        else:
            # Non-ZeRO-3: just iterate normally
            for model_param, ref_param in zip(model.parameters(), target_model.parameters()):
                MemoryEfficientSyncRefModelCallback._sync_param(model_param, ref_param, alpha)

    def on_step_end(self, args, state, control, **kwargs):
        model: PreTrainedModel = kwargs["model"]

        if self.ref_model is not None and state.global_step % args.ref_model_sync_steps == 0:
            if self.accelerator:
                model = self.accelerator.unwrap_model(model)
            self.sync_target_model_memory_efficient(model, self.ref_model, args.ref_model_mixup_alpha)

# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


class DistilTrainer(BaseTrainer):
    """Train a student against a contextualized reference distribution.

    The trainer receives already-instantiated student/reference models together
    with ``DistilConfig`` and a dataset produced by the contextualized training
    loader. Rollouts can come from the student or reference model. The reference
    prompt may be the dataset-defined teacher prompt or a static/dynamic context
    strategy constructed through ``ContextualizationManager``.

    For ``optim_loss="jsd"``, token-level teacher and student distributions are
    compared with forward KL (``alpha=0``), reverse KL (``alpha=1``), or the
    intermediate generalized Jensen-Shannon objective implemented by the trainer.
    ``optim_loss="cross_entropy"`` with ``optimal_policy_source="dataset"``
    provides the dataset-target SFT path.

    ``model_spec`` and ``ref_model_spec`` carry architecture-specific chat-template,
    vLLM, and HF-to-vLLM weight-synchronization policy from ``model_registry.py``.
    The normal project entry point is ``main.py``, which resolves these specs and
    constructs the configuration before instantiating the trainer.
    """

    _tag_names = ["trl", "distil"]
    _name = "Distil"

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        ref_model: Union[str, PreTrainedModel],
        args: Optional[DistilConfig] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[Union[PreTrainedTokenizerBase, ProcessorMixin]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        model_spec: Optional[ModelSpec] = None,
        ref_model_spec: Optional[ModelSpec] = None,
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = DistilConfig(f"{model_name}-Distil")

        # Models
        # Trained model
        model_init_kwargs = args.model_init_kwargs or {}
        if isinstance(model, str):
            model_id = model
            dtype = model_init_kwargs.get("dtype")
            if isinstance(dtype, torch.dtype) or dtype == "auto" or dtype is None:
                pass  # dtype is already a torch.dtype or "auto" or None
            elif isinstance(dtype, str):  # it's a str, but not "auto"
                dtype = getattr(torch, dtype)
                model_init_kwargs["dtype"] = dtype
            else:
                raise ValueError(
                    "Invalid `dtype` passed to `DistilConfig`. Expected either 'auto' or a string representing "
                    f"a `torch.dtype` (e.g., 'float32'), but got {dtype}."
                )
            # Disable caching if gradient checkpointing is enabled (not supported)
            config = AutoConfig.from_pretrained(model_id)
            architecture = getattr(transformers, config.architectures[0])
            model = architecture.from_pretrained(model_id, **model_init_kwargs)
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                logger.warning(
                    "You passed `model_init_kwargs` to the `DistilConfig`, but your model is already instantiated. "
                    "The `model_init_kwargs` will be ignored."
                )

        self.model_spec = model_spec or resolve_model_spec(model_id)
        self.chat_template_kwargs = dict(
            self.model_spec.chat_template_kwargs
        )

        # Some models (SmolVLM/Idefics3) don't support `logits_to_keep` argument and error out if we pass it
        # Inspect the forward method before we wrap the model with PEFT
        self.model_kwarg_keys = (
            inspect.signature(model.forward).parameters.keys()
            if not hasattr(model, "get_base_model")
            else inspect.signature(model.get_base_model().forward).parameters.keys()
        )

        if peft_config is not None or (is_peft_available() and isinstance(model, PeftModel)):
            from trl.models import prepare_peft_model # TODO: currently bugged - not supported for TRL 0.27.0
            model = prepare_peft_model(model, peft_config, args)

        # Processing class
        if processing_class is None:
            processing_class = AutoProcessor.from_pretrained(model.config._name_or_path, truncation_side="left")

        # Handle pad token for processors or tokenizers
        if isinstance(processing_class, ProcessorMixin):
            tokenizer = processing_class.tokenizer
        elif isinstance(processing_class, PreTrainedTokenizerBase):
            tokenizer = processing_class
        else:
            raise TypeError("The `processing_class` must be either a `PreTrainedTokenizerBase` or a `ProcessorMixin`")

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        self.pad_token = tokenizer.pad_token
        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length
        self.num_generations = args.num_generations
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.top_k = args.top_k
        self.min_p = args.min_p
        self.repetition_penalty = args.repetition_penalty
        self.use_transformers_paged = args.use_transformers_paged
        self.use_vllm = args.use_vllm
        self.vllm_mode = args.vllm_mode
        self.vllm_gpu_memory_utilization = args.vllm_gpu_memory_utilization  # only applies to colocation mode
        self.vllm_tensor_parallel_size = args.vllm_tensor_parallel_size  # only applies to colocation mode
        self.vllm_importance_sampling_correction = args.vllm_importance_sampling_correction
        self.vllm_importance_sampling_cap = args.vllm_importance_sampling_cap
        self.loss_type = args.loss_type
        self.scale_rewards = args.scale_rewards
        self.importance_sampling_level = args.importance_sampling_level
        self.mask_truncated_completions = args.mask_truncated_completions
        self.top_entropy_quantile = args.top_entropy_quantile
        self.num_loss_tokens_to_skip = args.num_loss_tokens_to_skip

        # Modifications to allow SFT
        self.optimal_policy_source = args.optimal_policy_source
        self.optim_loss = args.optim_loss

        self.use_dataset_targets = self.optimal_policy_source == "dataset"
        self.use_cross_entropy = self.optim_loss == "cross_entropy"

        # Datasets
        self.shuffle_dataset = args.shuffle_dataset

        if (
            isinstance(train_dataset, IterableDataset)
            or isinstance(eval_dataset, IterableDataset)
            or (
                isinstance(eval_dataset, dict) and any(isinstance(ds, IterableDataset) for ds in eval_dataset.values())
            )
        ):
            # See https://github.com/huggingface/trl/issues/3213
            raise NotImplementedError(
                "Iterable datasets are not yet supported in DistilTrainer. Please use a standard dataset instead."
            )

        # Multi-step
        self.num_iterations = args.num_iterations
        self.epsilon_low = args.epsilon
        self.epsilon_high = args.epsilon_high if args.epsilon_high is not None else args.epsilon
        # Tracks the number of iterations (forward + backward passes), including those within a grad accum cycle
        self._step = 0
        # Buffer the batch to reuse generated outputs across multiple updates. For more details, see
        # `_get_train_sampler` and `_prepare_inputs`.
        self._buffered_inputs = None

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in GRPO-like algorithms, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        warnings_issued = getattr(model, "warnings_issued", None)
        if isinstance(warnings_issued, dict):
            warnings_issued["estimate_tokens"] = True

        super().__init__(
            model=model,
            args=args,
            data_collator=identity,  # No data collation is needed in Distil
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
            # In Trainer, `training_step` scales the loss by `gradient_accumulation_steps` only if `compute_loss_func`
            # is None. For DAPO, loss scaling instead depends on the total number of completions tokens across the
            # global accumulated batch. To control scaling ourselves, we must disable Trainer’s built-in scaling. The
            # simplest (though a bit hacky) way is to set `compute_loss_func` to any non-None value, which bypasses
            # that behavior without rewriting `training_step`.
            compute_loss_func="non-None value to disable scaling",
        )

        # Reference model
        self.beta = args.beta
        self.alpha = args.alpha
        self.generate_from_teacher = args.generate_from_teacher
        if ref_model is not None:
            # If a reference model is provided, use it
            self.ref_model = ref_model
        elif self.beta == 0.0:
            # If beta is 0.0, the reference model is not needed
            self.ref_model = None
        elif is_peft_model(model):
            # If PEFT is used, the reference model is not needed since the adapter can be disabled
            # to revert to the initial model.
            self.ref_model = None
        else:
            # For deepspeed, fsdp or non-distributed models, create a reference model from scratch
            config = AutoConfig.from_pretrained(model_id)
            architecture = getattr(transformers, config.architectures[0])
            self.ref_model = architecture.from_pretrained(model_id, **model_init_kwargs)

        if ref_model_spec is not None:
            self.ref_model_spec = ref_model_spec
        elif self.ref_model is not None:
            ref_model_id = getattr(
                self.ref_model,
                "name_or_path",
                getattr(self.ref_model.config, "_name_or_path", model_id),
            )
            self.ref_model_spec = resolve_model_spec(ref_model_id)
        else:
            self.ref_model_spec = self.model_spec

        self.vllm_model_spec = (
            self.ref_model_spec
            if self.generate_from_teacher
            else self.model_spec
        )

        # Disable dropout in the models
        if args.disable_dropout:
            disable_dropout_in_model(model)
            if self.ref_model is not None:
                disable_dropout_in_model(self.ref_model)

        # Initialize the metrics
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self._total_train_tokens = 0
        self.log_completions = args.log_completions
        self.wandb_log_unique_prompts = args.wandb_log_unique_prompts
        self.num_completions_to_print = args.num_completions_to_print
        # Keep logs sized to the generation batch to record only outputs from the latest model update.
        self._logs = {
            "images": deque(maxlen=args.generation_batch_size),
            "prompt": deque(maxlen=args.generation_batch_size),
            "completion": deque(maxlen=args.generation_batch_size),
            "rewards": defaultdict(lambda: deque(maxlen=args.generation_batch_size)),
            "advantages": deque(maxlen=args.generation_batch_size),
        }

        # Ensure each process receives a unique seed to prevent duplicate completions when generating with
        # transformers if num_generations exceeds per_device_train_batch_size. We could skip it if we use vLLM, but
        # it's safer to set it in all cases.
        set_seed(args.seed, device_specific=True)

        if self.use_vllm:
            if not is_vllm_available():
                raise ImportError(
                    "vLLM is not available and `use_vllm` is set to True. Please install vLLM with "
                    "`pip install trl[vllm]` to use it."
                )

            if self.vllm_mode == "server":
                from trl.extras.vllm_client import VLLMClient #TODO: this import is currently bugged (vLLM 0.20.0 + torch 2.11.0 + TRL 0.27.0 versioning issues)
                if self.accelerator.is_main_process:
                    if args.vllm_server_base_url is not None:
                        base_url = args.vllm_server_base_url
                    else:
                        base_url = f"http://{args.vllm_server_host}:{args.vllm_server_port}"
                    self.vllm_client = VLLMClient(base_url=base_url, connection_timeout=args.vllm_server_timeout)
                    self.vllm_client.init_communicator(device=torch.cuda.current_device())

            elif self.vllm_mode == "colocate":
                # Make sure vllm_tensor_parallel_size group size evenly divides the world size - each group should have
                # the same number of ranks
                if not self.accelerator.num_processes % self.vllm_tensor_parallel_size == 0:
                    raise ValueError(
                        f"vllm_tensor_parallel_size ({self.vllm_tensor_parallel_size}) must divide world size "
                        f"({self.accelerator.num_processes}) evenly."
                    )

                if self.vllm_tensor_parallel_size > 1:
                    # Create subgroups of ranks for TP, each group with `vllm_tensor_parallel_size` ranks.
                    # For example, if world_size=8 and vllm_tensor_parallel_size=2 → groups: [0,1], [2,3], [4,5], [6,7]
                    self.tp_group, _ = torch.distributed.new_subgroups_by_enumeration(
                        [
                            list(range(i * self.vllm_tensor_parallel_size, (i + 1) * self.vllm_tensor_parallel_size))
                            for i in range(self.accelerator.num_processes // self.vllm_tensor_parallel_size)
                        ]
                    )

                # vLLM requires the environment variables to be set for distributed training.
                os.environ["RANK"] = str(self.accelerator.process_index)
                os.environ["LOCAL_RANK"] = str(self.accelerator.local_process_index)
                os.environ["WORLD_SIZE"] = str(self.accelerator.num_processes)
                # Ensure distributed rendezvous variables are set without colliding across concurrent runs
                ensure_master_addr_port()

                if self.max_prompt_length is not None and self.max_completion_length is not None:
                    max_model_len = self.max_prompt_length + self.max_completion_length
                else:
                    max_model_len = None
                # Use teacher model for vLLM when generate_from_teacher=True
                vllm_model_path = ref_model.name_or_path if self.generate_from_teacher and ref_model is not None else model.name_or_path
                logger.info(f"[DEBUG] Initializing vLLM with model: {vllm_model_path}, generate_from_teacher={self.generate_from_teacher}")
                # Model-family-specific engine kwargs (e.g. Mistral's own
                # tokenizer/checkpoint format). Empty by default, which keeps
                # this call identical to the historical, unparameterized one.
                vllm_extra_kwargs = dict(self.vllm_model_spec.vllm_extra_kwargs)
                vllm_extra_kwargs.update(
                    dict(getattr(self.args, "vllm_extra_kwargs", None) or {})
                )
                self.llm = LLM(
                    model=vllm_model_path,
                    tensor_parallel_size=args.vllm_tensor_parallel_size,
                    gpu_memory_utilization=self.vllm_gpu_memory_utilization,
                    max_num_seqs=self.args.per_device_train_batch_size
                    * self.vllm_tensor_parallel_size
                    * self.args.steps_per_generation,
                    max_model_len=max_model_len,
                    distributed_executor_backend="external_launcher",
                    # Feed identical seed for tp groups to ensure sampling results are the same across workers
                    seed=self.accelerator.process_index // self.vllm_tensor_parallel_size,
                    # Latest vLLM v1 memory profiler is misled by the high default value (i.e., 32768) - thinking there's not enough memory
                    max_num_batched_tokens=4096,
                    model_impl=self.args.vllm_model_impl,
                    enable_sleep_mode=self.args.vllm_enable_sleep_mode,
                    # Important so temperature scaling/logit tweaking affects the TIS log probs
                    logprobs_mode="processed_logprobs",
                    **vllm_extra_kwargs,
                )
                if self.args.vllm_enable_sleep_mode:
                    self.llm.sleep(level=1)
            else:
                raise ValueError(f"vllm_mode must be either 'server' or 'colocate', got '{self.vllm_mode}'.")

            self._last_loaded_step = -1  # tag to avoid useless loading during grad accumulation

            # When using vLLM, the main process is responsible for loading the model weights. This can cause process
            # desynchronization and seems to lead to DeepSpeed hanging during initialization. To prevent this, we
            # synchronize all processes after vLLM has been fully initialized.
            self.accelerator.wait_for_everyone()
        else:
            generation_kwargs = {
                "max_new_tokens": self.max_completion_length,
                "do_sample": True,
                "pad_token_id": tokenizer.pad_token_id,
                "bos_token_id": tokenizer.bos_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "min_p": self.min_p,
                "repetition_penalty": self.repetition_penalty,
                "cache_implementation": args.cache_implementation,
            }
            if args.generation_kwargs is not None:
                generation_kwargs.update(args.generation_kwargs)
            self.generation_config = GenerationConfig(**generation_kwargs)

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        # Add tags to the model
        self.model.add_model_tags(self._tag_names)

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            elif self.is_fsdp_enabled:
                self.ref_model = prepare_fsdp(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        if args.sync_ref_model:
            self.add_callback(MemoryEfficientSyncRefModelCallback(ref_model=self.ref_model, accelerator=self.accelerator))

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In DistilTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = [
                "prompt",
                "teacher_prompt",
                "target",
                "image",
                "images",
                "source_dataset",
                "context_strategy",
                "raw_example_json",
                "reference_json",
                "context_strategy_spec",
                "context_strategy_selection_spec",
                "context_prompt_variation",
                "context_base_prompt_variation",
                "context_stage1_variation",
                "context_row_index",
                "context_selection_seed",
                "context_prompt_variation_random",
                "context_stage1_variation_random",
            ]

    # This method overrides `Trainer.get_train_dataloader` to support our custom batching strategy.
    # Instead of returning a standard per-step batch (i.e., `per_device_batch_size), our dataloader loads an
    # *generation* batch (i.e., `per_device_batch_size × steps_per_generation`). This allows us to generate completions
    # once every steps_per_generation step—rather than once per accumulation step—which is significantly more
    # efficient. The only change from the original implementation is multiplying the batch size by
    # `steps_per_generation`. Thus, `_prepare_inputs` is called with this *generation* batch, and it handles the
    # splitting internally.
    # Maintenance note: This method is a copy-paste of the original `Trainer.get_train_dataloader` with only one line
    # modification. As a result, some parts of the method aren't relevant to Distil, but we keep them to stay one line
    # apart from the super method, ensuring easier maintenance in the future.
    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "batch_size": self._train_batch_size * self.args.steps_per_generation,  # < this is the change
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = partial(
                seed_worker, num_workers=self.args.dataloader_num_workers, rank=self.args.process_index
            )

            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

    def _get_train_sampler(self, dataset: Optional[Dataset] = None) -> Sampler:
        # Returns a sampler that
        # 1. ensures each prompt is repeated across multiple processes. This guarantees that identical prompts are
        #    distributed to different GPUs, allowing rewards to be computed and normalized correctly within each prompt
        #    group. Using the same seed across processes ensures consistent prompt assignment, preventing discrepancies
        #    in group formation.
        # 2. repeats the batch multiple times to allow reusing generations across multiple updates. Refer to
        #    _prepare_inputs to see how the generations are stored and reused.

        # In the following figure, the values are the prompt indices. The first row shows the first sampled batch, the
        # second row shows the second sampled batch, and so on.
        #
        #                                      |   GPU 0  |   GPU 1  |
        #
        #                 global_step   step    <-───>  num_generations=2
        #                                       <-───────> per_device_train_batch_size=3
        #  grad_accum    ▲  ▲  0          0     0   0   1   1   2   2   <- Generate for the first `steps_per_generation` (prompts 0 to 11); store the completions; use the first slice to compute the loss
        #     =2         ▼  |  0          1     3   3   4   4   5   5   <- Take the stored generations and use the second slice to compute the loss
        #                   |
        #                   |  1          2     6   6   7   7   8   8   <- Take the stored generations and use the third slice to compute the loss
        #  steps_per_gen=4  ▼  1          3     9   9  10  10  11  11   <- Take the stored generations and use the fourth slice to compute the loss
        #
        #                      2          4    12  12  13  13  14  14   <- Generate for the second `steps_per_generation` (prompts 12 to 23); store the completions; use the first slice to compute the loss
        #                      2          5    15  15  16  16  17  17   <- Take the stored generations and use the second slice to compute the loss
        #                                          ...
        if dataset is None:
            dataset = self.train_dataset
        return RepeatSampler(
            data_source=dataset,
            mini_repeat_count=self.num_generations,
            batch_size=self.args.generation_batch_size // self.num_generations,
            repeat_count=self.num_iterations * self.args.steps_per_generation,
            shuffle=self.shuffle_dataset,
            seed=self.args.seed,
        )

    def _get_eval_sampler(self, eval_dataset) -> Sampler:
        # See _get_train_sampler for an explanation of the sampler.
        return RepeatSampler(
            data_source=eval_dataset,
            mini_repeat_count=self.num_generations,
            seed=self.args.seed,
        )

    @profiling_decorator
    def _get_last_hidden_state(
        self,
        unwrapped_model,
        input_ids,
        attention_mask,
        logits_to_keep,
        pixel_values=None,
        image_grid_thw=None,
        pixel_attention_mask=None,
        image_sizes=None,
    ):
        if is_peft_model(unwrapped_model):
            unwrapped_model = unwrapped_model.base_model.model

        # Build model inputs - check if the model supports logits_to_keep (some models and VLMs don't)
        model_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}

        # For Qwen models:
        if image_grid_thw is not None and pixel_values is not None:
            model_inputs["image_grid_thw"] = image_grid_thw
        # For Gemma, SmolVLM2, LLaVa-Next etc.:
        if pixel_values is not None:
            model_inputs["pixel_values"] = pixel_values
        # For SmolVLM2
        if pixel_attention_mask is not None:
            model_inputs["pixel_attention_mask"] = pixel_attention_mask
        # For LLaVa-Next
        if image_sizes is not None:
            model_inputs["image_sizes"] = image_sizes

        # Only add logits_to_keep if the model supports it
        if "logits_to_keep" in self.model_kwarg_keys:
            # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
            model_inputs["logits_to_keep"] = logits_to_keep + 1

        model_inputs["use_cache"] = False  # only used in generation; set False to suppress warnings

        last_hidden_state = unwrapped_model.model(**model_inputs).last_hidden_state
        # Exclude the last value: it corresponds to the next token pred
        last_hidden_state = last_hidden_state[:, :-1, :]  # (B, L-1, H)
        # Only keep the last logits_to_keep. For model that support logits_to_keep, this is a no-op.
        last_hidden_state = last_hidden_state[:, -logits_to_keep:, :]  # (B, logits_to_keep, H)
        return last_hidden_state

    def get_high_entropy_mask(self, entropies: torch.Tensor, mask: torch.Tensor, threshold: float) -> torch.Tensor:
        """
        Returns a binary mask identifying tokens whose entropy exceeds a given quantile threshold.

        Args:
            entropies (`torch.Tensor`):
                Tensor of shape (batch_size, seq_len) with per-token entropy values.
            mask (`torch.Tensor`):
                Binary mask of the same shape as `entropies`, where `1` indicates valid tokens and `0` padding.
            threshold (`float`):
                Quantile threshold between `0.0` and `1.0` to select high-entropy tokens.

        Returns:
            `torch.Tensor`:
                Boolean mask of shape (batch_size, seq_len), where `True` indicates tokens with entropy >= threshold
                and `False` otherwise.
        """
        local = entropies[mask.bool()].float()

        # Use a negative pad_value as a sentinel because entropy values are always >= 0.
        # This guarantees that the sentinel cannot collide with any real entropy value.
        pad_value = -1e9

        # Pad across processes so that every rank has the same tensor length
        padded = self.accelerator.pad_across_processes(local, dim=0, pad_index=pad_value)
        gathered = self.accelerator.gather(padded)

        # Drop sentinel values (safe because no entropy can be negative)
        gathered = gathered[gathered != pad_value]

        if gathered.numel() == 0:
            return torch.zeros_like(entropies, dtype=torch.bool)

        entropy_threshold = torch.quantile(gathered, threshold)
        masked_entropies = entropies * mask.float()
        entropy_mask = masked_entropies >= entropy_threshold
        return entropy_mask & mask.bool()  # ensure padding tokens are always masked out

    @profiling_decorator
    def _get_per_token_logps_and_entropies(
        self,
        model,
        input_ids,
        attention_mask,
        logits_to_keep,
        batch_size=None,
        compute_entropy=False,
        pixel_values=None,
        image_grid_thw=None,
        num_images=None,
        pixel_attention_mask=None,
        image_sizes=None,
        token_type_ids=None,
        compute_all_logps=True,
    ) -> dict[str, Optional[torch.Tensor]]:
        """Compute log-probs and (optionally) entropies for each token."""
        batch_size = batch_size or input_ids.size(0)  # Chunk inputs into smaller batches to reduce memory peak
        all_selected_logps = []
        all_logps = []
        all_entropies = []
        for start in range(0, input_ids.size(0), batch_size):
            input_ids_batch = input_ids[start : start + batch_size]
            attention_mask_batch = attention_mask[start : start + batch_size]

            # Build model inputs - check if the model supports logits_to_keep (some models and VLMs don't)
            model_inputs = {"input_ids": input_ids_batch, "attention_mask": attention_mask_batch}
            if image_grid_thw is not None and pixel_values is not None:
                rows_per_image = image_grid_thw.prod(dim=-1)
                rows_per_sample = torch.split(rows_per_image, num_images)
                rows_per_sample = torch.stack([s.sum() for s in rows_per_sample])
                cum_rows = torch.cat([torch.tensor([0], device=rows_per_sample.device), rows_per_sample.cumsum(0)])
                row_start, row_end = cum_rows[start].item(), cum_rows[start + batch_size].item()
                model_inputs["pixel_values"] = pixel_values[row_start:row_end]
                cum_imgs = torch.tensor([0] + num_images).cumsum(0)
                img_start, img_end = cum_imgs[start], cum_imgs[start + batch_size]
                model_inputs["image_grid_thw"] = image_grid_thw[img_start:img_end]
            elif pixel_values is not None:
                model_inputs["pixel_values"] = pixel_values[start : start + batch_size]
            if pixel_attention_mask is not None:
                model_inputs["pixel_attention_mask"] = pixel_attention_mask[start : start + batch_size]
            if image_sizes is not None:
                model_inputs["image_sizes"] = image_sizes[start : start + batch_size]
            if token_type_ids is not None:
                model_inputs["token_type_ids"] = token_type_ids[start : start + batch_size]

            # Only add logits_to_keep if the model supports it
            if "logits_to_keep" in self.model_kwarg_keys:
                # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
                model_inputs["logits_to_keep"] = logits_to_keep + 1

            model_inputs["use_cache"] = False  # only used in generation; set False to suppress warnings

            logits = model(**model_inputs).logits
            # Exclude the last value: it corresponds to the next token pred
            logits = logits[:, :-1, :]  # (B, L-1, H)
            # Only keep the last logits_to_keep. For model that support logits_to_keep, this is a no-op.
            logits = logits[:, -logits_to_keep:, :]  # (B, logits_to_keep, H)
            # Divide logits by sampling temperature.
            # See https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo#policy-training-implementation-details
            logits = logits / self.temperature

            completion_ids = input_ids_batch[:, -logits_to_keep:]
            selected_logps = selective_log_softmax(logits, completion_ids)  # compute logprobs
            if compute_all_logps:
                logps = log_softmax(logits, dim=-1)
            else:
                logps = None
            all_selected_logps.append(selected_logps)
            all_logps.append(logps)

            if compute_entropy:
                with torch.no_grad():
                    entropies = entropy_from_logits(logits)
                all_entropies.append(entropies)

        selected_logps = torch.cat(all_selected_logps, dim=0)
        if compute_all_logps:
            logps = torch.cat(all_logps, dim=0)
        else:
            logps = None
        entropies = torch.cat(all_entropies, dim=0) if compute_entropy else None
        return selected_logps, logps, entropies

    def _fix_param_name_to_vllm(self, name, extra_prefixes: Optional[list[str]] = None):
        extra_prefixes = extra_prefixes or []
        prefixes = ["_checkpoint_wrapped_module."] + extra_prefixes
        for prefix in prefixes:
            name = name.replace(prefix, "")
        return name

    def _get_colocated_vllm_model(self):
        """Return the model object owned by the colocated vLLM engine."""
        if self.vllm_mode != "colocate":
            raise RuntimeError(
                "_get_colocated_vllm_model() is only valid in colocate mode."
            )

        return (
            self.llm.llm_engine
            .model_executor
            .driver_worker
            .model_runner
            .model
        )

    def _sync_named_param_to_vllm(
        self,
        name: str,
        tensor: torch.Tensor,
        *,
        extra_prefixes: Optional[list[str]] = None,
    ) -> None:
        """
        Synchronize one Hugging Face parameter into vLLM.

        Parameter-wrapper cleanup remains generic and trainer-owned.
        Architecture-specific HF -> vLLM naming differences are declared
        by ModelSpec.vllm_weight_sync.
        """
        name = self._fix_param_name_to_vllm(
            name,
            extra_prefixes=extra_prefixes,
        )

        mapped_name = self.vllm_model_spec.vllm_weight_sync.map_name(name)

        # Some vLLM configurations intentionally omit parts of the HF model,
        # e.g. Qwen3.5 with language_model_only=True.
        if mapped_name is None:
            return

        if self.vllm_mode == "server":
            if self.accelerator.is_main_process:
                self.vllm_client.update_named_param(mapped_name, tensor)
            return

        if self.vllm_mode == "colocate":
            llm_model = self._get_colocated_vllm_model()
            llm_model.load_weights([(mapped_name, tensor)])
            return

        raise ValueError(f"Unknown vLLM mode: {self.vllm_mode!r}")

    def _sync_fsdp1_params_to_vllm(self, module: nn.Module, prefix: str = "", visited=None):
        """Memory-efficient post-order traversal of FSDP modules to extract full parameters and sync with vLLM."""
        # For FSDP1, we need to recurse into children and also use summon_full_params
        if visited is None:
            visited = set()
        for child_name, child_module in module.named_children():
            child_prefix = f"{prefix}.{child_name}" if prefix else child_name
            self._sync_fsdp1_params_to_vllm(
                child_module, prefix=child_prefix, visited=visited
            )  # recurse into the child

        if isinstance(module, FSDP):
            with FSDP.summon_full_params(module, recurse=False, writeback=False):
                for param_name, param in module.named_parameters():
                    full_name = f"{prefix}.{param_name}" if prefix else param_name
                    full_name = self._fix_param_name_to_vllm(full_name, extra_prefixes=["_fsdp_wrapped_module."])

                    if full_name in visited:
                        continue  # skip FSDP subtrees already traversed
                    visited.add(full_name)

                    self._sync_named_param_to_vllm(
                        full_name,
                        param.data,
                    )

    def _sync_fsdp2_params_to_vllm(self, module: nn.Module):
        # For FSDP2, module.state_dict() already covers all parameters, so no need for recursion
        for name, param in module.state_dict().items():
            if param.is_cpu:
                param = param.to(torch.device("cuda"))
            param = param.full_tensor()

            self._sync_named_param_to_vllm(name, param)

    @profiling_decorator
    def _move_model_to_vllm(self):
        # Select which model to sync to vLLM: teacher (ref_model) or student (model)
        # When generate_from_teacher=True, sync the teacher model since vLLM was initialized with teacher weights
        model_to_sync = self.ref_model if self.generate_from_teacher else self.model
        
        # For DeepSpeed ZeRO-3 and FSDP, we need to gather all parameters before operations
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        zero_stage_3 = deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3
        if zero_stage_3:
            import deepspeed

            gather_if_zero3 = deepspeed.zero.GatheredParameters
        else:
            gather_if_zero3 = nullcontext

        if is_peft_model(self.model):
            if self.generate_from_teacher:
                raise ValueError("PEFT model handling only applies when syncing student model (teacher is typically not PEFT)")
            # With PEFT and FSDP/DeepSpeed ZeRO Stage 3, we must gather the full model at once before merging, as
            # merging adapters in a sharded manner is not supported.
            # TODO: does this work with FSDP?
            with gather_if_zero3(list(self.model.parameters())):
                self.model.merge_adapter()

                # Update vLLM weights while parameters are gathered
                if self.is_fsdp_enabled:  # note if using FSDP, gather_if_zero3 is nullcontext
                    # Update vLLM weights while parameters are gathered
                    # For PEFT with FSDP we need to use the memory efficient post-order traversal
                    fsdp_plugin = getattr(self.accelerator.state, "fsdp_plugin", None)
                    fsdp_version = getattr(fsdp_plugin, "fsdp_version", 1) if fsdp_plugin else 1
                    if fsdp_version == 1:
                        self._sync_fsdp1_params_to_vllm(
                            self.model
                        )  # use memory-efficient post-order traversal for FSDP
                    elif fsdp_version == 2:
                        self._sync_fsdp2_params_to_vllm(self.model)
                else:
                    # DeepSpeed ZeRO-3 with PEFT
                    for name, param in self.model.named_parameters():
                        # When using PEFT, we need to recover the original parameter name and discard some parameters
                        name = name.removeprefix("base_model.model.").replace(".base_layer", "")
                        if self.model.prefix in name:
                            continue
                        # When module to save, remove its prefix and discard the original module
                        if "original_module" in name:
                            continue
                        self._sync_named_param_to_vllm(
                            name,
                            param.data,
                            extra_prefixes=["modules_to_save.default."],
                        )
                # Unmerge adapters while parameters are still gathered
                self.model.unmerge_adapter()
                # Parameters will automatically be repartitioned when exiting the context
        else:
            # For non-PEFT models, simply gather (if needed) and update each parameter individually.
            if self.is_fsdp_enabled:
                fsdp_plugin = getattr(self.accelerator.state, "fsdp_plugin", None)
                fsdp_version = getattr(fsdp_plugin, "fsdp_version", 1) if fsdp_plugin else 1
                if fsdp_version == 1:
                    self._sync_fsdp1_params_to_vllm(model_to_sync)  # use memory-efficient post-order traversal for FSDP
                elif fsdp_version == 2:
                    self._sync_fsdp2_params_to_vllm(model_to_sync)
            else:
                for name, param in model_to_sync.named_parameters():
                    with gather_if_zero3([param]):
                        self._sync_named_param_to_vllm(
                            name,
                            param.data,
                        )

        # Reset cache on vLLM
        if self.vllm_mode == "server" and self.accelerator.is_main_process:
            self.vllm_client.reset_prefix_cache()
        elif self.vllm_mode == "colocate":
            self.llm.reset_prefix_cache()

    def _trim_completion_padding(
        self,
        batch: dict[str, Union[torch.Tensor, Any]],
    ) -> dict[str, Union[torch.Tensor, Any]]:
        """
        Remove generation-batch-wide right padding after the generation batch
        has been split into training microbatches.

        This is semantics-preserving: only columns beyond the longest real
        completion in the current microbatch are removed.

        Keeping explicit completion lengths rather than deriving them from
        completion_mask also makes this correct when
        mask_truncated_completions=True.
        """
        completion_lengths = batch.pop("_completion_lengths", None)
        completion_ids = batch.get("completion_ids")

        if completion_lengths is None or completion_ids is None:
            return batch

        if not torch.is_tensor(completion_lengths):
            raise TypeError(
                "_completion_lengths must be a tensor after split_tensor_dict."
            )

        if not torch.is_tensor(completion_ids) or completion_ids.ndim < 2:
            return batch

        if completion_lengths.numel() == 0:
            return batch

        old_completion_width = completion_ids.size(1)

        # Keep at least one position so downstream code always sees a
        # non-empty sequence dimension, including fully masked completions.
        new_completion_width = max(
            1,
            int(completion_lengths.max().item()),
        )

        if new_completion_width >= old_completion_width:
            return batch

        # Tensors whose second dimension corresponds directly to completion
        # positions. Slice only when that dimension matches the old padded
        # completion width, so sequence-level tensors (e.g. [B, 1]) are left
        # untouched.
        completion_aligned_keys = (
            "completion_ids",
            "completion_mask",
            "advantages",
            "old_per_token_logps",
            "importance_sampling_ratio",
            "ref_per_token_logps",
        )

        for key in completion_aligned_keys:
            value = batch.get(key)

            if (
                torch.is_tensor(value)
                and value.ndim >= 2
                and value.size(1) == old_completion_width
            ):
                batch[key] = value[:, :new_completion_width, ...]

        # token_type_ids, when present, cover prompt + completion rather than
        # only the completion. Trim only their completion suffix.
        token_type_ids = batch.get("token_type_ids")

        if torch.is_tensor(token_type_ids) and token_type_ids.ndim >= 2:
            prompt_ids = batch.get("prompt_ids")

            if torch.is_tensor(prompt_ids):
                prompt_width = prompt_ids.size(1)
                expected_width = prompt_width + old_completion_width

                if token_type_ids.size(1) == expected_width:
                    batch["token_type_ids"] = torch.cat(
                        [
                            token_type_ids[:, :prompt_width, ...],
                            token_type_ids[
                                :,
                                prompt_width : prompt_width + new_completion_width,
                                ...,
                            ],
                        ],
                        dim=1,
                    )

        return batch

    @profiling_decorator
    def _prepare_inputs(
        self, generation_batch: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        # Prepares inputs for model training/evaluation by managing completion generation and batch handling.
        # During training:
        #   - Receives the local generation batch (Per-GPU batch size × steps per generation)
        #     from the modified training dataloader instead of the standard local batch
        #   - Generates completions once for the entire generation batch and splits it into batches of size
        #     `per_device_train_batch_size`
        #   - Buffers these completions and returns the appropriate slice for the current accumulation step
        #   - Optimizes by regenerating completions only periodically (every steps_per_generation * num_iterations)
        # During evaluation:
        #   - The input is treated as a standard local batch (no accumulation, no multiple iterations)
        #   - Completions are generated for each batch without buffering or reuse
        # Returns a single local batch in both cases.

        mode = "train" if self.model.training else "eval"

        if self.use_dataset_targets and self._has_dynamic_context_strategy(generation_batch):
            raise ValueError(
                "Dynamic context strategies are not supported with "
                "optimal_policy_source='dataset' yet. "
                "Dynamic strategies require generated seed responses and possibly "
                "generated feedback/rationales. Use optimal_policy_source='teacher' "
                "or use a static/precomputed context_strategy."
            )

        prepare_fn = (
            self._prepare_dataset_target_inputs
            if self.use_dataset_targets
            else self._generate_and_score_completions
        )
        if mode == "train":
            generate_every = self.args.steps_per_generation * self.num_iterations
            if self._step % generate_every == 0 or self._buffered_inputs is None:
                # self._buffered_inputs=None can occur when resuming from a checkpoint.
                # Refresh opted-in wildcard prompt variations only when a new
                # training generation batch is actually consumed. Evaluation is
                # intentionally left stable.
                generation_batch = self._maybe_resample_context_for_epoch(generation_batch)
                generation_batch = prepare_fn(generation_batch)
                generation_batch = split_pixel_values_by_grid(generation_batch)
                if not self.use_dataset_targets:
                    generation_batch = shuffle_sequence_dict(generation_batch)
                generation_batches = split_tensor_dict(generation_batch, self.args.steps_per_generation)
                # self._buffered_inputs = [unsplit_pixel_values_by_grid(batch) for batch in generation_batches]
                self._buffered_inputs = [
                    self._trim_completion_padding(
                        unsplit_pixel_values_by_grid(batch)
                    )
                    for batch in generation_batches
                ]
            inputs = self._buffered_inputs[self._step % self.args.steps_per_generation]
            self._step += 1
        else:
            # In evaluation, there is neither batch grouping for generation, nor multiple iterations, hence
            # local generation batch == local eval batch
            inputs = prepare_fn(generation_batch)
        return inputs

    @profiling_decorator
    def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
        device = self.accelerator.device
        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)

        # Repeat all input columns (but "prompt", "completion", and "completion_ids") to match the num of generations
        keys = [key for key in inputs[0] if key not in ["prompt", "completion", "completion_ids"]]
        reward_kwargs = {key: [example[key] for example in inputs] for key in keys}

        # This allows for dynamic reward shaping based on training progress.
        reward_kwargs["trainer_state"] = self.state

        for i, (reward_func, reward_processing_class, reward_func_name) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes, self.reward_func_names)
        ):
            with profiling_context(self, reward_func_name):
                if isinstance(reward_func, nn.Module):  # Module (no PretrainedModel) for compat with compiled models
                    if is_conversational(inputs[0]):
                        messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                        texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                    else:
                        texts = [p + c for p, c in zip(prompts, completions)]
                    reward_inputs = reward_processing_class(
                        text=texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                    )
                    reward_inputs = super()._prepare_inputs(reward_inputs)
                    with torch.inference_mode():
                        rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
                else:
                    output_reward_func = reward_func(
                        prompts=prompts, completions=completions, completion_ids=completion_ids_list, **reward_kwargs
                    )
                    # Convert None values to NaN
                    output_reward_func = [reward if reward is not None else torch.nan for reward in output_reward_func]

                    rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # If all reward functions return None for a given row, issue a detailed warning
        if torch.isnan(rewards_per_func).all(dim=1).any():
            nan_row_idx = torch.isnan(rewards_per_func).all(dim=1).nonzero(as_tuple=True)[0][0]
            row_reward_kwargs = {
                key: value[nan_row_idx] for key, value in reward_kwargs.items() if key != "trainer_state"
            }
            row_reward_kwargs["prompt"] = prompts[nan_row_idx]
            row_reward_kwargs["completion"] = completions[nan_row_idx]
            logger.warning(
                f"All reward functions returned None for the following kwargs:\n{row_reward_kwargs}\n"
                "Please ensure that at least one reward function returns a valid reward."
            )

        # Gather the reward per function: this part is crucial, because the rewards are normalized per group and the
        # completions may be distributed across processes
        rewards_per_func = gather(rewards_per_func)
        return rewards_per_func

    @staticmethod
    def _chat_generation_template_kwargs(prompt: Any) -> dict[str, bool]:
        """Match TRL's prompt-only chat-template semantics."""
        if not is_conversational({"prompt": prompt}):
            return {}

        last_role = prompt[-1]["role"]
        if last_role in ("user", "tool"):
            return {
                "add_generation_prompt": True,
                "continue_final_message": False,
            }
        if last_role == "assistant":
            return {
                "add_generation_prompt": False,
                "continue_final_message": True,
            }

        raise ValueError(f"Invalid final chat role: {last_role!r}")

    def _render_prompt_text(self, prompt: Any) -> str:
        """Render a prompt to text while preserving the historical TEXT path."""
        if not is_conversational({"prompt": prompt}):
            return prompt

        return maybe_apply_chat_template(
            {"prompt": prompt},
            self.processing_class,
            **self.chat_template_kwargs,
        )["prompt"]

    def _tokenize_prompt_ids(self, prompt: Any) -> list[int]:
        """Tokenize one prompt directly, avoiding an unsafe text round-trip."""
        if not is_conversational({"prompt": prompt}):
            encoded = self.processing_class(
                text=prompt,
                add_special_tokens=False,
                truncation=True,
                max_length=self.max_prompt_length,
            )
            return list(encoded["input_ids"])

        controls = self._chat_generation_template_kwargs(prompt)
        encoded = self.processing_class.apply_chat_template(
            prompt,
            tokenize=True,
            truncation=True,
            max_length=self.max_prompt_length,
            return_dict=False,
            **controls,
            **self.chat_template_kwargs,
        )
        return list(encoded)

    def _prepare_text_prompt_batch(
        self,
        prompts: list[Any],
        *,
        padding_side: str = "left",
        processing_kwargs: Optional[dict[str, Any]] = None,
    ):
        """Prepare a prompt batch for an HF forward/generate call."""
        processing_kwargs = processing_kwargs or {}

        if self.model_spec.chat_template_mode == ChatTemplateMode.TEXT:
            prompt_texts = [self._render_prompt_text(prompt) for prompt in prompts]
            return self.processing_class(
                text=prompt_texts,
                return_tensors="pt",
                padding=True,
                padding_side=padding_side,
                max_length=self.max_prompt_length,
                truncation=True,
                add_special_tokens=False,
                **processing_kwargs,
            )

        if processing_kwargs:
            raise NotImplementedError(
                "TOKEN_IDS chat-template mode with multimodal training inputs "
                "has not yet been implemented. Text-only training is supported."
            )

        prompt_ids = [self._tokenize_prompt_ids(prompt) for prompt in prompts]
        input_tensors = [torch.tensor(ids, dtype=torch.long) for ids in prompt_ids]
        mask_tensors = [torch.ones(len(ids), dtype=torch.long) for ids in prompt_ids]

        return {
            "input_ids": pad(
                input_tensors,
                padding_value=self.pad_token_id,
                padding_side=padding_side,
            ),
            "attention_mask": pad(
                mask_tensors,
                padding_value=0,
                padding_side=padding_side,
            ),
        }

    def _prepare_vllm_prompts(self, prompts: list[Any]) -> list[Any]:
        """Prepare prompts for vLLM while enforcing max prompt length.

        vLLM 0.20 does not accept ``truncate_prompt_tokens`` in
        ``SamplingParams``. TOKEN_IDS models (e.g. MistralCommonBackend) are
        truncated by ``_tokenize_prompt_ids``. TEXT-mode prompts remain strings
        when they fit the configured limit; overlong prompts are converted to
        explicitly left-truncated token IDs.
        """
        if self.vllm_model_spec.chat_template_mode == ChatTemplateMode.TOKEN_IDS:
            return [
                {"prompt_token_ids": self._tokenize_prompt_ids(prompt)}
                for prompt in prompts
            ]

        prompt_texts = [self._render_prompt_text(prompt) for prompt in prompts]
        if self.max_prompt_length is None:
            return prompt_texts

        tokenizer = (
            self.processing_class.tokenizer
            if isinstance(self.processing_class, ProcessorMixin)
            else self.processing_class
        )

        vllm_prompts = []
        for prompt_text in prompt_texts:
            encoded = tokenizer(
                prompt_text,
                add_special_tokens=False,
                truncation=False,
            )
            prompt_token_ids = list(encoded["input_ids"])

            if len(prompt_token_ids) <= self.max_prompt_length:
                # Preserve the historical Qwen2.5/string path when no
                # truncation is required.
                vllm_prompts.append(prompt_text)
            else:
                # Preserve the old left-truncation semantics.
                vllm_prompts.append(
                    {
                        "prompt_token_ids": prompt_token_ids[
                            -self.max_prompt_length :
                        ]
                    }
                )

        return vllm_prompts

    def _generate_single_turn(
        self,
        prompts: list[str],
        images: Optional[list],
        *,
        max_tokens_override: int | None = None,
        temperature_override: float | None = None,
        top_p_override: float | None = None,
        disable_weight_sync: bool = False,
    ):
        device = self.accelerator.device

        effective_max_tokens = (
            self.max_completion_length
            if max_tokens_override is None
            else max_tokens_override
        )
        effective_temperature = (
            self.temperature
            if temperature_override is None
            else temperature_override
        )
        effective_top_p = (
            self.top_p
            if top_p_override is None
            else top_p_override
        )

        # If the prompts are conversational and the inputs contain images, we need to convert the prompts from
        # [{"role": "user", "content": "What color is the sky?"}] to
        # [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "What color is the sky?"}]}]
        kwargs = {}
        if images is not None:
            kwargs = {"images": images}
            for prompt, image_list in zip(prompts, images):
                if isinstance(prompt, list):  # i.e., when using conversational data
                    prepare_multimodal_messages(prompt, num_images=len(image_list))

        if images is not None and self.vllm_model_spec.chat_template_mode == ChatTemplateMode.TOKEN_IDS:
            raise NotImplementedError(
                "TOKEN_IDS chat-template mode with multimodal training inputs "
                "has not yet been implemented. Text-only training is supported."
            )

        if self.vllm_model_spec.chat_template_mode == ChatTemplateMode.TEXT:
            prompts_text = [self._render_prompt_text(prompt) for prompt in prompts]
            vllm_prompts = prompts_text
        else:
            prompts_text = None
            vllm_prompts = self._prepare_vllm_prompts(prompts) if self.use_vllm else None

        if images is not None:
            prompt_inputs = self.processing_class(text=prompts_text, padding=True, return_tensors="pt", **kwargs)
            prompt_inputs = super()._prepare_inputs(prompt_inputs)
            forward_kwargs = {k: v for k, v in prompt_inputs.items() if k not in ["input_ids", "attention_mask"]}
        else:
            forward_kwargs = {}

        # Generate completions using either vLLM or regular generation
        # Note: When generate_from_teacher=True, vLLM is initialized with teacher weights
        if self.use_vllm:
            if self.vllm_mode == "colocate" and self.args.vllm_enable_sleep_mode:
                # wake up colocated vLLM instances if needed
                torch.cuda.empty_cache()  # required to avoid OOM in some cases
                self.llm.wake_up()

            # First, update the vLLM weights if needed
            # When generate_from_teacher=True and sync_ref_model=False, teacher is static so no sync needed
            # (vLLM already loaded teacher weights at initialization)
            should_sync = self.state.global_step != self._last_loaded_step
            if self.generate_from_teacher and not self.args.sync_ref_model:
                should_sync = False  # Teacher is static, no need to sync
            if should_sync and not disable_weight_sync:
                self._move_model_to_vllm()
                self._last_loaded_step = self.state.global_step

            # Generate completions using vLLM: gather all prompts and use them in a single call in the main process
            if self.vllm_mode == "server":
                all_prompts_text = gather_object(vllm_prompts)
                if images is not None:
                    all_images = gather_object(images)

                if self.accelerator.is_main_process:
                    # Since 'prompts' contains 'num_generations' duplicates, we first take unique prompts, and generate
                    # num_generations outputs for each one. This is faster than generating outputs for each duplicate
                    # prompt individually.
                    ordered_set_of_prompts = all_prompts_text[:: self.num_generations]

                    if images is not None:
                        ordered_set_of_images = all_images[:: self.num_generations]
                    else:
                        ordered_set_of_images = None

                    with profiling_context(self, "vLLM.generate"):
                        output = self.vllm_client.generate(
                            prompts=ordered_set_of_prompts,
                            images=ordered_set_of_images,
                            n=self.num_generations,
                            repetition_penalty=self.repetition_penalty,
                            temperature=effective_temperature,
                            top_p=effective_top_p,
                            top_k=-1 if self.top_k is None else self.top_k,
                            min_p=0.0 if self.min_p is None else self.min_p,
                            max_tokens=effective_max_tokens,
                            truncate_prompt_tokens=self.max_prompt_length,
                            generation_kwargs=self.args.generation_kwargs,
                        )
                        payload = (output["prompt_ids"], output["completion_ids"], output["logprobs"])
                else:
                    payload = None

                # Broadcast the completions from the main process to all processes, ensuring each process receives its corresponding slice.
                obj_list = [payload]
                broadcast_object_list(obj_list, from_process=0)
                all_prompt_ids, all_completion_ids, all_logprobs = obj_list[0]

                # At this point, we only get 1 copy of each prompt, so we need to repeat them num_generations times
                all_prompt_ids = [ids for ids in all_prompt_ids for _ in range(self.num_generations)]

                process_slice = slice(
                    self.accelerator.process_index * len(prompts),
                    (self.accelerator.process_index + 1) * len(prompts),
                )
                prompt_ids = all_prompt_ids[process_slice]
                completion_ids = all_completion_ids[process_slice]
                logprobs = all_logprobs[process_slice]

            # Generate completions using colocated vLLM instances: each device holds vLLM copy and work on their own batch of prompts
            elif self.vllm_mode == "colocate":
                generation_kwargs = {
                    "n": 1,  # vLLM on each GPU generates only 1 in colocate mode
                    "repetition_penalty": self.repetition_penalty,
                    "temperature": effective_temperature,
                    "top_p": effective_top_p,
                    "top_k": -1 if self.top_k is None else self.top_k,
                    "min_p": 0.0 if self.min_p is None else self.min_p,
                    "max_tokens": effective_max_tokens,
                    "logprobs": 0,  # only return the logprob of the generated token
                }

                if self.args.generation_kwargs is not None:
                    generation_kwargs.update(self.args.generation_kwargs)
                sampling_params = SamplingParams(**generation_kwargs)

                if self.vllm_tensor_parallel_size > 1:
                    # Gather prompts from all ranks in the TP group and flatten.
                    # Each rank starts with its own prompts; after gathering, all ranks see the full group set.
                    orig_size = len(vllm_prompts)
                    gathered_prompts = [None for _ in range(self.vllm_tensor_parallel_size)]
                    torch.distributed.all_gather_object(gathered_prompts, vllm_prompts, group=self.tp_group)
                    all_prompts_text = [p for sublist in gathered_prompts for p in sublist]

                    if images is not None:
                        gathered_images = [None for _ in range(self.vllm_tensor_parallel_size)]
                        torch.distributed.all_gather_object(gathered_images, images, group=self.tp_group)
                        all_images = [img for sublist in gathered_images for img in sublist]
                    else:
                        all_images = None
                else:
                    all_prompts_text = vllm_prompts
                    all_images = images

                if images is not None and all_images:
                    vllm_inputs = []
                    for prompt, image_list in zip(all_prompts_text, all_images):
                        vllm_inputs.append({"prompt": prompt, "multi_modal_data": {"image": image_list}})

                else:
                    vllm_inputs = all_prompts_text

                with profiling_context(self, "vLLM.generate"):
                    all_outputs = self.llm.generate(vllm_inputs, sampling_params=sampling_params, use_tqdm=False)

                all_prompt_ids = [output.prompt_token_ids for output in all_outputs]
                all_completion_ids = [output.token_ids for outputs in all_outputs for output in outputs.outputs]
                all_logprobs = [
                    [next(iter(lp.values())).logprob for lp in output.logprobs]
                    for outputs in all_outputs
                    for output in outputs.outputs
                ]

                if self.vllm_tensor_parallel_size > 1:
                    # Slice completions for this rank within its TP group.
                    # Each rank generates all outputs — we keep only our share.
                    local_rank_in_group = torch.distributed.get_rank(group=self.tp_group)
                    tp_slice = slice(local_rank_in_group * orig_size, (local_rank_in_group + 1) * orig_size)
                    prompt_ids = all_prompt_ids[tp_slice]
                    completion_ids = all_completion_ids[tp_slice]
                    logprobs = all_logprobs[tp_slice]
                else:
                    prompt_ids = all_prompt_ids
                    completion_ids = all_completion_ids
                    logprobs = all_logprobs

                if self.args.vllm_enable_sleep_mode:
                    self.llm.sleep(level=1)

        elif self.use_transformers_paged:
            # Re-process inputs for paged generation if needed
            # Note: images are already validated and preprocessed above
            if self.model_spec.chat_template_mode == ChatTemplateMode.TEXT:
                paged_prompt_inputs = self.processing_class(text=prompts_text, **kwargs)
            else:
                paged_prompt_inputs = self._prepare_text_prompt_batch(
                    prompts,
                    padding_side="left",
                    processing_kwargs=kwargs,
                )
            previous_attn = self.model_wrapped.config._attn_implementation

            if is_flash_attn_2_available():
                self.model_wrapped.config._attn_implementation = "paged_attention"
            else:
                self.model_wrapped.config._attn_implementation = "sdpa_paged"
            with (
                profiling_context(self, "transformers.generate_batch"),
                unwrap_model_for_generation(
                    self.model_wrapped, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
                ) as unwrapped_model,
                torch.no_grad(),
                FSDP.summon_full_params(self.model_wrapped, recurse=False) if self.is_fsdp_enabled else nullcontext(),
            ):
                # Cast to the appropriate dtype based on training configuration
                if self.args.bf16:
                    unwrapped_model.to(torch.bfloat16)
                elif self.args.fp16:
                    unwrapped_model.to(torch.float16)
                with torch.inference_mode():
                    all_outputs = unwrapped_model.generate_batch(
                        paged_prompt_inputs["input_ids"], generation_config=self.generation_config, progress_bar=False
                    )
                    unwrapped_model.train()  # restore training mode, as generate_batch forces eval mode
            completion_ids = [output.generated_tokens for output in all_outputs.values()]
            prompt_ids = paged_prompt_inputs["input_ids"]
            # Restore the original attention implementation, training mode
            self.model_wrapped.config._attn_implementation = previous_attn
            logprobs = None  # not used in this case

        else:
            # Regular generation path
            generate_inputs = self._prepare_text_prompt_batch(
                prompts,
                padding_side="left",
                processing_kwargs=kwargs,
            )
            generate_inputs = super()._prepare_inputs(generate_inputs)

            with (
                profiling_context(self, "transformers.generate"),
                unwrap_model_for_generation(
                    self.model_wrapped, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
                ) as unwrapped_model,
                torch.no_grad(),
                FSDP.summon_full_params(self.model_wrapped, recurse=False) if self.is_fsdp_enabled else nullcontext(),
            ):
                prompt_completion_ids = unwrapped_model.generate(
                    **generate_inputs, generation_config=self.generation_config, disable_compile=True
                )
            # Compute prompt length and extract completion ids
            prompt_ids, prompt_mask = generate_inputs["input_ids"], generate_inputs["attention_mask"]
            prompt_length = prompt_ids.size(1)
            completion_ids = prompt_completion_ids[:, prompt_length:]

            # Mask everything after the first EOS token
            is_eos = completion_ids == self.eos_token_id
            eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
            eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
            sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
            completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
            prompt_ids = [p[m].tolist() for p, m in zip(prompt_ids, prompt_mask.bool())]
            completion_ids = [c[m].tolist() for c, m in zip(completion_ids, completion_mask.bool())]
            logprobs = None  # not used in this case

        return prompt_ids, completion_ids, logprobs, forward_kwargs

    def _generate(self, prompts: list[str], images: Optional[list]):
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        prompt_ids, completion_ids, logprobs, forward_kwargs = self._generate_single_turn(prompts, images)

        # Get completion length per sequence, used for logging
        prompt_lengths = torch.tensor([len(ids) for ids in prompt_ids], device=device)
        completion_lengths = torch.tensor([len(ids) for ids in completion_ids], device=device)
        agg_prompt_lengths = self.accelerator.gather(prompt_lengths)
        agg_completion_lengths = self.accelerator.gather(completion_lengths)
        total_prompt_tokens = agg_prompt_lengths.sum()
        total_completion_tokens = agg_completion_lengths.sum()  # = num_items_in_batch, required for the DAPO loss

        # Log the metrics
        if mode == "train":
            self.state.num_input_tokens_seen += (total_prompt_tokens + total_completion_tokens).item()
        self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]

        # Log completion lengths, mean, min, max
        self._metrics[mode]["completions/mean_length"].append(agg_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_length"].append(agg_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_length"].append(agg_completion_lengths.float().max().item())

        # Identify sequences that terminated with EOS and log their lengths
        eos_and_pad = [self.eos_token_id, self.pad_token_id]
        is_truncated = torch.tensor([ids[-1] not in eos_and_pad for ids in completion_ids], device=device)
        agg_is_truncated = self.accelerator.gather(is_truncated)
        self._metrics[mode]["completions/clipped_ratio"].append(agg_is_truncated.float().mean().item())
        term_completion_lengths = agg_completion_lengths[~agg_is_truncated]
        if len(term_completion_lengths) == 0:  # edge case where no terminated sequences are found
            term_completion_lengths = torch.zeros(1, device=device)
        self._metrics[mode]["completions/mean_terminated_length"].append(term_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_terminated_length"].append(term_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_terminated_length"].append(term_completion_lengths.float().max().item())

        return prompt_ids, completion_ids, total_completion_tokens, logprobs, forward_kwargs

    @profiling_decorator
    def _prepare_dataset_target_inputs(
        self, inputs: list[dict[str, Union[torch.Tensor, Any]]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        prompts = [x["prompt"] for x in inputs]
        teacher_prompts = [x["teacher_prompt"] for x in inputs]
        self._maybe_dump_teacher_prompts( # Maybe remove this later?
            inputs=inputs,
            teacher_prompts=teacher_prompts,
            tag="dataset_target",
        )

        targets = [x["target"] for x in inputs]

        if "images" in inputs[0]:
            images = [example.get("images") for example in inputs]
        elif "image" in inputs[0]:
            images = [[example.get("image")] if example.get("image") is not None else None for example in inputs]
        else:
            images = None

        if images is not None and all(img_list == [] for img_list in images):
            images = None

        kwargs = {}
        if images is not None:
            kwargs = {"images": images}
            for prompt, image_list in zip(prompts, images):
                if isinstance(prompt, list):
                    prepare_multimodal_messages(prompt, num_images=len(image_list))
            for prompt, image_list in zip(teacher_prompts, images):
                if isinstance(prompt, list):
                    prepare_multimodal_messages(prompt, num_images=len(image_list))

        # Student prompts
        if self.use_vllm:
            self.processing_class.truncation_side = "left"
        student_inputs = self._prepare_text_prompt_batch(
            prompts,
            padding_side="left",
            processing_kwargs=kwargs,
        )
        student_inputs = super()._prepare_inputs(student_inputs)
        student_prompt_ids, student_prompt_mask = student_inputs["input_ids"], student_inputs["attention_mask"]
        prompt_ids_list = [p[m].tolist() for p, m in zip(student_prompt_ids, student_prompt_mask.bool())]

        forward_kwargs = {k: v for k, v in student_inputs.items() if k not in ["input_ids", "attention_mask"]}

        # Teacher prompts
        teacher_inputs = self._prepare_text_prompt_batch(
            teacher_prompts,
            padding_side="left",
            processing_kwargs=kwargs,
        )
        teacher_inputs = super()._prepare_inputs(teacher_inputs)
        teacher_prompt_ids, teacher_prompt_mask = teacher_inputs["input_ids"], teacher_inputs["attention_mask"]
        teacher_prompt_ids_list = [p[m].tolist() for p, m in zip(teacher_prompt_ids, teacher_prompt_mask.bool())]

        if self.use_vllm:
            self.processing_class.truncation_side = "right"

        # Targets -> completion ids
        targets_text = targets

        target_inputs = self.processing_class(
            text=targets_text,
            return_tensors="pt",
            padding=True,
            padding_side="right",
            max_length=self.max_completion_length,
            truncation=True,
            add_special_tokens=False,
        )
        target_inputs = super()._prepare_inputs(target_inputs)
        completion_ids_raw, completion_mask_raw = target_inputs["input_ids"], target_inputs["attention_mask"]
        completion_ids_list = [c[m].tolist() for c, m in zip(completion_ids_raw, completion_mask_raw.bool())]

        # Convert to padded tensors
        prompt_ids = [torch.tensor(ids, device=device) for ids in prompt_ids_list]
        prompt_mask = [torch.ones_like(ids, dtype=torch.long) for ids in prompt_ids]
        prompt_ids = pad(prompt_ids, padding_value=self.pad_token_id, padding_side="left")
        prompt_mask = pad(prompt_mask, padding_value=0, padding_side="left")

        teacher_prompt_ids = [torch.tensor(ids, device=device) for ids in teacher_prompt_ids_list]
        teacher_prompt_mask = [torch.ones_like(ids, dtype=torch.long) for ids in teacher_prompt_ids]
        teacher_prompt_ids = pad(teacher_prompt_ids, padding_value=self.pad_token_id, padding_side="left")
        teacher_prompt_mask = pad(teacher_prompt_mask, padding_value=0, padding_side="left")

        completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids_list]
        completion_mask = [torch.ones_like(ids, dtype=torch.long) for ids in completion_ids]
        completion_ids = pad(completion_ids, padding_value=self.pad_token_id, padding_side="right")
        completion_mask = pad(completion_mask, padding_value=0, padding_side="right")

        if self.mask_truncated_completions:
            eos_and_pad = [self.eos_token_id, self.pad_token_id]
            is_truncated = torch.tensor(
                [len(ids) > 0 and ids[-1] not in eos_and_pad for ids in completion_ids_list],
                device=device,
            )
            completion_mask = completion_mask * (~is_truncated).unsqueeze(1).int()
        else:
            is_truncated = torch.zeros(len(completion_ids_list), dtype=torch.bool, device=device)

        # If token_type_ids are used, extend them with zeros for the completion part
        if "token_type_ids" in forward_kwargs:
            token_type_ids = forward_kwargs["token_type_ids"]
            forward_kwargs["token_type_ids"] = torch.cat(
                [token_type_ids, token_type_ids.new_zeros(completion_ids.shape)], dim=1
            )

        # Logging/token accounting
        prompt_lengths = torch.tensor([len(ids) for ids in prompt_ids_list], device=device)
        completion_lengths = torch.tensor([len(ids) for ids in completion_ids_list], device=device)
        agg_prompt_lengths = self.accelerator.gather(prompt_lengths)
        agg_completion_lengths = self.accelerator.gather(completion_lengths)
        agg_is_truncated = self.accelerator.gather(is_truncated)

        total_prompt_tokens = agg_prompt_lengths.sum()
        total_completion_tokens = agg_completion_lengths.sum()

        if mode == "train":
            self.state.num_input_tokens_seen += (total_prompt_tokens + total_completion_tokens).item()
        self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]
        self._metrics[mode]["completions/mean_length"].append(agg_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_length"].append(agg_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_length"].append(agg_completion_lengths.float().max().item())
        self._metrics[mode]["completions/clipped_ratio"].append(agg_is_truncated.float().mean().item())

        term_completion_lengths = agg_completion_lengths[~agg_is_truncated]
        if len(term_completion_lengths) == 0:
            term_completion_lengths = torch.zeros(1, device=device)
        self._metrics[mode]["completions/mean_terminated_length"].append(term_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_terminated_length"].append(term_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_terminated_length"].append(term_completion_lengths.float().max().item())

        prompts_decoded = self.processing_class.batch_decode(prompt_ids, skip_special_tokens=True)
        completions_decoded = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)

        rewards = torch.zeros_like(completion_ids, dtype=torch.float32)
        advantages = rewards

        self._logs["prompt"].extend(gather_object(prompts_decoded))
        self._logs["completion"].extend(gather_object(completions_decoded))
        self._logs["rewards"]["main"].extend(gather_object(rewards.mean(dim=-1).tolist()))
        self._logs["advantages"].extend(gather_object(advantages.mean(dim=-1).tolist()))

        reward_to_log = rewards[completion_mask.bool()]
        mean_reward = torch.mean(reward_to_log) if reward_to_log.numel() > 0 else torch.tensor(0.0, device=device)
        self._metrics[mode]["rewards"].append(self.accelerator.gather(mean_reward).mean().item())

        output = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "teacher_prompt_ids": teacher_prompt_ids,
            "teacher_prompt_mask": teacher_prompt_mask,
            "advantages": advantages,
            "num_items_in_batch": total_completion_tokens,
            "_completion_lengths": completion_lengths.cpu(),
        }

        if "pixel_values" in forward_kwargs:
            output["pixel_values"] = forward_kwargs["pixel_values"]
        if "image_grid_thw" in forward_kwargs:
            output["image_grid_thw"] = forward_kwargs["image_grid_thw"]
        if "pixel_attention_mask" in forward_kwargs:
            output["pixel_attention_mask"] = forward_kwargs["pixel_attention_mask"]
        if "image_sizes" in forward_kwargs:
            output["image_sizes"] = forward_kwargs["image_sizes"]
        if "token_type_ids" in forward_kwargs:
            output["token_type_ids"] = forward_kwargs["token_type_ids"]
        if images is not None:
            output["num_images"] = [len(img_list) for img_list in images]

        return output

    def _get_example_strategy_name(self, example: dict) -> str:
        return example.get(
            "context_strategy",
            getattr(self.args, "context_strategy", "dataset_default"),
        )

    def _get_example_source_dataset(self, example: dict) -> str | None:
        return example.get("source_dataset", None)

    def _has_dynamic_context_strategy(self, inputs: list[dict[str, Any]]) -> bool:
        for example in inputs:
            strategy_name = self._get_example_strategy_name(example)
            manager = ContextualizationManager(strategy_name)
            if not manager.strategy.can_precompute:
                return True
        return False

    # -------------------------------------------------------------------------
    # Per-epoch prompt-variation resampling
    # -------------------------------------------------------------------------

    def _current_epoch_index(self) -> int:
        """Return the checkpoint-restored, 0-based training epoch index."""
        epoch = getattr(self.state, "epoch", None)
        if epoch is None:
            return 0
        try:
            # TrainerState.epoch is fractional inside an epoch and is restored
            # from checkpoints. A tiny epsilon avoids 0.999999999 artifacts.
            return max(0, math.floor(float(epoch) + 1e-8))
        except (TypeError, ValueError, OverflowError):
            return 0

    def _has_epoch_resampled_context_strategy(
        self,
        inputs: list[dict[str, Any]],
    ) -> bool:
        """Return whether any row belongs to an explicitly opted-in strategy."""
        for example in inputs:
            strategy_name = self._get_example_strategy_name(example)
            manager = ContextualizationManager(strategy_name)
            if bool(getattr(manager.strategy, "resample_prompt_variation_per_epoch", False)):
                return True
        return False

    def _resample_context_example_for_epoch(
        self,
        example: dict[str, Any],
        *,
        epoch_index: int,
    ) -> dict[str, Any]:
        """Rebuild one mapped row for ``epoch_index`` when its wildcard opts in.

        The trainer deliberately consumes concrete map-time metadata instead of
        replaying the original strategy-selection expression. This keeps the
        epoch mechanism independent of future parser/registry changes.
        """
        strategy_name = self._get_example_strategy_name(example)
        manager = ContextualizationManager(strategy_name)
        strategy = manager.strategy

        if not bool(getattr(strategy, "resample_prompt_variation_per_epoch", False)):
            return example

        if "context_prompt_variation_random" not in example:
            raise ValueError(
                f"Context strategy {strategy_name!r} opts into per-epoch prompt "
                "resampling, but this row has no context_prompt_variation_random "
                "metadata. Rebuild the training dataset with the current "
                "contextualized_train_loader.py."
            )

        # Fixed/default variation specs are deliberate controls and must never
        # be changed merely because the strategy class supports resampling.
        if not bool(example.get("context_prompt_variation_random")):
            return example

        source_dataset = self._get_example_source_dataset(example)
        selection_spec = example.get("context_strategy_selection_spec")
        row_index = example.get("context_row_index")
        selection_seed = example.get("context_selection_seed")
        base_variation = example.get(
            "context_base_prompt_variation",
            example.get("context_prompt_variation"),
        )
        stage1_variation = example.get("context_stage1_variation")

        try:
            row_index = int(row_index)
            selection_seed = int(selection_seed)
            base_variation = int(base_variation)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Context strategy {strategy_name!r} requests per-epoch prompt "
                "resampling, but its row/seed/base-variation metadata is missing "
                "or invalid. Rebuild the training dataset with the current loader."
            ) from exc

        if not source_dataset or row_index < 0 or selection_seed < 0:
            raise ValueError(
                f"Context strategy {strategy_name!r} requests per-epoch prompt "
                "resampling, but source_dataset/context_row_index/"
                "context_selection_seed is missing or invalid."
            )

        prompt_variation = strategy.resolve_epoch_prompt_variation(
            dataset_name=source_dataset,
            base_variation=base_variation,
            row_index=row_index,
            seed=selection_seed,
            epoch_index=epoch_index,
        )

        if prompt_variation == example.get("context_prompt_variation"):
            return example

        raw_example = json_loads_safe(example.get("raw_example_json"))
        if not isinstance(raw_example, dict):
            raise ValueError(
                f"Per-epoch context resampling for strategy {strategy_name!r} "
                "requires raw_example_json to decode to a dict."
            )

        rebuilt_manager = ContextualizationManager(
            strategy_name,
            prompt_variation=prompt_variation,
            stage1_variation=stage1_variation,
        )
        rebuilt = rebuilt_manager.format_train_example(
            dataset_name=source_dataset,
            raw_example=raw_example,
            prompt_variation=prompt_variation,
            stage1_variation=stage1_variation,
            selection_spec=(
                str(selection_spec) if selection_spec else strategy_name
            ),
            base_prompt_variation=base_variation,
            row_index=row_index,
            selection_seed=selection_seed,
            prompt_variation_random=True,
            stage1_variation_random=bool(
                example.get("context_stage1_variation_random", False)
            ),
        )

        # Preserve any extra/future columns not emitted by the manager while
        # replacing all prompt/context fields with the freshly rebuilt values.
        return {**example, **rebuilt}

    def _maybe_resample_context_for_epoch(
        self,
        inputs: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Apply opted-in wildcard prompt variation changes for this epoch."""
        if not inputs or not isinstance(inputs, list):
            return inputs

        if not self._has_epoch_resampled_context_strategy(inputs):
            return inputs

        epoch_index = self._current_epoch_index()
        if epoch_index <= 0:
            # Epoch 0 is the exact map-time assignment.
            return inputs

        return [
            self._resample_context_example_for_epoch(
                example,
                epoch_index=epoch_index,
            )
            for example in inputs
        ]

    def _decode_completion_ids_list(self, completion_ids_list: list[list[int]]) -> list[str]:
        return [
            self.processing_class.decode(ids, skip_special_tokens=True)
            for ids in completion_ids_list
        ]

    def _build_runtime_teacher_prompts(
        self,
        *,
        inputs: list[dict[str, Any]],
        seed_responses: list[str],
        feedback_texts: list[str | None],
        sibling_texts: list[str | None] | None = None,
    ) -> list[Any]:
        teacher_prompts = []

        if sibling_texts is None:
            sibling_texts = [None for _ in inputs]

        if len(feedback_texts) != len(inputs):
            raise ValueError(
                "feedback_texts must have the same length as inputs."
            )

        if len(sibling_texts) != len(inputs):
            raise ValueError(
                "sibling_texts must have the same length as inputs."
            )

        for example, seed_response, feedback_text, sibling_text in zip(
            inputs,
            seed_responses,
            feedback_texts,
            sibling_texts,
        ):
            strategy_name = self._get_example_strategy_name(example)

            if strategy_name == "dataset_default":
                teacher_prompts.append(example["teacher_prompt"])
                continue

            source_dataset = self._get_example_source_dataset(example)
            raw_example = json_loads_safe(example.get("raw_example_json"))
            reference = json_loads_safe(example.get("reference_json"))

            prompt_variation = example.get("context_prompt_variation")

            if not source_dataset or raw_example is None:
                raise ValueError(
                    f"Dynamic strategy {strategy_name!r} requires source_dataset, "
                    "raw_example_json, and reference_json."
                )

            if not isinstance(raw_example, dict):
                raise ValueError(
                    f"Dynamic strategy {strategy_name!r} expected raw_example_json to decode "
                    f"to a dict, but got {type(raw_example).__name__}. "
                    "Fix contextualization_manager.json_dumps_safe so it preserves dict/list structure."
                )

            manager = ContextualizationManager(strategy_name)
            teacher_prompt = manager.build_teacher_prompt_runtime(
                dataset_name=source_dataset,
                raw_example=raw_example,
                reference=reference,
                student_response=seed_response,
                feedback_text=feedback_text,
                sibling_text=sibling_text,
                prompt_variation=prompt_variation,
            )
            teacher_prompts.append(teacher_prompt)

        return teacher_prompts

    def _build_runtime_feedback_prompts(
        self,
        *,
        inputs: list[dict[str, Any]],
        seed_responses: list[str],
    ) -> list[Any | None]:
        feedback_prompts = []

        for example, seed_response in zip(inputs, seed_responses):
            strategy_name = self._get_example_strategy_name(example)
            manager = ContextualizationManager(strategy_name)

            if not manager.strategy.requires_feedback_generation:
                feedback_prompts.append(None)
                continue

            source_dataset = self._get_example_source_dataset(example)
            raw_example = json_loads_safe(example.get("raw_example_json"))
            reference = json_loads_safe(example.get("reference_json"))

            stage1_variation = example.get("context_stage1_variation")

            if not source_dataset or raw_example is None:
                raise ValueError(
                    f"Feedback strategy {strategy_name!r} requires source_dataset, "
                    "raw_example_json, and reference_json."
                )

            if not isinstance(raw_example, dict):
                raise ValueError(
                    f"Feedback strategy {strategy_name!r} expected raw_example_json to decode "
                    f"to a dict, but got {type(raw_example).__name__}. "
                    "Fix contextualization_manager.json_dumps_safe so it preserves dict/list structure."
                )

            feedback_prompt = manager.build_feedback_prompt_runtime(
                dataset_name=source_dataset,
                raw_example=raw_example,
                reference=reference,
                student_response=seed_response,
                stage1_variation=stage1_variation,
            )
            feedback_prompts.append(feedback_prompt)

        return feedback_prompts

    def _generate_auxiliary_text_with_vllm_loaded_model(
        self,
        *,
        prompts: list[Any],
        images: Optional[list],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> list[str]:
        """
        Generate auxiliary text with the model currently loaded in vLLM.

        Important:
        - uses _generate_single_turn, not _generate;
        - therefore it does not update training token counters or completion metrics;
        - supports feedback-specific max tokens and sampling parameters.
        """
        _prompt_ids, completion_ids, _logprobs, _forward_kwargs = self._generate_single_turn(
            prompts,
            images,
            max_tokens_override=max_new_tokens,
            temperature_override=temperature,
            top_p_override=top_p,
            disable_weight_sync=True,
        )
        return self._decode_completion_ids_list(completion_ids)

    @torch.no_grad()
    def _generate_auxiliary_text_with_hf_model(
        self,
        *,
        model,
        prompts: list[Any],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        batch_size: int | None = None,
    ) -> list[str]:
        """
        HF fallback used only when the requested auxiliary model is not the one
        currently loaded in vLLM.
        """
        if model is None:
            raise ValueError("Cannot generate auxiliary text with model=None.")

        batch_size = batch_size or self.args.per_device_train_batch_size
        model_was_training = model.training
        model.eval()

        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.pad_token_id,
            "eos_token_id": self.eos_token_id,
        }

        if temperature > 0:
            generation_kwargs["do_sample"] = True
            generation_kwargs["temperature"] = temperature
            generation_kwargs["top_p"] = top_p
        else:
            generation_kwargs["do_sample"] = False

        # If this is the trainable model, use model_wrapped. Otherwise use the
        # already-prepared ref_model.
        model_for_generation = self.model_wrapped if model is self.model else model

        outputs_text = []

        with unwrap_model_for_generation(
            model_for_generation,
            self.accelerator,
            gather_deepspeed3_params=self.args.ds3_gather_for_generation,
        ) as unwrapped_model:
            for start in range(0, len(prompts), batch_size):
                batch_prompts = prompts[start : start + batch_size]

                tokenized = self._prepare_text_prompt_batch(
                    batch_prompts,
                    padding_side="left",
                )
                tokenized = super()._prepare_inputs(tokenized)

                generated = unwrapped_model.generate(
                    **tokenized,
                    **generation_kwargs,
                )

                prompt_length = tokenized["input_ids"].shape[1]

                for i in range(generated.size(0)):
                    completion_ids = generated[i, prompt_length:]
                    outputs_text.append(
                        self.processing_class.decode(
                            completion_ids,
                            skip_special_tokens=True,
                        )
                    )

        if model_was_training:
            model.train()

        return outputs_text

    def _can_use_vllm_for_auxiliary_source(self, source: str) -> bool:
        if not self.use_vllm:
            return False

        if source == "generation_model":
            return True

        if source == "teacher":
            return bool(self.generate_from_teacher)

        if source == "student":
            return not bool(self.generate_from_teacher)

        return False

    def _generate_auxiliary_text(
        self,
        *,
        source: str,
        prompts: list[Any],
        images: Optional[list],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> list[str]:
        """
        Generate auxiliary text from teacher/student/generation_model.

        vLLM is used only when the requested source matches the model currently
        loaded in vLLM. Otherwise we use HF fallback to avoid loading a second
        vLLM engine.
        """
        if self._can_use_vllm_for_auxiliary_source(source):
            return self._generate_auxiliary_text_with_vllm_loaded_model(
                prompts=prompts,
                images=images,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )

        if source == "teacher":
            return self._generate_auxiliary_text_with_hf_model(
                model=self.ref_model,
                prompts=prompts,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )

        if source == "student":
            return self._generate_auxiliary_text_with_hf_model(
                model=self.model,
                prompts=prompts,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )

        if source == "generation_model":
            raise ValueError(
                "feedback_model_source='generation_model' requires use_vllm=True "
                "or a generation backend matching the currently loaded model."
            )

        raise ValueError(
            f"Unknown feedback_model_source={source!r}. "
            "Expected 'teacher', 'student', or 'generation_model'."
        )
    
    def _is_context_response_correct(
        self,
        *,
        example: dict[str, Any],
        response: str | None,
    ) -> bool:
        """
        Return whether a generated response is correct for its source dataset.

        Used by SDPO-like strategies to avoid generating siblings or feedback
        when the original student rollout is already successful.
        """
        response = str(response or "").strip()
        if not response:
            return False

        source_dataset, raw_example, reference = self._load_context_example_parts(
            example,
            purpose="Runtime response correctness check",
        )
        adapter = get_dataset_adapter(source_dataset)

        scores = adapter.score_responses_for_context(
            [response],
            raw_example=raw_example,
            reference=reference,
        )
        return bool(scores and int(scores[0]) == 1)

    def _load_context_example_parts(
        self,
        example: dict[str, Any],
        *,
        purpose: str,
    ) -> tuple[str, dict[str, Any], Any]:
        source_dataset = self._get_example_source_dataset(example)
        raw_example = json_loads_safe(example.get("raw_example_json"))
        reference = json_loads_safe(example.get("reference_json"))

        if not source_dataset or raw_example is None:
            raise ValueError(
                f"{purpose} requires source_dataset, raw_example_json, "
                "and reference_json."
            )

        if not isinstance(raw_example, dict):
            raise ValueError(
                f"{purpose} expected raw_example_json to decode to a dict, "
                f"but got {type(raw_example).__name__}. "
                "Fix contextualization_manager.json_dumps_safe so it preserves "
                "dict/list structure."
            )

        return source_dataset, raw_example, reference

    def _generate_sibling_candidate_responses(
        self,
        *,
        prompts: list[Any],
        images: Optional[list],
        active_indices: list[int],
    ) -> dict[int, list[str]]:
        """
        Generate additional sibling responses from the same prompts and the same
        generation backend/parameters used for the ordinary student rollout.

        This intentionally does not use:
          - feedback_model_source;
          - feedback prompts;
          - gold/reference information;
          - previous student responses;
          - sibling-specific temperature/top-p/max-token overrides.

        The implementation repeats prompts and performs one batched generation
        call. With use_vllm=True, this uses the currently loaded vLLM model.
        """
        num_rollouts = int(getattr(self.args, "sibling_num_rollouts", 8))
        if num_rollouts < 1:
            raise ValueError(
                f"sibling_num_rollouts must be >= 1, got {num_rollouts}."
            )

        repeated_prompts: list[Any] = []
        owner_indices: list[int] = []

        repeated_images = None if images is None else []

        for idx in active_indices:
            for _ in range(num_rollouts):
                repeated_prompts.append(prompts[idx])
                owner_indices.append(idx)

                if repeated_images is not None:
                    repeated_images.append(images[idx])

        if not repeated_prompts:
            return {idx: [] for idx in active_indices}

        _prompt_ids, completion_ids, _logprobs, _forward_kwargs = self._generate_single_turn(
            repeated_prompts,
            repeated_images,
            disable_weight_sync=True,
        )

        decoded = self._decode_completion_ids_list(completion_ids)

        candidates_by_index: dict[int, list[str]] = {
            idx: [] for idx in active_indices
        }
        for idx, text in zip(owner_indices, decoded):
            candidates_by_index[idx].append((text or "").strip())

        return candidates_by_index

    def _select_sibling_response_texts(
        self,
        *,
        inputs: list[dict[str, Any]],
        prompts: list[Any],
        images: Optional[list],
        active_indices: list[int],
    ) -> dict[int, str | None]:
        """
        Select one correct sibling response per active example.

        Selection policy:
          1. Generate sibling candidates from the original prompt.
          2. Score candidates with the dataset adapter.
          3. Keep the first candidate that is both correct and structurally
             usable as an in-context example.
          4. Either fall back to the full gold/reference response or return None, depending on the strategy's sibling_fallback_to_gold setting.
        """
        if not active_indices:
            return {}

        if self.generate_from_teacher:
            raise ValueError(
                "Sibling-based contextualization strategies are designed for unprivileged "
                "student-rollout exploration and requires "
                "generate_from_teacher=False."
            )

        candidates_by_index = self._generate_sibling_candidate_responses(
            prompts=prompts,
            images=images,
            active_indices=active_indices,
        )

        selected_by_index: dict[int, str | None] = {}

        stats = getattr(
            self,
            "_sibling_selection_stats",
            {
                "total": 0,
                "selected_generated": 0,
                "fallback_gold": 0,
                "total_candidates": 0,
                "total_correct_candidates": 0,
                "total_correct_usable_candidates": 0,
            },
        )

        for idx in active_indices:
            example = inputs[idx]
            source_dataset, raw_example, reference = self._load_context_example_parts(
                example,
                purpose="Sibling response selection",
            )
            adapter = get_dataset_adapter(source_dataset)

            candidates = [
                candidate
                for candidate in candidates_by_index.get(idx, [])
                if candidate
            ]

            scores = adapter.score_responses_for_context(
                candidates,
                raw_example=raw_example,
                reference=reference,
            )

            selected_text: str | None = None
            correct_count = 0
            correct_usable_count = 0

            for candidate, score in zip(candidates, scores):
                is_correct = int(score) == 1
                if is_correct:
                    correct_count += 1

                if not is_correct:
                    continue

                is_usable = adapter.is_usable_context_response_example(
                    candidate,
                    raw_example=raw_example,
                    reference=reference,
                )
                if not is_usable:
                    continue

                correct_usable_count += 1
                selected_text = candidate
                break

            stats["total"] += 1
            stats["total_candidates"] += len(candidates)
            stats["total_correct_candidates"] += correct_count
            stats["total_correct_usable_candidates"] += correct_usable_count

            if selected_text:
                selected_by_index[idx] = selected_text
                stats["selected_generated"] += 1
            else:
                strategy_name = self._get_example_strategy_name(example)
                manager = ContextualizationManager(strategy_name)

                if getattr(manager.strategy, "sibling_fallback_to_gold", True):
                    selected_by_index[idx] = adapter.get_fallback_response_for_context(
                        raw_example,
                        reference,
                    )
                    stats["fallback_gold"] += 1
                else:
                    selected_by_index[idx] = None
                    stats["no_sibling_available"] = (
                        stats.get("no_sibling_available", 0) + 1
                    )

        self._sibling_selection_stats = stats
        return selected_by_index
    
    def _generate_sibling_context_texts(
        self,
        *,
        inputs: list[dict[str, Any]],
        seed_responses: list[str],
        prompts: list[Any],
        images: Optional[list],
    ) -> list[str | None]:
        """
        Build selected sibling context responses for sibling-response strategies.

        Sibling selection is independent from feedback generation. Sibling
        candidates are sampled from the original prompt under the same rollout
        conditions as the ordinary student completion, without privileged
        information.
        """
        sibling_indices: list[int] = []

        for idx, example in enumerate(inputs):
            strategy_name = self._get_example_strategy_name(example)
            manager = ContextualizationManager(strategy_name)
            strategy = manager.strategy

            if not getattr(strategy, "requires_sibling_selection", False):
                continue

            if getattr(strategy, "sibling_only_if_student_incorrect", False):
                if self._is_context_response_correct(
                    example=example,
                    response=seed_responses[idx],
                ):
                    continue

            sibling_indices.append(idx)

        sibling_texts: list[str | None] = [None for _ in inputs]

        if not sibling_indices:
            return sibling_texts

        selected_siblings = self._select_sibling_response_texts(
            inputs=inputs,
            prompts=prompts,
            images=images,
            active_indices=sibling_indices,
        )

        for idx, text in selected_siblings.items():
            sibling_texts[idx] = text

        return sibling_texts
    
    def _resolve_feedback_max_new_tokens_for_example(
        self,
        example: dict[str, Any],
    ) -> int:
        """
        Resolve per-example max_new_tokens for auxiliary feedback/rationale/rewrite
        generation.

        Default resolution:
            strategy.feedback_max_new_tokens is None
            -> args.feedback_max_new_tokens

        Strategy-specific overrides can use:
            - an integer;
            - "dataset_default", meaning adapter.default_max_new_tokens.
        """
        strategy_name = self._get_example_strategy_name(example)
        source_dataset, _raw_example, _reference = self._load_context_example_parts(
            example,
            purpose="Feedback max_new_tokens resolution",
        )

        adapter = get_dataset_adapter(source_dataset)
        manager = ContextualizationManager(strategy_name)

        configured_max_new_tokens = int(
            getattr(self.args, "feedback_max_new_tokens", 256)
        )

        return manager.strategy.resolve_feedback_max_new_tokens(
            dataset_name=source_dataset,
            adapter=adapter,
            configured_max_new_tokens=configured_max_new_tokens,
        )
    
    def _generate_feedback_texts(
        self,
        *,
        inputs: list[dict[str, Any]],
        seed_responses: list[str],
        sibling_texts: list[str | None] | None = None,
    ) -> list[str | None]:
        """
        Generate feedback/rationale/rewrite text for strategies that require
        feedback generation.

        For normal feedback strategies, behavior is unchanged.

        For SDPO-like strategies with
        feedback_only_without_correct_solution=True, feedback is generated only
        when the main rollout is incorrect and no correct sibling solution is
        available.
        """
        feedback_prompts = self._build_runtime_feedback_prompts(
            inputs=inputs,
            seed_responses=seed_responses,
        )

        active_indices = [
            i for i, prompt in enumerate(feedback_prompts)
            if prompt is not None
        ]

        if sibling_texts is None:
            sibling_texts = [None for _ in inputs]

        if len(sibling_texts) != len(inputs):
            raise ValueError(
                "sibling_texts must have the same length as inputs."
            )

        filtered_indices: list[int] = []

        for idx in active_indices:
            strategy_name = self._get_example_strategy_name(inputs[idx])
            manager = ContextualizationManager(strategy_name)
            strategy = manager.strategy

            if getattr(strategy, "feedback_only_without_correct_solution", False):
                if self._is_context_response_correct(
                    example=inputs[idx],
                    response=seed_responses[idx],
                ):
                    continue

                if str(sibling_texts[idx] or "").strip():
                    continue

            filtered_indices.append(idx)

        active_indices = filtered_indices

        if not active_indices:
            return [None for _ in inputs]

        source = getattr(self.args, "feedback_model_source", "teacher")
        temperature = getattr(self.args, "feedback_temperature", 0.0)
        top_p = getattr(self.args, "feedback_top_p", 1.0)

        # Different strategies in the same batch can require different
        # feedback generation lengths, especially when context_strategy is a
        # strategy pool. Group by resolved max_new_tokens so each generation
        # call uses the right cap while preserving batch efficiency when values
        # are shared.
        grouped_indices: dict[int, list[int]] = defaultdict(list)
        for idx in active_indices:
            max_new_tokens = self._resolve_feedback_max_new_tokens_for_example(
                inputs[idx]
            )
            grouped_indices[max_new_tokens].append(idx)

        feedback_texts = [None for _ in inputs]

        for max_new_tokens, group_indices in grouped_indices.items():
            group_prompts = [feedback_prompts[i] for i in group_indices]

            group_texts = self._generate_auxiliary_text(
                source=source,
                prompts=group_prompts,
                images=None,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )

            for idx, text in zip(group_indices, group_texts):
                feedback_texts[idx] = text

        return feedback_texts

    def _generate_and_score_completions(
        self, inputs: list[dict[str, Union[torch.Tensor, Any]]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        prompts = [x["prompt"] for x in inputs]
        teacher_prompts = [x["teacher_prompt"] for x in inputs]

        if "images" in inputs[0]:
            images = [example.get("images") for example in inputs]
        elif "image" in inputs[0]:
            images = [[example.get("image")] if example.get("image") is not None else None for example in inputs]
        else:
            images = None
        # Transformers requires at least one image in the batch, otherwise it throws an error
        if images is not None and all(img_list == [] for img_list in images):
            images = None

        has_dynamic_strategy = self._has_dynamic_context_strategy(inputs)

        if not has_dynamic_strategy:
            # Original behavior.
            generation_prompts = teacher_prompts if self.generate_from_teacher else prompts

            (
                _generation_prompt_ids_list,
                completion_ids_list,
                num_items_in_batch,
                sampling_per_token_logps_list,
                forward_kwargs,
            ) = self._generate(generation_prompts, images)

        elif not self.generate_from_teacher:
            # Student-generated self-distillation path:
            # 1. Generate the actual training completion from the normal prompt.
            # 2. Use that completion as the seed response for dynamic teacher context.
            # 3. Generate feedback/rationale if the strategy requires it.
            # 4. Replace teacher_prompts before teacher log-prob computation.
            (
                _generation_prompt_ids_list,
                completion_ids_list,
                num_items_in_batch,
                sampling_per_token_logps_list,
                forward_kwargs,
            ) = self._generate(prompts, images)
            seed_responses = self._decode_completion_ids_list(completion_ids_list)

            sibling_texts = self._generate_sibling_context_texts(
                inputs=inputs,
                seed_responses=seed_responses,
                prompts=prompts,
                images=images,
            )

            feedback_texts = self._generate_feedback_texts(
                inputs=inputs,
                seed_responses=seed_responses,
                sibling_texts=sibling_texts,
            )

            teacher_prompts = self._build_runtime_teacher_prompts(
                inputs=inputs,
                seed_responses=seed_responses,
                feedback_texts=feedback_texts,
                sibling_texts=sibling_texts,
            )
        else:
            # Teacher-generated path:
            # 1. Generate a non-contextualized seed response only for feedback.
            # 2. Build final dynamic teacher prompts.
            # 3. Generate the actual training completion from those teacher prompts.
            if self._has_sibling_context_strategy(inputs):
                raise ValueError(
                    "Sibling-based contextualization strategies require "
                    "generate_from_teacher=False. Sibling candidates must be "
                    "sampled from the same unprivileged student rollout "
                    "conditions as the ordinary training completion."
                )
            source = getattr(self.args, "feedback_model_source", "teacher")

            seed_responses = self._generate_auxiliary_text(
                source=source,
                prompts=prompts,
                images=images,
                max_new_tokens=self.max_completion_length,
                temperature=self.temperature,
                top_p=self.top_p,
            )

            sibling_texts = [None for _ in inputs]

            feedback_texts = self._generate_feedback_texts(
                inputs=inputs,
                seed_responses=seed_responses,
                sibling_texts=sibling_texts,
            )

            teacher_prompts = self._build_runtime_teacher_prompts(
                inputs=inputs,
                seed_responses=seed_responses,
                feedback_texts=feedback_texts,
                sibling_texts=sibling_texts,
            )

            (
                _generation_prompt_ids_list,
                completion_ids_list,
                num_items_in_batch,
                sampling_per_token_logps_list,
                forward_kwargs,
            ) = self._generate(teacher_prompts, images)

        completion_lengths = torch.tensor(
            [len(ids) for ids in completion_ids_list],
            #device=device,
            dtype=torch.long,
        )

        self._maybe_dump_teacher_prompts( # Maybe remove this later
            inputs=inputs,
            teacher_prompts=teacher_prompts,
            tag="generate_and_score",
            feedback_texts=feedback_texts if has_dynamic_strategy else None,
            sibling_texts=sibling_texts if has_dynamic_strategy else None,
        )

        # Process student prompts (always used for student training, regardless of generation source)
        if self.use_vllm:
            self.processing_class.truncation_side = "left"
        student_inputs = self._prepare_text_prompt_batch(
            prompts,
            padding_side="left",
        )
        student_inputs = super()._prepare_inputs(student_inputs)
        student_prompt_ids, student_prompt_mask = student_inputs["input_ids"], student_inputs["attention_mask"]
        prompt_ids_list = [p[m].tolist() for p, m in zip(student_prompt_ids, student_prompt_mask.bool())]

        # Process teacher prompts (always used for teacher, regardless of generation source)
        teacher_inputs = self._prepare_text_prompt_batch(
            teacher_prompts,
            padding_side="left",
        )
        teacher_inputs = super()._prepare_inputs(teacher_inputs)
        if self.use_vllm:
            self.processing_class.truncation_side = "right"
        teacher_prompt_ids, teacher_prompt_mask = teacher_inputs["input_ids"], teacher_inputs["attention_mask"]
        teacher_prompt_ids_list = [p[m].tolist() for p, m in zip(teacher_prompt_ids, teacher_prompt_mask.bool())]

        # Convert lists of token IDs to padded tensors
        prompt_ids = [torch.tensor(ids, device=device) for ids in prompt_ids_list]
        prompt_mask = [torch.ones_like(ids, dtype=torch.long) for ids in prompt_ids]
        prompt_ids = pad(prompt_ids, padding_value=self.pad_token_id, padding_side="left")
        prompt_mask = pad(prompt_mask, padding_value=0, padding_side="left")
        teacher_prompt_ids = [torch.tensor(ids, device=device) for ids in teacher_prompt_ids_list]
        teacher_prompt_mask = [torch.ones_like(ids, dtype=torch.long) for ids in teacher_prompt_ids]
        teacher_prompt_ids = pad(teacher_prompt_ids, padding_value=self.pad_token_id, padding_side="left")
        teacher_prompt_mask = pad(teacher_prompt_mask, padding_value=0, padding_side="left")
        completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids_list]
        completion_mask = [torch.ones_like(ids, dtype=torch.long) for ids in completion_ids]
        completion_ids = pad(completion_ids, padding_value=self.pad_token_id, padding_side="right")
        completion_mask = pad(completion_mask, padding_value=0, padding_side="right")
        if sampling_per_token_logps_list is not None:
            sampling_per_token_logps = [torch.tensor(logps, device=device) for logps in sampling_per_token_logps_list]
            sampling_per_token_logps = pad(sampling_per_token_logps, padding_value=0.0, padding_side="right")
        else:
            sampling_per_token_logps = None

        # If mask_truncated_completions is enabled, zero out truncated completions in completion_mask
        if self.mask_truncated_completions:
            eos_and_pad = [self.eos_token_id, self.pad_token_id]
            is_truncated = torch.tensor([ids[-1] not in eos_and_pad for ids in completion_ids_list], device=device)
            completion_mask = completion_mask * (~is_truncated).unsqueeze(1).int()

        # Concatenate prompt_mask with completion_mask for logit computation
        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)  # (B, P+C)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)
        teacher_prompt_completion_ids = torch.cat([teacher_prompt_ids, completion_ids], dim=1)  # (B, P+C)
        teacher_attention_mask = torch.cat([teacher_prompt_mask, completion_mask], dim=1)  # (B, P+C)
        # If token_type_ids are used, extend them with zeros for the completion part
        if "token_type_ids" in forward_kwargs:
            token_type_ids = forward_kwargs["token_type_ids"]
            forward_kwargs["token_type_ids"] = torch.cat(
                [token_type_ids, token_type_ids.new_zeros(completion_ids.shape)], dim=1
            )

        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens
        batch_size = self.args.per_device_train_batch_size if mode == "train" else self.args.per_device_eval_batch_size

        num_images = [len(img_list) for img_list in images] if images is not None else None

        with torch.no_grad():
            # If the generation and optimization steps are misaligned—i.e., if generation does not occur at the end of
            # a full optimizer step (when gradient_accumulation_steps is not a multiple of generate_every)—then the
            # samples may come from an earlier version of the model. In that case, we need to track old_per_token_logps
            # for importance sampling. If the steps are aligned, importance sampling isn't necessary and we set
            # old_per_token_logps to None.
            # When using vLLM, we always compute old_per_token_logps for importance sampling, it was shown that the
            # distribution mismatch between vLLM and the training model can be large and harm the training.
            # Skip when generate_from_teacher=True since importance sampling is not used in that case.
            generate_every = self.args.steps_per_generation * self.num_iterations  # generation frequency
            if not self.generate_from_teacher and (
                self.args.gradient_accumulation_steps % generate_every != 0 or (
                self.use_vllm and self.vllm_importance_sampling_correction)):
                old_per_token_logps, _, _ = self._get_per_token_logps_and_entropies(
                    self.model,
                    prompt_completion_ids,
                    attention_mask,
                    logits_to_keep,
                    batch_size,
                    num_images=num_images,
                    compute_all_logps=False,
                    **forward_kwargs,  # may contain pixel_values, image_grid_thw, pixel_attention_mask and image_sizes
                )
            else:
                old_per_token_logps = None

            # Compute the importance sampling ratio when using vLLM, to correct for potential distribution mismatch
            # Skip when generate_from_teacher=True since vLLM has teacher weights (no mismatch to correct)
            if self.use_vllm and self.vllm_importance_sampling_correction and not self.generate_from_teacher:
                importance_sampling_ratio = torch.exp(old_per_token_logps - sampling_per_token_logps)
                importance_sampling_ratio = torch.clamp(
                    importance_sampling_ratio, max=self.vllm_importance_sampling_cap
                )
            else:
                importance_sampling_ratio = None

            # Compute the per-token log probabilities for the reference model
            if self.beta != 0.0:
                if self.ref_model is not None:
                    ref_per_token_logps, _, _ = self._get_per_token_logps_and_entropies(
                        self.ref_model,
                        prompt_completion_ids,
                        attention_mask,
                        logits_to_keep,
                        batch_size=batch_size,
                        num_images=num_images,
                        compute_all_logps=False,
                        **forward_kwargs,  # may contain pixel_values, image_grid_thw, pixel_attention_mask and image_sizes
                    )
                else:
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        ref_per_token_logps, _, _ = self._get_per_token_logps_and_entropies(
                            self.model,
                            prompt_completion_ids,
                            attention_mask,
                            logits_to_keep,
                            batch_size=batch_size,
                            num_images=num_images,
                            compute_all_logps=False,
                            **forward_kwargs,  # may contain pixel_values, image_grid_thw, pixel_attention_mask and image_sizes
                        )   
            else:
                ref_per_token_logps = None

        # Decode
        prompts_text = self.processing_class.batch_decode(prompt_ids, skip_special_tokens=True)
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                prompt_copy = list(prompt)
                bootstrap = (
                    prompt_copy.pop()["content"]
                    if prompt_copy and prompt_copy[-1]["role"] == "assistant"
                    else ""
                )
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text
        
        # Not really necessary, but keeping for now
        rewards = torch.zeros_like(completion_ids, dtype=torch.float32)
        advantages = rewards
        
        # Keep a copy for logging (data is already local to each process, no slicing needed)
        all_process_advantages = advantages.clone()

        # Log prompt and completion texts
        self._logs["prompt"].extend(gather_object(prompts_text))
        self._logs["completion"].extend(gather_object(completions_text))
        self._logs["rewards"]["main"].extend(gather_object(rewards.mean(dim=-1).tolist()))
        self._logs["advantages"].extend(gather_object(all_process_advantages.mean(dim=-1).tolist()))
        reward_to_log = rewards.clone()
        reward_to_log = reward_to_log[completion_mask.bool()]
        mean_reward = torch.mean(reward_to_log) if reward_to_log.numel() > 0 else torch.tensor(0.0, device=device)
        self._metrics[mode]["rewards"].append(self.accelerator.gather(mean_reward).mean().item())

        if images is not None:
            self._logs["images"].extend(gather_object(images))

        if importance_sampling_ratio is not None:
            delta = torch.abs(old_per_token_logps - sampling_per_token_logps)
            delta = delta[completion_mask.bool()]
            mean_delta = torch.mean(delta) if delta.numel() > 0 else torch.tensor(0.0, device=device)
            max_delta = torch.max(delta) if delta.numel() > 0 else torch.tensor(0.0, device=device)
            self._metrics[mode]["sampling/sampling_logp_difference/mean"].append(
                self.accelerator.gather(mean_delta).mean().item()
            )
            self._metrics[mode]["sampling/sampling_logp_difference/max"].append(
                self.accelerator.gather(max_delta).max().item()
            )

            flat_is_ratio = importance_sampling_ratio[completion_mask.bool()]
            min_importance_sampling_ratio = (
                torch.min(flat_is_ratio) if flat_is_ratio.numel() > 0 else torch.tensor(0.0, device=device)
            )
            mean_importance_sampling_ratio = (
                torch.mean(flat_is_ratio) if flat_is_ratio.numel() > 0 else torch.tensor(0.0, device=device)
            )
            max_importance_sampling_ratio = (
                torch.max(flat_is_ratio) if flat_is_ratio.numel() > 0 else torch.tensor(0.0, device=device)
            )
            self._metrics[mode]["sampling/importance_sampling_ratio/min"].append(
                nanmin(self.accelerator.gather(min_importance_sampling_ratio)).item()
            )
            self._metrics[mode]["sampling/importance_sampling_ratio/mean"].append(
                self.accelerator.gather(mean_importance_sampling_ratio).nanmean().item()
            )
            self._metrics[mode]["sampling/importance_sampling_ratio/max"].append(
                nanmax(self.accelerator.gather(max_importance_sampling_ratio)).item()
            )

        output = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "teacher_prompt_ids": teacher_prompt_ids,
            "teacher_prompt_mask": teacher_prompt_mask,
            "advantages": advantages,
            "num_items_in_batch": num_items_in_batch,
            "_completion_lengths": completion_lengths,
        }
        if old_per_token_logps is not None:
            output["old_per_token_logps"] = old_per_token_logps
        if importance_sampling_ratio is not None:
            output["importance_sampling_ratio"] = importance_sampling_ratio
        if ref_per_token_logps is not None:
            output["ref_per_token_logps"] = ref_per_token_logps
        if "pixel_values" in forward_kwargs:
            output["pixel_values"] = forward_kwargs["pixel_values"]
        if "image_grid_thw" in forward_kwargs:
            output["image_grid_thw"] = forward_kwargs["image_grid_thw"]
        if "pixel_attention_mask" in forward_kwargs:
            output["pixel_attention_mask"] = forward_kwargs["pixel_attention_mask"]
        if "image_sizes" in forward_kwargs:
            output["image_sizes"] = forward_kwargs["image_sizes"]
        if "token_type_ids" in forward_kwargs:
            output["token_type_ids"] = forward_kwargs["token_type_ids"]
        if images is not None:
            output["num_images"] = num_images

        return output


    @profiling_decorator
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The DistilTrainer does not support returning outputs")
        return self._compute_loss(model, inputs)

    def _compute_loss(self, model, inputs):
        # Compute the per-token log probabilities for the model
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        teacher_prompt_ids, teacher_prompt_mask = inputs["teacher_prompt_ids"], inputs["teacher_prompt_mask"]
        
        # Create a separate mask for loss computation that skips the first N tokens
        # Note: completion_mask is used for both attention (forward pass) and loss computation
        # We need to keep the original for attention, but create a modified one for loss
        loss_completion_mask = completion_mask
        if self.num_loss_tokens_to_skip > 0:
            batch_size, seq_len = completion_mask.shape
            # Create a mask that is 0 for the first num_loss_tokens_to_skip tokens and 1 elsewhere
            token_positions = torch.arange(seq_len, device=completion_mask.device).unsqueeze(0).expand(batch_size, -1)
            skip_mask = (token_positions >= self.num_loss_tokens_to_skip).int()
            # Apply the skip mask (only mask tokens that were originally unmasked)
            loss_completion_mask = completion_mask * skip_mask
        
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        teacher_input_ids = torch.cat([teacher_prompt_ids, completion_ids], dim=1)
        teacher_attention_mask = torch.cat([teacher_prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        # Compute the per_token_logps and the entropy at each position in the completion
        per_token_logps, all_logps, entropies = self._get_per_token_logps_and_entropies(
            model,
            input_ids,
            attention_mask,
            logits_to_keep,
            compute_entropy=True,
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            num_images=inputs.get("num_images"),
            pixel_attention_mask=inputs.get("pixel_attention_mask"),
            image_sizes=inputs.get("image_sizes"),
            token_type_ids=inputs.get("token_type_ids"),
        )

        if self.top_entropy_quantile < 1.0:
            entropy_mask = self.get_high_entropy_mask(entropies, loss_completion_mask, 1 - self.top_entropy_quantile)
        else:
            entropy_mask = None

        if self.use_cross_entropy:
            per_token_loss = -per_token_logps

            if entropy_mask is not None:
                per_token_loss = per_token_loss * entropy_mask

            loss = ((per_token_loss * loss_completion_mask).sum(-1) / loss_completion_mask.sum(-1).clamp(min=1.0)).mean()
            loss = loss / self.current_gradient_accumulation_steps

            mode = "train" if self.model.training else "eval"
            mean_entropy = ((entropies * loss_completion_mask).sum() / loss_completion_mask.sum().clamp(min=1.0))
            self._metrics[mode]["entropy"].append(self.accelerator.gather(mean_entropy).nanmean().item())
            self._metrics[mode]["ce_loss"].append(self.accelerator.gather(loss.detach()).mean().item())

            return loss

        with torch.no_grad():
            teacher_per_token_logps, teacher_all_logps, teacher_entropies = self._get_per_token_logps_and_entropies(
                self.ref_model,
                teacher_input_ids,
                teacher_attention_mask,
                logits_to_keep,
                compute_entropy=True,
                pixel_values=inputs.get("pixel_values"),
                image_grid_thw=inputs.get("image_grid_thw"),
                num_images=inputs.get("num_images"),
                pixel_attention_mask=inputs.get("pixel_attention_mask"),
                image_sizes=inputs.get("image_sizes"),
                token_type_ids=inputs.get("token_type_ids"),
            )


        # Compute the KL divergence between the model and the reference model
        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )
        
        # Compute KL divergences using F.kl_div
        # PyTorch differs from the standard mathematical definition, so the order of the probability distributions is swapped compared to that defined in the paper.
        if self.alpha == 0: #Forward KL
            kl_loss = kl_div(all_logps, teacher_all_logps, reduction="none", log_target=True)
        elif self.alpha == 1: #Reverse KL
            kl_loss = kl_div(teacher_all_logps, all_logps, reduction="none", log_target=True)
        else:
            # Compute the log of the mixture distribution
            # log(a + b) = log(exp(log(a)) + exp(log(b))) -> for mixture
            alpha = torch.tensor(self.alpha, dtype=all_logps.dtype)
            mixture_log_probs = torch.logsumexp(
                torch.stack([all_logps + torch.log(1 - alpha), teacher_all_logps + torch.log(alpha)]),
                dim=0,
            )

            kl_teacher = kl_div(mixture_log_probs, teacher_all_logps, reduction="none", log_target=True)
            kl_student = kl_div(mixture_log_probs, all_logps, reduction="none", log_target=True)

            # Compute the Generalized Jensen-Shannon Divergence
            kl_loss = alpha * kl_teacher + (1 - alpha) * kl_student
        per_token_loss = kl_loss.sum(-1)

        if self.use_vllm and self.vllm_importance_sampling_correction and not self.generate_from_teacher:
            ratio = inputs["importance_sampling_ratio"]
            importance_weights = (ratio * loss_completion_mask).sum(-1) / loss_completion_mask.sum(-1).clamp(min=1.0)
            importance_weights = importance_weights.unsqueeze(-1)
            per_token_loss = per_token_loss * importance_weights

        if entropy_mask is not None:
            per_token_loss = per_token_loss * entropy_mask

        loss = ((per_token_loss * loss_completion_mask).sum(-1) / loss_completion_mask.sum(-1).clamp(min=1.0)).mean()
        loss = loss / self.current_gradient_accumulation_steps

        # Log the metrics
        mode = "train" if self.model.training else "eval"

        with torch.no_grad():
            kl_approx = (per_token_logps - teacher_per_token_logps) + torch.exp(teacher_per_token_logps - per_token_logps) - 1
            kl_approx_mean = (kl_approx * loss_completion_mask).sum() / loss_completion_mask.sum()
        self._metrics[mode]["kl_approx"].append(self.accelerator.gather(kl_approx_mean).nanmean().item())
        
        loss_completion_token_count = loss_completion_mask.sum().clamp(min=1.0)

        def masked_batch_mean(x):
            if x.shape[1] == 1:  # when importance_sampling_level == "sequence"
                return x.mean()
            else:
                return (x * loss_completion_mask).sum() / loss_completion_token_count

        if self.beta != 0.0:
            mean_kl = masked_batch_mean(per_token_kl)
            self._metrics[mode]["kl_to_base_model"].append(self.accelerator.gather(mean_kl).nanmean().item())

        mean_entropy = masked_batch_mean(entropies)
        mean_teacher_entropy = masked_batch_mean(teacher_entropies)
        self._metrics[mode]["entropy"].append(self.accelerator.gather(mean_entropy).nanmean().item())
        self._metrics[mode]["teacher_entropy"].append(
            self.accelerator.gather(mean_teacher_entropy).nanmean().item()
        )
        self._metrics[mode]["entropy_delta_teacher_minus_student"].append(
            self.accelerator.gather(mean_teacher_entropy - mean_entropy).nanmean().item()
        )

        return loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys: Optional[list[str]] = None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            loss = loss.mean().detach()
        return loss, None, None

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        mode = "train" if self.model.training else "eval"
        metrics = {key: sum(val) / len(val) for key, val in self._metrics[mode].items()}  # average the metrics

        # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
        # start with "eval_". We need to add the prefix "eval_" to the keys in `metrics` to match the format.
        if mode == "eval":
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs = {**logs, **metrics}
        super().log(logs, start_time)
        self._metrics[mode].clear()

        if self.accelerator.is_main_process and self.log_completions:
            if is_rich_available():
                print_prompt_completions_sample(
                    self._logs["prompt"],
                    self._logs["completion"],
                    self._logs["rewards"],
                    self._logs["advantages"],
                    self.state.global_step,
                    self.num_completions_to_print,
                )

            if self.args.report_to and "wandb" in self.args.report_to and wandb.run is not None:
                import pandas as pd

                table = {
                    "step": [str(self.state.global_step)] * len(self._logs["prompt"]),
                    "prompt": self._logs["prompt"],
                    "completion": self._logs["completion"],
                    **self._logs["rewards"],
                    "advantage": self._logs["advantages"],
                }

                if self._logs["images"]:
                    table["images"] = []
                    for image_list in self._logs["images"]:
                        # Convert images to wandb Image objects for proper visualization
                        table["images"].append([wandb.Image(image) for image in image_list])

                df = pd.DataFrame(table)
                if self.wandb_log_unique_prompts:
                    df = df.drop_duplicates(subset=["prompt"])
                wandb.log({"completions": wandb.Table(dataframe=df)})

    # Ensure the model card is saved along with the checkpoint
    def _save_checkpoint(self, model, trial):
        if self.args.hub_model_id is None:
            model_name = Path(self.args.output_dir).name
        else:
            model_name = self.args.hub_model_id.split("/")[-1]
        self.create_model_card(model_name=model_name)
        super()._save_checkpoint(model, trial)

    def _maybe_dump_teacher_prompts(
        self,
        *,
        inputs,
        teacher_prompts,
        tag: str,
        feedback_texts=None,
        sibling_texts=None,
    ) -> None:  # Maybe delete this in the future
        """
        Minimal JSONL dump of a subset of teacher prompts used by the trainer.

        Enable with:
            DUMP_TEACHER_PROMPTS=1

        Useful options:
            DUMP_TEACHER_PROMPTS_PER_STEP=5   # save at most 5 rows per global_step
            DUMP_TEACHER_PROMPTS_MAX=1000     # global cap; -1 means no global cap
        """
        if os.environ.get("DUMP_TEACHER_PROMPTS", "0") != "1":
            return

        if hasattr(self, "is_world_process_zero") and not self.is_world_process_zero():
            return

        step = int(getattr(self.state, "global_step", -1))

        per_step = int(os.environ.get("DUMP_TEACHER_PROMPTS_PER_STEP", "5"))
        max_examples = int(os.environ.get("DUMP_TEACHER_PROMPTS_MAX", "-1"))

        if per_step == 0:
            return

        total_count = getattr(self, "_teacher_prompt_dump_count", 0)
        if max_examples >= 0 and total_count >= max_examples:
            return

        per_step_counts = getattr(self, "_teacher_prompt_dump_counts_by_step", None)
        if per_step_counts is None:
            per_step_counts = {}
            self._teacher_prompt_dump_counts_by_step = per_step_counts

        already_this_step = per_step_counts.get(step, 0)

        if per_step > 0 and already_this_step >= per_step:
            return

        remaining_step = None if per_step < 0 else per_step - already_this_step
        remaining_total = None if max_examples < 0 else max_examples - total_count

        n = len(teacher_prompts)

        if remaining_step is not None:
            n = min(n, remaining_step)

        if remaining_total is not None:
            n = min(n, remaining_total)

        if n <= 0:
            return

        path = Path(self.args.output_dir) / "analysis" / "teacher_prompts.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)

        def make_jsonable(x):
            if x is None or isinstance(x, (str, int, float, bool)):
                return x
            if isinstance(x, bytes):
                return x.decode("utf-8", errors="replace")
            if isinstance(x, dict):
                return {str(k): make_jsonable(v) for k, v in x.items()}
            if isinstance(x, (list, tuple)):
                return [make_jsonable(v) for v in x]
            if hasattr(x, "item"):
                try:
                    return make_jsonable(x.item())
                except Exception:
                    pass
            if hasattr(x, "tolist"):
                try:
                    return make_jsonable(x.tolist())
                except Exception:
                    pass
            return str(x)

        with path.open("a", encoding="utf-8") as f:
            for i in range(n):
                example = inputs[i]

                record = {
                    "global_step": step,
                    "local_index": i,
                    "tag": tag,
                    "context_strategy": example.get("context_strategy"),
                    "context_strategy_spec": example.get("context_strategy_spec"),
                    "context_strategy_selection_spec": example.get("context_strategy_selection_spec"),
                    "context_prompt_variation": example.get("context_prompt_variation"),
                    "context_base_prompt_variation": example.get("context_base_prompt_variation"),
                    "context_stage1_variation": example.get("context_stage1_variation"),
                    "context_prompt_variation_random": example.get("context_prompt_variation_random"),
                    "context_row_index": example.get("context_row_index"),
                    "context_epoch_index": self._current_epoch_index(),
                    "source_dataset": example.get("source_dataset"),
                    "student_prompt": example.get("prompt"),
                    "teacher_prompt": teacher_prompts[i],
                    "feedback_text": feedback_texts[i] if feedback_texts is not None else None,
                    "sibling_text": sibling_texts[i] if sibling_texts is not None else None,
                }

                f.write(json.dumps(make_jsonable(record), ensure_ascii=False) + "\n")

        self._teacher_prompt_dump_count = total_count + n
        per_step_counts[step] = already_this_step + n
    
    def _has_sibling_context_strategy(self, inputs: list[dict[str, Any]]) -> bool:
        for example in inputs:
            strategy_name = self._get_example_strategy_name(example)
            manager = ContextualizationManager(strategy_name)
            if getattr(manager.strategy, "requires_sibling_selection", False):
                return True
        return False