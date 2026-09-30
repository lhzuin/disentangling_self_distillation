"""Direct single-phase training entry point.

This module trains one dataset/phase with ``DistilTrainer``. It resolves the
student/reference/rollout models through ``model_registry.py``, validates the
training world size against the requested colocated-vLLM tensor-parallel size,
builds the contextualized training dataset, and constructs ``DistilConfig``.
For CSV-driven multi-phase or sweep orchestration, use ``run_experiments.py``.
"""

import os

from distil_trainer import DistilTrainer
from distil_config import DistilConfig
from dataset_adapters import get_dataset_adapter, get_dataset_names
from contextualizer.contextualized_train_loader import load_contextualized_train_dataset
from model_registry import (
    DEFAULT_MODEL_KEY,
    check_tensor_parallel_compatibility,
    load_model_and_tokenizer_hf,
    resolve_model_spec,
)
import torch
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Distil Trainer")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument(
        "--num_prompts_per_batch",
        type=int,
        default=32,
        help=(
            "Global (world-size-independent) number of prompts per optimizer step. "
            "This value is kept constant regardless of --nproc_per_node / "
            "--vllm_tensor_parallel_size: the per-process gradient_accumulation_steps "
            "is derived as num_prompts_per_batch // world_size, so scaling up the "
            "number of training processes (e.g. to satisfy a tensor-parallel "
            "vLLM group) does not silently change the effective batch size."
        ),
    )
    parser.add_argument("--max_completion_length", type=int, default=2048, help="Maximum number of completion tokens") # TODO: find out the best behavior. Currently this is the standard value always used in training, while in eval time the limits are decided by the dataset adapter classes.
    parser.add_argument("--max_prompt_length", type=int, default=2048, help="Maximum number of tokens in prompt") # TODO: idem
    parser.add_argument("--ref_model_mixup_alpha", type=float, default=0.01, help="Reference model mixup alpha")
    parser.add_argument("--output_dir", type=str, help="Output directory")
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help=(
            "Model name or path. Accepts a registered short key (see "
            "model_registry.known_model_keys(), e.g. 'qwen3.5-4b', "
            "'ministral-3-3b'), a Hugging Face repo id, or a local checkpoint "
            "directory. Qwen2.5 is the project default; the BF16 Ministral path is "
            "also validated in the current text-only workflow. Qwen3.5 is registered "
            "but still requires end-to-end validation."
        ),
    )
    parser.add_argument(
        "--vllm_tensor_parallel_size",
        type=int,
        default=1,
        help=(
            "Tensor-parallel size for the colocated vLLM generation engine. "
            "Must evenly divide the training world size (--nproc_per_node). "
            "Default: 1."
        ),
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="tooluse",
        help="Dataset name",
        choices=get_dataset_names(include_training_only=True),
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed")

    parser.add_argument("--save_strategy", type=str, default="steps", choices=["no", "steps", "epoch"], help="Checkpoint save strategy")
    parser.add_argument("--save_steps", type=int, default=100, help="Save checkpoint every N steps")
    parser.add_argument("--alpha", type=float, default=0.0, help="KL direction: 0.0=forward, 1.0=reverse")

    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Optional checkpoint path to resume training from."
    )

    parser.add_argument(
        "--sync_ref_model",
        type=lambda x: str(x).lower() in ["true", "1", "yes"],
        default=True,
        help="Whether to update the teacher/reference model during training"
    )

    parser.add_argument(
        "--teacher_name",
        type=str,
        default=None,
        help="Optional teacher/reference model name. Used only when --sync_ref_model is false. "
             "Otherwise defaults to --model_name."
    )


    parser.add_argument(
        "--generate_from_teacher",
        type=lambda x: str(x).lower() in ["true", "1", "yes"],
        default=False,
        help="Whether to sample completions from the teacher instead of the student"
    )
    parser.add_argument(
        "--ref_model_sync_steps",
        type=int,
        default=1,
        help="How often to sync the teacher/reference model"
    )

    parser.add_argument(
        "--optim",
        type=str,
        default="adamw_torch",
        help="Optimizer name, e.g. adamw_torch or paged_adamw_8bit"
    )

    parser.add_argument( 
        "--optimal_policy_source",
        type=str,
        default="teacher",
        choices=["teacher", "dataset"],
        help="defines what will be the source of the optimal policy supervision, which can be either the teacher or gold labels directly"
    )

    parser.add_argument( 
        "--optim_loss",
        type=str,
        default="jsd",
        choices=["jsd", "cross_entropy"],
        help="defines the loss function which will be used"
    )

    parser.add_argument(
        "--num_loss_tokens_to_skip",
        type=int,
        default=3,
        help="Number of completion tokens at the beginning to exclude from the loss"
    )

    parser.add_argument(
        "--context_strategy",
        type=str,
        default="dataset_default",
        help=(
            "Contextualization strategy. Examples: "
            "'dataset_default', 'gold_hint', 'gold_hint(2)', "
            "'self_feedback_concrete(1,2)', 'self_feedback_concrete(*,*)', "
            "or 'gold_hint|self_feedback_concrete'. "
            "If no variation is specified, variation 1 is used."
        ),
    )

    parser.add_argument(
        "--sibling_num_rollouts",
        type=int,
        default=8,
        help=(
            "Number of additional unprivileged sibling responses to sample "
            "from the original prompt for sibling_response_as_example."
        ),
    )

    parser.add_argument(
        "--feedback_model_source",
        type=str,
        default="teacher",
        choices=["teacher", "student", "generation_model"],
        help=(
            "Model used for first-step feedback/rationale generation. "
            "teacher uses the EMA/reference model by default."
        ),
    )

    parser.add_argument(
        "--feedback_max_new_tokens",
        type=int,
        default=256,
        help="Maximum new tokens for feedback/rationale generation.",
    )

    parser.add_argument(
        "--feedback_temperature",
        type=float,
        default=0.0,
        help="Temperature for feedback/rationale generation.",
    )

    parser.add_argument(
        "--feedback_top_p",
        type=float,
        default=1.0,
        help="Top-p for feedback/rationale generation.",
    )
    
    args = parser.parse_args()
    # Resolve the actual teacher model to load
    if args.sync_ref_model or args.teacher_name is None:
        args.effective_teacher_name = args.model_name
    else:
        args.effective_teacher_name = args.teacher_name

    return args


if __name__ == "__main__":
    args = parse_args()
    adapter = get_dataset_adapter(args.dataset_name)

    student_spec = resolve_model_spec(args.model_name)
    teacher_spec = resolve_model_spec(args.effective_teacher_name)
    generation_spec = (
        teacher_spec
        if args.generate_from_teacher
        else student_spec
    )

    if student_spec.key != DEFAULT_MODEL_KEY:
        import logging as _logging

        _logging.getLogger(__name__).warning(
            "Training with model_name=%s (registry key=%s). This model is "
            "a non-default model. Review model_registry.py and docs/ARCHITECTURE.md "
            "for model-specific runtime settings. Registry notes: %s",
            args.model_name,
            student_spec.key,
            student_spec.notes or "(no notes)",
        )

    # World size is set by the torchrun/torch.distributed.run launcher before
    # this process starts. We resolve it here (rather than trusting a fixed
    # gradient_accumulation_steps) so that --num_prompts_per_batch always
    # means the same *global* per-optimizer-step prompt count regardless of
    # how many processes are launched -- e.g. turning on a tensor-parallel
    # vLLM engine (which requires nproc_per_node to be a multiple of
    # --vllm_tensor_parallel_size) no longer silently changes the effective
    # batch size the way a fixed gradient_accumulation_steps would.
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if args.num_prompts_per_batch % world_size != 0:
        raise ValueError(
            f"--num_prompts_per_batch ({args.num_prompts_per_batch}) must be "
            f"evenly divisible by the training world size ({world_size}, from "
            "the WORLD_SIZE env var set by the launcher). Otherwise the "
            "effective global batch size would silently change with the "
            "number of processes/GPUs used."
        )
    per_process_gradient_accumulation_steps = args.num_prompts_per_batch // world_size

    if world_size % args.vllm_tensor_parallel_size != 0:
        raise ValueError(
            f"--vllm_tensor_parallel_size ({args.vllm_tensor_parallel_size}) must "
            f"evenly divide the training world size ({world_size}). Launch with "
            "a --nproc_per_node that is a multiple of --vllm_tensor_parallel_size."
        )
    check_tensor_parallel_compatibility(
        generation_spec,
        args.vllm_tensor_parallel_size,
    )

    model, processing_class = load_model_and_tokenizer_hf(
        student_spec,
        torch_dtype=torch.bfloat16,
    )
    teacher_model, _ = load_model_and_tokenizer_hf(teacher_spec, torch_dtype=torch.bfloat16)
    dataset, _ = load_contextualized_train_dataset(
        dataset_name=args.dataset_name,
        seed=args.seed,
        context_strategy=args.context_strategy,
        optimal_policy_source=args.optimal_policy_source,
    )

    config = DistilConfig(
        seed=args.seed,
        use_vllm = True,
        vllm_mode="colocate",
        vllm_tensor_parallel_size=args.vllm_tensor_parallel_size,
        vllm_gpu_memory_utilization=0.3,
        vllm_enable_sleep_mode=True, 
        learning_rate = args.learning_rate,
        #warmup_ratio = 0.1,
        warmup_steps = 10,
        lr_scheduler_type = "cosine",
        logging_steps = 1,
        bf16 = True,
        fp16 = False,
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = per_process_gradient_accumulation_steps,
        max_prompt_length = args.max_prompt_length,
        max_completion_length = args.max_completion_length,
        num_train_epochs = args.num_train_epochs,
        num_iterations = 1,
        num_generations = 1,
        max_grad_norm = 1,
        report_to = "wandb",
        output_dir = args.output_dir,
        log_completions = False, # True for debugging
        sync_ref_model=args.sync_ref_model,
        generate_from_teacher=args.generate_from_teacher,
        ref_model_sync_steps=args.ref_model_sync_steps,
        ref_model_mixup_alpha = args.ref_model_mixup_alpha,
        vllm_importance_sampling_correction = True,
        num_loss_tokens_to_skip = args.num_loss_tokens_to_skip,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        optim=args.optim,
        alpha=args.alpha,
        optimal_policy_source=args.optimal_policy_source,
        optim_loss=args.optim_loss,
        context_strategy=args.context_strategy,
        feedback_model_source=args.feedback_model_source,
        feedback_max_new_tokens=args.feedback_max_new_tokens,
        feedback_temperature=args.feedback_temperature,
        feedback_top_p=args.feedback_top_p,
        sibling_num_rollouts=args.sibling_num_rollouts,
    )
    trainer = DistilTrainer(
        model=model,
        ref_model=teacher_model,
        args=config,
        train_dataset=dataset,
        processing_class=processing_class,
        model_spec=student_spec,
        ref_model_spec=teacher_spec,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
