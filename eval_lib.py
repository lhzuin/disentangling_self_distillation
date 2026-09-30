"""Shared project-dataset evaluation utilities.

The module provides model-aware vLLM loading, chat-template preparation,
generation, adapter-based scoring, result serialization, and a standalone
single-dataset evaluation entry point. Architecture-specific behavior is
resolved from ``model_registry.py``; task-specific prompting/scoring is delegated
to ``dataset_adapters.py``.
"""

from collections.abc import Mapping
import json
import os
from typing import Any

import torch
from vllm import LLM, SamplingParams

from dataset_adapters import get_dataset_adapter
from model_registry import (
    ChatTemplateMode,
    check_tensor_parallel_compatibility,
    load_tokenizer_for_eval,
    resolve_model_spec,
    vllm_eval_engine_kwargs,
)


def load_model_and_tokenizer_vllm(
    model_path,
    gpu_memory_utilization=0.8,
    max_model_len=None,
    tensor_parallel_size: int = 1,
    seed: int | None = None,
):
    """Load a vLLM engine + tokenizer for standalone evaluation.

    `model_path` may be:
      - a registered short model key;
      - a registered Hugging Face repo id;
      - a recognized local HF Trainer checkpoint;
      - an arbitrary unregistered model/path.

    Recognized local checkpoints inherit the registered model family's
    tokenizer/runtime behavior. Unrecognized paths use the generic causal-LM
    fallback.
    """

    print(f"Loading model from {model_path}")

    spec = resolve_model_spec(model_path)
    check_tensor_parallel_compatibility(
        spec,
        tensor_parallel_size,
    )

    resolved_model_path = spec.hf_repo_id
    tokenizer = load_tokenizer_for_eval(spec)
    engine_kwargs = vllm_eval_engine_kwargs(spec)

    print(
        "Resolved eval model: "
        f"key={spec.key}, "
        f"family={spec.family.value}, "
        f"path={resolved_model_path}, "
        f"vllm_kwargs={engine_kwargs}"
    )

    llm_kwargs = dict(
        model=resolved_model_path,
        gpu_memory_utilization=gpu_memory_utilization,
        dtype=torch.bfloat16,

        # Generic evaluation defaults.
        trust_remote_code=True,

        tensor_parallel_size=tensor_parallel_size,
        **engine_kwargs,
    )

    if max_model_len is not None:
        llm_kwargs["max_model_len"] = max_model_len

    # Optional explicit vLLM seed. Baseline calibration supplies it so replicate
    # runs share the same model-loading path as normal evaluation.
    if seed is not None:
        llm_kwargs["seed"] = int(seed)

    llm = LLM(**llm_kwargs)
    return llm, tokenizer


def _extract_input_ids(tokenized_prompt: Any) -> list[int]:
    """Normalize chat-template tokenization output to one flat list of token IDs.

    Transformers tokenizers/processors may return:
      - a BatchEncoding / mapping containing ``input_ids``;
      - a tensor;
      - a flat list/tuple of token IDs;
      - a batch containing exactly one token-ID sequence.

    Evaluation templates are applied one example at a time, so batched output
    with more than one sequence is considered an error.
    """

    if isinstance(tokenized_prompt, Mapping):
        if "input_ids" not in tokenized_prompt:
            raise ValueError(
                "Tokenized chat template output does not contain 'input_ids'."
            )
        input_ids = tokenized_prompt["input_ids"]
    else:
        input_ids = tokenized_prompt

    if torch.is_tensor(input_ids):
        input_ids = input_ids.detach().cpu().tolist()
    elif (
        hasattr(input_ids, "tolist")
        and not isinstance(input_ids, (list, tuple))
    ):
        input_ids = input_ids.tolist()

    if isinstance(input_ids, tuple):
        input_ids = list(input_ids)

    if not isinstance(input_ids, list):
        raise TypeError(
            "Expected tokenized chat template output to contain a list of "
            f"token IDs, got {type(input_ids).__name__}."
        )

    # Some tokenizers/processors return a one-item batch.
    if input_ids and isinstance(input_ids[0], (list, tuple)):
        if len(input_ids) != 1:
            raise ValueError(
                "Expected one tokenized prompt, but chat template returned "
                f"a batch of {len(input_ids)} prompts."
            )
        input_ids = list(input_ids[0])

    try:
        normalized_ids = [int(token_id) for token_id in input_ids]
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "Chat template produced non-integer token IDs."
        ) from exc

    if not normalized_ids:
        raise ValueError("Chat template produced an empty token-ID sequence.")

    return normalized_ids


def _prepare_vllm_eval_prompts(
    *,
    adapter,
    tokenizer,
    eval_output,
    model_spec,
) -> list[Any]:
    """Prepare model-facing prompts according to the registered model policy.

    TEXT uses adapter-rendered prompts unless the model defines additional
    chat-template kwargs.

    TOKEN_IDS applies the chat template directly with ``tokenize=True`` and
    passes the resulting IDs to vLLM, avoiding unsafe text round-trips for
    tokenizers such as MistralCommonBackend.
    """

    mode = model_spec.chat_template_mode

    if mode == ChatTemplateMode.TEXT:
        # Reuse adapter-rendered prompts when there are no model-specific
        # chat-template requirements.
        if not model_spec.chat_template_kwargs:
            return eval_output.prompts

        # Models such as Qwen3.5 need extra template kwargs (for example,
        # enable_thinking=False), so render them here from the original
        # messages rather than using the adapter's generic rendered string.
        prepared_prompts = []

        for index, raw_example in enumerate(eval_output.raw_examples):
            messages = adapter.get_prompt_messages(raw_example)
            if messages is None:
                raise ValueError(
                    f"Dataset adapter {adapter.name!r} cannot recover chat "
                    f"messages for evaluation example {index}."
                )

            template_kwargs = dict(model_spec.chat_template_kwargs)
            template_kwargs.update(
                tokenize=False,
                add_generation_prompt=True,
            )

            prepared_prompts.append(
                tokenizer.apply_chat_template(
                    messages,
                    **template_kwargs,
                )
            )

        return prepared_prompts

    if mode == ChatTemplateMode.TOKEN_IDS:
        prepared_prompts = []

        for index, raw_example in enumerate(eval_output.raw_examples):
            messages = adapter.get_prompt_messages(raw_example)
            if messages is None:
                raise ValueError(
                    f"Dataset adapter {adapter.name!r} cannot recover chat "
                    f"messages required for TOKEN_IDS evaluation at "
                    f"example {index}."
                )

            template_kwargs = dict(model_spec.chat_template_kwargs)
            template_kwargs.update(
                tokenize=True,
                add_generation_prompt=True,
            )

            tokenized_prompt = tokenizer.apply_chat_template(
                messages,
                **template_kwargs,
            )

            prompt_token_ids = _extract_input_ids(tokenized_prompt)

            prepared_prompts.append(
                {
                    "prompt_token_ids": prompt_token_ids,
                }
            )

        return prepared_prompts

    raise ValueError(
        f"Unsupported chat template mode: {mode!r}"
    )


def generate_responses_vllm(llm, tokenizer, prompts, max_new_tokens=1024, temperature=0.0): #TODO: remove tokenizer from this function
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_new_tokens,
        #stop_token_ids=[tokenizer.eos_token_id] if tokenizer.eos_token_id else None,
    )
    print(f"Generating responses for {len(prompts)} prompts...")
    outputs = llm.generate(prompts, sampling_params)
    return [output.outputs[0].text for output in outputs]


def generate_responses_hf(model, tokenizer, prompts, max_new_tokens=1024, temperature=0.0, batch_size=4):
    device = next(model.parameters()).device
    model_was_training = model.training
    model.eval()

    all_responses = []

    generation_kwargs = dict(
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    if temperature > 0:
        generation_kwargs["do_sample"] = True
        generation_kwargs["temperature"] = temperature
    else:
        generation_kwargs["do_sample"] = False

    with torch.no_grad():
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(device)

            outputs = model.generate(
                **inputs,
                **generation_kwargs,
            )

            input_lengths = inputs["attention_mask"].sum(dim=1)
            for j in range(outputs.size(0)):
                generated_ids = outputs[j, input_lengths[j]:]
                text = tokenizer.decode(generated_ids, skip_special_tokens=True)
                all_responses.append(text)

    if model_was_training:
        model.train()

    return all_responses


def evaluate_dataset_with_responses(dataset_name: str, responses: list[str], tokenizer):
    adapter = get_dataset_adapter(dataset_name)
    eval_data = adapter.load_eval()
    eval_output = adapter.build_eval_output(eval_data, tokenizer)

    summary = adapter.evaluate(responses, eval_output.references)
    response_records = adapter.build_response_records(
        prompts=eval_output.prompts,
        responses=responses,
        references=eval_output.references,
        scores=summary["per_sample_scores"],
        raw_examples=eval_output.raw_examples,
    )
    return summary, response_records, eval_output


def save_eval_outputs(
    dataset_name: str,
    output_dir: str,
    summary: dict[str, Any],
    response_records: list[dict[str, Any]],
    model_path: str,
    max_new_tokens: int,
    temperature: float,
):
    adapter = get_dataset_adapter(dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    results_to_save = {
        **summary,
        "config": {
            "dataset_name": dataset_name,
            "model_path": model_path,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
        },
    }

    output_path = os.path.join(output_dir, adapter.results_filename)
    with open(output_path, "w") as f:
        json.dump(results_to_save, f, indent=2)

    responses_path = os.path.join(output_dir, adapter.responses_filename)
    with open(responses_path, "w") as f:
        json.dump(response_records, f, indent=2)

    return output_path, responses_path

def run_loaded_eval(
    dataset_name: str,
    llm,
    tokenizer,
    model_path: str,
    output_dir: str,
    max_new_tokens: int | None = None,
    temperature: float = 0.0,
):
    adapter = get_dataset_adapter(dataset_name)
    max_new_tokens = max_new_tokens or adapter.default_max_new_tokens

    eval_data = adapter.load_eval()
    eval_output = adapter.build_eval_output(eval_data, tokenizer)

    model_spec = resolve_model_spec(model_path)

    generation_prompts = _prepare_vllm_eval_prompts(
        adapter=adapter,
        tokenizer=tokenizer,
        eval_output=eval_output,
        model_spec=model_spec,
    )

    responses = generate_responses_vllm(
        llm=llm,
        tokenizer=tokenizer,
        prompts=generation_prompts,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )

    print("\nEvaluating responses...")
    summary = adapter.evaluate(responses, eval_output.references)

    response_records = adapter.build_response_records(
        prompts=eval_output.prompts,
        responses=responses,
        references=eval_output.references,
        scores=summary["per_sample_scores"],
        raw_examples=eval_output.raw_examples,
    )

    output_path, responses_path = save_eval_outputs(
        dataset_name=dataset_name,
        output_dir=output_dir,
        summary=summary,
        response_records=response_records,
        model_path=model_path,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )

    print(f"\nSaved results to {output_path}")
    print(f"Saved responses to {responses_path}")

    return summary

def run_standalone_eval(
    dataset_name: str,
    model_path: str,
    output_dir: str | None = None,
    max_new_tokens: int | None = None,
    temperature: float = 0.0,
    gpu_memory_utilization: float = 0.8,
    max_model_len: int | None = None,
    tensor_parallel_size: int = 1,
):
    adapter = get_dataset_adapter(dataset_name)
    effective_max_model_len = (
        max_model_len if max_model_len is not None else adapter.default_max_model_len
    )
    max_new_tokens = max_new_tokens or adapter.default_max_new_tokens

    llm, tokenizer = load_model_and_tokenizer_vllm(
        model_path,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=effective_max_model_len,
        tensor_parallel_size=tensor_parallel_size,
    )

    eval_data = adapter.load_eval()
    eval_output = adapter.build_eval_output(eval_data, tokenizer)

    model_spec = resolve_model_spec(model_path)

    generation_prompts = _prepare_vllm_eval_prompts(
        adapter=adapter,
        tokenizer=tokenizer,
        eval_output=eval_output,
        model_spec=model_spec,
    )

    responses = generate_responses_vllm(
        llm=llm,
        tokenizer=tokenizer,
        prompts=generation_prompts,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )

    print("\nEvaluating responses...")
    summary = adapter.evaluate(responses, eval_output.references)
    response_records = adapter.build_response_records(
        prompts=eval_output.prompts,
        responses=responses,
        references=eval_output.references,
        scores=summary["per_sample_scores"],
        raw_examples=eval_output.raw_examples,
    )

    final_output_dir = output_dir if output_dir is not None else model_path
    output_path, responses_path = save_eval_outputs(
        dataset_name=dataset_name,
        output_dir=final_output_dir,
        summary=summary,
        response_records=response_records,
        model_path=model_path,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )

    print(f"\nSaved results to {output_path}")
    print(f"Saved responses to {responses_path}")

    return summary