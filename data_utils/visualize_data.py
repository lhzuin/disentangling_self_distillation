#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import load_from_disk
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset_adapters import get_dataset_adapter, get_evaluation_dataset_names

from contextualizer.contextualization_manager import ContextualizationManager

KNOWN_DATASETS = list(
    get_evaluation_dataset_names()
)


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize raw or formatted training datasets.")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="science",
        choices=KNOWN_DATASETS,
        help="Which dataset to inspect.",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="data",
        help="Root folder containing dataset directories.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train_data",
        help="Dataset split folder name.",
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=3,
        help="How many examples to print.",
    )
    parser.add_argument(
        "--formatted",
        action="store_true",
        help="Apply the same formatting logic used in training and inspect prompt/teacher_prompt.",
    )
    parser.add_argument(
        "--show_chat_template",
        action="store_true",
        help="Render prompt and teacher_prompt using the tokenizer chat template.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Tokenizer/model name used to render chat templates.",
    )
    parser.add_argument(
        "--show_full",
        action="store_true",
        help="Print full fields instead of truncated previews.",
    )

    parser.add_argument(
        "--context_strategy",
        type=str,
        default="dataset_default",
        help="Contextualization strategy used for formatted examples.",
    )
    return parser.parse_args()


def find_dataset_path(data_root: str, dataset_name: str, split: str) -> Path:
    candidates = [
        Path(data_root) / f"{dataset_name}_data" / split,
        Path(data_root) / dataset_name / split,
        Path(data_root) / dataset_name,
    ]
    for path in candidates:
        if path.exists():
            return path
    candidate_str = "\n".join(str(p) for p in candidates)
    raise FileNotFoundError(
        f"Could not find dataset '{dataset_name}'. Tried:\n{candidate_str}"
    )


def preview(value: Any, max_len: int = 1200, show_full: bool = False) -> str:
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, indent=2, ensure_ascii=False)

    if show_full or len(text) <= max_len:
        return text
    return text[:max_len] + "\n... [truncated]"


def print_header(title: str):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def apply_formatting(
    dataset_name: str,
    dataset,
    *,
    context_strategy: str,
):
    """Apply the same adapter/contextualization path used for training."""

    if context_strategy == "dataset_default":
        adapter = get_dataset_adapter(dataset_name)
        return dataset.map(
            adapter.format_train_example,
            remove_columns=dataset.column_names,
        )

    manager = ContextualizationManager(context_strategy)

    return dataset.map(
        lambda example: manager.format_train_example(
            dataset_name=dataset_name,
            raw_example=example,
            selection_spec=context_strategy,
        ),
        remove_columns=dataset.column_names,
        load_from_cache_file=False,
    )


def show_dataset_summary(dataset, dataset_path: Path):
    print_header("DATASET SUMMARY")
    print(f"Path: {dataset_path}")
    print(dataset)
    print(f"Columns: {dataset.column_names}")
    print(f"Number of rows: {len(dataset)}")


def show_raw_examples(dataset, num_examples: int, show_full: bool):
    print_header("RAW EXAMPLES")
    n = min(num_examples, len(dataset))
    for i in range(n):
        print(f"\n--- Example {i} ---")
        row = dataset[i]
        for key in dataset.column_names:
            print(f"\n[{key}]")
            print(preview(row[key], show_full=show_full))


def show_formatted_examples(dataset, num_examples: int, show_full: bool):
    print_header("FORMATTED EXAMPLES")
    n = min(num_examples, len(dataset))
    for i in range(n):
        print(f"\n--- Example {i} ---")
        row = dataset[i]

        print("\n[prompt]")
        print(preview(row["prompt"], show_full=show_full))

        print("\n[teacher_prompt]")
        print(preview(row["teacher_prompt"], show_full=show_full))

        if "target" in row:
            print("\n[target]")
            print(preview(row["target"], show_full=show_full))


def show_chat_template_examples(dataset, tokenizer, num_examples: int, show_full: bool):
    print_header("CHAT TEMPLATE RENDERING")
    n = min(num_examples, len(dataset))
    for i in range(n):
        row = dataset[i]

        print(f"\n--- Example {i}: prompt rendered ---")
        prompt_text = tokenizer.apply_chat_template(
            row["prompt"],
            tokenize=False,
            add_generation_prompt=True,
        )
        print(preview(prompt_text, max_len=2500, show_full=show_full))

        print(f"\n--- Example {i}: teacher_prompt rendered ---")
        teacher_prompt_text = tokenizer.apply_chat_template(
            row["teacher_prompt"],
            tokenize=False,
            add_generation_prompt=True,
        )
        print(preview(teacher_prompt_text, max_len=2500, show_full=show_full))


def main():
    args = parse_args()

    dataset_path = find_dataset_path(args.data_root, args.dataset_name, args.split)
    raw_dataset = load_from_disk(str(dataset_path))

    show_dataset_summary(raw_dataset, dataset_path)
    show_raw_examples(raw_dataset, args.num_examples, args.show_full)

    if args.formatted or args.show_chat_template:
        formatted_dataset = apply_formatting(
            args.dataset_name,
            raw_dataset,
            context_strategy=args.context_strategy,
        )
        show_dataset_summary(formatted_dataset, dataset_path)
        show_formatted_examples(formatted_dataset, args.num_examples, args.show_full)

        if args.show_chat_template:
            tokenizer = AutoTokenizer.from_pretrained(args.model_name)
            show_chat_template_examples(
                formatted_dataset,
                tokenizer,
                args.num_examples,
                args.show_full,
            )


if __name__ == "__main__":
    main()
