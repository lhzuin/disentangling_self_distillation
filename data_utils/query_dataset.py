#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Callable

from datasets import load_from_disk

from dataset_adapters import get_dataset_adapter


SUPPORTED_OPS = {
    "eq": lambda a, b: str(a) == b,
    "neq": lambda a, b: str(a) != b,
    "contains": lambda a, b: b.lower() in stringify(a).lower(),
    "not_contains": lambda a, b: b.lower() not in stringify(a).lower(),
    "regex": lambda a, b: re.search(b, stringify(a), flags=re.IGNORECASE) is not None,
    "gt": lambda a, b: to_float(a) > float(b),
    "gte": lambda a, b: to_float(a) >= float(b),
    "lt": lambda a, b: to_float(a) < float(b),
    "lte": lambda a, b: to_float(a) <= float(b),
}


def stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, indent=2)


def to_float(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return float(str(value))


def preview(value: Any, max_len: int = 1200, show_full: bool = False) -> str:
    text = stringify(value)
    if show_full or len(text) <= max_len:
        return text
    return text[:max_len] + "\n... [truncated]"


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


def parse_condition(raw: str) -> tuple[str, str, str]:
    """
    Expected format:
        column:op:value

    Examples:
        question:contains:Newton
        answer:eq:B
        difficulty:gte:3
        prompt:regex:cardio.*patient
    """
    parts = raw.split(":", 2)
    if len(parts) != 3:
        raise ValueError(
            f"Invalid condition: {raw!r}. Expected format column:op:value"
        )

    column, op, value = parts

    if op not in SUPPORTED_OPS:
        raise ValueError(
            f"Unsupported operator {op!r}. Supported: {sorted(SUPPORTED_OPS)}"
        )

    return column, op, value


def row_matches_condition(row: dict[str, Any], condition: tuple[str, str, str]) -> bool:
    column, op, value = condition

    if column not in row:
        return False

    try:
        return bool(SUPPORTED_OPS[op](row[column], value))
    except Exception:
        return False


def row_matches_global_query(row: dict[str, Any], query: str) -> bool:
    query = query.lower()
    return any(query in stringify(value).lower() for value in row.values())


def row_matches_regex_query(row: dict[str, Any], pattern: str) -> bool:
    return any(
        re.search(pattern, stringify(value), flags=re.IGNORECASE) is not None
        for value in row.values()
    )


def apply_formatting(dataset_name: str, dataset):
    adapter = get_dataset_adapter(dataset_name)
    return dataset.map(adapter.format_train_example, remove_columns=dataset.column_names)


def print_row(
    row: dict[str, Any],
    index: int,
    columns: list[str] | None,
    show_full: bool,
    max_len: int,
):
    print("\n" + "=" * 100)
    print(f"ROW {index}")
    print("=" * 100)

    selected_columns = columns or list(row.keys())

    for col in selected_columns:
        if col not in row:
            print(f"\n[{col}]")
            print("<missing column>")
            continue

        print(f"\n[{col}]")
        print(preview(row[col], max_len=max_len, show_full=show_full))


def export_results(
    rows: list[dict[str, Any]],
    indices: list[int],
    output_path: str,
):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    records = []
    for idx, row in zip(indices, rows):
        record = {"_index": idx}
        record.update(row)
        records.append(record)

    if path.suffix == ".jsonl":
        with path.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    else:
        with path.open("w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"\nSaved {len(records)} rows to {path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Query HuggingFace datasets saved with load_from_disk."
    )

    parser.add_argument(
        "--dataset_name",
        type=str,
        default="science",
        choices=["science", "tooluse"],
        help="Which dataset to query.",
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
        "--formatted",
        action="store_true",
        help="Apply dataset adapter formatting before querying.",
    )

    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Case-insensitive text search over all columns.",
    )
    parser.add_argument(
        "--regex",
        type=str,
        default=None,
        help="Regex search over all columns.",
    )
    parser.add_argument(
        "--where",
        action="append",
        default=[],
        help=(
            "Filter condition in the format column:op:value. "
            "Can be passed multiple times. "
            "Ops: eq, neq, contains, not_contains, regex, gt, gte, lt, lte."
        ),
    )

    parser.add_argument(
        "--columns",
        type=str,
        nargs="+",
        default=None,
        help="Columns to print. By default, prints all columns.",
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=10,
        help="Maximum number of matching rows to print.",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Skip the first N matching rows before printing.",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="Randomly sample from matching rows.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used with --random.",
    )

    parser.add_argument(
        "--show_full",
        action="store_true",
        help="Print full values instead of truncated previews.",
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=1200,
        help="Maximum printed characters per field unless --show_full is passed.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save matching printed rows as .json or .jsonl.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    dataset_path = find_dataset_path(args.data_root, args.dataset_name, args.split)
    dataset = load_from_disk(str(dataset_path))

    if args.formatted:
        dataset = apply_formatting(args.dataset_name, dataset)

    conditions = [parse_condition(raw) for raw in args.where]

    matching_indices: list[int] = []

    for i, row in enumerate(dataset):
        row = dict(row)

        if args.query is not None and not row_matches_global_query(row, args.query):
            continue

        if args.regex is not None and not row_matches_regex_query(row, args.regex):
            continue

        if any(not row_matches_condition(row, cond) for cond in conditions):
            continue

        matching_indices.append(i)

    print("\n" + "=" * 100)
    print("QUERY SUMMARY")
    print("=" * 100)
    print(f"Dataset path: {dataset_path}")
    print(f"Formatted: {args.formatted}")
    print(f"Columns: {dataset.column_names}")
    print(f"Total rows: {len(dataset)}")
    print(f"Matching rows: {len(matching_indices)}")

    if args.random:
        rng = random.Random(args.seed)
        selected_indices = matching_indices[:]
        rng.shuffle(selected_indices)
        selected_indices = selected_indices[: args.num_examples]
    else:
        selected_indices = matching_indices[args.offset : args.offset + args.num_examples]

    selected_rows = [dict(dataset[i]) for i in selected_indices]

    for idx, row in zip(selected_indices, selected_rows):
        print_row(
            row=row,
            index=idx,
            columns=args.columns,
            show_full=args.show_full,
            max_len=args.max_len,
        )

    if args.output is not None:
        export_results(selected_rows, selected_indices, args.output)


if __name__ == "__main__":
    main()
