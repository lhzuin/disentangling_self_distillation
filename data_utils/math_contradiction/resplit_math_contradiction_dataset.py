#!/usr/bin/env python3
from __future__ import annotations

"""Recombine, optionally audit-filter, and deterministically resplit a dataset.

The input directory must contain Hugging Face datasets saved as::

    <input-root>/train_data
    <input-root>/eval_data

The script combines both source splits, optionally keeps only rows marked
``status=ok``, ``label=consistent``, and ``recommendation=keep`` in an audit
``filter_manifest.jsonl``, shuffles the eligible rows deterministically, and
writes new ``train_data``, ``eval_data``, and ``test_data`` directories.

The source datasets are never modified. A separate assignment manifest records
where every selected row came from and which new split received it.
"""

import argparse
import hashlib
import json
import logging
import random
import shutil
import sys
import tempfile
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

VERSION = "1.1.0"
DEFAULT_TRAIN_SIZE = 4_000
DEFAULT_EVAL_SIZE = 500
DEFAULT_SEED = 42
SOURCE_SPLITS = ("train", "eval")
IDENTITY_FIELDS = ("source_id", "id_in_dataset", "uuid", "problem_hash")

# Canonical row schema for the final math-contradiction dataset.  It keeps all
# fields needed for training/evaluation, original-world controls, stable
# provenance, stratified analysis, and the base-conversion definition.
CLEAN_COLUMNS = (
    "messages",
    "problem",
    "answer",
    "output_text",
    "original_messages",
    "original_problem",
    "original_answer",
    "original_output_text",
    "dataset_name",
    "source_dataset_name",
    "source",
    "source_id",
    "problem_hash",
    "module",
    "difficulty",
    "bucket",
    "base",
    "transformation",
    "transformation_version",
    "mod_equals_original_answer",
    "teacher_question_had_decimals",
    "output_tokens",
    "split",
)

# These aliases were intentionally emitted by the builder for compatibility
# with OpenR1-style tools and for explicit transformed/original naming.  Before
# clean-up, every present alias is compared against its canonical column across
# the complete combined dataset.  A mismatch aborts instead of silently losing
# information.
EXACT_ALIAS_GROUPS = {
    "messages": ("mod_messages",),
    "problem": ("question", "mod_problem"),
    "answer": ("golden_answer", "mod_answer"),
    "output_text": (
        "visible_output_text",
        "golden_response",
        "mod_output_text",
        "mod_visible_output_text",
    ),
    "original_answer": ("original_golden_answer",),
    "original_output_text": (
        "original_visible_output_text",
        "original_golden_response",
    ),
    "source_id": ("id_in_dataset", "uuid"),
    "module": ("problem_type",),
    "difficulty": ("question_type",),
}

# Construction/debug fields that are useful in the builder's reports and raw
# accepted-record archive, but not in the final train/eval/test Arrow datasets.
CONSTRUCTION_ONLY_COLUMNS = {
    "correctness_count",
    "correctness_llama",
    "correctness_math_verify",
    "decimal_consistency_filter_enabled",
    "decimal_interpretation_answer",
    "decimal_interpretation_solver",
    "finish_reason",
    "generation_idx",
    "is_reasoning_complete",
    "local_extracted_answer",
    "local_verify_method",
    "mod_decimal_gold",
    "mod_decimal_prediction",
    "mod_local_extracted_answer",
    "mod_local_verify_method",
    "prompt_tokens",
    "teacher_candidate_policy_reason",
    "teacher_candidate_policy_valid",
}


def utc_now() -> str:
    """Return the current UTC timestamp in a stable ISO-8601 format."""

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def configure_logging(level: str) -> None:
    """Configure concise console logging."""

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, value: Any) -> None:
    """Atomically write one pretty-printed JSON document."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_jsonl_atomic(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    """Atomically write JSONL rows and return the number written."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
            count += 1
    temporary.replace(path)
    return count


@dataclass(frozen=True)
class AuditDecision:
    """One row-level decision loaded from the audit filter manifest."""

    split: str
    row_index: int
    status: str
    label: str
    recommendation: str
    source_id: str | None
    problem_hash: str | None

    @property
    def key(self) -> tuple[str, int]:
        """Return the original dataset location used for exact matching."""

        return self.split, self.row_index

    @property
    def should_keep(self) -> bool:
        """Return whether this row satisfies the conservative keep policy."""

        return (
            self.status == "ok"
            and self.label == "consistent"
            and self.recommendation == "keep"
        )


def read_jsonl_objects(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file, requiring one object per non-empty line."""

    if not path.is_file():
        raise FileNotFoundError(f"JSONL file not found: {path}")

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(value, dict):
                raise ValueError(
                    f"Expected a JSON object at {path}:{line_number}, "
                    f"got {type(value).__name__}"
                )
            rows.append(value)
    return rows


def load_audit_decisions(path: Path) -> dict[tuple[str, int], AuditDecision]:
    """Load and validate decisions from an audit ``filter_manifest.jsonl``."""

    decisions: dict[tuple[str, int], AuditDecision] = {}
    for line_number, row in enumerate(read_jsonl_objects(path), start=1):
        try:
            split = str(row["split"])
            row_index = int(row["row_index"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Audit manifest row {line_number} must contain valid "
                "'split' and 'row_index' fields"
            ) from exc

        if split not in SOURCE_SPLITS:
            raise ValueError(
                f"Unsupported split {split!r} at audit manifest row {line_number}; "
                f"expected one of {SOURCE_SPLITS}"
            )
        if row_index < 0:
            raise ValueError(
                f"Negative row_index at audit manifest row {line_number}: {row_index}"
            )

        decision = AuditDecision(
            split=split,
            row_index=row_index,
            status=str(row.get("status") or ""),
            label=str(row.get("label") or ""),
            recommendation=str(row.get("recommendation") or ""),
            source_id=(
                str(row["source_id"])
                if row.get("source_id") not in (None, "")
                else None
            ),
            problem_hash=(
                str(row["problem_hash"])
                if row.get("problem_hash") not in (None, "")
                else None
            ),
        )
        if decision.key in decisions:
            raise ValueError(
                "Duplicate audit decision for "
                f"split={split!r}, row_index={row_index}"
            )
        decisions[decision.key] = decision

    if not decisions:
        raise ValueError(f"Audit manifest is empty: {path}")
    return decisions


def deterministic_partition(
    eligible_indices: Sequence[int],
    *,
    train_size: int,
    eval_size: int,
    seed: int,
) -> dict[str, list[int]]:
    """Shuffle global row indices deterministically and partition them."""

    if train_size <= 0:
        raise ValueError("train_size must be positive")
    if eval_size <= 0:
        raise ValueError("eval_size must be positive")
    required = train_size + eval_size
    if len(eligible_indices) < required:
        raise ValueError(
            f"Need at least {required} eligible rows, but only "
            f"{len(eligible_indices)} are available"
        )

    shuffled = list(eligible_indices)
    random.Random(seed).shuffle(shuffled)
    return {
        "train": shuffled[:train_size],
        "eval": shuffled[train_size:required],
        "test": shuffled[required:],
    }


def select_identity_field(column_names: Sequence[str]) -> str | None:
    """Choose the strongest available field for duplicate detection."""

    for field in IDENTITY_FIELDS:
        if field in column_names:
            return field
    return None


def replace_split_column(dataset: Any, split_name: str) -> Any:
    """Replace a stale ``split`` column with the newly assigned split name."""

    if "split" in dataset.column_names:
        dataset = dataset.remove_columns("split")
    return dataset.add_column("split", [split_name] * len(dataset))


def validate_exact_aliases(dataset: Any) -> dict[str, str]:
    """Verify every present compatibility alias equals its canonical column.

    Returns a mapping from alias to canonical column.  Full-column equality is
    intentionally checked before any pruning; the dataset is small enough that
    this is inexpensive and safer than relying on builder assumptions.
    """

    columns = set(dataset.column_names)
    validated: dict[str, str] = {}
    for canonical, aliases in EXACT_ALIAS_GROUPS.items():
        present_aliases = [alias for alias in aliases if alias in columns]
        if not present_aliases:
            continue
        if canonical not in columns:
            raise ValueError(
                f"Cannot validate aliases {present_aliases}: canonical column "
                f"{canonical!r} is absent"
            )

        canonical_values = dataset[canonical]
        for alias in present_aliases:
            alias_values = dataset[alias]
            mismatch_index = next(
                (
                    i
                    for i, (left, right) in enumerate(
                        zip(canonical_values, alias_values)
                    )
                    if left != right
                ),
                None,
            )
            if len(alias_values) != len(canonical_values):
                mismatch_index = min(len(alias_values), len(canonical_values))
            if mismatch_index is not None:
                raise ValueError(
                    f"Refusing to drop non-identical alias {alias!r} for "
                    f"{canonical!r}; first mismatch at row {mismatch_index}"
                )
            validated[alias] = canonical
    return dict(sorted(validated.items()))


def resolve_output_columns(
    column_names: Sequence[str],
    *,
    profile: str,
    extra_keep_columns: Sequence[str],
) -> tuple[list[str], list[str]]:
    """Resolve ordered kept/dropped columns for the selected schema profile."""

    available = list(column_names)
    if profile == "all":
        if extra_keep_columns:
            raise ValueError("--extra-keep-columns is unnecessary with --column-profile all")
        return available, []
    if profile != "clean":
        raise ValueError(f"Unsupported column profile: {profile!r}")

    missing_core = [column for column in CLEAN_COLUMNS if column not in available]
    if missing_core:
        raise ValueError(
            "The clean profile requires columns missing from the source schema: "
            f"{missing_core}"
        )

    unknown_extra = [column for column in extra_keep_columns if column not in available]
    if unknown_extra:
        raise ValueError(
            f"--extra-keep-columns contains unknown source columns: {unknown_extra}"
        )

    requested = set(CLEAN_COLUMNS) | set(extra_keep_columns)
    kept = [column for column in available if column in requested]
    dropped = [column for column in available if column not in requested]
    return kept, dropped


def apply_column_profile(
    dataset: Any,
    *,
    profile: str,
    extra_keep_columns: Sequence[str],
) -> tuple[Any, dict[str, Any]]:
    """Apply an explicit output schema and return a detailed clean-up report."""

    aliases = validate_exact_aliases(dataset) if profile == "clean" else {}
    kept, dropped = resolve_output_columns(
        dataset.column_names,
        profile=profile,
        extra_keep_columns=extra_keep_columns,
    )
    cleaned = dataset.remove_columns(dropped) if dropped else dataset

    drop_reasons: dict[str, str] = {}
    for column in dropped:
        if column in aliases:
            drop_reasons[column] = f"exact alias of {aliases[column]}"
        elif column in CONSTRUCTION_ONLY_COLUMNS:
            drop_reasons[column] = "construction/debug-only metadata"
        else:
            drop_reasons[column] = "not part of selected canonical schema"

    return cleaned, {
        "profile": profile,
        "input_columns": list(dataset.column_names),
        "output_columns": list(cleaned.column_names),
        "extra_keep_columns": list(extra_keep_columns),
        "validated_exact_aliases": aliases,
        "dropped_columns": dropped,
        "drop_reasons": drop_reasons,
    }


def count_values(dataset: Any, field: str) -> dict[str, int]:
    """Count values for a dataset column when it exists."""

    if field not in dataset.column_names:
        return {}
    return dict(sorted(Counter(str(value) for value in dataset[field]).items()))


def prepare_output_root(output_root: Path, overwrite: bool) -> None:
    """Create the output root and enforce safe overwrite semantics."""

    output_root.mkdir(parents=True, exist_ok=True)
    targets = [
        output_root / "train_data",
        output_root / "eval_data",
        output_root / "test_data",
        output_root / "split_assignments.jsonl",
        output_root / "resplit_report.json",
    ]
    existing = [path for path in targets if path.exists()]
    if existing and not overwrite:
        rendered = "\n  - ".join(str(path) for path in existing)
        raise FileExistsError(
            "Refusing to overwrite existing outputs. Re-run with --overwrite:\n"
            f"  - {rendered}"
        )

    if overwrite:
        for path in existing:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()


def save_splits_atomically(
    output_root: Path,
    datasets_by_split: Mapping[str, Any],
) -> None:
    """Save all split datasets to temporary directories before renaming."""

    token = uuid.uuid4().hex[:12]
    temporary_paths: dict[str, Path] = {}
    try:
        for split_name, dataset in datasets_by_split.items():
            temporary = output_root / f".{split_name}_data.tmp-{token}"
            if temporary.exists():
                shutil.rmtree(temporary)
            logging.info(
                "Saving %s_data (%d rows) to temporary path",
                split_name,
                len(dataset),
            )
            dataset.save_to_disk(str(temporary))
            temporary_paths[split_name] = temporary

        for split_name, temporary in temporary_paths.items():
            destination = output_root / f"{split_name}_data"
            temporary.replace(destination)
    except Exception:
        for temporary in temporary_paths.values():
            if temporary.exists():
                shutil.rmtree(temporary, ignore_errors=True)
        raise


def validate_schema(train_dataset: Any, eval_dataset: Any) -> None:
    """Require compatible source schemas before concatenation."""

    if train_dataset.column_names != eval_dataset.column_names:
        raise ValueError(
            "train_data and eval_data have different column order or names:\n"
            f"train={train_dataset.column_names}\n"
            f"eval={eval_dataset.column_names}"
        )
    if train_dataset.features != eval_dataset.features:
        raise ValueError("train_data and eval_data have incompatible features")


def validate_unique_identities(
    train_dataset: Any,
    eval_dataset: Any,
    identity_field: str | None,
) -> None:
    """Reject duplicate stable identities that could leak across new splits."""

    if identity_field is None:
        logging.warning(
            "No identity field found among %s; duplicate-content validation skipped",
            IDENTITY_FIELDS,
        )
        return

    identities = [
        str(value)
        for value in list(train_dataset[identity_field])
        + list(eval_dataset[identity_field])
    ]
    duplicates = [
        identity
        for identity, count in Counter(identities).items()
        if identity and count > 1
    ]
    if duplicates:
        preview = ", ".join(repr(value) for value in duplicates[:10])
        raise ValueError(
            f"Found {len(duplicates)} duplicated values in {identity_field!r}; "
            f"examples: {preview}. Refusing to create splits with leakage."
        )


def validate_audit_against_sources(
    decisions: Mapping[tuple[str, int], AuditDecision],
    train_dataset: Any,
    eval_dataset: Any,
    *,
    require_complete_coverage: bool,
) -> None:
    """Verify audit locations and metadata against the exact source datasets."""

    datasets = {"train": train_dataset, "eval": eval_dataset}
    all_source_keys = {
        (split_name, row_index)
        for split_name, dataset in datasets.items()
        for row_index in range(len(dataset))
    }
    decision_keys = set(decisions)

    unknown = decision_keys - all_source_keys
    if unknown:
        preview = sorted(unknown)[:10]
        raise ValueError(
            f"Audit manifest contains {len(unknown)} locations absent from the "
            f"source datasets; examples: {preview}"
        )

    missing = all_source_keys - decision_keys
    if missing and require_complete_coverage:
        preview = sorted(missing)[:10]
        raise ValueError(
            f"Audit manifest does not cover {len(missing)} source rows; "
            f"examples: {preview}. Use --allow-partial-audit only intentionally."
        )

    for key, decision in decisions.items():
        dataset = datasets[decision.split]
        row = dataset[decision.row_index]
        if decision.source_id is not None and "source_id" in dataset.column_names:
            actual_source_id = str(row.get("source_id") or "")
            if actual_source_id != decision.source_id:
                raise ValueError(
                    f"source_id mismatch for {key}: audit={decision.source_id!r}, "
                    f"dataset={actual_source_id!r}"
                )
        if decision.problem_hash is not None and "problem_hash" in dataset.column_names:
            actual_problem_hash = str(row.get("problem_hash") or "")
            if actual_problem_hash != decision.problem_hash:
                raise ValueError(
                    f"problem_hash mismatch for {key}: "
                    f"audit={decision.problem_hash!r}, "
                    f"dataset={actual_problem_hash!r}"
                )


def build_eligible_indices(
    *,
    train_size: int,
    eval_size: int,
    decisions: Mapping[tuple[str, int], AuditDecision] | None,
) -> tuple[list[int], dict[str, int]]:
    """Return eligible global indices and audit decision counts."""

    offsets = {"train": 0, "eval": train_size}
    source_lengths = {"train": train_size, "eval": eval_size}

    if decisions is None:
        return list(range(train_size + eval_size)), {"not_filtered": train_size + eval_size}

    counts: Counter[str] = Counter()
    eligible: list[int] = []
    for split_name in SOURCE_SPLITS:
        for row_index in range(source_lengths[split_name]):
            decision = decisions.get((split_name, row_index))
            if decision is None:
                counts["missing_decision"] += 1
                continue
            key = (
                f"{decision.status or 'missing_status'}/"
                f"{decision.label or 'missing_label'}/"
                f"{decision.recommendation or 'missing_recommendation'}"
            )
            counts[key] += 1
            if decision.should_keep:
                eligible.append(offsets[split_name] + row_index)

    return eligible, dict(sorted(counts.items()))


def assignment_rows(
    partitions: Mapping[str, Sequence[int]],
    *,
    train_source_size: int,
    combined_dataset: Any,
) -> Iterable[dict[str, Any]]:
    """Yield traceable original-to-new split assignments."""

    for new_split, global_indices in partitions.items():
        for new_index, global_index in enumerate(global_indices):
            if global_index < train_source_size:
                source_split = "train"
                source_row_index = global_index
            else:
                source_split = "eval"
                source_row_index = global_index - train_source_size

            row = combined_dataset[global_index]
            assignment: dict[str, Any] = {
                "new_split": new_split,
                "new_row_index": new_index,
                "source_split": source_split,
                "source_row_index": source_row_index,
            }
            for field in IDENTITY_FIELDS:
                if field in combined_dataset.column_names:
                    assignment[field] = row.get(field)
            yield assignment


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Combine train_data/eval_data, optionally keep only audited "
            "consistent rows, and create new train/eval/test splits."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=None,
        help="Directory containing train_data and eval_data.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Directory in which new train_data/eval_data/test_data are saved.",
    )
    parser.add_argument(
        "--audit-manifest",
        type=Path,
        default=None,
        help=(
            "Optional audit filter_manifest.jsonl. When supplied, only rows "
            "with status=ok, label=consistent, recommendation=keep are eligible."
        ),
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=DEFAULT_TRAIN_SIZE,
        help=f"Number of new training rows (default: {DEFAULT_TRAIN_SIZE}).",
    )
    parser.add_argument(
        "--eval-size",
        type=int,
        default=DEFAULT_EVAL_SIZE,
        help=f"Number of new evaluation rows (default: {DEFAULT_EVAL_SIZE}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Deterministic shuffle seed (default: {DEFAULT_SEED}).",
    )
    parser.add_argument(
        "--allow-partial-audit",
        action="store_true",
        help=(
            "Allow an audit manifest that omits source rows. Omitted rows are "
            "excluded. By default, complete audit coverage is required."
        ),
    )
    parser.add_argument(
        "--column-profile",
        choices=("clean", "all"),
        default="clean",
        help=(
            "Output schema: 'clean' keeps the documented canonical columns and "
            "drops verified aliases/construction metadata; 'all' preserves the "
            "source schema unchanged (default: clean)."
        ),
    )
    parser.add_argument(
        "--extra-keep-columns",
        nargs="*",
        default=(),
        metavar="COLUMN",
        help=(
            "Additional source columns to retain with --column-profile clean. "
            "Unknown names fail fast."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing generated splits and reports in output-root.",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run dependency-free unit tests and exit.",
    )
    return parser.parse_args()


def run_self_tests() -> None:
    """Run dependency-free tests for the core selection logic."""

    indices = list(range(20))
    first = deterministic_partition(indices, train_size=10, eval_size=5, seed=7)
    second = deterministic_partition(indices, train_size=10, eval_size=5, seed=7)
    assert first == second
    assert len(first["train"]) == 10
    assert len(first["eval"]) == 5
    assert len(first["test"]) == 5
    assert set().union(*map(set, first.values())) == set(indices)
    assert sum(len(values) for values in first.values()) == len(indices)

    changed = deterministic_partition(indices, train_size=10, eval_size=5, seed=8)
    assert first != changed

    clean_source_columns = list(CLEAN_COLUMNS) + [
        "golden_answer",
        "correctness_llama",
    ]
    kept, dropped = resolve_output_columns(
        clean_source_columns, profile="clean", extra_keep_columns=()
    )
    assert kept == list(CLEAN_COLUMNS)
    assert dropped == ["golden_answer", "correctness_llama"]

    kept_extra, dropped_extra = resolve_output_columns(
        clean_source_columns,
        profile="clean",
        extra_keep_columns=("correctness_llama",),
    )
    assert "correctness_llama" in kept_extra
    assert dropped_extra == ["golden_answer"]

    kept_all, dropped_all = resolve_output_columns(
        clean_source_columns, profile="all", extra_keep_columns=()
    )
    assert kept_all == clean_source_columns and dropped_all == []

    try:
        deterministic_partition([0, 1], train_size=2, eval_size=1, seed=0)
    except ValueError as exc:
        assert "at least 3" in str(exc)
    else:
        raise AssertionError("Insufficient-row validation did not trigger")

    consistent = AuditDecision("train", 0, "ok", "consistent", "keep", "a", "ha")
    inconsistent = AuditDecision("train", 1, "ok", "inconsistent", "drop", "b", "hb")
    errored = AuditDecision("eval", 0, "error", "", "review", "c", "hc")
    assert consistent.should_keep
    assert not inconsistent.should_keep
    assert not errored.should_keep

    decisions = {decision.key: decision for decision in (consistent, inconsistent, errored)}
    eligible, counts = build_eligible_indices(
        train_size=2,
        eval_size=1,
        decisions=decisions,
    )
    assert eligible == [0]
    assert counts["ok/consistent/keep"] == 1
    assert counts["ok/inconsistent/drop"] == 1
    assert counts["error/missing_label/review"] == 1

    class FakeDataset:
        def __init__(self, rows: list[dict[str, Any]]):
            self.rows = rows
            self.column_names = list(rows[0]) if rows else []

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, index: int | str) -> Any:
            if isinstance(index, str):
                return [row[index] for row in self.rows]
            return self.rows[index]

        def remove_columns(self, columns: Sequence[str]) -> "FakeDataset":
            removed = set(columns)
            return FakeDataset([
                {key: value for key, value in row.items() if key not in removed}
                for row in self.rows
            ])

    fake_train = FakeDataset([
        {"source_id": "a", "problem_hash": "ha"},
        {"source_id": "b", "problem_hash": "hb"},
    ])
    fake_eval = FakeDataset([{"source_id": "c", "problem_hash": "hc"}])

    alias_dataset = FakeDataset([
        {"problem": "p1", "question": "p1", "answer": "a1", "golden_answer": "a1"},
        {"problem": "p2", "question": "p2", "answer": "a2", "golden_answer": "a2"},
    ])
    assert validate_exact_aliases(alias_dataset) == {
        "golden_answer": "answer",
        "question": "problem",
    }
    bad_alias_dataset = FakeDataset([
        {"problem": "p1", "question": "different"},
    ])
    try:
        validate_exact_aliases(bad_alias_dataset)
    except ValueError as exc:
        assert "non-identical alias" in str(exc)
    else:
        raise AssertionError("Non-identical alias validation did not trigger")

    validate_audit_against_sources(
        decisions, fake_train, fake_eval, require_complete_coverage=True
    )

    bad_decisions = dict(decisions)
    bad_decisions[("train", 0)] = AuditDecision(
        "train", 0, "ok", "consistent", "keep", "wrong", "ha"
    )
    try:
        validate_audit_against_sources(
            bad_decisions, fake_train, fake_eval, require_complete_coverage=True
        )
    except ValueError as exc:
        assert "source_id mismatch" in str(exc)
    else:
        raise AssertionError("Audit metadata mismatch validation did not trigger")

    with tempfile.TemporaryDirectory() as temporary_directory:
        manifest_path = Path(temporary_directory) / "filter_manifest.jsonl"
        write_jsonl_atomic(
            manifest_path,
            [
                {
                    "split": decision.split,
                    "row_index": decision.row_index,
                    "status": decision.status,
                    "label": decision.label,
                    "recommendation": decision.recommendation,
                    "source_id": decision.source_id,
                    "problem_hash": decision.problem_hash,
                }
                for decision in (consistent, inconsistent, errored)
            ],
        )
        loaded = load_audit_decisions(manifest_path)
        assert loaded == decisions

    print("All self-tests passed.")


def main() -> None:
    """Load, validate, filter, resplit, save, and verify the datasets."""

    args = parse_args()
    configure_logging(args.log_level)

    if args.self_test:
        run_self_tests()
        return

    if args.input_root is None or args.output_root is None:
        raise ValueError("--input-root and --output-root are required unless --self-test is used")

    input_root = args.input_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if input_root == output_root:
        raise ValueError(
            "input-root and output-root must differ so the source datasets are "
            "never modified in place"
        )

    train_path = input_root / "train_data"
    eval_path = input_root / "eval_data"
    if not train_path.is_dir() or not eval_path.is_dir():
        raise FileNotFoundError(
            "Expected Hugging Face datasets at both:\n"
            f"  - {train_path}\n"
            f"  - {eval_path}"
        )

    try:
        from datasets import concatenate_datasets, load_from_disk
    except ImportError as exc:
        raise ImportError(
            "The 'datasets' package is required. Install it with: "
            "pip install -U datasets"
        ) from exc

    logging.info("Loading source splits from %s", input_root)
    source_train = load_from_disk(str(train_path))
    source_eval = load_from_disk(str(eval_path))
    validate_schema(source_train, source_eval)

    logging.info(
        "Loaded %d train rows and %d eval rows",
        len(source_train),
        len(source_eval),
    )
    identity_field = select_identity_field(source_train.column_names)
    validate_unique_identities(source_train, source_eval, identity_field)

    audit_decisions: dict[tuple[str, int], AuditDecision] | None = None
    audit_manifest_path: Path | None = None
    if args.audit_manifest is not None:
        audit_manifest_path = args.audit_manifest.expanduser().resolve()
        audit_decisions = load_audit_decisions(audit_manifest_path)
        validate_audit_against_sources(
            audit_decisions,
            source_train,
            source_eval,
            require_complete_coverage=not args.allow_partial_audit,
        )
        logging.info(
            "Loaded and validated %d audit decisions from %s",
            len(audit_decisions),
            audit_manifest_path,
        )
    else:
        logging.warning(
            "No --audit-manifest supplied: all source rows are eligible, "
            "including rows not approved by the reasoning audit"
        )

    combined_full = concatenate_datasets([source_train, source_eval])
    combined, column_cleanup = apply_column_profile(
        combined_full,
        profile=args.column_profile,
        extra_keep_columns=tuple(args.extra_keep_columns),
    )
    logging.info(
        "Column profile %s: kept %d / %d columns",
        args.column_profile,
        len(combined.column_names),
        len(combined_full.column_names),
    )
    if column_cleanup["dropped_columns"]:
        logging.info(
            "Dropped columns: %s",
            ", ".join(column_cleanup["dropped_columns"]),
        )

    eligible_indices, audit_counts = build_eligible_indices(
        train_size=len(source_train),
        eval_size=len(source_eval),
        decisions=audit_decisions,
    )
    logging.info(
        "Eligible rows: %d / %d",
        len(eligible_indices),
        len(combined),
    )

    partitions = deterministic_partition(
        eligible_indices,
        train_size=args.train_size,
        eval_size=args.eval_size,
        seed=args.seed,
    )
    logging.info(
        "New split sizes: train=%d eval=%d test=%d",
        len(partitions["train"]),
        len(partitions["eval"]),
        len(partitions["test"]),
    )

    datasets_by_split: dict[str, Any] = {}
    for split_name, indices in partitions.items():
        selected = combined.select(indices)
        datasets_by_split[split_name] = replace_split_column(selected, split_name)

    prepare_output_root(output_root, args.overwrite)
    save_splits_atomically(output_root, datasets_by_split)

    assignments_path = output_root / "split_assignments.jsonl"
    assignment_count = write_jsonl_atomic(
        assignments_path,
        assignment_rows(
            partitions,
            train_source_size=len(source_train),
            combined_dataset=combined,
        ),
    )
    if assignment_count != len(eligible_indices):
        raise RuntimeError(
            f"Assignment manifest contains {assignment_count} rows, expected "
            f"{len(eligible_indices)}"
        )

    report = {
        "created_at": utc_now(),
        "script_version": VERSION,
        "input_root": str(input_root),
        "output_root": str(output_root),
        "seed": args.seed,
        "requested_sizes": {
            "train": args.train_size,
            "eval": args.eval_size,
            "test": "all_remaining_eligible_rows",
        },
        "source": {
            "train_rows": len(source_train),
            "eval_rows": len(source_eval),
            "total_rows": len(combined),
            "columns": list(combined_full.column_names),
            "output_columns": list(combined.column_names),
            "identity_field_checked": identity_field,
            "train_fingerprint": str(getattr(source_train, "_fingerprint", "")),
            "eval_fingerprint": str(getattr(source_eval, "_fingerprint", "")),
        },
        "column_cleanup": column_cleanup,
        "audit_filter": {
            "enabled": audit_manifest_path is not None,
            "manifest_path": str(audit_manifest_path) if audit_manifest_path else None,
            "manifest_sha256": (
                sha256_file(audit_manifest_path) if audit_manifest_path else None
            ),
            "policy": (
                "status=ok AND label=consistent AND recommendation=keep"
                if audit_manifest_path
                else "no audit filtering"
            ),
            "require_complete_coverage": not args.allow_partial_audit,
            "decision_counts": audit_counts,
            "eligible_rows": len(eligible_indices),
            "excluded_rows": len(combined) - len(eligible_indices),
        },
        "new_splits": {
            split_name: {
                "rows": len(dataset),
                "module_counts": count_values(dataset, "module"),
                "difficulty_counts": count_values(dataset, "difficulty"),
                "bucket_counts": count_values(dataset, "bucket"),
                "path": str(output_root / f"{split_name}_data"),
            }
            for split_name, dataset in datasets_by_split.items()
        },
        "output_files": {
            "train_data": str(output_root / "train_data"),
            "eval_data": str(output_root / "eval_data"),
            "test_data": str(output_root / "test_data"),
            "split_assignments": str(assignments_path),
            "report": str(output_root / "resplit_report.json"),
        },
    }
    report_path = output_root / "resplit_report.json"
    write_json_atomic(report_path, report)

    logging.info("Reloading saved splits for final verification")
    reloaded = {
        split_name: load_from_disk(str(output_root / f"{split_name}_data"))
        for split_name in ("train", "eval", "test")
    }
    expected_lengths = {name: len(indices) for name, indices in partitions.items()}
    for split_name, dataset in reloaded.items():
        if len(dataset) != expected_lengths[split_name]:
            raise RuntimeError(
                f"Saved {split_name}_data has {len(dataset)} rows; expected "
                f"{expected_lengths[split_name]}"
            )
        expected_columns = list(datasets_by_split[split_name].column_names)
        if dataset.column_names != expected_columns:
            raise RuntimeError(
                f"Saved {split_name}_data columns differ from the selected schema: "
                f"saved={dataset.column_names}, expected={expected_columns}"
            )
        if "split" not in dataset.column_names:
            raise RuntimeError(f"Saved {split_name}_data is missing the split column")
        if set(dataset["split"]) != {split_name}:
            raise RuntimeError(
                f"Saved {split_name}_data contains stale split labels: "
                f"{sorted(set(dataset['split']))}"
            )

    logging.info("Resplitting completed successfully")
    logging.info("Report: %s", report_path)


if __name__ == "__main__":
    main()