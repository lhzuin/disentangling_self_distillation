#!/usr/bin/env python3
"""Build a filtered DeepMind Mathematics Dataset subset for LLM demonstrations.

This script downloads the original DeepMind Mathematics Dataset archive, reads only
selected module/difficulty files, filters examples whose answers are simple numeric
objects, samples a balanced subset, and writes the result as JSONL.

The intended downstream use is to take the saved prompts, ask a strong model for
step-by-step demonstrations, and then optionally transform all numerals into an
alternative arithmetic world (for example, base 9) before fine-tuning/evaluation.

Default behavior:
  * Uses symbolic/numeric arithmetic and algebra modules.
  * Preserves difficulty labels: train-easy, train-medium, train-hard.
  * Samples up to 10,000 examples, approximately balanced across module and
    difficulty buckets.
  * Saves:
      - filtered_math_dataset.jsonl
      - metadata.json

Example:
    python build_deepmind_math_filtered.py --output-dir data/deepmind_math_filtered

Notes:
    The original archive is large (~2.17 GiB). The script keeps the downloaded
    tar.gz under <output-dir>/cache by default and does not extract the full
    archive to disk.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import random
import re
import tarfile
import urllib.request
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

DATASET_URL = "https://storage.googleapis.com/mathematics-dataset/mathematics_dataset-v1.0.tar.gz"
DATASET_VERSION_DIR = "mathematics_dataset-v1.0"

DEFAULT_MODULES = (
    "arithmetic__mixed",
    "arithmetic__add_sub_multiple",
    "arithmetic__mul_div_multiple",
    "arithmetic__div",
    "algebra__linear_1d",
    "algebra__linear_2d",
)

DEFAULT_DIFFICULTIES = ("train-easy", "train-medium", "train-hard")

# Conservative numeric-answer filter. This intentionally keeps answers that are
# easy to verify with exact-match or math_verify-style parsing, and rejects
# natural-language/list/polynomial answers.
NUMERIC_ANSWER_RE = re.compile(
    r"""
    ^\s*
    [-+]?(
        (\d+(\.\d*)?|\.\d+)(/[+-]?\d+(\.\d*)?)?  # int/decimal, optional /number
        |
        \d+\s*/\s*[-+]?\d+                        # rational with spaces
    )
    \s*$
    """,
    re.VERBOSE,
)

DIGIT_RE = re.compile(r"\d")


@dataclass(frozen=True)
class Example:
    """One filtered dataset example."""

    id: str
    question: str
    answer: str
    module: str
    difficulty: str
    source: str


@dataclass(frozen=True)
class BucketStats:
    """Summary statistics for one module/difficulty bucket."""

    module: str
    difficulty: str
    seen: int
    kept_after_filtering: int
    sampled: int


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Download and filter DeepMind Mathematics Dataset examples."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where the filtered dataset and metadata will be saved.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=10_000,
        help="Maximum number of examples to save after filtering and sampling.",
    )
    parser.add_argument(
        "--modules",
        nargs="+",
        default=list(DEFAULT_MODULES),
        help="DeepMind Mathematics Dataset module names to include.",
    )
    parser.add_argument(
        "--difficulties",
        nargs="+",
        default=list(DEFAULT_DIFFICULTIES),
        help="Difficulty folders to include, usually train-easy train-medium train-hard.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=37,
        help="Random seed used for reproducible sampling and shuffling.",
    )
    parser.add_argument(
        "--archive-path",
        type=Path,
        default=None,
        help=(
            "Optional path to an existing mathematics_dataset-v1.0.tar.gz. "
            "If omitted, it is downloaded into <output-dir>/cache."
        ),
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download the archive even if it already exists.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging verbosity.",
    )
    return parser.parse_args()


def configure_logging(level: str) -> None:
    """Configure process-wide logging."""

    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def download_archive(output_dir: Path, archive_path: Path | None, force: bool) -> Path:
    """Return a local archive path, downloading the dataset if necessary."""

    if archive_path is not None:
        if not archive_path.exists():
            raise FileNotFoundError(f"Archive not found: {archive_path}")
        return archive_path

    cache_dir = output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    destination = cache_dir / "mathematics_dataset-v1.0.tar.gz"

    if destination.exists() and not force:
        logging.info("Using existing archive: %s", destination)
        return destination

    logging.info("Downloading %s", DATASET_URL)
    logging.info("Saving archive to %s", destination)
    with urllib.request.urlopen(DATASET_URL) as response, destination.open("wb") as out_file:
        total_bytes = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            out_file.write(chunk)
            total_bytes += len(chunk)
            if total_bytes and total_bytes % (256 * 1024 * 1024) < 1024 * 1024:
                logging.info("Downloaded %.2f GiB", total_bytes / 2**30)
    return destination


def member_name(module: str, difficulty: str) -> str:
    """Return the expected archive member path for a module/difficulty pair."""

    return f"{DATASET_VERSION_DIR}/{difficulty}/{module}.txt"


def is_numeric_answer(answer: str) -> bool:
    """Return True if answer is a simple numeric value accepted by the filter."""

    normalized = answer.strip().replace(",", "")
    return NUMERIC_ANSWER_RE.fullmatch(normalized) is not None


def should_keep(question: str, answer: str) -> bool:
    """Apply content-level filters to one question-answer pair."""

    if not question.strip() or not answer.strip():
        return False
    if DIGIT_RE.search(question) is None:
        return False
    return is_numeric_answer(answer)


def stable_id(module: str, difficulty: str, question: str, answer: str) -> str:
    """Create a deterministic ID for one example."""

    payload = f"{module}\n{difficulty}\n{question}\n{answer}".encode("utf-8")
    digest = hashlib.sha1(payload).hexdigest()[:16]
    return f"dm_math_{digest}"


def parse_lines_as_examples(
    lines: Sequence[str], module: str, difficulty: str
) -> tuple[list[Example], BucketStats]:
    """Parse alternating question/answer lines and return filtered examples."""

    cleaned = [line.strip() for line in lines if line.strip()]
    if len(cleaned) % 2 != 0:
        raise ValueError(
            f"Expected an even number of non-empty lines for {module}/{difficulty}, "
            f"got {len(cleaned)}."
        )

    examples: list[Example] = []
    seen = 0
    for question, answer in zip(cleaned[0::2], cleaned[1::2]):
        seen += 1
        if not should_keep(question, answer):
            continue
        examples.append(
            Example(
                id=stable_id(module, difficulty, question, answer),
                question=question,
                answer=answer.strip().replace(",", ""),
                module=module,
                difficulty=difficulty,
                source="deepmind_mathematics_dataset_v1.0",
            )
        )

    stats = BucketStats(
        module=module,
        difficulty=difficulty,
        seen=seen,
        kept_after_filtering=len(examples),
        sampled=0,
    )
    return examples, stats


def read_bucket(
    tar: tarfile.TarFile, module: str, difficulty: str
) -> tuple[list[Example], BucketStats]:
    """Read and filter one module/difficulty file from the dataset archive."""

    name = member_name(module, difficulty)
    try:
        member = tar.getmember(name)
    except KeyError as exc:
        raise FileNotFoundError(f"Archive member not found: {name}") from exc

    file_obj = tar.extractfile(member)
    if file_obj is None:
        raise OSError(f"Could not extract archive member: {name}")

    with file_obj:
        text = file_obj.read().decode("utf-8")
    lines = text.splitlines()
    return parse_lines_as_examples(lines, module=module, difficulty=difficulty)


def target_counts(buckets: Sequence[tuple[str, str]], max_examples: int) -> dict[tuple[str, str], int]:
    """Compute approximately balanced per-bucket target counts."""

    if max_examples <= 0:
        raise ValueError("--max-examples must be positive")
    if not buckets:
        raise ValueError("No buckets were requested")

    base = max_examples // len(buckets)
    remainder = max_examples % len(buckets)
    counts: dict[tuple[str, str], int] = {}
    for i, bucket in enumerate(buckets):
        counts[bucket] = base + (1 if i < remainder else 0)
    return counts


def load_and_sample_examples(
    archive: Path,
    modules: Sequence[str],
    difficulties: Sequence[str],
    max_examples: int,
    seed: int,
) -> tuple[list[Example], list[BucketStats]]:
    """Load requested buckets, filter examples, and sample a balanced subset."""

    rng = random.Random(seed)
    buckets = [(module, difficulty) for module in modules for difficulty in difficulties]
    targets = target_counts(buckets, max_examples)

    sampled_examples: list[Example] = []
    all_stats: list[BucketStats] = []

    with tarfile.open(archive, mode="r:gz") as tar:
        for module, difficulty in buckets:
            logging.info("Reading bucket module=%s difficulty=%s", module, difficulty)
            examples, stats = read_bucket(tar, module=module, difficulty=difficulty)
            rng.shuffle(examples)
            selected = examples[: targets[(module, difficulty)]]
            sampled_examples.extend(selected)
            all_stats.append(
                BucketStats(
                    module=stats.module,
                    difficulty=stats.difficulty,
                    seen=stats.seen,
                    kept_after_filtering=stats.kept_after_filtering,
                    sampled=len(selected),
                )
            )
            logging.info(
                "Bucket done: seen=%d kept=%d sampled=%d",
                stats.seen,
                stats.kept_after_filtering,
                len(selected),
            )

    # If some buckets were underfilled, top up from all remaining eligible buckets.
    deficit = max_examples - len(sampled_examples)
    if deficit > 0:
        logging.warning(
            "Initial balanced sample has %d examples; target was %d. "
            "No top-up is performed because selected buckets may be exhausted.",
            len(sampled_examples),
            max_examples,
        )

    rng.shuffle(sampled_examples)
    return sampled_examples[:max_examples], all_stats


def write_jsonl(examples: Iterable[Example], path: Path) -> int:
    """Write examples as JSONL and return the number of rows written."""

    count = 0
    with path.open("w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(asdict(example), ensure_ascii=False) + "\n")
            count += 1
    return count


def write_metadata(
    path: Path,
    args: argparse.Namespace,
    archive: Path,
    num_examples: int,
    stats: Sequence[BucketStats],
) -> None:
    """Write a metadata JSON file describing the generated subset."""

    module_counts = Counter(stat.module for stat in stats for _ in range(stat.sampled))
    difficulty_counts = Counter(stat.difficulty for stat in stats for _ in range(stat.sampled))
    bucket_counts = {
        f"{stat.module}/{stat.difficulty}": stat.sampled for stat in stats
    }

    metadata = {
        "dataset": "DeepMind Mathematics Dataset",
        "source_url": DATASET_URL,
        "archive_path": str(archive),
        "num_examples": num_examples,
        "max_examples_requested": args.max_examples,
        "seed": args.seed,
        "modules": list(args.modules),
        "difficulties": list(args.difficulties),
        "answer_filter": "simple numeric answers only: integers, decimals, or rational numbers",
        "output_format": "jsonl with fields: id, question, answer, module, difficulty, source",
        "module_counts": dict(module_counts),
        "difficulty_counts": dict(difficulty_counts),
        "bucket_counts": bucket_counts,
        "bucket_stats": [asdict(stat) for stat in stats],
    }
    path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def validate_args(args: argparse.Namespace) -> None:
    """Validate user-provided arguments early."""

    if args.max_examples <= 0:
        raise ValueError("--max-examples must be positive")
    if not args.modules:
        raise ValueError("At least one module must be provided")
    if not args.difficulties:
        raise ValueError("At least one difficulty must be provided")


def main() -> None:
    """CLI entry point."""

    args = parse_args()
    configure_logging(args.log_level)
    validate_args(args)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    archive = download_archive(args.output_dir, args.archive_path, args.force_download)

    examples, stats = load_and_sample_examples(
        archive=archive,
        modules=args.modules,
        difficulties=args.difficulties,
        max_examples=args.max_examples,
        seed=args.seed,
    )

    dataset_path = args.output_dir / "filtered_math_dataset.jsonl"
    metadata_path = args.output_dir / "metadata.json"
    num_written = write_jsonl(examples, dataset_path)
    write_metadata(metadata_path, args, archive, num_written, stats)

    logging.info("Saved %d examples to %s", num_written, dataset_path)
    logging.info("Saved metadata to %s", metadata_path)


if __name__ == "__main__":
    main()
