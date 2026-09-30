#!/usr/bin/env python3
from __future__ import annotations

"""Temporary SFT ablation suite for the richer Spatial Contradiction v3 dataset.

This runner is intentionally a thin orchestration layer around the existing
project training/evaluation stack.  It does not patch ``dataset_adapters.py``
or permanently register diagnostic datasets.  Instead it:

1. imports the generator from ``data_utils/spatial_contradiction2``;
2. constructs deterministic Arrow caches for the requested train/eval views;
3. registers process-local adapters that reuse ``SpatialContradiction2Adapter``;
4. launches the existing ``main.py`` SFT path;
5. evaluates only the final checkpoint (the only checkpoint written);
6. optionally runs final-only LM-Eval and its existing summarizer;
7. archives metrics to resumable JSON reports; and
8. runs the repository checkpoint cleanup only after every requested evaluation.

The suite is designed to answer three questions:

* Sample efficiency: does richer surface variation delay the abrupt SFT
  saturation seen in the original benchmark?
* Semantic-family transfer: does the learned R90 rule generalize to a lexical
  family that never appears in training?
* Stateful transfer: how much harder is the per-hop rotating-frame variant,
  under the same family-holdout protocol?

OOD protocol
------------
The default held-out family is ``clock_face``.  It is completely removed from
both questions and reference solutions during OOD training.  The OOD eval uses
only clock-face realizations, while the paired ID eval uses the remaining
families.  Both eval views have the same latent graphs, directions, distances,
anchors, statement order, and prompt-shell template IDs.  This makes the OOD
comparison a controlled surface-semantic transfer test rather than a chain-
length test.

Training defaults intentionally follow the user's requested ablation:
Qwen/Qwen2.5-7B-Instruct, CE/SFT on dataset targets, seed 13, global batch 32
(on one process), cosine schedule/warmup inherited from main.py, and LR 2e-4.
The learning rate is exposed as a CLI override because 2e-4 is deliberately
aggressive relative to the earlier core sweep.

Experiments
-----------
3_1_sft_500
    500 rows, 1 epoch, fixed R90, no LM-Eval.
3_2_sft_4k_ep1
    4,000 rows, 1 epoch, fixed R90, LM-Eval.
3_3_sft_4k_ep2
    4,000 rows, 2 epochs, fixed R90, LM-Eval.
3_4_sft_4k_ep2_ood_r90
    4,000 rows x 2 epochs, clock_face held out in training; paired ID/OOD
    evaluation under fixed R90 semantics; LM-Eval.
3_5_sft_4k_ep2_ood_ordinary
    Same family holdout, but ordinary (non-rotated) semantics; LM-Eval.
3_6_sft_4k_ep2_ood_rotating
    Same family holdout, but hop i uses R90^i semantics; LM-Eval.
3_7_sft_2k
    2,000 rows, 1 epoch, fixed R90, no LM-Eval.

The script is resumable at stage boundaries.  The central report is rewritten
atomically after dataset preparation and after train/eval/lm_eval/cleanup for
every experiment.  Cleanup never runs before all required task evaluations and
LM-Eval have been archived.
"""

import argparse
import dataclasses
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import runpy
import shlex
import shutil
import subprocess
import sys
import tempfile
import traceback
from typing import Any, Iterable, Mapping, Sequence


SCRIPT_VERSION = "1.0.0"
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
TRAIN_SEED = 13
GENERATION_SEED = 37
SUBSET_SELECTION_SEED = 20260812
DEFAULT_LEARNING_RATE = "2e-4"
NUM_PROMPTS_PER_BATCH = 32
DEFAULT_HELDOUT_FAMILY = "clock_face"
DEFAULT_BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_DIR = DEFAULT_BASE_DIR / "data_utils" / "spatial_contradiction2"
DEFAULT_OUTPUT_ROOT = DEFAULT_BASE_DIR / "outputs_spatial_contradiction2_ablation"
DEFAULT_LM_EVAL_TASKS = "hellaswag,mmlu,truthfulqa,winogrande,humaneval,ifeval"
REPORT_FILENAME = "spatial_contradiction2_ablation_report.json"
DATASET_MANIFEST_FILENAME = "dataset_manifest.json"

WORLD_R90 = "r90"
WORLD_ORDINARY = "normal"
WORLD_ROTATING = "rotating_r90"


@dataclass(frozen=True)
class ExperimentSpec:
    key: str
    train_rows: int
    epochs: int
    world: str
    family_holdout: bool
    run_lm_eval: bool
    description: str

    @property
    def expected_optimizer_steps(self) -> int:
        # This matches the user's prior one-process diagnostics: 500 -> 15,
        # 2,000 -> 62, 4,000 -> 125 optimizer steps per epoch.  DistilTrainer's
        # generation/training sampler drops the incomplete global batch.
        return (self.train_rows // NUM_PROMPTS_PER_BATCH) * self.epochs


EXPERIMENTS: tuple[ExperimentSpec, ...] = (
    ExperimentSpec(
        "3_1_sft_500", 500, 1, WORLD_R90, False, False,
        "Fixed-R90 SFT sample-efficiency probe with 500 training rows.",
    ),
    ExperimentSpec(
        "3_2_sft_4k_ep1", 4000, 1, WORLD_R90, False, True,
        "Fixed-R90 SFT on the full 4k train set for one epoch.",
    ),
    ExperimentSpec(
        "3_3_sft_4k_ep2", 4000, 2, WORLD_R90, False, True,
        "Fixed-R90 SFT on 4k rows for two epochs (8k row exposures).",
    ),
    ExperimentSpec(
        "3_4_sft_4k_ep2_ood_r90", 4000, 2, WORLD_R90, True, True,
        "Fixed-R90 SFT with one semantic family held out from training.",
    ),
    ExperimentSpec(
        "3_5_sft_4k_ep2_ood_ordinary", 4000, 2, WORLD_ORDINARY, True, True,
        "Ordinary-world SFT with the same semantic-family holdout.",
    ),
    ExperimentSpec(
        "3_6_sft_4k_ep2_ood_rotating", 4000, 2, WORLD_ROTATING, True, True,
        "Per-hop rotating-frame SFT with the same semantic-family holdout.",
    ),
    ExperimentSpec(
        "3_7_sft_2k", 2000, 1, WORLD_R90, False, False,
        "Fixed-R90 SFT sample-efficiency probe with 2,000 training rows.",
    ),
)
EXPERIMENT_BY_KEY = {spec.key: spec for spec in EXPERIMENTS}


@dataclass(frozen=True)
class DatasetBinding:
    train_dataset_name: str
    train_path: Path
    eval_bindings: tuple[tuple[str, Path, str], ...]
    # Each eval tuple is: (temporary dataset name, Arrow path, role)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def str_to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value != 0
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent), text=True
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except Exception:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass
        raise


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def marker(path: Path, name: str) -> Path:
    return path / name


def is_marked(path: Path, name: str) -> bool:
    return marker(path, name).exists()


def mark(path: Path, name: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    marker(path, name).write_text(f"{utc_now()}\n", encoding="utf-8")


def write_command(path: Path, filename: str, cmd: Sequence[str]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / filename).write_text(shlex.join([str(x) for x in cmd]) + "\n", encoding="utf-8")


def run_subprocess(
    cmd: Sequence[str],
    *,
    cwd: Path,
    env_updates: Mapping[str, Any] | None = None,
    log_path: Path | None = None,
) -> None:
    command = [str(x) for x in cmd]
    print("\n" + "=" * 100)
    print(shlex.join(command))
    print("=" * 100, flush=True)

    env = os.environ.copy()
    if env_updates:
        env.update({str(k): str(v) for k, v in env_updates.items()})

    if log_path is None:
        subprocess.run(command, check=True, cwd=str(cwd), env=env)
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", buffering=1) as log_handle:
        log_handle.write(f"\n[{utc_now()}] COMMAND: {shlex.join(command)}\n\n")
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_handle.write(line)
        returncode = process.wait()
        log_handle.write(f"\n[{utc_now()}] EXIT CODE: {returncode}\n")
        if returncode != 0:
            raise subprocess.CalledProcessError(returncode, command)


def _import_generator(source_dir: Path):
    """Import the local v3 core/builder without requiring project installation."""
    source_dir = source_dir.resolve()
    core_path = source_dir / "spatial_contradiction_core.py"
    builder_path = source_dir / "build_spatial_contradiction_dataset.py"
    if not core_path.exists() or not builder_path.exists():
        raise FileNotFoundError(
            "Expected spatial_contradiction_core.py and "
            f"build_spatial_contradiction_dataset.py under {source_dir}"
        )

    source_text = core_path.read_text(encoding="utf-8")
    if "rotating_r90" not in source_text or "rotate_quarter_turns" not in source_text:
        raise RuntimeError(
            "The source core does not contain the rotating-frame ablation support. "
            "Copy the final core file supplied with this runner before running the suite."
        )

    source_str = str(source_dir)
    if source_str not in sys.path:
        sys.path.insert(0, source_str)

    # Avoid silently binding the builder to a same-named module from another
    # Spatial Contradiction directory.
    existing = sys.modules.get("spatial_contradiction_core")
    if existing is not None:
        existing_file = Path(getattr(existing, "__file__", "")).resolve()
        if existing_file != core_path.resolve():
            del sys.modules["spatial_contradiction_core"]

    core = importlib.import_module("spatial_contradiction_core")

    existing_builder = sys.modules.get("build_spatial_contradiction_dataset")
    if existing_builder is not None:
        existing_file = Path(getattr(existing_builder, "__file__", "")).resolve()
        if existing_file != builder_path.resolve():
            del sys.modules["build_spatial_contradiction_dataset"]
    builder = importlib.import_module("build_spatial_contradiction_dataset")
    return core, builder


def _family_inventory(core) -> dict[str, Any]:
    phrase_counts: dict[str, int] = {family: 0 for family in core.PHRASE_SEMANTIC_FAMILIES}
    for spec in core.DIRECTION_SPECS.values():
        for phrase in (*spec.relative_phrases, *spec.motion_phrases):
            phrase_counts[phrase.semantic_family] += 1
    return {
        "families": list(core.PHRASE_SEMANTIC_FAMILIES),
        "num_families": len(core.PHRASE_SEMANTIC_FAMILIES),
        "phrase_specs_by_family": dict(sorted(phrase_counts.items())),
        "total_phrase_specs": sum(phrase_counts.values()),
        "num_statement_families": len(core.STATEMENT_FAMILIES),
        "num_solution_step_families": len(core.SOLUTION_STEP_FAMILIES),
        "num_anchor_templates": len(core.ANCHOR_TEMPLATES),
        "num_query_templates": len(core.QUERY_TEMPLATES),
        "num_solution_start_templates": len(core.SOLUTION_START_TEMPLATES),
        "num_solution_final_templates": len(core.SOLUTION_FINAL_TEMPLATES),
    }


def _families_without(core, heldout_family: str) -> frozenset[str]:
    all_families = frozenset(str(x) for x in core.PHRASE_SEMANTIC_FAMILIES)
    if heldout_family not in all_families:
        raise ValueError(
            f"Unknown held-out family {heldout_family!r}; available: {sorted(all_families)}"
        )
    remaining = frozenset(all_families - {heldout_family})
    issues = core.validate_phrase_family_coverage(remaining)
    if issues:
        raise ValueError(
            f"Training family set is not semantically complete after holding out "
            f"{heldout_family!r}: {issues}"
        )
    ood_issues = core.validate_phrase_family_coverage(frozenset({heldout_family}))
    if ood_issues:
        raise ValueError(
            f"Held-out family {heldout_family!r} cannot stand alone as a paired OOD "
            f"evaluation family: {ood_issues}"
        )
    return remaining


def _build_train_eval_records(
    *,
    builder,
    train_families: frozenset[str] | None,
    eval_families: frozenset[str] | None,
    generation_seed: int,
    origin_probability: float,
    anchor_max_abs: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    seen_problem_hashes: set[str] = set()
    seen_graph_hashes: set[str] = set()
    kwargs = dict(
        root_seed=generation_seed,
        min_hops=builder.DEFAULT_MIN_HOPS,
        max_hops=builder.DEFAULT_MAX_HOPS,
        reverse_probability=builder.DEFAULT_REVERSE_PROB,
        distance_weights=builder.DEFAULT_DISTANCE_WEIGHTS,
        origin_probability=origin_probability,
        anchor_max_abs=anchor_max_abs,
    )
    train = builder.build_split(
        split="train",
        size=4000,
        seen_problem_hashes=seen_problem_hashes,
        seen_graph_hashes=seen_graph_hashes,
        allowed_phrase_semantic_families=train_families,
        **kwargs,
    )
    eval_records = builder.build_split(
        split="eval",
        size=500,
        seen_problem_hashes=seen_problem_hashes,
        seen_graph_hashes=seen_graph_hashes,
        allowed_phrase_semantic_families=eval_families,
        **kwargs,
    )
    return train, eval_records


def _build_paired_ood_eval_records(
    *,
    builder,
    train_records: Sequence[Mapping[str, Any]],
    heldout_family: str,
    generation_seed: int,
    origin_probability: float,
    anchor_max_abs: int,
) -> list[dict[str, Any]]:
    """Regenerate eval with held-out wording while preserving latent instances.

    ``build_split`` sees the same train graph hashes as the ID generation.  The
    patched builder uses independent RNG streams for graph, statement order,
    and prompt shell, so changing the phrase allowlist cannot perturb those
    latent/presentation-control variables.
    """
    seen_graph_hashes = {str(row["graph_hash"]) for row in train_records}
    return builder.build_split(
        split="eval",
        size=500,
        root_seed=generation_seed,
        min_hops=builder.DEFAULT_MIN_HOPS,
        max_hops=builder.DEFAULT_MAX_HOPS,
        reverse_probability=builder.DEFAULT_REVERSE_PROB,
        distance_weights=builder.DEFAULT_DISTANCE_WEIGHTS,
        seen_problem_hashes=set(),
        seen_graph_hashes=set(seen_graph_hashes),
        origin_probability=origin_probability,
        anchor_max_abs=anchor_max_abs,
        allowed_phrase_semantic_families=frozenset({heldout_family}),
    )


def _row_identity(row: Mapping[str, Any], index: int) -> str:
    for key in ("graph_hash", "problem_hash", "source_id"):
        value = row.get(key)
        if value not in (None, ""):
            return f"{key}:{value}"
    return f"index:{index}"


def _subset_rank(row: Mapping[str, Any], index: int) -> str:
    payload = (
        f"spatial-contradiction2-ablation::{SUBSET_SELECTION_SEED}::"
        f"{_row_identity(row, index)}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def stratified_nested_subset(
    records: Sequence[Mapping[str, Any]],
    *,
    size: int,
    hops: Iterable[int] = range(1, 7),
) -> list[dict[str, Any]]:
    """Deterministic hop-balanced subset; 500 is nested in the 2k subset."""
    hop_values = tuple(sorted({int(x) for x in hops}))
    if size <= 0:
        raise ValueError("size must be positive")
    base_quota, remainder = divmod(size, len(hop_values))
    quotas = {
        hop: base_quota + (1 if i < remainder else 0)
        for i, hop in enumerate(hop_values)
    }
    grouped: dict[int, list[tuple[str, int]]] = {hop: [] for hop in hop_values}
    for index, row in enumerate(records):
        hop = int(row["hop_count"])
        if hop in grouped:
            grouped[hop].append((_subset_rank(row, index), index))

    chosen: list[tuple[str, int]] = []
    for hop in hop_values:
        candidates = sorted(grouped[hop])
        quota = quotas[hop]
        if len(candidates) < quota:
            raise RuntimeError(
                f"Need {quota} rows for hop={hop}, only {len(candidates)} available"
            )
        chosen.extend(candidates[:quota])
    chosen.sort()
    return [dict(records[index]) for _, index in chosen]


def _world_row(
    row: Mapping[str, Any],
    *,
    world: str,
    core,
    builder,
) -> dict[str, Any]:
    """Create one model-facing row for a diagnostic semantic world."""
    result = dict(builder.final_record(row))
    result["diagnostic_world"] = world

    if world == WORLD_R90:
        return result

    if world == WORLD_ORDINARY:
        result.update(
            {
                "messages": result.get("original_messages", result["messages"]),
                "problem": result.get("original_problem", result["problem"]),
                "answer": result["original_answer"],
                "output_text": result["original_output_text"],
                "visible_output_text": result["original_output_text"],
                "golden_answer": result["original_answer"],
                "golden_response": result["original_output_text"],
                "transformation": "ordinary_direction_semantics_control",
            }
        )
        return result

    if world == WORLD_ROTATING:
        anchor_values = list(result["anchor_coordinate"])
        anchor_coord = (int(anchor_values[0]), int(anchor_values[1]))
        allowlist = frozenset(str(x) for x in result["surface_phrase_family_allowlist"])
        solution_seed = core.stable_int(int(result["instance_seed"]), "solution_style")
        output_text, final_coord, trace = core.render_solution(
            anchor=str(result["anchor_entity"]),
            anchor_coord=anchor_coord,
            target=str(result["target_entity"]),
            edges=result["edges"],
            world=WORLD_ROTATING,
            solution_seed=solution_seed,
            allowed_semantic_families=allowlist,
        )
        answer = core.format_coord(final_coord)
        result.update(
            {
                "answer": answer,
                "output_text": output_text,
                "visible_output_text": output_text,
                "golden_answer": answer,
                "golden_response": output_text,
                "mod_answer": answer,
                "mod_output_text": output_text,
                "transformation": core.ROTATING_TRANSFORMATION_NAME,
                "diagnostic_frame_quarter_turns": [
                    int(step["frame_quarter_turns"]) for step in trace
                ],
            }
        )
        verifier = core.SpatialCoordinateVerifier()
        check = verifier.verify_with_details(answer, output_text)
        if not check.correct:
            raise AssertionError(
                f"Generated rotating-frame target failed exact verification: {check.error}"
            )
        return result

    raise ValueError(f"Unknown world: {world!r}")


def _convert_records(
    records: Sequence[Mapping[str, Any]],
    *,
    world: str,
    core,
    builder,
) -> list[dict[str, Any]]:
    return [_world_row(row, world=world, core=core, builder=builder) for row in records]


def _save_arrow(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    try:
        from datasets import Dataset  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Hugging Face 'datasets' is required for prepare/run on the training machine. "
            "The pure logic self-test does not require it."
        ) from exc
    if path.exists():
        shutil.rmtree(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list([dict(row) for row in records]).save_to_disk(str(path))


def _paired_surface_checks(
    id_rows: Sequence[Mapping[str, Any]],
    ood_rows: Sequence[Mapping[str, Any]],
    *,
    heldout_family: str,
    training_families: frozenset[str],
) -> dict[str, Any]:
    if len(id_rows) != len(ood_rows):
        raise AssertionError("Paired ID/OOD eval lengths differ")
    counters = {
        "graph_hash": 0,
        "directions": 0,
        "distances": 0,
        "anchor_coordinate": 0,
        "statement_order": 0,
        "anchor_template_id": 0,
        "query_template_id": 0,
    }
    for id_row, ood_row in zip(id_rows, ood_rows):
        for field in counters:
            if id_row[field] == ood_row[field]:
                counters[field] += 1
        if not set(id_row["phrase_semantic_families"]).issubset(training_families):
            raise AssertionError("ID eval contains the held-out question family")
        if not set(id_row["solution_phrase_semantic_families"]).issubset(training_families):
            raise AssertionError("ID eval target contains the held-out solution family")
        if set(ood_row["phrase_semantic_families"]) != {heldout_family}:
            raise AssertionError("OOD eval question is not exclusively held-out-family wording")
        if set(ood_row["solution_phrase_semantic_families"]) != {heldout_family}:
            raise AssertionError("OOD eval target is not exclusively held-out-family wording")

    if any(value != len(id_rows) for value in counters.values()):
        raise AssertionError(f"Paired ID/OOD controls are not invariant: {counters}")
    return {field: {"matches": count, "total": len(id_rows)} for field, count in counters.items()}


def _family_usage(records: Sequence[Mapping[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in records:
        for value in row.get(key, []):
            text = str(value)
            counts[text] = counts.get(text, 0) + 1
    return dict(sorted(counts.items()))


def _record_stats(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    hop_counts: dict[str, int] = {}
    origin_count = 0
    for row in records:
        hop = str(int(row["hop_count"]))
        hop_counts[hop] = hop_counts.get(hop, 0) + 1
        if list(row["anchor_coordinate"]) == [0, 0]:
            origin_count += 1
    return {
        "num_rows": len(records),
        "hop_counts": dict(sorted(hop_counts.items(), key=lambda item: int(item[0]))),
        "question_family_counts": _family_usage(records, "phrase_semantic_families"),
        "solution_family_counts": _family_usage(records, "solution_phrase_semantic_families"),
        "statement_syntax_counts": _family_usage(records, "statement_syntax_families"),
        "solution_syntax_counts": _family_usage(records, "solution_step_syntax_families"),
        "origin_anchor_count": origin_count,
        "origin_anchor_fraction": origin_count / len(records) if records else None,
    }


def _dataset_cache_root(output_root: Path) -> Path:
    return output_root / "_dataset_cache"


def _dataset_paths(output_root: Path) -> dict[str, Path]:
    root = _dataset_cache_root(output_root)
    return {
        "all_r90_train_4k": root / "all_r90" / "train_4k",
        "all_r90_train_2k": root / "all_r90" / "train_2k",
        "all_r90_train_500": root / "all_r90" / "train_500",
        "all_r90_eval": root / "all_r90" / "eval_500",
        "ood_r90_train": root / "ood_r90" / "train_no_heldout_4k",
        "ood_r90_id": root / "ood_r90" / "eval_id_500",
        "ood_r90_ood": root / "ood_r90" / "eval_ood_500",
        "ood_normal_train": root / "ood_normal" / "train_no_heldout_4k",
        "ood_normal_id": root / "ood_normal" / "eval_id_500",
        "ood_normal_ood": root / "ood_normal" / "eval_ood_500",
        "ood_rotating_train": root / "ood_rotating" / "train_no_heldout_4k",
        "ood_rotating_id": root / "ood_rotating" / "eval_id_500",
        "ood_rotating_ood": root / "ood_rotating" / "eval_ood_500",
    }


def _dataset_bindings(output_root: Path) -> dict[str, DatasetBinding]:
    paths = _dataset_paths(output_root)
    return {
        "3_1_sft_500": DatasetBinding(
            "sp2ab_31_train", paths["all_r90_train_500"],
            (("sp2ab_31_eval", paths["all_r90_eval"], "eval"),),
        ),
        "3_2_sft_4k_ep1": DatasetBinding(
            "sp2ab_32_train", paths["all_r90_train_4k"],
            (("sp2ab_32_eval", paths["all_r90_eval"], "eval"),),
        ),
        "3_3_sft_4k_ep2": DatasetBinding(
            "sp2ab_33_train", paths["all_r90_train_4k"],
            (("sp2ab_33_eval", paths["all_r90_eval"], "eval"),),
        ),
        "3_4_sft_4k_ep2_ood_r90": DatasetBinding(
            "sp2ab_34_train", paths["ood_r90_train"],
            (
                ("sp2ab_34_id", paths["ood_r90_id"], "id"),
                ("sp2ab_34_ood", paths["ood_r90_ood"], "ood"),
            ),
        ),
        "3_5_sft_4k_ep2_ood_ordinary": DatasetBinding(
            "sp2ab_35_train", paths["ood_normal_train"],
            (
                ("sp2ab_35_id", paths["ood_normal_id"], "id"),
                ("sp2ab_35_ood", paths["ood_normal_ood"], "ood"),
            ),
        ),
        "3_6_sft_4k_ep2_ood_rotating": DatasetBinding(
            "sp2ab_36_train", paths["ood_rotating_train"],
            (
                ("sp2ab_36_id", paths["ood_rotating_id"], "id"),
                ("sp2ab_36_ood", paths["ood_rotating_ood"], "ood"),
            ),
        ),
        "3_7_sft_2k": DatasetBinding(
            "sp2ab_37_train", paths["all_r90_train_2k"],
            (("sp2ab_37_eval", paths["all_r90_eval"], "eval"),),
        ),
    }


def prepare_datasets(
    *,
    output_root: Path,
    source_dir: Path,
    heldout_family: str,
    generation_seed: int,
    origin_probability: float,
    anchor_max_abs: int,
    force: bool,
    write_arrow: bool = True,
) -> dict[str, Any]:
    """Build and validate all reusable dataset views for the suite."""
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = _dataset_cache_root(output_root) / DATASET_MANIFEST_FILENAME
    core, builder = _import_generator(source_dir)

    source_hashes = {
        "core_sha256": sha256_file(source_dir / "spatial_contradiction_core.py"),
        "builder_sha256": sha256_file(source_dir / "build_spatial_contradiction_dataset.py"),
    }
    requested_signature = {
        "script_version": SCRIPT_VERSION,
        "generation_seed": generation_seed,
        "heldout_family": heldout_family,
        "origin_probability": origin_probability,
        "anchor_max_abs": anchor_max_abs,
        **source_hashes,
    }

    if manifest_path.exists() and not force:
        existing = read_json(manifest_path)
        if existing.get("cache_signature") == requested_signature:
            if not write_arrow or all(path.exists() for path in _dataset_paths(output_root).values()):
                print(f"[dataset cache] reusing validated cache under {_dataset_cache_root(output_root)}")
                return existing
        raise RuntimeError(
            "An incompatible dataset cache already exists. Use --force-prepare to rebuild it.\n"
            f"Existing signature: {existing.get('cache_signature')}\n"
            f"Requested signature: {requested_signature}"
        )

    if force and _dataset_cache_root(output_root).exists():
        shutil.rmtree(_dataset_cache_root(output_root))

    all_families = frozenset(str(x) for x in core.PHRASE_SEMANTIC_FAMILIES)
    training_families = _families_without(core, heldout_family)

    print("[prepare] generating all-family train/eval records", flush=True)
    all_train, all_eval = _build_train_eval_records(
        builder=builder,
        train_families=None,
        eval_families=None,
        generation_seed=generation_seed,
        origin_probability=origin_probability,
        anchor_max_abs=anchor_max_abs,
    )

    print("[prepare] generating family-holdout ID train/eval records", flush=True)
    id_train, id_eval = _build_train_eval_records(
        builder=builder,
        train_families=training_families,
        eval_families=training_families,
        generation_seed=generation_seed,
        origin_probability=origin_probability,
        anchor_max_abs=anchor_max_abs,
    )
    print("[prepare] generating paired held-out-family OOD eval", flush=True)
    ood_eval = _build_paired_ood_eval_records(
        builder=builder,
        train_records=id_train,
        heldout_family=heldout_family,
        generation_seed=generation_seed,
        origin_probability=origin_probability,
        anchor_max_abs=anchor_max_abs,
    )

    # Independent exact validation before any world conversion.
    for label, records in (
        ("all_train", all_train), ("all_eval", all_eval),
        ("id_train", id_train), ("id_eval", id_eval), ("ood_eval", ood_eval),
    ):
        invalid: list[dict[str, Any]] = []
        for i, record in enumerate(records):
            issues = builder.validate_record(record)
            if issues:
                invalid.append({"index": i, "issues": issues})
                if len(invalid) >= 10:
                    break
        if invalid:
            raise AssertionError(f"{label} contains invalid generated records: {invalid}")

    all_train_graphs = {str(row["graph_hash"]) for row in all_train}
    all_eval_graphs = {str(row["graph_hash"]) for row in all_eval}
    id_train_graphs = {str(row["graph_hash"]) for row in id_train}
    id_eval_graphs = {str(row["graph_hash"]) for row in id_eval}
    if all_train_graphs & all_eval_graphs:
        raise AssertionError("All-family train/eval graph leakage")
    if id_train_graphs & id_eval_graphs:
        raise AssertionError("Family-holdout train/eval graph leakage")

    pair_checks = _paired_surface_checks(
        id_eval, ood_eval,
        heldout_family=heldout_family,
        training_families=training_families,
    )

    train_500 = stratified_nested_subset(all_train, size=500)
    train_2000 = stratified_nested_subset(all_train, size=2000)
    ids_500 = {str(row["graph_hash"]) for row in train_500}
    ids_2000 = {str(row["graph_hash"]) for row in train_2000}
    if not ids_500.issubset(ids_2000):
        raise AssertionError("500-row training subset is not nested in 2k subset")

    converted = {
        "all_r90_train_4k": _convert_records(all_train, world=WORLD_R90, core=core, builder=builder),
        "all_r90_train_2k": _convert_records(train_2000, world=WORLD_R90, core=core, builder=builder),
        "all_r90_train_500": _convert_records(train_500, world=WORLD_R90, core=core, builder=builder),
        "all_r90_eval": _convert_records(all_eval, world=WORLD_R90, core=core, builder=builder),
        "ood_r90_train": _convert_records(id_train, world=WORLD_R90, core=core, builder=builder),
        "ood_r90_id": _convert_records(id_eval, world=WORLD_R90, core=core, builder=builder),
        "ood_r90_ood": _convert_records(ood_eval, world=WORLD_R90, core=core, builder=builder),
        "ood_normal_train": _convert_records(id_train, world=WORLD_ORDINARY, core=core, builder=builder),
        "ood_normal_id": _convert_records(id_eval, world=WORLD_ORDINARY, core=core, builder=builder),
        "ood_normal_ood": _convert_records(ood_eval, world=WORLD_ORDINARY, core=core, builder=builder),
        "ood_rotating_train": _convert_records(id_train, world=WORLD_ROTATING, core=core, builder=builder),
        "ood_rotating_id": _convert_records(id_eval, world=WORLD_ROTATING, core=core, builder=builder),
        "ood_rotating_ood": _convert_records(ood_eval, world=WORLD_ROTATING, core=core, builder=builder),
    }

    paths = _dataset_paths(output_root)
    if write_arrow:
        print("[prepare] writing Arrow caches", flush=True)
        for key, records in converted.items():
            _save_arrow(paths[key], records)

    inventory = _family_inventory(core)
    heldout_phrase_specs = inventory["phrase_specs_by_family"][heldout_family]
    retained_phrase_specs = inventory["total_phrase_specs"] - heldout_phrase_specs
    manifest = {
        "created_at": utc_now(),
        "cache_signature": requested_signature,
        "source": {
            "source_dir": str(source_dir),
            "core_version": getattr(core, "VERSION", None),
            "builder_version": getattr(builder, "VERSION", None),
            **source_hashes,
        },
        "policy": {
            "generation_seed": generation_seed,
            "heldout_family": heldout_family,
            "all_families": sorted(all_families),
            "training_families": sorted(training_families),
            "num_training_families": len(training_families),
            "origin_probability": origin_probability,
            "anchor_max_abs": anchor_max_abs,
            "family_holdout_retained_phrase_specs": retained_phrase_specs,
            "family_holdout_total_phrase_specs": inventory["total_phrase_specs"],
            "family_holdout_retained_phrase_fraction": (
                retained_phrase_specs / inventory["total_phrase_specs"]
            ),
        },
        "inventory": inventory,
        "checks": {
            "generator_exact_validation": True,
            "all_family_train_eval_graph_overlap": 0,
            "family_holdout_train_eval_graph_overlap": 0,
            "paired_id_ood_surface_controls": pair_checks,
            "train_500_nested_in_train_2000": True,
        },
        "stats": {
            "all_train_4k": _record_stats(all_train),
            "all_eval_500": _record_stats(all_eval),
            "train_500": _record_stats(train_500),
            "train_2000": _record_stats(train_2000),
            "family_holdout_train_4k": _record_stats(id_train),
            "family_holdout_id_eval_500": _record_stats(id_eval),
            "heldout_family_ood_eval_500": _record_stats(ood_eval),
        },
        "arrow_written": write_arrow,
        "paths": {key: str(path) for key, path in paths.items()},
    }
    atomic_write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2))
    print(f"[prepare] saved manifest: {manifest_path}")
    return manifest


def _install_temporary_adapters(output_root: Path) -> dict[str, DatasetBinding]:
    """Register all temporary dataset names inside the current Python process."""
    try:
        import dataset_adapters as da
    except ImportError as exc:
        raise ImportError(
            "Could not import dataset_adapters. Run the script with --base-dir pointing "
            "to the sdft repository and from its Python environment."
        ) from exc

    base_spec = da.DATASET_SPECS.get("spatial_contradiction2")
    if base_spec is None:
        raise RuntimeError("Base 'spatial_contradiction2' adapter is not registered")
    base_cls = type(base_spec.adapter)

    bindings = _dataset_bindings(output_root)
    registered: dict[str, tuple[Path, Path]] = {}

    def register(name: str, train_path: Path, eval_path: Path) -> None:
        if not train_path.exists():
            raise FileNotFoundError(f"Missing temporary train Arrow dataset: {train_path}")
        if not eval_path.exists():
            raise FileNotFoundError(f"Missing temporary eval Arrow dataset: {eval_path}")

        class TemporarySpatial2Adapter(base_cls):
            pass

        adapter = TemporarySpatial2Adapter()
        adapter.name = name
        adapter.train_path = str(train_path)
        adapter.eval_path = str(eval_path)
        adapter.test_path = ""
        adapter.results_filename = f"eval_{name}_results.json"
        adapter.responses_filename = f"eval_{name}_responses.json"
        da.DATASET_SPECS[name] = da.DatasetSpec(
            adapter=adapter,
            # Compatibility-only placeholder. The central ablation report uses
            # absolute accuracy and does not interpret this as a measured base.
            baseline_accuracy=0.0,
            expert_accuracy=1.0,
            floor_accuracy=0.0,
            training_only=False,
        )
        da.DATASET_ADAPTERS[name] = adapter
        registered[name] = (train_path, eval_path)

    for binding in bindings.values():
        default_eval_path = binding.eval_bindings[0][1]
        register(binding.train_dataset_name, binding.train_path, default_eval_path)
        for eval_name, eval_path, _role in binding.eval_bindings:
            register(eval_name, binding.train_path, eval_path)

    return bindings


def _run_target_script(
    target: Path,
    forwarded_args: Sequence[str],
    *,
    output_root: Path,
) -> None:
    _install_temporary_adapters(output_root)
    if not target.exists():
        raise FileNotFoundError(f"Missing project script: {target}")
    old_argv = sys.argv[:]
    try:
        sys.argv = [str(target), *[str(x) for x in forwarded_args]]
        runpy.run_path(str(target), run_name="__main__")
    finally:
        sys.argv = old_argv


def _train_internal(args: argparse.Namespace) -> None:
    base_dir = args.base_dir.resolve()
    output_root = args.output_root.resolve()
    train_args = [
        "--dataset_name", args.dataset_name,
        "--model_name", args.model_name,
        "--output_dir", str(args.phase_dir),
        "--alpha", "1",
        "--generate_from_teacher", "FALSE",
        "--ref_model_mixup_alpha", "0.0",
        "--learning_rate", str(args.learning_rate),
        "--sync_ref_model", "FALSE",
        "--optimal_policy_source", "dataset",
        "--optim_loss", "cross_entropy",
        "--num_loss_tokens_to_skip", "0",
        "--num_train_epochs", str(args.epochs),
        "--num_prompts_per_batch", str(NUM_PROMPTS_PER_BATCH),
        "--optim", "paged_adamw_8bit",
        "--save_strategy", "steps",
        "--save_steps", str(args.save_steps),
        "--context_strategy", "dataset_default",
        "--seed", str(TRAIN_SEED),
    ]
    _run_target_script(base_dir / "main.py", train_args, output_root=output_root)


def _eval_internal(args: argparse.Namespace) -> None:
    base_dir = args.base_dir.resolve()
    _run_target_script(
        base_dir / "run_all_checkpoint_evals.py",
        args.forwarded,
        output_root=args.output_root.resolve(),
    )


def _experiment_root(output_root: Path, spec: ExperimentSpec) -> Path:
    return output_root / spec.key


def _phase_dir(output_root: Path, spec: ExperimentSpec) -> Path:
    binding = _dataset_bindings(output_root)[spec.key]
    return _experiment_root(output_root, spec) / binding.train_dataset_name


def _checkpoint_dirs(phase_dir: Path) -> list[tuple[int, Path]]:
    if not phase_dir.exists():
        return []
    out: list[tuple[int, Path]] = []
    for child in phase_dir.iterdir():
        if not child.is_dir() or not child.name.startswith("checkpoint-"):
            continue
        try:
            step = int(child.name.split("checkpoint-", 1)[1])
        except ValueError:
            continue
        out.append((step, child))
    return sorted(out)


def _is_usable_checkpoint(path: Path) -> bool:
    if not path.is_dir() or not (path / "trainer_state.json").exists():
        return False
    if any((path / name).exists() for name in (
        "pytorch_model.bin", "model.safetensors", "adapter_model.bin", "adapter_model.safetensors"
    )):
        return True
    return bool(list(path.glob("*.safetensors")) or list(path.glob("*.bin")))


def _latest_usable_checkpoint(phase_dir: Path) -> Path | None:
    usable = [(step, path) for step, path in _checkpoint_dirs(phase_dir) if _is_usable_checkpoint(path)]
    return usable[-1][1] if usable else None


def _only_final_checkpoint_check(phase_dir: Path, expected_step: int) -> Path:
    checkpoints = [(step, path) for step, path in _checkpoint_dirs(phase_dir) if _is_usable_checkpoint(path)]
    if len(checkpoints) != 1:
        raise RuntimeError(
            f"Expected exactly one usable final checkpoint under {phase_dir}, found "
            f"{[(step, str(path)) for step, path in checkpoints]}"
        )
    step, path = checkpoints[0]
    if step != expected_step:
        raise RuntimeError(
            f"Expected final checkpoint-{expected_step}, found checkpoint-{step}. "
            "This indicates the effective optimizer-step count differs from the "
            "one-process batch-32 diagnostic convention."
        )
    return path


def _report_path(output_root: Path) -> Path:
    return output_root / REPORT_FILENAME


def _initial_report(
    *,
    output_root: Path,
    source_dir: Path,
    learning_rate: str,
    heldout_family: str,
    generation_seed: int,
    origin_probability: float,
    anchor_max_abs: int,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "runner_version": SCRIPT_VERSION,
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "suite": {
            "model": MODEL_NAME,
            "train_seed": TRAIN_SEED,
            "generation_seed": generation_seed,
            "learning_rate": learning_rate,
            "num_prompts_per_batch": NUM_PROMPTS_PER_BATCH,
            "optimizer": "paged_adamw_8bit",
            "optimal_policy_source": "dataset",
            "optim_loss": "cross_entropy",
            "context_strategy": "dataset_default",
            "save_policy": "exactly one final checkpoint",
            "heldout_family": heldout_family,
            "origin_probability": origin_probability,
            "anchor_max_abs": anchor_max_abs,
            "source_dir": str(source_dir),
            "output_root": str(output_root),
            "note": (
                "Temporary registry baseline_accuracy=0 is used only for evaluator "
                "compatibility; this report analyzes absolute task accuracy."
            ),
        },
        "dataset_manifest": None,
        "experiments": {
            spec.key: {
                "spec": dataclasses.asdict(spec),
                "expected_final_step": spec.expected_optimizer_steps,
                "status": "pending",
                "stages": {},
                "results": {},
            }
            for spec in EXPERIMENTS
        },
        "comparisons": {},
    }


def _load_or_create_report(
    *,
    output_root: Path,
    source_dir: Path,
    learning_rate: str,
    heldout_family: str,
    generation_seed: int,
    origin_probability: float,
    anchor_max_abs: int,
) -> dict[str, Any]:
    path = _report_path(output_root)
    if path.exists():
        report = read_json(path)
        # Fail rather than mixing results from incompatible suite settings.
        expected = {
            "learning_rate": str(learning_rate),
            "heldout_family": heldout_family,
            "generation_seed": generation_seed,
            "origin_probability": origin_probability,
            "anchor_max_abs": anchor_max_abs,
        }
        actual = {key: report.get("suite", {}).get(key) for key in expected}
        if actual != expected:
            raise RuntimeError(
                "Existing central report belongs to a different suite configuration. "
                "Use a different --output-root or remove/rebuild the old report.\n"
                f"Existing: {actual}\nRequested: {expected}"
            )
        return report
    report = _initial_report(
        output_root=output_root,
        source_dir=source_dir,
        learning_rate=learning_rate,
        heldout_family=heldout_family,
        generation_seed=generation_seed,
        origin_probability=origin_probability,
        anchor_max_abs=anchor_max_abs,
    )
    atomic_write_json(path, report)
    return report


def _safe_float(value: Any) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _compute_comparisons(report: dict[str, Any]) -> dict[str, Any]:
    exps = report.get("experiments", {})

    def accuracy(key: str, role: str = "eval") -> float | None:
        return _safe_float(
            exps.get(key, {}).get("results", {}).get("task_evaluations", {}).get(role, {}).get("accuracy")
        )

    out: dict[str, Any] = {}
    a500 = accuracy("3_1_sft_500")
    a2k = accuracy("3_7_sft_2k")
    a4k1 = accuracy("3_2_sft_4k_ep1")
    a4k2 = accuracy("3_3_sft_4k_ep2")
    if any(x is not None for x in (a500, a2k, a4k1, a4k2)):
        out["sample_efficiency"] = {
            "500_rows_1ep": a500,
            "2000_rows_1ep": a2k,
            "4000_rows_1ep": a4k1,
            "4000_rows_2ep": a4k2,
            "gain_500_to_2000": None if a500 is None or a2k is None else a2k - a500,
            "gain_2000_to_4000": None if a2k is None or a4k1 is None else a4k1 - a2k,
            "gain_second_epoch": None if a4k1 is None or a4k2 is None else a4k2 - a4k1,
        }

    ood_gaps: dict[str, Any] = {}
    for key, label in (
        ("3_4_sft_4k_ep2_ood_r90", "fixed_r90"),
        ("3_5_sft_4k_ep2_ood_ordinary", "ordinary"),
        ("3_6_sft_4k_ep2_ood_rotating", "rotating_frame"),
    ):
        id_acc = accuracy(key, "id")
        ood_acc = accuracy(key, "ood")
        if id_acc is not None or ood_acc is not None:
            ood_gaps[label] = {
                "id_accuracy": id_acc,
                "ood_accuracy": ood_acc,
                "ood_minus_id": None if id_acc is None or ood_acc is None else ood_acc - id_acc,
            }
    if ood_gaps:
        out["semantic_family_ood"] = ood_gaps
        fixed_gap = ood_gaps.get("fixed_r90", {}).get("ood_minus_id")
        ordinary_gap = ood_gaps.get("ordinary", {}).get("ood_minus_id")
        if fixed_gap is not None and ordinary_gap is not None:
            out["semantic_family_ood"]["contradiction_specific_gap_difference"] = (
                fixed_gap - ordinary_gap
            )

    return out


def _save_report(output_root: Path, report: dict[str, Any]) -> None:
    report["updated_at"] = utc_now()
    report["comparisons"] = _compute_comparisons(report)
    atomic_write_json(_report_path(output_root), report)


def _update_experiment_stage(
    *,
    output_root: Path,
    report: dict[str, Any],
    spec: ExperimentSpec,
    stage: str,
    status: str,
    extra: Mapping[str, Any] | None = None,
) -> None:
    entry = report["experiments"][spec.key]
    payload: dict[str, Any] = {"status": status, "updated_at": utc_now()}
    if extra:
        payload.update(dict(extra))
    entry.setdefault("stages", {})[stage] = payload
    if status == "failed":
        entry["status"] = "failed"
    elif stage == "cleanup" and status == "done":
        entry["status"] = "done"
    else:
        entry["status"] = "running"
    _save_report(output_root, report)
    atomic_write_json(_experiment_root(output_root, spec) / "experiment_report.json", entry)


def _evaluation_result_path(checkpoint: Path, dataset_name: str) -> Path:
    return checkpoint / f"eval_{dataset_name}_results.json"


def _derive_per_hop_from_responses(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        records = read_json(path)
    except Exception:
        return None
    if not isinstance(records, list):
        return None
    grouped: dict[int, list[int]] = {}
    for row in records:
        if not isinstance(row, dict):
            continue
        try:
            hop = int(row.get("hop_count"))
        except (TypeError, ValueError):
            continue
        score = int(bool(row.get("correct", row.get("score", 0))))
        grouped.setdefault(hop, []).append(score)
    if not grouped:
        return None
    return {
        str(hop): {
            "num_rows": len(scores),
            "num_correct": sum(scores),
            "accuracy": sum(scores) / len(scores),
        }
        for hop, scores in sorted(grouped.items())
    }


def _archive_task_results(
    *,
    output_root: Path,
    report: dict[str, Any],
    spec: ExperimentSpec,
    checkpoint: Path,
) -> dict[str, Any]:
    binding = _dataset_bindings(output_root)[spec.key]
    task_results: dict[str, Any] = {}
    for dataset_name, _eval_path, role in binding.eval_bindings:
        result_path = _evaluation_result_path(checkpoint, dataset_name)
        if not result_path.exists():
            raise FileNotFoundError(
                f"Missing final task-evaluation result for role={role}: {result_path}"
            )
        data = read_json(result_path)
        responses_path = checkpoint / f"eval_{dataset_name}_responses.json"
        per_hop = data.get("per_hop") or _derive_per_hop_from_responses(responses_path)
        task_results[role] = {
            "dataset_name": dataset_name,
            "accuracy": data.get("accuracy"),
            "num_correct": data.get("num_correct"),
            "num_total": data.get("num_total"),
            "per_hop": per_hop,
            "result_path": str(result_path),
            "responses_path": str(responses_path),
        }
    report["experiments"][spec.key].setdefault("results", {})["task_evaluations"] = task_results
    report["experiments"][spec.key]["results"]["final_checkpoint"] = str(checkpoint)
    _save_report(output_root, report)
    atomic_write_json(
        _experiment_root(output_root, spec) / "task_evaluation_summary.json",
        task_results,
    )
    return task_results


def run_training(
    *,
    base_dir: Path,
    output_root: Path,
    report: dict[str, Any],
    spec: ExperimentSpec,
    learning_rate: str,
    master_port: int,
    force_train: bool,
) -> Path:
    binding = _dataset_bindings(output_root)[spec.key]
    exp_root = _experiment_root(output_root, spec)
    phase_dir = _phase_dir(output_root, spec)

    if force_train and exp_root.exists():
        print(f"[force train] deleting {exp_root}")
        shutil.rmtree(exp_root)
        # Restore report entry after deletion.
        report["experiments"][spec.key] = {
            "spec": dataclasses.asdict(spec),
            "expected_final_step": spec.expected_optimizer_steps,
            "status": "pending",
            "stages": {},
            "results": {},
        }
        _save_report(output_root, report)

    phase_dir.mkdir(parents=True, exist_ok=True)
    if is_marked(exp_root, ".cleanup_done") and not force_train:
        raise RuntimeError(
            f"{spec.key} is already cleaned/completed. Use --force-train only if you "
            "intentionally want to rerun it."
        )

    if is_marked(phase_dir, ".train_done") and not force_train:
        checkpoint = _only_final_checkpoint_check(phase_dir, spec.expected_optimizer_steps)
        print(f"[skip train] {spec.key}: {checkpoint}")
        return checkpoint

    # Crash-safe recovery for the narrow window after Trainer wrote the final
    # checkpoint but before this parent process wrote .train_done.  Because the
    # suite intentionally saves only at the expected final step, a usable
    # checkpoint with exactly that step is sufficient evidence that training
    # completed.
    if not force_train:
        existing_usable = [
            (step, path)
            for step, path in _checkpoint_dirs(phase_dir)
            if _is_usable_checkpoint(path)
        ]
        if existing_usable:
            checkpoint = _only_final_checkpoint_check(phase_dir, spec.expected_optimizer_steps)
            mark(phase_dir, ".train_done")
            _update_experiment_stage(
                output_root=output_root, report=report, spec=spec,
                stage="train", status="done",
                extra={
                    "recovered_from_existing_final_checkpoint": True,
                    "final_checkpoint": str(checkpoint),
                },
            )
            print(f"[recover train] {spec.key}: {checkpoint}")
            return checkpoint

    _update_experiment_stage(
        output_root=output_root, report=report, spec=spec, stage="train", status="running",
        extra={"started_at": utc_now()},
    )

    cmd = [
        sys.executable,
        "-m", "torch.distributed.run",
        "--master_port", str(master_port),
        "--nproc_per_node", "1",
        str(Path(__file__).resolve()),
        "__train__",
        "--base-dir", str(base_dir),
        "--output-root", str(output_root),
        "--dataset-name", binding.train_dataset_name,
        "--phase-dir", str(phase_dir),
        "--model-name", MODEL_NAME,
        "--learning-rate", str(learning_rate),
        "--epochs", str(spec.epochs),
        "--save-steps", str(spec.expected_optimizer_steps),
    ]
    write_command(phase_dir, "train_command.txt", cmd)
    run_subprocess(
        cmd,
        cwd=base_dir,
        env_updates={
            "WANDB_MODE": os.environ.get("WANDB_MODE", "offline"),
            "HF_ALLOW_CODE_EVAL": os.environ.get("HF_ALLOW_CODE_EVAL", "1"),
        },
        log_path=exp_root / "logs" / "train.log",
    )

    checkpoint = _only_final_checkpoint_check(phase_dir, spec.expected_optimizer_steps)
    mark(phase_dir, ".train_done")
    _update_experiment_stage(
        output_root=output_root, report=report, spec=spec, stage="train", status="done",
        extra={
            "finished_at": utc_now(),
            "final_checkpoint": str(checkpoint),
            "usable_checkpoint_count": 1,
        },
    )
    return checkpoint


def run_task_evaluation(
    *,
    base_dir: Path,
    output_root: Path,
    report: dict[str, Any],
    spec: ExperimentSpec,
    visible_devices: str,
    gpu_memory_utilization: float,
    checkpoint: Path,
    force_eval: bool,
) -> None:
    exp_root = _experiment_root(output_root, spec)
    binding = _dataset_bindings(output_root)[spec.key]

    if is_marked(exp_root, ".checkpoint_eval_done") and not force_eval:
        print(f"[skip task eval] {spec.key}")
        _archive_task_results(
            output_root=output_root, report=report, spec=spec, checkpoint=checkpoint
        )
        return

    _update_experiment_stage(
        output_root=output_root, report=report, spec=spec, stage="task_eval", status="running",
        extra={"started_at": utc_now()},
    )
    eval_names = [name for name, _path, _role in binding.eval_bindings]
    dataset_order = [binding.train_dataset_name, *eval_names]
    # Remove duplicates while preserving order.
    dataset_order = list(dict.fromkeys(dataset_order))
    forwarded = [
        "--experiment_root", str(exp_root),
        "--skip_existing",
        "--dataset_order", *dataset_order,
        "--eval_subset", *eval_names,
        "--gpu_memory_utilization", str(gpu_memory_utilization),
    ]
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "__eval__",
        "--base-dir", str(base_dir),
        "--output-root", str(output_root),
        "--",
        *forwarded,
    ]
    write_command(exp_root, "eval_command.txt", cmd)
    run_subprocess(
        cmd,
        cwd=base_dir,
        env_updates={"CUDA_VISIBLE_DEVICES": visible_devices},
        log_path=exp_root / "logs" / "task_eval.log",
    )
    task_results = _archive_task_results(
        output_root=output_root, report=report, spec=spec, checkpoint=checkpoint
    )
    mark(exp_root, ".checkpoint_eval_done")
    _update_experiment_stage(
        output_root=output_root, report=report, spec=spec, stage="task_eval", status="done",
        extra={"finished_at": utc_now(), "results": task_results},
    )


def _find_lm_eval_summary(exp_root: Path) -> Path | None:
    direct = exp_root / "lm_eval_summary.json"
    if direct.exists():
        return direct
    candidates = sorted(exp_root.glob("**/lm_eval_summary.json"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def _valid_raw_lm_eval_result(checkpoint: Path) -> Path | None:
    lm_dir = checkpoint / "lm_eval"
    if not lm_dir.exists():
        return None
    candidates = sorted(lm_dir.glob("**/results_*.json"), key=lambda path: path.stat().st_mtime)
    for candidate in reversed(candidates):
        try:
            data = read_json(candidate)
        except Exception:
            continue
        if isinstance(data, dict) and data.get("results"):
            return candidate
    return None


def run_lm_eval(
    *,
    base_dir: Path,
    output_root: Path,
    report: dict[str, Any],
    spec: ExperimentSpec,
    visible_devices: str,
    gpu_memory_utilization: float,
    lm_eval_tasks: str,
    checkpoint: Path,
    force_lm_eval: bool,
) -> None:
    exp_root = _experiment_root(output_root, spec)
    if not spec.run_lm_eval:
        report["experiments"][spec.key].setdefault("results", {})["lm_eval"] = {
            "skipped_by_design": True,
            "reason": "LM-Eval omitted for the 500/2k sample-efficiency probes.",
        }
        _update_experiment_stage(
            output_root=output_root, report=report, spec=spec, stage="lm_eval", status="skipped"
        )
        return

    if is_marked(exp_root, ".lm_eval_done") and not force_lm_eval:
        print(f"[skip lm_eval] {spec.key}")
        summary_path = _find_lm_eval_summary(exp_root)
        if summary_path is not None:
            report["experiments"][spec.key].setdefault("results", {})["lm_eval"] = read_json(summary_path)
            report["experiments"][spec.key]["results"]["lm_eval_summary_path"] = str(summary_path)
            _save_report(output_root, report)
        return

    lmeval_script = base_dir / "scripts" / "lmeval.sh"
    if not lmeval_script.exists():
        raise FileNotFoundError(f"Missing LM-Eval script: {lmeval_script}")
    lmeval_script.chmod(lmeval_script.stat().st_mode | 0o111)

    _update_experiment_stage(
        output_root=output_root, report=report, spec=spec, stage="lm_eval", status="running",
        extra={"started_at": utc_now()},
    )

    # Another narrow crash-recovery window: if LM-Eval already wrote a valid
    # results JSON but the parent process died before summarization/marking, do
    # not spend GPU time repeating LM-Eval.
    raw_lm_result = None if force_lm_eval else _valid_raw_lm_eval_result(checkpoint)
    if raw_lm_result is None:
        lm_cmd = [str(lmeval_script), str(checkpoint)]
        write_command(exp_root, "lmeval_command.txt", lm_cmd)
        run_subprocess(
            lm_cmd,
            cwd=base_dir,
            env_updates={
                "VISIBLE_DEVICES": visible_devices,
                "GPU_MEMORY_UTILIZATION": str(gpu_memory_utilization),
                "TASKS": lm_eval_tasks,
            },
            log_path=exp_root / "logs" / "lm_eval.log",
        )
        raw_lm_result = _valid_raw_lm_eval_result(checkpoint)
    else:
        print(f"[recover lm_eval] reusing {raw_lm_result}")

    summarizer = base_dir / "summarize_lmeval.py"
    summary_path: Path | None = None
    if summarizer.exists():
        summarize_cmd = [
            sys.executable,
            str(summarizer),
            "--checkpoint-path", str(checkpoint),
            "--lm-eval-dir", str(checkpoint / "lm_eval"),
            "--exp-root", str(exp_root),
            "--output-name", "lm_eval_summary.json",
        ]
        write_command(exp_root, "summarize_lmeval_command.txt", summarize_cmd)
        run_subprocess(
            summarize_cmd,
            cwd=base_dir,
            log_path=exp_root / "logs" / "summarize_lm_eval.log",
        )
        summary_path = _find_lm_eval_summary(exp_root)

    lm_payload: dict[str, Any]
    if summary_path is not None:
        lm_payload = read_json(summary_path)
    else:
        raw_candidates = sorted(
            (checkpoint / "lm_eval").glob("**/results_*.json"),
            key=lambda path: path.stat().st_mtime,
        ) if (checkpoint / "lm_eval").exists() else []
        lm_payload = {
            "summary_unavailable": True,
            "raw_lm_eval_json": str(raw_candidates[-1]) if raw_candidates else None,
            "note": "summarize_lmeval.py was unavailable; raw LM-Eval output is retained.",
        }

    report["experiments"][spec.key].setdefault("results", {})["lm_eval"] = lm_payload
    if summary_path is not None:
        report["experiments"][spec.key]["results"]["lm_eval_summary_path"] = str(summary_path)
    _save_report(output_root, report)
    mark(exp_root, ".lm_eval_done")
    _update_experiment_stage(
        output_root=output_root, report=report, spec=spec, stage="lm_eval", status="done",
        extra={"finished_at": utc_now(), "summary_path": str(summary_path) if summary_path else None},
    )


def run_cleanup(
    *,
    base_dir: Path,
    output_root: Path,
    report: dict[str, Any],
    spec: ExperimentSpec,
    enabled: bool,
) -> None:
    exp_root = _experiment_root(output_root, spec)
    if not enabled:
        _update_experiment_stage(
            output_root=output_root, report=report, spec=spec, stage="cleanup", status="skipped",
            extra={"reason": "--run-cleanup false"},
        )
        return
    if is_marked(exp_root, ".cleanup_done"):
        print(f"[skip cleanup] {spec.key}")
        report["experiments"][spec.key]["status"] = "done"
        _save_report(output_root, report)
        return

    if not is_marked(exp_root, ".checkpoint_eval_done"):
        raise RuntimeError("Refusing cleanup before task evaluation is complete")
    if spec.run_lm_eval and not is_marked(exp_root, ".lm_eval_done"):
        raise RuntimeError("Refusing cleanup before required LM-Eval is complete")

    cleanup_script = base_dir / "scripts" / "cleanup_checkpoints.sh"
    if not cleanup_script.exists():
        raise FileNotFoundError(f"Missing cleanup script: {cleanup_script}")
    cleanup_script.chmod(cleanup_script.stat().st_mode | 0o111)

    _update_experiment_stage(
        output_root=output_root, report=report, spec=spec, stage="cleanup", status="running",
        extra={"started_at": utc_now()},
    )
    cmd = [str(cleanup_script), str(exp_root)]
    write_command(exp_root, "cleanup_command.txt", cmd)
    run_subprocess(
        cmd,
        cwd=base_dir,
        log_path=exp_root / "logs" / "cleanup.log",
    )
    mark(exp_root, ".cleanup_done")
    _update_experiment_stage(
        output_root=output_root, report=report, spec=spec, stage="cleanup", status="done",
        extra={"finished_at": utc_now()},
    )


def _rebuild_report_from_artifacts(
    *,
    output_root: Path,
    report: dict[str, Any],
) -> dict[str, Any]:
    manifest_path = _dataset_cache_root(output_root) / DATASET_MANIFEST_FILENAME
    if manifest_path.exists():
        report["dataset_manifest"] = read_json(manifest_path)

    for spec in EXPERIMENTS:
        exp_root = _experiment_root(output_root, spec)
        phase_dir = _phase_dir(output_root, spec)
        entry = report["experiments"][spec.key]
        checkpoints = _checkpoint_dirs(phase_dir)
        entry["artifacts"] = {
            "experiment_root": str(exp_root),
            "phase_dir": str(phase_dir),
            "checkpoint_dirs": [str(path) for _step, path in checkpoints],
            "cleanup_done": is_marked(exp_root, ".cleanup_done"),
        }

        # Prefer the archived task summary, which survives cleanup.
        task_summary = exp_root / "task_evaluation_summary.json"
        if task_summary.exists():
            entry.setdefault("results", {})["task_evaluations"] = read_json(task_summary)
        lm_summary = _find_lm_eval_summary(exp_root)
        if lm_summary is not None:
            entry.setdefault("results", {})["lm_eval"] = read_json(lm_summary)
            entry["results"]["lm_eval_summary_path"] = str(lm_summary)
        elif not spec.run_lm_eval:
            entry.setdefault("results", {})["lm_eval"] = {
                "skipped_by_design": True,
                "reason": "LM-Eval omitted for the 500/2k sample-efficiency probes.",
            }

        if is_marked(exp_root, ".cleanup_done"):
            entry["status"] = "done"
        elif is_marked(exp_root, ".checkpoint_eval_done"):
            entry["status"] = "evaluated"
        elif is_marked(phase_dir, ".train_done"):
            entry["status"] = "trained"

    _save_report(output_root, report)
    return report


def run_suite(args: argparse.Namespace) -> None:
    base_dir = args.base_dir.resolve()
    output_root = args.output_root.resolve()
    source_dir = args.source_dir.resolve()
    if not base_dir.exists():
        raise FileNotFoundError(f"--base-dir does not exist: {base_dir}")
    os.chdir(base_dir)

    manifest = prepare_datasets(
        output_root=output_root,
        source_dir=source_dir,
        heldout_family=args.heldout_family,
        generation_seed=args.generation_seed,
        origin_probability=args.origin_probability,
        anchor_max_abs=args.anchor_max_abs,
        force=args.force_prepare,
        write_arrow=True,
    )
    report = _load_or_create_report(
        output_root=output_root,
        source_dir=source_dir,
        learning_rate=args.learning_rate,
        heldout_family=args.heldout_family,
        generation_seed=args.generation_seed,
        origin_probability=args.origin_probability,
        anchor_max_abs=args.anchor_max_abs,
    )
    report["dataset_manifest"] = manifest
    _save_report(output_root, report)

    selected = [EXPERIMENT_BY_KEY[key] for key in args.experiments]
    for offset, spec in enumerate(selected):
        exp_root = _experiment_root(output_root, spec)
        if is_marked(exp_root, ".cleanup_done") and not args.force_train:
            print(f"[skip complete] {spec.key}: .cleanup_done exists")
            report["experiments"][spec.key]["status"] = "done"
            _save_report(output_root, report)
            continue
        try:
            checkpoint = run_training(
                base_dir=base_dir,
                output_root=output_root,
                report=report,
                spec=spec,
                learning_rate=args.learning_rate,
                master_port=args.master_port + offset,
                force_train=args.force_train,
            )
            run_task_evaluation(
                base_dir=base_dir,
                output_root=output_root,
                report=report,
                spec=spec,
                visible_devices=args.visible_devices,
                gpu_memory_utilization=args.gpu_memory_utilization,
                checkpoint=checkpoint,
                force_eval=args.force_eval,
            )
            run_lm_eval(
                base_dir=base_dir,
                output_root=output_root,
                report=report,
                spec=spec,
                visible_devices=args.visible_devices,
                gpu_memory_utilization=args.gpu_memory_utilization,
                lm_eval_tasks=args.lm_eval_tasks,
                checkpoint=checkpoint,
                force_lm_eval=args.force_lm_eval,
            )
            run_cleanup(
                base_dir=base_dir,
                output_root=output_root,
                report=report,
                spec=spec,
                enabled=args.run_cleanup,
            )
        except Exception as exc:
            _update_experiment_stage(
                output_root=output_root,
                report=report,
                spec=spec,
                stage="failure",
                status="failed",
                extra={
                    "exception_type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )
            if not args.continue_on_error:
                raise
            print(f"[ERROR] {spec.key}: {exc}", file=sys.stderr)

    _rebuild_report_from_artifacts(output_root=output_root, report=report)
    print(f"\nCentral report: {_report_path(output_root)}")


def logic_self_test(source_dir: Path) -> dict[str, Any]:
    """Pure-Python tests that do not need HF Datasets, Torch, or a GPU."""
    core, builder = _import_generator(source_dir)
    if builder.validate_vocabulary():
        raise AssertionError(f"Vocabulary errors: {builder.validate_vocabulary()}")

    training_families = _families_without(core, DEFAULT_HELDOUT_FAMILY)
    tiny_train, tiny_id = _build_train_eval_records(
        builder=builder,
        train_families=training_families,
        eval_families=training_families,
        generation_seed=GENERATION_SEED,
        origin_probability=builder.DEFAULT_ORIGIN_PROBABILITY,
        anchor_max_abs=builder.DEFAULT_ANCHOR_MAX_ABS,
    )
    # Keep the test fast after exercising 4k/500 deterministic construction:
    # validate pairing and world conversion on the first 48 eval rows.
    tiny_ood_full = _build_paired_ood_eval_records(
        builder=builder,
        train_records=tiny_train,
        heldout_family=DEFAULT_HELDOUT_FAMILY,
        generation_seed=GENERATION_SEED,
        origin_probability=builder.DEFAULT_ORIGIN_PROBABILITY,
        anchor_max_abs=builder.DEFAULT_ANCHOR_MAX_ABS,
    )
    pair = _paired_surface_checks(
        tiny_id,
        tiny_ood_full,
        heldout_family=DEFAULT_HELDOUT_FAMILY,
        training_families=training_families,
    )

    for row in tiny_id[:48]:
        rotating = _world_row(row, world=WORLD_ROTATING, core=core, builder=builder)
        ordinary = _world_row(row, world=WORLD_ORDINARY, core=core, builder=builder)
        fixed = _world_row(row, world=WORLD_R90, core=core, builder=builder)
        if ordinary["problem"] != fixed["problem"] or rotating["problem"] != fixed["problem"]:
            raise AssertionError("World conversion changed the model-facing problem")
        hop_count = int(row["hop_count"])
        expected_turns = [step % 4 for step in range(1, hop_count + 1)]
        if rotating["diagnostic_frame_quarter_turns"] != expected_turns:
            raise AssertionError("Rotating frame state trace is incorrect")

    subset500 = stratified_nested_subset(tiny_train, size=500)
    subset2000 = stratified_nested_subset(tiny_train, size=2000)
    if not {r["graph_hash"] for r in subset500}.issubset({r["graph_hash"] for r in subset2000}):
        raise AssertionError("Nested subset invariant failed")

    expected_steps = {spec.key: spec.expected_optimizer_steps for spec in EXPERIMENTS}
    if expected_steps != {
        "3_1_sft_500": 15,
        "3_2_sft_4k_ep1": 125,
        "3_3_sft_4k_ep2": 250,
        "3_4_sft_4k_ep2_ood_r90": 250,
        "3_5_sft_4k_ep2_ood_ordinary": 250,
        "3_6_sft_4k_ep2_ood_rotating": 250,
        "3_7_sft_2k": 62,
    }:
        raise AssertionError(f"Unexpected optimizer-step plan: {expected_steps}")

    result = {
        "passed": True,
        "runner_version": SCRIPT_VERSION,
        "core_version": core.VERSION,
        "builder_version": builder.VERSION,
        "heldout_family": DEFAULT_HELDOUT_FAMILY,
        "training_families": sorted(training_families),
        "paired_controls": pair,
        "expected_final_steps": expected_steps,
        "sampled_world_conversion_rows": min(48, len(tiny_id)),
    }
    print(json.dumps(result, indent=2))
    return result


def _add_common_dataset_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--heldout-family", default=DEFAULT_HELDOUT_FAMILY)
    parser.add_argument("--generation-seed", type=int, default=GENERATION_SEED)
    parser.add_argument("--origin-probability", type=float, default=0.20)
    parser.add_argument("--anchor-max-abs", type=int, default=4)


def public_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Spatial Contradiction v3 SFT ablation suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    self_test = sub.add_parser("self-test", help="Run pure generator/runner logic tests.")
    self_test.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)

    prepare = sub.add_parser("prepare", help="Build and validate reusable Arrow dataset views.")
    _add_common_dataset_args(prepare)
    prepare.add_argument("--force-prepare", action="store_true")

    run = sub.add_parser("run", help="Prepare, train, evaluate, LM-Eval, archive, and cleanup.")
    _add_common_dataset_args(run)
    run.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    run.add_argument(
        "--experiments", nargs="+", choices=list(EXPERIMENT_BY_KEY),
        default=[spec.key for spec in EXPERIMENTS],
    )
    run.add_argument("--learning-rate", default=DEFAULT_LEARNING_RATE)
    run.add_argument(
        "--visible-devices",
        default=(os.environ.get("VISIBLE_DEVICES") or os.environ.get("CUDA_VISIBLE_DEVICES") or "0"),
    )
    run.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    run.add_argument("--master-port", type=int, default=29731)
    run.add_argument("--lm-eval-tasks", default=DEFAULT_LM_EVAL_TASKS)
    run.add_argument("--run-cleanup", type=str_to_bool, default=True)
    run.add_argument("--continue-on-error", type=str_to_bool, default=False)
    run.add_argument("--force-prepare", action="store_true")
    run.add_argument("--force-train", action="store_true")
    run.add_argument("--force-eval", action="store_true")
    run.add_argument("--force-lm-eval", action="store_true")

    report = sub.add_parser("report", help="Rebuild the central JSON summary from saved artifacts.")
    _add_common_dataset_args(report)
    report.add_argument("--learning-rate", default=DEFAULT_LEARNING_RATE)

    return parser


def internal_parser_train(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--phase-dir", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--learning-rate", required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--save-steps", type=int, required=True)
    return parser.parse_args(list(argv))


def internal_parser_eval(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("forwarded", nargs=argparse.REMAINDER)
    args = parser.parse_args(list(argv))
    if args.forwarded and args.forwarded[0] == "--":
        args.forwarded = args.forwarded[1:]
    return args


def main() -> None:
    if len(sys.argv) >= 2 and sys.argv[1] == "__train__":
        _train_internal(internal_parser_train(sys.argv[2:]))
        return
    if len(sys.argv) >= 2 and sys.argv[1] == "__eval__":
        _eval_internal(internal_parser_eval(sys.argv[2:]))
        return

    parser = public_parser()
    args = parser.parse_args()

    if args.command == "self-test":
        logic_self_test(args.source_dir.resolve())
        return

    if not 0.0 <= args.origin_probability <= 1.0:
        parser.error("--origin-probability must be in [0,1]")
    if args.anchor_max_abs < 1:
        parser.error("--anchor-max-abs must be >=1")

    if args.command == "prepare":
        prepare_datasets(
            output_root=args.output_root.resolve(),
            source_dir=args.source_dir.resolve(),
            heldout_family=args.heldout_family,
            generation_seed=args.generation_seed,
            origin_probability=args.origin_probability,
            anchor_max_abs=args.anchor_max_abs,
            force=args.force_prepare,
            write_arrow=True,
        )
        return

    if args.command == "run":
        if not 0 < args.gpu_memory_utilization <= 1:
            parser.error("--gpu-memory-utilization must be in (0,1]")
        run_suite(args)
        return

    if args.command == "report":
        output_root = args.output_root.resolve()
        report_data = _load_or_create_report(
            output_root=output_root,
            source_dir=args.source_dir.resolve(),
            learning_rate=args.learning_rate,
            heldout_family=args.heldout_family,
            generation_seed=args.generation_seed,
            origin_probability=args.origin_probability,
            anchor_max_abs=args.anchor_max_abs,
        )
        report_data = _rebuild_report_from_artifacts(
            output_root=output_root,
            report=report_data,
        )
        print(json.dumps(report_data, indent=2))
        print(f"\nCentral report: {_report_path(output_root)}")
        return

    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
