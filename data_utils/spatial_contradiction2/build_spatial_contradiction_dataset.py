#!/usr/bin/env python3
from __future__ import annotations

"""Build the Spatial Contradiction benchmark.

The final model-facing task is an implicit alternative-world spatial reasoning
problem.  Each example has exactly the same natural-language problem in the
ordinary and contradiction worlds, but the semantic interpretation of all eight
spatial directions is rotated 90 degrees clockwise in the contradiction world.
The coordinate readout is held fixed.

Default outputs
---------------
<output-root>/train_data     4,000 HF Arrow rows
<output-root>/eval_data        500 HF Arrow rows (validation split)
<output-root>/test_data      2,000 HF Arrow rows
<output-root>/metadata.json
<output-root>/README.generated.md
<output-root>/intermediate/
    all_generated_records.jsonl
    split_assignments.jsonl
    vocabulary.json
    generation_report.json
    validation_report.json
    preview_examples.md
    build.log

The builder does not call an LLM.  Both worlds and their complete reasoning
traces are generated from an exact symbolic graph.  This removes a source of
noise present in free-form expert generation while retaining natural-language
variation through a controlled vocabulary and sentence grammar.
"""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import random
import shutil
import sys
from typing import Any, Iterable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_utils.spatial_contradiction2.spatial_contradiction_core import (
    VERSION as CORE_VERSION,
    TRANSFORMATION_NAME,
    TRANSFORMATION_VERSION,
    DIRECTION_KEYS,
    DIRECTION_SPECS,
    DIAGONAL_DIRECTIONS,
    PHRASE_SEMANTIC_FAMILIES,
    ENTITY_NAMES,
    EdgeRecord,
    SpatialCoordinateVerifier,
    add_coord,
    build_messages,
    edge_to_dict,
    format_coord,
    render_problem,
    render_solution,
    render_statement,
    rotate_r90,
    scaled,
    stable_hash,
    stable_int,
    validate_phrase_family_coverage,
    validate_vocabulary,
    vector_for_direction,
    vocabulary_manifest,
)

VERSION = "1.3.1"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "spatial_contradiction2_data"
DEFAULT_SEED = 37
DEFAULT_TRAIN_SIZE = 4_000
DEFAULT_EVAL_SIZE = 500
DEFAULT_TEST_SIZE = 2_000
DEFAULT_MIN_HOPS = 1
DEFAULT_MAX_HOPS = 6
DEFAULT_REVERSE_PROB = 0.35
DEFAULT_ORIGIN_PROBABILITY = 0.20
DEFAULT_ANCHOR_MAX_ABS = 4
DEFAULT_DISTANCE_WEIGHTS = {1: 0.48, 2: 0.30, 3: 0.16, 4: 0.06}
MAX_GENERATION_ATTEMPTS_PER_ROW = 500


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def configure_logging(intermediate_dir: Path, level: str) -> None:
    intermediate_dir.mkdir(parents=True, exist_ok=True)
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    handlers.append(logging.FileHandler(intermediate_dir / "build.log", encoding="utf-8"))
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
            count += 1
    temporary.replace(path)
    return count


def balanced_hop_schedule(size: int, min_hops: int, max_hops: int, rng: random.Random) -> list[int]:
    if size <= 0:
        raise ValueError("split size must be positive")
    if min_hops < 1 or max_hops < min_hops:
        raise ValueError("invalid hop range")
    values = list(range(min_hops, max_hops + 1))
    base, remainder = divmod(size, len(values))
    schedule: list[int] = []
    for offset, hop in enumerate(values):
        schedule.extend([hop] * (base + (1 if offset < remainder else 0)))
    rng.shuffle(schedule)
    return schedule


def weighted_distance(rng: random.Random, weights: Mapping[int, float]) -> int:
    distances = sorted(weights)
    probabilities = [float(weights[d]) for d in distances]
    if any(p < 0 for p in probabilities) or sum(probabilities) <= 0:
        raise ValueError("distance weights must be non-negative and sum to > 0")
    return int(rng.choices(distances, weights=probabilities, k=1)[0])


def sample_anchor_coordinate(
    rng: random.Random,
    *,
    origin_probability: float = DEFAULT_ORIGIN_PROBABILITY,
    max_abs: int = DEFAULT_ANCHOR_MAX_ABS,
) -> tuple[int, int]:
    """Sample the explicit starting coordinate without affecting graph sampling.

    The origin is retained as a minority control case.  Otherwise we sample
    uniformly from the integer square ``[-max_abs, max_abs]^2`` excluding
    ``(0, 0)``.  A dedicated RNG stream is used so changing this policy does
    not alter names, directions, distances, statement inversions, or wording.
    """

    if not 0.0 <= origin_probability <= 1.0:
        raise ValueError("origin_probability must be in [0, 1]")
    if max_abs < 1:
        raise ValueError("anchor max_abs must be >= 1")
    if rng.random() < origin_probability:
        return (0, 0)

    candidates = [
        (x, y)
        for x in range(-max_abs, max_abs + 1)
        for y in range(-max_abs, max_abs + 1)
        if (x, y) != (0, 0)
    ]
    return rng.choice(candidates)


def canonical_graph_payload(anchor: str, edges: Sequence[EdgeRecord]) -> dict[str, Any]:
    return {
        "anchor": anchor,
        "edges": [
            {
                "parent": edge.parent,
                "child": edge.child,
                "path_direction": edge.path_direction,
                "distance": edge.distance,
            }
            for edge in edges
        ],
    }


def abstract_path_payload(edges: Sequence[EdgeRecord]) -> list[dict[str, Any]]:
    """Name/template-free path signature used for analysis, not deduplication."""

    return [
        {
            "direction": edge.path_direction,
            "distance": edge.distance,
            "reversed": edge.statement_reversed,
        }
        for edge in edges
    ]


def _sample_backbone(
    *,
    graph_rng: random.Random,
    surface_rng: random.Random,
    anchor_coord: tuple[int, int],
    hop_count: int,
    reverse_probability: float,
    distance_weights: Mapping[int, float],
    allowed_phrase_semantic_families: frozenset[str] | None = None,
) -> tuple[str, str, list[EdgeRecord], dict[str, tuple[int, int]]]:
    names = graph_rng.sample(list(ENTITY_NAMES), hop_count + 1)
    anchor = names[0]
    target = names[-1]
    coords: dict[str, tuple[int, int]] = {anchor: tuple(anchor_coord)}
    used_coords = {tuple(anchor_coord)}
    edges: list[EdgeRecord] = []

    for step in range(hop_count):
        parent = names[step]
        child = names[step + 1]
        parent_coord = coords[parent]

        for _ in range(100):
            direction = graph_rng.choice(DIRECTION_KEYS)
            distance = weighted_distance(graph_rng, distance_weights)
            displacement = scaled(vector_for_direction(direction, world="normal"), distance)
            child_coord = add_coord(parent_coord, displacement)
            if child_coord not in used_coords:
                break
        else:
            raise RuntimeError("Could not sample a non-overlapping backbone coordinate")

        reverse = graph_rng.random() < reverse_probability
        if reverse:
            statement_direction = DIRECTION_SPECS[direction].opposite
            statement_subject = parent
            statement_reference = child
        else:
            statement_direction = direction
            statement_subject = child
            statement_reference = parent

        rendered = render_statement(
            subject=statement_subject,
            reference=statement_reference,
            direction=statement_direction,
            distance=distance,
            rng=surface_rng,
            allowed_semantic_families=allowed_phrase_semantic_families,
        )

        edge = EdgeRecord(
            parent=parent,
            child=child,
            path_direction=direction,
            distance=distance,
            statement_reversed=reverse,
            statement_direction=statement_direction,
            statement_subject=statement_subject,
            statement_reference=statement_reference,
            statement_text=rendered.text,
            statement_family_id=rendered.family_id,
            statement_family_key=rendered.family_key,
            statement_syntax_family=rendered.syntax_family,
            phrase_id=rendered.phrase_id,
            phrase_kind=rendered.phrase_kind,
            phrase_semantic_family=rendered.phrase_semantic_family,
            phrase_grammar_role=rendered.phrase_grammar_role,
            amount_text=rendered.amount_text,
        )
        edges.append(edge)
        coords[child] = child_coord
        used_coords.add(child_coord)

    return anchor, target, edges, coords


def generate_record(
    *,
    split: str,
    split_index: int,
    hop_count: int,
    root_seed: int,
    attempt: int,
    reverse_probability: float,
    distance_weights: Mapping[int, float],
    origin_probability: float = DEFAULT_ORIGIN_PROBABILITY,
    anchor_max_abs: int = DEFAULT_ANCHOR_MAX_ABS,
    allowed_phrase_semantic_families: frozenset[str] | None = None,
) -> dict[str, Any]:
    instance_seed = stable_int("spatial_contradiction", root_seed, split, split_index, attempt)
    graph_rng = random.Random(stable_int(instance_seed, "graph"))
    surface_rng = random.Random(stable_int(instance_seed, "surface"))
    statement_order_rng = random.Random(stable_int(instance_seed, "statement_order"))
    problem_shell_rng = random.Random(stable_int(instance_seed, "problem_shell"))
    anchor_rng = random.Random(stable_int(instance_seed, "anchor_coordinate"))
    anchor_coord = sample_anchor_coordinate(
        anchor_rng,
        origin_probability=origin_probability,
        max_abs=anchor_max_abs,
    )

    anchor, target, edges, normal_coords = _sample_backbone(
        graph_rng=graph_rng,
        surface_rng=surface_rng,
        anchor_coord=anchor_coord,
        hop_count=hop_count,
        reverse_probability=reverse_probability,
        distance_weights=distance_weights,
        allowed_phrase_semantic_families=allowed_phrase_semantic_families,
    )

    # Present relations in a shuffled order.  The exact symbolic reasoning path
    # remains the backbone stored in ``edges``.
    statement_order = list(range(len(edges)))
    statement_order_rng.shuffle(statement_order)
    statements = [edges[i].statement_text for i in statement_order]
    problem, anchor_template_id, query_template_id = render_problem(
        anchor=anchor,
        anchor_coord=anchor_coord,
        target=target,
        statements=statements,
        rng=problem_shell_rng,
    )

    # Use the same wording/template choices in both worlds.  This keeps the
    # paired ordinary/R90 references surface-aligned so that their only
    # substantive difference is the coordinate semantics.
    solution_style_seed = stable_int(instance_seed, "solution_style")
    normal_output, normal_final, normal_trace = render_solution(
        anchor=anchor,
        anchor_coord=anchor_coord,
        target=target,
        edges=edges,
        world="normal",
        solution_seed=solution_style_seed,
        allowed_semantic_families=allowed_phrase_semantic_families,
    )
    r90_output, r90_final, r90_trace = render_solution(
        anchor=anchor,
        anchor_coord=anchor_coord,
        target=target,
        edges=edges,
        world="r90",
        solution_seed=solution_style_seed,
        allowed_semantic_families=allowed_phrase_semantic_families,
    )

    # A nonzero displacement cannot be a fixed point of R90.  The explicitly
    # supplied anchor coordinate is held fixed in both worlds, so target==anchor
    # is the relevant degeneracy rather than target==(0,0).
    if normal_final == anchor_coord or r90_final == normal_final:
        raise ValueError("degenerate target coordinate under R90")

    graph_payload = canonical_graph_payload(anchor, edges)
    graph_hash = stable_hash(graph_payload, length=32)
    abstract_hash = stable_hash(abstract_path_payload(edges), length=24)
    problem_hash = stable_hash(problem, length=32)
    source_id = f"spatial2_{split}_{split_index:05d}_{problem_hash[:10]}"

    edge_dicts = [edge_to_dict(edge) for edge in edges]
    directions = [edge.path_direction for edge in edges]
    distances = [edge.distance for edge in edges]
    num_diagonal = sum(direction in DIAGONAL_DIRECTIONS for direction in directions)
    num_multi_cell = sum(distance > 1 for distance in distances)
    num_reversed = sum(edge.statement_reversed for edge in edges)
    phrase_semantic_families = [edge.phrase_semantic_family for edge in edges]
    phrase_grammar_roles = [edge.phrase_grammar_role for edge in edges]
    statement_syntax_families = [edge.statement_syntax_family for edge in edges]
    statement_family_keys = [edge.statement_family_key for edge in edges]
    phrase_kinds = [edge.phrase_kind for edge in edges]
    solution_phrase_semantic_families = [str(step["solution_phrase_semantic_family"]) for step in normal_trace]
    solution_phrase_grammar_roles = [str(step["solution_phrase_grammar_role"]) for step in normal_trace]
    solution_step_syntax_families = [str(step["solution_step_syntax_family"]) for step in normal_trace]
    solution_step_family_keys = [str(step["solution_step_family_key"]) for step in normal_trace]
    surface_phrase_family_allowlist = (
        list(PHRASE_SEMANTIC_FAMILIES)
        if allowed_phrase_semantic_families is None
        else sorted(allowed_phrase_semantic_families)
    )

    answer_r90 = format_coord(r90_final)
    answer_normal = format_coord(normal_final)
    messages = build_messages(problem)

    record: dict[str, Any] = {
        # Canonical model-facing contradiction-world columns.
        "messages": messages,
        "problem": problem,
        "answer": answer_r90,
        "output_text": r90_output,
        "visible_output_text": r90_output,
        "golden_answer": answer_r90,
        "golden_response": r90_output,
        # Paired ordinary-world controls.
        "original_messages": messages,
        "original_problem": problem,
        "original_answer": answer_normal,
        "original_output_text": normal_output,
        "original_golden_answer": answer_normal,
        "original_golden_response": normal_output,
        # Explicit transformed aliases parallel to Math Contradiction.
        "mod_messages": messages,
        "mod_problem": problem,
        "mod_answer": answer_r90,
        "mod_output_text": r90_output,
        # Provenance / task metadata.
        "dataset_name": "spatial_contradiction2",
        "source_dataset_name": "synthetic_stepgame_inspired",
        "source": "symbolic_spatial_generator_v3",
        "source_id": source_id,
        "problem_hash": problem_hash,
        "graph_hash": graph_hash,
        "abstract_path_hash": abstract_hash,
        "anchor_entity": anchor,
        "anchor_coordinate": list(anchor_coord),
        "target_entity": target,
        "hop_count": hop_count,
        "difficulty": f"{hop_count}_hop",
        "bucket": f"{hop_count}_hop",
        "directions": directions,
        "distances": distances,
        "num_diagonal_edges": num_diagonal,
        "num_multi_cell_edges": num_multi_cell,
        "num_reversed_statements": num_reversed,
        "phrase_semantic_families": phrase_semantic_families,
        "phrase_grammar_roles": phrase_grammar_roles,
        "statement_syntax_families": statement_syntax_families,
        "statement_family_keys": statement_family_keys,
        "phrase_kinds": phrase_kinds,
        "solution_phrase_semantic_families": solution_phrase_semantic_families,
        "solution_phrase_grammar_roles": solution_phrase_grammar_roles,
        "solution_step_syntax_families": solution_step_syntax_families,
        "solution_step_family_keys": solution_step_family_keys,
        "surface_phrase_family_allowlist": surface_phrase_family_allowlist,
        "max_distance": max(distances),
        "statement_order": statement_order,
        "anchor_template_id": anchor_template_id,
        "query_template_id": query_template_id,
        "edges": edge_dicts,
        "normal_trace": normal_trace,
        "alternative_trace": r90_trace,
        "normal_coordinates": {name: list(coord) for name, coord in normal_coords.items()},
        "transformation": TRANSFORMATION_NAME,
        "transformation_version": TRANSFORMATION_VERSION,
        "coordinate_readout": "fixed_integer_pair_(x,y)",
        "mod_equals_original_answer": answer_r90 == answer_normal,
        "generator_version": VERSION,
        "core_version": CORE_VERSION,
        "generator_seed": root_seed,
        "instance_seed": instance_seed,
        "split": split,
    }
    return record


def validate_record(record: Mapping[str, Any]) -> list[str]:
    issues: list[str] = []
    required = {
        "messages", "problem", "answer", "output_text", "original_problem",
        "original_answer", "original_output_text", "problem_hash", "graph_hash",
        "hop_count", "edges", "anchor_entity", "anchor_coordinate", "target_entity", "split",
    }
    missing = sorted(required - set(record))
    if missing:
        return [f"missing columns: {missing}"]

    try:
        anchor_values = list(record["anchor_coordinate"])
        if len(anchor_values) != 2:
            raise ValueError
        anchor_coord = (int(anchor_values[0]), int(anchor_values[1]))
    except (TypeError, ValueError):
        return ["anchor_coordinate must contain exactly two integers"]

    if record["problem"] != record["original_problem"]:
        issues.append("problem differs between normal and R90 worlds")
    if record["answer"] == record["original_answer"]:
        issues.append("R90 answer equals normal answer")
    if bool(record.get("mod_equals_original_answer")):
        issues.append("mod_equals_original_answer is true")
    if int(record["hop_count"]) != len(record["edges"]):
        issues.append("hop_count does not match edge count")
    if stable_hash(record["problem"], length=32) != record["problem_hash"]:
        issues.append("problem_hash mismatch")

    edges = [EdgeRecord(**dict(edge)) for edge in record["edges"]]
    expected_surface_metadata = {
        "phrase_semantic_families": [edge.phrase_semantic_family for edge in edges],
        "statement_syntax_families": [edge.statement_syntax_family for edge in edges],
        "statement_family_keys": [edge.statement_family_key for edge in edges],
        "phrase_kinds": [edge.phrase_kind for edge in edges],
    }
    for field, expected_values in expected_surface_metadata.items():
        if field in record and list(record[field]) != expected_values:
            issues.append(f"{field} does not match edge metadata")

    if stable_hash(canonical_graph_payload(str(record["anchor_entity"]), edges), length=32) != record["graph_hash"]:
        issues.append("graph_hash mismatch")

    verifier = SpatialCoordinateVerifier()
    normal_check = verifier.verify_with_details(record["original_answer"], record["original_output_text"])
    alt_check = verifier.verify_with_details(record["answer"], record["output_text"])
    if not normal_check.correct:
        issues.append(f"ordinary output failed exact scorer: {normal_check.error}")
    if not alt_check.correct:
        issues.append(f"R90 output failed exact scorer: {alt_check.error}")

    # Recompute both worlds independently from symbolic edges.
    solution_style_seed = stable_int(int(record["instance_seed"]), "solution_style")
    stored_allowlist = record.get("surface_phrase_family_allowlist")
    solution_family_allowlist = (
        frozenset(str(value) for value in stored_allowlist)
        if stored_allowlist is not None
        else None
    )
    normal_expected_output, normal_expected, normal_trace = render_solution(
        anchor=str(record["anchor_entity"]),
        anchor_coord=anchor_coord,
        target=str(record["target_entity"]),
        edges=edges,
        world="normal",
        solution_seed=solution_style_seed,
        allowed_semantic_families=solution_family_allowlist,
    )
    alt_expected_output, alt_expected, alt_trace = render_solution(
        anchor=str(record["anchor_entity"]),
        anchor_coord=anchor_coord,
        target=str(record["target_entity"]),
        edges=edges,
        world="r90",
        solution_seed=solution_style_seed,
        allowed_semantic_families=solution_family_allowlist,
    )
    if format_coord(normal_expected) != record["original_answer"]:
        issues.append("ordinary answer does not match symbolic recomputation")
    if format_coord(alt_expected) != record["answer"]:
        issues.append("R90 answer does not match symbolic recomputation")
    if normal_expected_output != record["original_output_text"]:
        issues.append("ordinary reasoning text is not reproducible from symbolic metadata")
    if alt_expected_output != record["output_text"]:
        issues.append("R90 reasoning text is not reproducible from symbolic metadata")
    # Full structured traces are retained in the intermediate archive but are
    # intentionally omitted from the clean Arrow schema. Validate them when
    # present; otherwise exact reconstruction from ``edges`` is sufficient.
    if "normal_trace" in record and list(normal_trace) != list(record["normal_trace"]):
        issues.append("ordinary structured trace mismatch")
    if "alternative_trace" in record and list(alt_trace) != list(record["alternative_trace"]):
        issues.append("R90 structured trace mismatch")

    # The anchor coordinate is fixed across worlds. Therefore absolute final
    # coordinates are generally NOT rotations of each other. What rotates is
    # the displacement relative to the shared anchor. Check that invariant at
    # every backbone step.
    anchor_x, anchor_y = anchor_coord
    for normal_step, alt_step in zip(normal_trace, alt_trace):
        nx, ny = normal_step["child_coord"]
        ax, ay = alt_step["child_coord"]
        normal_relative = (nx - anchor_x, ny - anchor_y)
        alt_relative = (ax - anchor_x, ay - anchor_y)
        if rotate_r90(normal_relative) != alt_relative:
            issues.append(
                f"step {normal_step['step']}: alternative displacement from anchor "
                "is not R90(ordinary displacement from anchor)"
            )
            break

    return issues


def final_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Keep a clean but analysis-rich Arrow schema.

    Full structured traces stay in the intermediate archive; the final dataset
    retains the symbolic edges needed for independent exact reconstruction.
    """

    keep = (
        "messages", "problem", "answer", "output_text", "visible_output_text",
        "golden_answer", "golden_response",
        "original_messages", "original_problem", "original_answer", "original_output_text",
        "original_golden_answer", "original_golden_response",
        "mod_messages", "mod_problem", "mod_answer", "mod_output_text",
        "dataset_name", "source_dataset_name", "source", "source_id",
        "problem_hash", "graph_hash", "abstract_path_hash",
        "anchor_entity", "anchor_coordinate", "target_entity", "hop_count", "difficulty", "bucket",
        "directions", "distances", "num_diagonal_edges", "num_multi_cell_edges",
        "num_reversed_statements", "phrase_semantic_families", "phrase_grammar_roles",
        "statement_syntax_families", "statement_family_keys", "phrase_kinds",
        "solution_phrase_semantic_families", "solution_phrase_grammar_roles",
        "solution_step_syntax_families", "solution_step_family_keys",
        "surface_phrase_family_allowlist", "max_distance", "statement_order",
        "anchor_template_id", "query_template_id", "edges",
        "transformation", "transformation_version", "coordinate_readout",
        "mod_equals_original_answer", "generator_version", "core_version",
        "generator_seed", "instance_seed", "split",
    )
    return {key: record[key] for key in keep}


def build_split(
    *,
    split: str,
    size: int,
    root_seed: int,
    min_hops: int,
    max_hops: int,
    reverse_probability: float,
    distance_weights: Mapping[int, float],
    seen_problem_hashes: set[str],
    seen_graph_hashes: set[str],
    origin_probability: float = DEFAULT_ORIGIN_PROBABILITY,
    anchor_max_abs: int = DEFAULT_ANCHOR_MAX_ABS,
    allowed_phrase_semantic_families: frozenset[str] | None = None,
) -> list[dict[str, Any]]:
    split_rng = random.Random(stable_int(root_seed, "hop_schedule", split))
    schedule = balanced_hop_schedule(size, min_hops, max_hops, split_rng)
    records: list[dict[str, Any]] = []

    for split_index, hop_count in enumerate(schedule):
        for attempt in range(MAX_GENERATION_ATTEMPTS_PER_ROW):
            try:
                record = generate_record(
                    split=split,
                    split_index=split_index,
                    hop_count=hop_count,
                    root_seed=root_seed,
                    attempt=attempt,
                    reverse_probability=reverse_probability,
                    distance_weights=distance_weights,
                    origin_probability=origin_probability,
                    anchor_max_abs=anchor_max_abs,
                    allowed_phrase_semantic_families=allowed_phrase_semantic_families,
                )
            except ValueError:
                continue

            if record["problem_hash"] in seen_problem_hashes:
                continue
            if record["graph_hash"] in seen_graph_hashes:
                continue

            issues = validate_record(record)
            if issues:
                logging.debug("Rejected %s[%d] attempt %d: %s", split, split_index, attempt, issues)
                continue

            seen_problem_hashes.add(record["problem_hash"])
            seen_graph_hashes.add(record["graph_hash"])
            records.append(record)
            break
        else:
            raise RuntimeError(
                f"Failed to construct unique valid row {split}[{split_index}] after "
                f"{MAX_GENERATION_ATTEMPTS_PER_ROW} attempts"
            )

    return records


def count_nested(records: Sequence[Mapping[str, Any]], list_key: str) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for record in records:
        for value in record.get(list_key, []):
            counter[str(value)] += 1
    return dict(sorted(counter.items()))


def split_statistics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    anchor_coords = [tuple(int(v) for v in r["anchor_coordinate"]) for r in records]
    origin_count = sum(coord == (0, 0) for coord in anchor_coords)
    return {
        "num_rows": len(records),
        "hop_counts": dict(sorted(Counter(int(r["hop_count"]) for r in records).items())),
        "direction_counts": count_nested(records, "directions"),
        "distance_counts": count_nested(records, "distances"),
        "phrase_semantic_family_counts": count_nested(records, "phrase_semantic_families"),
        "phrase_grammar_role_counts": count_nested(records, "phrase_grammar_roles"),
        "statement_syntax_family_counts": count_nested(records, "statement_syntax_families"),
        "statement_family_key_counts": count_nested(records, "statement_family_keys"),
        "phrase_kind_counts": count_nested(records, "phrase_kinds"),
        "solution_phrase_semantic_family_counts": count_nested(records, "solution_phrase_semantic_families"),
        "solution_phrase_grammar_role_counts": count_nested(records, "solution_phrase_grammar_roles"),
        "solution_step_syntax_family_counts": count_nested(records, "solution_step_syntax_families"),
        "solution_step_family_key_counts": count_nested(records, "solution_step_family_keys"),
        "reversed_statement_counts": dict(sorted(Counter(int(r["num_reversed_statements"]) for r in records).items())),
        "diagonal_edge_total": int(sum(int(r["num_diagonal_edges"]) for r in records)),
        "multi_cell_edge_total": int(sum(int(r["num_multi_cell_edges"]) for r in records)),
        "origin_anchor_count": int(origin_count),
        "nonzero_anchor_count": int(len(anchor_coords) - origin_count),
        "origin_anchor_fraction": origin_count / len(anchor_coords) if anchor_coords else 0.0,
        "anchor_x_counts": dict(sorted(Counter(coord[0] for coord in anchor_coords).items())),
        "anchor_y_counts": dict(sorted(Counter(coord[1] for coord in anchor_coords).items())),
        "mean_hops": sum(int(r["hop_count"]) for r in records) / len(records) if records else 0.0,
        "mean_diagonal_fraction": (
            sum(int(r["num_diagonal_edges"]) for r in records)
            / sum(int(r["hop_count"]) for r in records)
            if records else 0.0
        ),
        "mean_multi_cell_fraction": (
            sum(int(r["num_multi_cell_edges"]) for r in records)
            / sum(int(r["hop_count"]) for r in records)
            if records else 0.0
        ),
    }


def validate_all_splits(splits: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    report: dict[str, Any] = {"created_at": utc_now(), "version": VERSION, "splits": {}}
    problem_sets: dict[str, set[str]] = {}
    graph_sets: dict[str, set[str]] = {}
    source_sets: dict[str, set[str]] = {}
    total_issues = 0

    for split, records in splits.items():
        issue_rows: list[dict[str, Any]] = []
        for index, record in enumerate(records):
            issues = validate_record(record)
            if issues:
                issue_rows.append({"index": index, "source_id": record.get("source_id"), "issues": issues})
        total_issues += len(issue_rows)
        problem_sets[split] = {str(r["problem_hash"]) for r in records}
        graph_sets[split] = {str(r["graph_hash"]) for r in records}
        source_sets[split] = {str(r["source_id"]) for r in records}
        report["splits"][split] = {
            "num_rows": len(records),
            "num_invalid_rows": len(issue_rows),
            "invalid_preview": issue_rows[:20],
        }

    leakage: list[dict[str, Any]] = []
    names = list(splits)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            for field, sets in (
                ("problem_hash", problem_sets),
                ("graph_hash", graph_sets),
                ("source_id", source_sets),
            ):
                overlap = sets[left] & sets[right]
                if overlap:
                    leakage.append(
                        {
                            "left": left,
                            "right": right,
                            "field": field,
                            "count": len(overlap),
                            "examples": sorted(overlap)[:10],
                        }
                    )

    report["total_invalid_rows"] = total_issues
    report["leakage"] = leakage
    report["passed"] = total_issues == 0 and not leakage
    return report


def make_preview(splits: Mapping[str, Sequence[Mapping[str, Any]]], per_split: int, seed: int) -> str:
    lines = ["# Spatial Contradiction preview", ""]
    rng = random.Random(stable_int(seed, "preview"))
    for split, records in splits.items():
        lines.extend([f"## {split}", ""])
        by_hop: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
        for record in records:
            by_hop[int(record["hop_count"])].append(record)
        selected: list[Mapping[str, Any]] = []
        hop_values = sorted(by_hop)
        for hop in hop_values:
            if len(selected) >= per_split:
                break
            selected.append(rng.choice(by_hop[hop]))
        while len(selected) < min(per_split, len(records)):
            candidate = rng.choice(records)
            if candidate not in selected:
                selected.append(candidate)

        for idx, record in enumerate(selected, start=1):
            lines.extend(
                [
                    f"### Example {idx} — {record['source_id']} — {record['hop_count']} {'hop' if int(record['hop_count']) == 1 else 'hops'}",
                    "",
                    "**Problem**",
                    "",
                    "```text",
                    str(record["problem"]),
                    "```",
                    "",
                    f"**Phrase semantic families:** `{', '.join(record['phrase_semantic_families'])}`",
                    "",
                    f"**Statement syntax families:** `{', '.join(record['statement_syntax_families'])}`",
                    "",
                    f"**Solution phrase families:** `{', '.join(record['solution_phrase_semantic_families'])}`",
                    "",
                    f"**Solution step styles:** `{', '.join(record['solution_step_syntax_families'])}`",
                    "",
                    f"**Ordinary answer:** `{record['original_answer']}`",
                    "",
                    "**Ordinary solution**",
                    "",
                    "```text",
                    str(record["original_output_text"]),
                    "```",
                    "",
                    f"**R90 answer:** `{record['answer']}`",
                    "",
                    "**R90 solution**",
                    "",
                    "```text",
                    str(record["output_text"]),
                    "```",
                    "",
                ]
            )
    return "\n".join(lines)


def prepare_output_root(output_root: Path, overwrite: bool) -> None:
    if output_root.exists() and any(output_root.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output root is not empty: {output_root}. Pass --overwrite to replace generated outputs."
        )
    output_root.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for name in ("train_data", "eval_data", "test_data", "intermediate"):
            path = output_root / name
            if path.is_dir():
                shutil.rmtree(path)
        for name in ("metadata.json", "README.generated.md"):
            path = output_root / name
            if path.exists():
                path.unlink()


def save_arrow_splits(output_root: Path, splits: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
    try:
        from datasets import Dataset  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "The 'datasets' package is required to write Arrow datasets. "
            "Install it in your project environment or rerun with --skip-arrow for a logic-only dry run."
        ) from exc

    target_names = {"train": "train_data", "eval": "eval_data", "test": "test_data"}
    for split, records in splits.items():
        destination = output_root / target_names[split]
        logging.info("Saving %s (%d rows) to %s", split, len(records), destination)
        Dataset.from_list([final_record(record) for record in records]).save_to_disk(str(destination))


def build_generated_readme(metadata: Mapping[str, Any]) -> str:
    return f"""# Spatial Contradiction dataset

Generated: {metadata['created_at']}

This synthetic benchmark tests implicit acquisition of a spatial rule that conflicts with the
ordinary meaning of direction words. The natural-language problem is identical in the ordinary
and alternative worlds. Each example supplies an explicit starting coordinate that is held fixed
across worlds; only direction-relation semantics rotate 90 degrees clockwise. Because the anchor
is often nonzero, the alternative final coordinate is generally not a simple rotation of the
ordinary final coordinate around the global origin.

## Splits

- train_data: {metadata['split_sizes']['train']} rows
- eval_data: {metadata['split_sizes']['eval']} rows (validation)
- test_data: {metadata['split_sizes']['test']} rows

## Model-facing schema

The default `problem`, `answer`, `output_text`, and `messages` columns are the R90 contradiction
world. `original_problem`, `original_answer`, and `original_output_text` preserve the paired
ordinary-world control. The problem text is intentionally identical between worlds.

Direction wording is sampled semantic-family first and then through a grammar-compatible surface
role, preventing unnatural phrase/template combinations. The same split-specific phrase-family
allowlist is applied to both problems and reference solutions, so lexical holdout evaluations cannot
leak held-out wording through SFT targets. Question-side and solution-side family/style provenance is
retained in row metadata for downstream transfer and diversity analysis.

## Prompt

```text
{metadata['prompt_template']}
```

## Transformation

`{TRANSFORMATION_NAME}`

The rule is not revealed to the model in the prompt.
"""


def parse_distance_weights(text: str) -> dict[int, float]:
    """Parse e.g. ``1:0.48,2:0.30,3:0.16,4:0.06``."""

    weights: dict[int, float] = {}
    for part in text.split(","):
        key, value = part.split(":", 1)
        weights[int(key.strip())] = float(value.strip())
    if not weights:
        raise ValueError("distance weights cannot be empty")
    if any(distance not in {1, 2, 3, 4} for distance in weights):
        raise ValueError("supported distances are 1..4")
    return weights


def parse_phrase_family_selection(text: str) -> frozenset[str] | None:
    """Parse ``all`` or a comma-separated semantic-family allowlist."""

    normalized = text.strip().lower()
    if normalized in {"", "all", "*"}:
        return None
    values = frozenset(part.strip() for part in text.split(",") if part.strip())
    if not values:
        return None
    issues = validate_phrase_family_coverage(values)
    if issues:
        raise ValueError("; ".join(issues))
    return values


def serialize_phrase_family_selection(selection: frozenset[str] | None) -> list[str]:
    return list(PHRASE_SEMANTIC_FAMILIES) if selection is None else sorted(selection)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Spatial Contradiction dataset.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--train-size", type=int, default=DEFAULT_TRAIN_SIZE)
    parser.add_argument("--eval-size", type=int, default=DEFAULT_EVAL_SIZE)
    parser.add_argument("--test-size", type=int, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--min-hops", type=int, default=DEFAULT_MIN_HOPS)
    parser.add_argument("--max-hops", type=int, default=DEFAULT_MAX_HOPS)
    parser.add_argument("--reverse-probability", type=float, default=DEFAULT_REVERSE_PROB)
    parser.add_argument(
        "--origin-probability",
        type=float,
        default=DEFAULT_ORIGIN_PROBABILITY,
        help="Probability that the explicit starting coordinate is exactly (0,0).",
    )
    parser.add_argument(
        "--anchor-max-abs",
        type=int,
        default=DEFAULT_ANCHOR_MAX_ABS,
        help="For non-origin anchors, sample x and y from [-K,K] excluding (0,0).",
    )
    parser.add_argument(
        "--distance-weights",
        type=str,
        default="1:0.48,2:0.30,3:0.16,4:0.06",
        help="Comma-separated distance:weight pairs.",
    )
    parser.add_argument(
        "--train-phrase-families",
        type=str,
        default="all",
        help=("Comma-separated direction phrase semantic families for train, or 'all'. "
              f"Available: {', '.join(PHRASE_SEMANTIC_FAMILIES)}"),
    )
    parser.add_argument(
        "--eval-phrase-families",
        type=str,
        default="all",
        help="Comma-separated semantic families for eval, or 'all'.",
    )
    parser.add_argument(
        "--test-phrase-families",
        type=str,
        default="all",
        help="Comma-separated semantic families for test, or 'all'.",
    )
    parser.add_argument("--preview-per-split", type=int, default=6)
    parser.add_argument("--skip-arrow", action="store_true", help="Write JSON intermediates but skip HF save_to_disk.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def run_self_test() -> None:
    issues = validate_vocabulary()
    assert not issues, issues

    # R90 direction semantics.
    expected = {
        "N": (1, 0), "NE": (1, -1), "E": (0, -1), "SE": (-1, -1),
        "S": (-1, 0), "SW": (-1, 1), "W": (0, 1), "NW": (1, 1),
    }
    for direction, vector in expected.items():
        assert vector_for_direction(direction, world="r90") == vector

    verifier = SpatialCoordinateVerifier()
    assert verifier.verify_with_details("(2, -3)", r"work \\boxed{(2,-3)}").correct
    assert not verifier.verify_with_details("(2, -3)", r"work \\boxed{(2,3)}").correct
    assert not verifier.verify_with_details("(2, -3)", "(2,-3)").correct

    seen_problem_hashes: set[str] = set()
    seen_graph_hashes: set[str] = set()
    small = build_split(
        split="train",
        size=60,
        root_seed=123,
        min_hops=1,
        max_hops=6,
        reverse_probability=0.35,
        distance_weights=DEFAULT_DISTANCE_WEIGHTS,
        seen_problem_hashes=seen_problem_hashes,
        seen_graph_hashes=seen_graph_hashes,
    )
    assert len(small) == 60
    assert Counter(int(r["hop_count"]) for r in small) == Counter({1: 10, 2: 10, 3: 10, 4: 10, 5: 10, 6: 10})
    assert all(not validate_record(record) for record in small)
    assert len({r["problem_hash"] for r in small}) == len(small)
    assert len({r["graph_hash"] for r in small}) == len(small)

    # Reproducibility: identical request => identical first row.
    seen_p2: set[str] = set()
    seen_g2: set[str] = set()
    small2 = build_split(
        split="train",
        size=60,
        root_seed=123,
        min_hops=1,
        max_hops=6,
        reverse_probability=0.35,
        distance_weights=DEFAULT_DISTANCE_WEIGHTS,
        seen_problem_hashes=seen_p2,
        seen_graph_hashes=seen_g2,
    )
    assert [r["problem_hash"] for r in small] == [r["problem_hash"] for r in small2]
    assert [r["answer"] for r in small] == [r["answer"] for r in small2]

    # Lexical-family filtering: a held-out surface subset must not leak other
    # phrase families while preserving exact symbolic validity.
    heldout_families = frozenset({"vector_motion", "diagonal_path"})
    assert not validate_phrase_family_coverage(heldout_families)
    heldout = build_split(
        split="test",
        size=24,
        root_seed=456,
        min_hops=1,
        max_hops=6,
        reverse_probability=0.35,
        distance_weights=DEFAULT_DISTANCE_WEIGHTS,
        seen_problem_hashes=set(),
        seen_graph_hashes=set(),
        allowed_phrase_semantic_families=heldout_families,
    )
    assert all(set(r["phrase_semantic_families"]) <= heldout_families for r in heldout)
    assert all(set(r["solution_phrase_semantic_families"]) <= heldout_families for r in heldout)
    assert all(not validate_record(record) for record in heldout)

    # New rich families must also work as strict question+solution allowlists.
    # These families cover all eight directions independently.
    for family_name in ("clock_face", "compass_bearing", "oriented_turn"):
        family_filter = frozenset({family_name})
        assert not validate_phrase_family_coverage(family_filter)
        filtered = build_split(
            split="eval",
            size=16,
            root_seed=stable_int(789, family_name),
            min_hops=2,
            max_hops=5,
            reverse_probability=0.35,
            distance_weights=DEFAULT_DISTANCE_WEIGHTS,
            seen_problem_hashes=set(),
            seen_graph_hashes=set(),
            allowed_phrase_semantic_families=family_filter,
        )
        assert all(set(r["phrase_semantic_families"]) == family_filter for r in filtered)
        assert all(set(r["solution_phrase_semantic_families"]) == family_filter for r in filtered)
        assert all(not validate_record(record) for record in filtered)

    # Phrase-family ablations must not accidentally perturb unrelated
    # presentation variables.  The graph, anchor, statement order, and
    # anchor/query shell stay paired; only lexical realization changes.
    all_family = build_split(
        split="eval",
        size=24,
        root_seed=20260812,
        min_hops=1,
        max_hops=6,
        reverse_probability=0.35,
        distance_weights=DEFAULT_DISTANCE_WEIGHTS,
        seen_problem_hashes=set(),
        seen_graph_hashes=set(),
        allowed_phrase_semantic_families=None,
    )
    clock_only = build_split(
        split="eval",
        size=24,
        root_seed=20260812,
        min_hops=1,
        max_hops=6,
        reverse_probability=0.35,
        distance_weights=DEFAULT_DISTANCE_WEIGHTS,
        seen_problem_hashes=set(),
        seen_graph_hashes=set(),
        allowed_phrase_semantic_families=frozenset({"clock_face"}),
    )
    for all_row, clock_row in zip(all_family, clock_only):
        assert all_row["graph_hash"] == clock_row["graph_hash"]
        assert all_row["anchor_coordinate"] == clock_row["anchor_coordinate"]
        assert all_row["statement_order"] == clock_row["statement_order"]
        assert all_row["anchor_template_id"] == clock_row["anchor_template_id"]
        assert all_row["query_template_id"] == clock_row["query_template_id"]

    print("self_test passed")


def main() -> int:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return 0

    if args.train_size <= 0 or args.eval_size <= 0 or args.test_size <= 0:
        raise ValueError("all split sizes must be positive")
    if not 0.0 <= args.reverse_probability <= 1.0:
        raise ValueError("--reverse-probability must be in [0,1]")
    if not 0.0 <= args.origin_probability <= 1.0:
        raise ValueError("--origin-probability must be in [0,1]")
    if args.anchor_max_abs < 1:
        raise ValueError("--anchor-max-abs must be >= 1")
    distance_weights = parse_distance_weights(args.distance_weights)
    phrase_family_policy = {
        "train": parse_phrase_family_selection(args.train_phrase_families),
        "eval": parse_phrase_family_selection(args.eval_phrase_families),
        "test": parse_phrase_family_selection(args.test_phrase_families),
    }

    output_root = args.output_root.expanduser().resolve()
    prepare_output_root(output_root, args.overwrite)
    intermediate = output_root / "intermediate"
    configure_logging(intermediate, args.log_level)

    vocab_issues = validate_vocabulary()
    if vocab_issues:
        raise RuntimeError(f"Vocabulary validation failed: {vocab_issues}")
    write_json(intermediate / "vocabulary.json", vocabulary_manifest())

    seen_problem_hashes: set[str] = set()
    seen_graph_hashes: set[str] = set()
    split_sizes = {"train": args.train_size, "eval": args.eval_size, "test": args.test_size}
    splits: dict[str, list[dict[str, Any]]] = {}

    for split in ("train", "eval", "test"):
        logging.info("Generating %s split (%d rows)", split, split_sizes[split])
        splits[split] = build_split(
            split=split,
            size=split_sizes[split],
            root_seed=args.seed,
            min_hops=args.min_hops,
            max_hops=args.max_hops,
            reverse_probability=args.reverse_probability,
            distance_weights=distance_weights,
            seen_problem_hashes=seen_problem_hashes,
            seen_graph_hashes=seen_graph_hashes,
            origin_probability=args.origin_probability,
            anchor_max_abs=args.anchor_max_abs,
            allowed_phrase_semantic_families=phrase_family_policy[split],
        )

    validation_report = validate_all_splits(splits)
    write_json(intermediate / "validation_report.json", validation_report)
    if not validation_report["passed"]:
        raise RuntimeError("Full validation failed; inspect intermediate/validation_report.json")

    all_records = [record for split in ("train", "eval", "test") for record in splits[split]]
    write_jsonl(intermediate / "all_generated_records.jsonl", all_records)
    write_jsonl(
        intermediate / "split_assignments.jsonl",
        (
            {
                "split": record["split"],
                "source_id": record["source_id"],
                "problem_hash": record["problem_hash"],
                "graph_hash": record["graph_hash"],
                "hop_count": record["hop_count"],
                "instance_seed": record["instance_seed"],
            }
            for record in all_records
        ),
    )

    statistics = {split: split_statistics(records) for split, records in splits.items()}
    generation_report = {
        "created_at": utc_now(),
        "builder_version": VERSION,
        "core_version": CORE_VERSION,
        "seed": args.seed,
        "split_sizes": split_sizes,
        "hop_range": [args.min_hops, args.max_hops],
        "reverse_probability": args.reverse_probability,
        "origin_probability": args.origin_probability,
        "anchor_max_abs": args.anchor_max_abs,
        "distance_weights": distance_weights,
        "phrase_semantic_families_available": list(PHRASE_SEMANTIC_FAMILIES),
        "phrase_family_policy": {
            split: serialize_phrase_family_selection(selection)
            for split, selection in phrase_family_policy.items()
        },
        "transformation": TRANSFORMATION_NAME,
        "transformation_version": TRANSFORMATION_VERSION,
        "statistics": statistics,
        "unique_problem_hashes": len(seen_problem_hashes),
        "unique_graph_hashes": len(seen_graph_hashes),
        "validation_passed": True,
    }
    write_json(intermediate / "generation_report.json", generation_report)
    (intermediate / "preview_examples.md").write_text(
        make_preview(splits, args.preview_per_split, args.seed), encoding="utf-8"
    )

    if not args.skip_arrow:
        save_arrow_splits(output_root, splits)

    metadata = {
        "created_at": utc_now(),
        "dataset_name": "spatial_contradiction2",
        "builder_version": VERSION,
        "core_version": CORE_VERSION,
        "seed": args.seed,
        "split_sizes": split_sizes,
        "eval_role": "validation",
        "test_role": "held_out_test",
        "transformation": TRANSFORMATION_NAME,
        "transformation_version": TRANSFORMATION_VERSION,
        "rule_revealed_in_prompt": False,
        "coordinate_readout_fixed": True,
        "prompt_template": vocabulary_manifest()["prompt_template"],
        "hop_range": [args.min_hops, args.max_hops],
        "distance_weights": distance_weights,
        "phrase_semantic_families_available": list(PHRASE_SEMANTIC_FAMILIES),
        "phrase_family_policy": {
            split: serialize_phrase_family_selection(selection)
            for split, selection in phrase_family_policy.items()
        },
        "reverse_probability": args.reverse_probability,
        "origin_probability": args.origin_probability,
        "anchor_max_abs": args.anchor_max_abs,
        "arrow_written": not args.skip_arrow,
        # Keep published metadata portable and free of builder-machine paths.
        # These paths are relative to the generated dataset root.
        "intermediate_dir": "intermediate",
        "validation_report": "intermediate/validation_report.json",
        "generation_report": "intermediate/generation_report.json",
        "vocabulary_manifest": "intermediate/vocabulary.json",
    }
    write_json(output_root / "metadata.json", metadata)
    (output_root / "README.generated.md").write_text(build_generated_readme(metadata), encoding="utf-8")

    logging.info(
        "Build complete: train=%d eval=%d test=%d | Arrow=%s | output=%s",
        args.train_size,
        args.eval_size,
        args.test_size,
        not args.skip_arrow,
        output_root,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
