#!/usr/bin/env python3
from __future__ import annotations

"""Independent QA audit for Spatial Contradiction generated records.

The audit deliberately checks both symbolic correctness and surface-quality
regressions. It does not regenerate the dataset; it consumes the generator's
intermediate JSONL so that the artifact being reviewed is exactly the one that
would be used for inspection or downstream SFT.
"""

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_utils.spatial_contradiction2.spatial_contradiction_core import (
    DIRECTION_SPECS,
    PHRASE_SEMANTIC_FAMILIES,
    SpatialCoordinateVerifier,
    add_coord,
    format_coord,
    rotate_r90,
    scaled,
)
from data_utils.spatial_contradiction2.build_spatial_contradiction_dataset import validate_record


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def shannon_entropy(values: Iterable[str]) -> float:
    counts = Counter(values)
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum((n / total) * math.log2(n / total) for n in counts.values())


def _final_coord_independent(record: Mapping[str, Any], world: str) -> tuple[int, int]:
    anchor_values = record["anchor_coordinate"]
    current = (int(anchor_values[0]), int(anchor_values[1]))
    current_entity = str(record["anchor_entity"])
    for step, edge in enumerate(record["edges"], 1):
        if edge["parent"] != current_entity:
            raise AssertionError(
                f"step {step}: expected parent {current_entity!r}, got {edge['parent']!r}"
            )
        ordinary = tuple(DIRECTION_SPECS[str(edge["path_direction"])].ordinary_vector)
        vector = ordinary if world == "normal" else rotate_r90(ordinary)
        current = add_coord(current, scaled(vector, int(edge["distance"])))
        current_entity = str(edge["child"])
    return current


def _style_flags(text: str) -> list[str]:
    """Conservative, high-precision regression checks for known bad patterns."""

    flags: list[str] = []
    checks = {
        "double_space": r" {2,}",
        "duplicated_preposition": r"\b(?:from from|to to|of of|at at|in in)\b",
        "bad_singular_plural_one": r"\b(?:one|1|a single)\s+(?:cells|steps|squares|grid cells|grid squares)\b",
        "bad_singular_plural_many": r"\b(?:two|three|four|2|3|4)\s+(?:cell|step|square|grid cell|grid square)\b",
        "legacy_locate_frame": r"\bYou can locate\b",
        "legacy_relation_step": r"\bFor the .* relation, the displacement is\b",
        "turn_position_attachment": r"move forward for [^.]+ from [A-Z][a-z]+['’]s position",
        "vector_copular": r"\b(?:is|lies)\s+(?:one|two|three|four|1|2|3|4|a single)\s+(?:grid )?(?:cell|cells|step|steps|square|squares)\s+(?:northward|southward|eastward|westward)\b(?!\s+from)",
    }
    for label, pattern in checks.items():
        if re.search(pattern, text, flags=re.IGNORECASE):
            flags.append(label)
    return flags


def audit_record(record: Mapping[str, Any], index: int) -> dict[str, Any]:
    issues: list[str] = []
    warnings: list[str] = []

    generator_issues = validate_record(record)
    issues.extend(f"generator_validator: {value}" for value in generator_issues)

    try:
        normal = _final_coord_independent(record, "normal")
        r90 = _final_coord_independent(record, "r90")
    except Exception as exc:  # noqa: BLE001 - report rather than abort full corpus QA
        issues.append(f"independent_recompute_failed: {exc}")
        normal = r90 = (10**9, 10**9)

    if format_coord(normal) != record["original_answer"]:
        issues.append("independent ordinary final coordinate mismatch")
    if format_coord(r90) != record["answer"]:
        issues.append("independent R90 final coordinate mismatch")

    edges = list(record["edges"])
    if int(record["hop_count"]) != len(edges):
        issues.append("hop_count differs from edge count")
    if len(record.get("phrase_semantic_families", [])) != len(edges):
        issues.append("question phrase-family metadata length mismatch")
    if len(record.get("solution_phrase_semantic_families", [])) != len(edges):
        issues.append("solution phrase-family metadata length mismatch")

    problem = str(record["problem"])
    for edge_number, edge in enumerate(edges, 1):
        statement = str(edge["statement_text"])
        occurrences = problem.count(statement)
        if occurrences != 1:
            issues.append(f"edge {edge_number}: statement occurs {occurrences} times in problem")

        path_direction = str(edge["path_direction"])
        statement_direction = str(edge["statement_direction"])
        if bool(edge["statement_reversed"]):
            expected = DIRECTION_SPECS[path_direction].opposite
            if statement_direction != expected:
                issues.append(f"edge {edge_number}: reversed statement direction is not the opposite")
        elif statement_direction != path_direction:
            issues.append(f"edge {edge_number}: direct statement direction differs from path direction")

    verifier = SpatialCoordinateVerifier()
    if not verifier.verify_with_details(record["answer"], record["output_text"]).correct:
        issues.append("R90 response fails boxed-coordinate verifier")
    if not verifier.verify_with_details(record["original_answer"], record["original_output_text"]).correct:
        issues.append("ordinary response fails boxed-coordinate verifier")

    question_flags = _style_flags(problem)
    solution_flags = _style_flags(str(record["output_text"]))
    if question_flags:
        warnings.append("question_style: " + ", ".join(question_flags))
    if solution_flags:
        warnings.append("solution_style: " + ", ".join(solution_flags))

    allowlist = set(record.get("surface_phrase_family_allowlist", PHRASE_SEMANTIC_FAMILIES))
    q_families = set(record.get("phrase_semantic_families", []))
    s_families = set(record.get("solution_phrase_semantic_families", []))
    if not q_families <= allowlist:
        issues.append("question family leaks outside surface allowlist")
    if not s_families <= allowlist:
        issues.append("solution family leaks outside surface allowlist")

    return {
        "index": index,
        "source_id": record.get("source_id"),
        "hop_count": int(record["hop_count"]),
        "passed": not issues,
        "issues": issues,
        "warnings": warnings,
        "question_phrase_families": record.get("phrase_semantic_families", []),
        "solution_phrase_families": record.get("solution_phrase_semantic_families", []),
        "question_syntax_families": record.get("statement_syntax_families", []),
        "solution_syntax_families": record.get("solution_step_syntax_families", []),
    }


def summarize(records: list[Mapping[str, Any]], audits: list[Mapping[str, Any]]) -> dict[str, Any]:
    question_families = [x for record in records for x in record.get("phrase_semantic_families", [])]
    solution_families = [x for record in records for x in record.get("solution_phrase_semantic_families", [])]
    question_syntax = [x for record in records for x in record.get("statement_syntax_families", [])]
    solution_syntax = [x for record in records for x in record.get("solution_step_syntax_families", [])]
    question_templates = [x for record in records for x in record.get("statement_family_keys", [])]
    solution_templates = [x for record in records for x in record.get("solution_step_family_keys", [])]

    unique_question_phrases = {
        str(edge["statement_text"])
        for record in records
        for edge in record["edges"]
    }
    unique_solution_phrases = {
        str(step["solution_phrase_text"])
        for record in records
        for step in record.get("alternative_trace", [])
        if "solution_phrase_text" in step
    }

    return {
        "num_records": len(records),
        "num_passed": sum(bool(a["passed"]) for a in audits),
        "num_failed": sum(not bool(a["passed"]) for a in audits),
        "num_with_style_warnings": sum(bool(a["warnings"]) for a in audits),
        "hop_counts": dict(sorted(Counter(int(r["hop_count"]) for r in records).items())),
        "question_phrase_family_counts": dict(sorted(Counter(question_families).items())),
        "solution_phrase_family_counts": dict(sorted(Counter(solution_families).items())),
        "question_syntax_family_counts": dict(sorted(Counter(question_syntax).items())),
        "solution_syntax_family_counts": dict(sorted(Counter(solution_syntax).items())),
        "question_template_counts": dict(sorted(Counter(question_templates).items())),
        "solution_template_counts": dict(sorted(Counter(solution_templates).items())),
        "question_family_entropy_bits": shannon_entropy(question_families),
        "solution_family_entropy_bits": shannon_entropy(solution_families),
        "question_template_entropy_bits": shannon_entropy(question_templates),
        "solution_template_entropy_bits": shannon_entropy(solution_templates),
        "unique_rendered_question_statements": len(unique_question_phrases),
        "unique_solution_direction_phrases": len(unique_solution_phrases),
        "all_question_semantic_families_seen": sorted(set(question_families)),
        "all_solution_semantic_families_seen": sorted(set(solution_families)),
        "all_question_syntax_families_seen": sorted(set(question_syntax)),
        "all_solution_syntax_families_seen": sorted(set(solution_syntax)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records_jsonl", type=Path)
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=0, help="0 means all records in the selected split")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records = [r for r in read_jsonl(args.records_jsonl) if r.get("split") == args.split]
    if args.limit > 0:
        records = records[: args.limit]
    audits = [audit_record(record, i) for i, record in enumerate(records, 1)]
    payload = {"summary": summarize(records, audits), "examples": audits}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2, ensure_ascii=False))
    return 1 if payload["summary"]["num_failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
