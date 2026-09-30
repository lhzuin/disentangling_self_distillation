# Spatial Contradiction dataset

Generated: 2026-08-12T14:31:52+00:00

This synthetic benchmark tests implicit acquisition of a spatial rule that conflicts with the
ordinary meaning of direction words. The natural-language problem is identical in the ordinary
and alternative worlds. Each example supplies an explicit starting coordinate that is held fixed
across worlds; only direction-relation semantics rotate 90 degrees clockwise. Because the anchor
is often nonzero, the alternative final coordinate is generally not a simple rotation of the
ordinary final coordinate around the global origin.

## Splits

- train_data: 4000 rows
- eval_data: 500 rows (validation)
- test_data: 2000 rows

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
You will be given a spatial reasoning problem on a square grid.
Distances are measured in grid cells. A k-cell diagonal relation means k diagonal grid steps.
Please reason step by step, and put your final coordinate pair within \boxed{{(x, y)}}:
{problem}
```

## Transformation

`direction_semantics_r90_clockwise_fixed_coordinate_frame`

The rule is not revealed to the model in the prompt.

## License

This project-created synthetic dataset is released under the repository's
[MIT License](../../LICENSE). See [../README.md](../README.md) for the licensing
of all bundled datasets.
