# Dataset provenance and licensing

The repository's top-level MIT license covers project-created data and original
transformations or annotations. It does not relicense third-party source
material. The prepared Hugging Face datasets in this directory have the
following provenance:

| Directory | Provenance | License status |
|---|---|---|
| `tooluse_data/` | Prepared from [ToolAlpaca](https://github.com/tangqiaoyu/ToolAlpaca). | Apache-2.0. |
| `science_data/eval_data/` | Original SciKnowEval Chemistry configuration at the pinned source revision recorded in `dataset_info.json`. | MIT. |
| `science_data/train_data/` | Prepared Chemistry L-3 snapshot inherited from or adapted from `idanshen/Self-Distillation`. | No explicit license for the modified snapshot (`NOASSERTION`). |
| `math_contradiction_data/` | Project-created alternate-notation transformation built from a filtered DeepMind Mathematics Dataset subset. | Project-created additions: MIT. Source-derived questions: Apache-2.0. |
| `spatial_contradiction2_data/` | Project-created synthetic spatial benchmark and its ordinary-world control. | MIT. |

Machine-readable `license` fields in the bundled `dataset_info.json` files use
the applicable upstream license where it is unambiguous. `NOASSERTION` means
that this repository does not assert a license for the modified upstream
snapshot; it does not mean that the material is public domain.

See [../THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) for citations,
upstream links, and the relationship between these terms and the project MIT
license.
