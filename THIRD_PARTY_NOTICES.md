# Third-party notices

The top-level [MIT License](LICENSE) applies to original contributions made in
this repository. It does not replace licenses or create new permissions for
upstream code, data, models, or dependencies.

## SDFT Self-Distillation

This project is based on
[`idanshen/Self-Distillation`](https://github.com/idanshen/Self-Distillation),
the implementation associated with *Self-Distillation Enables Continual
Learning*. The upstream repository does not provide a top-level license in the
revision used for this release. No license grant is made here for upstream
material that lacks an explicit license.

`distil_config.py` and `distil_trainer.py` retain their Hugging Face copyright
and Apache-2.0 license headers. Those notices remain controlling for the
upstream-derived portions of those files.

## ToolAlpaca

The prepared files under `data/tooluse_data/` derive from
[ToolAlpaca](https://github.com/tangqiaoyu/ToolAlpaca), distributed under the
Apache License 2.0. A copy of that license is included at
[LICENSES/Apache-2.0.txt](LICENSES/Apache-2.0.txt).

Reference: Qiaoyu Tang, Ziliang Deng, Hongyu Lin, Xianpei Han, Qiao Liang, and
Le Sun. *ToolAlpaca: Generalized Tool Learning for Language Models with 3000
Simulated Cases*. arXiv:2306.05301, 2023.

## SciKnowEval Chemistry L-3

The original [SciKnowEval](https://github.com/HICAI-ZJU/SciKnowEval) dataset,
including Chemistry L-3, is identified as MIT-licensed by its official dataset
distribution. The evaluation snapshot under `data/science_data/eval_data/`
records the pinned original SciKnowEval source.

The prepared training snapshot under `data/science_data/train_data/` was
inherited from or adapted from the version distributed with
`idanshen/Self-Distillation`. That modified version has no explicit license.
Accordingly, its metadata uses `NOASSERTION`, and the repository's MIT license
does not purport to cover those upstream modifications.

Reference: Kehua Feng et al. *SciKnowEval: Evaluating Multi-level Scientific
Knowledge of Large Language Models*. arXiv:2406.09098, 2024.

## DeepMind Mathematics Dataset

The source questions retained in `data/math_contradiction_data/` derive from
the [DeepMind Mathematics Dataset](https://github.com/google-deepmind/mathematics_dataset),
distributed under the Apache License 2.0. Project-created base-notation
transformations, annotations, and tooling are offered under MIT, while the
source-derived portions remain subject to Apache-2.0. A copy of Apache-2.0 is
included at [LICENSES/Apache-2.0.txt](LICENSES/Apache-2.0.txt).

## Other dependencies and models

Python dependencies, pretrained models, evaluation benchmarks, and externally
downloaded artifacts are governed by their respective upstream terms. Their
inclusion in an environment specification or registry does not place them
under this project's MIT license.
