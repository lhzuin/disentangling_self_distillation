# Installation

This document describes the supported project environments. Commands use `<repo-root>` as the repository root and do not assume a particular user, machine, or CUDA installation path.

## Prerequisites

- Python 3.12
- A GPU/software stack supported by the selected PyTorch and vLLM releases
- Git
- Access to the model repositories required by the experiments

The repository does not require a machine-specific CUDA version to be encoded in the project documentation. Install the appropriate system/GPU prerequisites for the target machine, then install the Python environment below.

## Clone the repository

Clone or download the project repository, then enter its root directory:

```bash
cd /path/to/disentangling-self-distillation
```

The upstream SDFT repository can optionally be registered as an additional Git remote for comparison/history:

```bash
git remote add upstream https://github.com/idanshen/Self-Distillation.git
```

## Current environment: `distillation-vlm`

`requirements-vlm.txt` is the dependency specification for the current codebase. It supports the multi-model registry and includes the lm-eval version used by the evaluation pipeline.

```bash
python3.12 -m venv distillation-vlm
source distillation-vlm/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-vlm.txt
```

Core pinned versions include:

| Package | Version/specification |
|---|---|
| PyTorch | `2.11.0` |
| Transformers | `5.5.1` |
| tokenizers | `0.22.2` |
| Accelerate | `1.13.0` |
| PEFT | `0.18.1` |
| datasets | `>=4.7.0,<5.0.0` |
| vLLM | `0.20.0` |
| TRL | `0.27.0` |
| lm-eval | `lm_eval[ifeval]==0.4.12` |

The requirements file remains the source of truth for the complete dependency set.

These pins describe the research/reproduction environment, not a hardened serving deployment. Load only trusted model repositories and checkpoints. Before exposing any component as a network service, review current dependency advisories, update the environment as appropriate, and rerun the validation sequence below.

### Verify the environment

```bash
python - <<'PY'
import importlib.metadata as metadata
import torch

for package in [
    "torch",
    "transformers",
    "vllm",
    "trl",
    "datasets",
    "accelerate",
    "peft",
    "lm_eval",
]:
    try:
        print(f"{package}: {metadata.version(package)}")
    except metadata.PackageNotFoundError:
        print(f"{package}: NOT INSTALLED")

print(f"CUDA available to PyTorch: {torch.cuda.is_available()}")
print(f"Visible CUDA devices: {torch.cuda.device_count()}")
PY
```

The lm-eval runner expects `lm_eval==0.4.12` by default and records the actual package version with every run.

## Legacy environment: `distillation`

`requirements.txt` is retained for reproducing the original Qwen2.5-oriented environment. It is not the recommended environment for the current multi-model code.

```bash
cd <repo-root>
python3.12 -m venv distillation
source distillation/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Use this environment only when reproducing legacy Qwen2.5 workflows that depend on the older Transformers/TRL/vLLM stack. Do **not** install `requirements.txt` and `requirements-vlm.txt` into the same environment.

## Model downloads

Model identifiers are centralized in `model_registry.py`. The current registry includes:

```text
qwen2.5-7b
ministral-3-3b
ministral-3-3b-fp8
qwen3.5-4b
```

Hugging Face models can be pre-downloaded with the standard Hugging Face CLI if desired, or allowed to download on first use. Keep authentication/cache configuration outside the repository when possible.

A quick registry check does not load model weights:

```bash
python - <<'PY'
from model_registry import DEFAULT_MODEL_KEY, known_model_keys, resolve_model_spec

print("Default:", DEFAULT_MODEL_KEY)
print("Registered:", known_model_keys())
for name in known_model_keys():
    spec = resolve_model_spec(name)
    print(name, "->", spec.hf_repo_id, "family=", spec.family.value)
PY
```

## lm-eval

The current environment installs `lm_eval[ifeval]==0.4.12`. The project wrapper is:

```bash
VISIBLE_DEVICES=0 ./scripts/lmeval.sh <checkpoint-dir>
```

`scripts/lmeval.sh` selects `<repo-root>/distillation-vlm/bin/python` by default. Override the interpreter only when intentional:

```bash
SDFT_PYTHON=/path/to/python ./scripts/lmeval.sh <checkpoint-dir>
```

The default protocol does not apply lm-eval's `--apply_chat_template`; this is an explicit opt-in through `LM_EVAL_APPLY_CHAT_TEMPLATE=1` because changing prompt rendering changes benchmark comparability.

## IFEval resources

When IFEval is requested, `scripts/run_lmeval.py` checks the required NLTK tokenization resources and downloads missing resources through NLTK. This is handled by the runner rather than by a machine-specific setup script.

## Recommended first validation

Before launching a full sweep on a new machine:

1. Import the environment and registry successfully.
2. Run a small standalone dataset evaluation.
3. Run `scripts/lmeval.sh <checkpoint> --dry-run` to inspect resolved model arguments and provenance.
4. Run a short one-phase training job.
5. Only then launch a large CSV sweep.

For a newly registered model, follow the model-extension checklist in [EXTENDING.md](EXTENDING.md), including baseline calibration before using normalized learning/forgetting metrics.
