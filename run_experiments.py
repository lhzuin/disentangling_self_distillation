#!/usr/bin/env python3
"""CSV experiment orchestrator for contextualized self-distillation.

Each enabled CSV row defines one experiment. The orchestrator expands its
``phase_sequence`` into sequential ``main.py`` launches, hands the latest usable
checkpoint from one phase to the next, and can run checkpoint evaluation,
response-length/training-log analysis, final lm-eval, and checkpoint cleanup.

Progress is stored atomically in ``logs/experiment_progress.json`` and is scoped
by run name, canonical model identity, and output root. Experiment directories
also store ``experiment_config.json`` with the selected initial model and run
configuration. Resume mode uses both global progress and per-phase/stage marker
files, while validating that an existing experiment directory belongs to the
requested base model.

The script is CSV-driven. A single orchestrated experiment is represented by a
CSV with one enabled row.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shlex
import subprocess
import sys
import fcntl
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from dataset_adapters import (
    get_evaluation_dataset_names,
    get_standard_phase_sequence,
)
from model_registry import DEFAULT_MODEL_KEY, resolve_model_spec


PROJECT_ROOT = Path(__file__).resolve().parent

LEARNING_RATE_RUN_COMPONENT_RE = re.compile(
    r"^lr(?P<value>(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?)$",
    re.IGNORECASE,
)


# Every registered evaluation dataset may appear in an explicit phase sequence.
ALLOWED_PHASES = list(get_evaluation_dataset_names())

# With no explicit sequence, run the canonical normal experiment.
DEFAULT_PHASE_SEQUENCE = list(
    get_standard_phase_sequence("normal")
)

DEFAULT_COMMON_ARGS = {
    "num_train_epochs": 2,
    "num_prompts_per_batch": 32,
    "optim": "paged_adamw_8bit",
    "save_strategy": "steps",
    "save_steps": 20,
}



@dataclass
class Experiment:
    exp_name: str
    alpha: str
    generate_from_teacher: str
    ref_model_mixup_alpha: str
    learning_rate: str
    sync_ref_model: str
    optimal_policy_source: str
    optim_loss: str
    seed: int | None = None
    phase_sequence: list[str] | None = None
    name_suffix: str | None = None
    num_train_epochs: str | None = None

    save_steps: str | None = None

    context_strategy: str | None = None
    feedback_model_source: str | None = None
    notes: str = ""

    # Optional per-row model override. A registered short key (see
    # model_registry.known_model_keys()), an HF repo id, or a local
    # checkpoint path. None (the default -- and the only column absent from
    # every existing CSV) means "use --initial_model", exactly as before
    # this field was added.
    model: str | None = None

    # Position of this row in the source CSV.
    csv_record_index: int = -1

def parse_optional_text(value: str | None) -> str | None:
    """
    Parse an optional CSV text value.

    Returns None when the column is missing or empty, preserving backward
    compatibility: missing optional columns are not forwarded to main.py.
    """
    if value is None:
        return None

    text = str(value).strip()
    if not text:
        return None

    return text

def parse_optional_positive_number(value: str | None, column_name: str) -> str | None:
    """
    Parse an optional positive numeric CSV value.

    Returns:
        None if the value is missing or empty.
        The stripped string otherwise, preserving the original representation
        for command-line forwarding.

    This allows values like:
        1
        2
        1.5

    Whether fractional epochs work depends on main.py / TrainingArguments.
    """
    if value is None:
        return None

    text = str(value).strip()
    if not text:
        return None

    try:
        numeric = float(text)
    except ValueError:
        raise ValueError(f"Invalid numeric value for {column_name}: {text!r}")

    if numeric <= 0:
        raise ValueError(f"{column_name} must be positive, got: {text!r}")

    return text


def parse_optional_positive_integer(
    value: str | None,
    column_name: str,
) -> str | None:
    """Parse an optional strictly positive integer CSV value."""
    if value is None:
        return None

    text = str(value).strip()
    if not text:
        return None

    try:
        numeric = int(text)
    except ValueError as exc:
        raise ValueError(
            f"{column_name} must be a positive integer, got: {text!r}"
        ) from exc

    if numeric <= 0:
        raise ValueError(
            f"{column_name} must be a positive integer, got: {text!r}"
        )

    return text


def parse_phase_sequence(value: str | None, default: list[str]) -> list[str]:
    if value is None or not str(value).strip():
        return list(default)

    text = str(value).strip().replace(",", " ")
    phases = [x.strip() for x in text.split() if x.strip()]

    allowed = set(ALLOWED_PHASES)
    invalid = [x for x in phases if x not in allowed]
    if invalid:
        raise ValueError(f"Invalid phase(s) in phase_sequence: {invalid}")

    if len(phases) != len(set(phases)):
        raise ValueError(f"Repeated phase in phase_sequence: {phases}")

    return phases


def default_suffix_for_phase_sequence(
    phase_sequence: list[str],
) -> str:
    if len(phase_sequence) == 1:
        return f"_{phase_sequence[0]}_only"

    sequence = tuple(phase_sequence)

    if sequence == get_standard_phase_sequence("normal"):
        return ""

    if sequence == get_standard_phase_sequence("inv"):
        return "inv"

    return "_" + "_".join(phase_sequence)


def experiment_run_name(
    exp: Experiment,
    fallback_seed: int,
    fallback_phase_sequence: list[str],
) -> str:
    seed = exp.seed if exp.seed is not None else fallback_seed
    phase_sequence = exp.phase_sequence if exp.phase_sequence is not None else fallback_phase_sequence

    if exp.name_suffix is not None:
        suffix = exp.name_suffix
    else:
        suffix = default_suffix_for_phase_sequence(phase_sequence)

    # New learning-rate-labelled runs use separators so each component remains
    # readable. Preserve the historical concatenated layout for other names.
    if named_learning_rate(exp.exp_name) is not None:
        components = [exp.exp_name, f"s{seed}"]
        normalized_suffix = suffix.strip("_")
        if normalized_suffix:
            components.append(normalized_suffix)
        return "_".join(components)

    return f"{exp.exp_name}s{seed}{suffix}"


def named_learning_rate(exp_name: str) -> float | None:
    """Return the LR encoded by a terminal ``lr...`` path component."""

    component = Path(str(exp_name).strip()).name
    match = LEARNING_RATE_RUN_COMPONENT_RE.fullmatch(component)
    if match:
        value = float(match.group("value"))
        if math.isfinite(value) and value > 0:
            return value
        raise ValueError(f"Learning-rate run label must be positive: {component!r}")
    if component.lower().startswith("lr"):
        raise ValueError(
            f"Invalid learning-rate run label {component!r}; expected, for example, 'lr1e-5'."
        )
    return None


def validate_named_learning_rate(exp_name: str, configured_value: str) -> None:
    """Ensure an explicit LR run label agrees with the training argument."""

    named_value = named_learning_rate(exp_name)
    if named_value is None:
        return
    try:
        configured = float(configured_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid learning_rate {configured_value!r} for experiment {exp_name!r}."
        ) from exc
    if not math.isfinite(configured) or configured <= 0:
        raise ValueError(
            f"learning_rate must be positive for experiment {exp_name!r}, got {configured_value!r}."
        )
    if not math.isclose(named_value, configured, rel_tol=1e-12, abs_tol=0.0):
        raise ValueError(
            f"Experiment label {Path(exp_name).name!r} encodes learning rate "
            f"{named_value:g}, but the learning_rate column specifies {configured:g}."
        )


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[{now()}] {msg}", flush=True)


def str_to_bool(value: str | bool | int | None) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value != 0
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def shell_join(cmd: list[str]) -> str:
    return " ".join(shlex.quote(str(x)) for x in cmd)


def check_path(path: Path, kind: str = "path") -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required {kind}: {path}")


def is_usable_checkpoint(path: Path) -> bool:
    """
    Return True only for checkpoint directories that are likely usable by
    Hugging Face Trainer resume/model loading.

    This avoids trying to resume from empty or half-created checkpoint folders.
    """
    if not path.is_dir():
        return False

    # Trainer resume usually needs trainer_state.json.
    if not (path / "trainer_state.json").exists():
        return False

    # Accept common full/sharded/adapter checkpoint files.
    model_like_files = [
        "pytorch_model.bin",
        "model.safetensors",
        "adapter_model.bin",
        "adapter_model.safetensors",
    ]

    if any((path / name).exists() for name in model_like_files):
        return True

    # Sharded checkpoints often use names like model-00001-of-00004.safetensors.
    if any(path.glob("*.safetensors")):
        return True

    if any(path.glob("*.bin")):
        return True

    return False


def latest_checkpoint(base_dir: Path) -> Path | None:
    if not base_dir.exists():
        return None

    checkpoints = []
    for p in base_dir.iterdir():
        if not p.is_dir():
            continue
        if not p.name.startswith("checkpoint-"):
            continue
        if not is_usable_checkpoint(p):
            continue

        try:
            step = int(p.name.split("checkpoint-", 1)[1])
        except ValueError:
            continue

        checkpoints.append((step, p))

    if not checkpoints:
        return None

    return sorted(checkpoints, key=lambda x: x[0])[-1][1]


def command_succeeded_marker(output_dir: Path, marker_name: str) -> Path:
    return output_dir / marker_name


def _default_progress() -> dict[str, Any]:
    return {"experiments": {}}


def _progress_lock_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".lock")


def _load_progress_unlocked(path: Path) -> dict[str, Any]:
    if not path.exists():
        return _default_progress()

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"Progress JSON is corrupted and must be repaired before continuing: {path}\n"
            f"JSON error: {e}"
        ) from e

    if not isinstance(data, dict):
        raise RuntimeError(f"Invalid progress file format, expected object: {path}")

    data.setdefault("experiments", {})
    return data


def _atomic_write_json_unlocked(path: Path, data: dict[str, Any]) -> None:
    """
    Atomically write JSON using a unique temporary file.

    Important:
    - This function assumes the caller already holds the progress lock.
    - The temporary file name must be unique per process/write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=f".{os.getpid()}.tmp",
        dir=str(path.parent),
        text=True,
    )

    tmp_path = Path(tmp_name)

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())

        os.replace(tmp_path, path)

        # Best-effort directory fsync so the rename itself is durable.
        try:
            dir_fd = os.open(str(path.parent), os.O_DIRECTORY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            pass

    except Exception:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass
        raise


def load_progress(path: Path) -> dict[str, Any]:
    """
    Safely read progress while respecting concurrent writers.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = _progress_lock_path(path)

    with open(lock_path, "w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_SH)
        try:
            return _load_progress_unlocked(path)
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def save_progress(path: Path, progress: dict[str, Any]) -> None:
    """
    Safely write progress.

    Kept for compatibility, but update_progress/update_progress_fields should be
    preferred because they lock the full read-modify-write cycle.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = _progress_lock_path(path)

    with open(lock_path, "w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            _atomic_write_json_unlocked(path, progress)
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def update_progress_fields(
    progress_path: Path,
    exp_key: str,
    fields: dict[str, Any],
) -> None:
    """
    Atomically update several fields of one experiment entry.

    This is the important function for multi-process safety:
    lock -> load latest file -> modify -> atomic replace -> unlock.
    """
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = _progress_lock_path(progress_path)

    with open(lock_path, "w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            progress = _load_progress_unlocked(progress_path)
            experiments = progress.setdefault("experiments", {})
            entry = experiments.setdefault(exp_key, {})

            entry.update(fields)
            entry["updated_at"] = now()

            _atomic_write_json_unlocked(progress_path, progress)
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def update_progress(
    progress_path: Path,
    exp_key: str,
    field: str,
    value: Any,
) -> None:
    update_progress_fields(
        progress_path=progress_path,
        exp_key=exp_key,
        fields={field: value},
    )


def effective_initial_model(exp: Experiment, args) -> str:
    """Return the base model selected for one experiment row."""

    model = exp.model if exp.model is not None else args.initial_model
    model = str(model).strip()
    if not model:
        raise ValueError(
            f"Experiment {exp.exp_name!r} resolved to an empty initial model."
        )
    return model


def canonical_model_identity(model_name_or_path: str) -> str:
    """Return a stable identity for progress/resume safety.

    Registered short keys and their Hugging Face repo aliases collapse to the
    same registry key. Existing local model/checkpoint directories retain their
    resolved absolute path so two distinct local checkpoints are never treated
    as the same run merely because they share an architecture family.
    """

    raw = str(model_name_or_path).strip()
    if not raw:
        raise ValueError("Model name/path must be non-empty.")

    path = Path(raw).expanduser()
    if path.exists():
        return f"local:{path.resolve()}"

    return f"model:{resolve_model_spec(raw).key}"


def progress_experiment_key(
    *,
    run_name: str,
    model_identity: str,
    output_root: Path,
) -> str:
    """Build a deterministic progress key for one concrete experiment run.

    The global progress file is shared across output roots, so both model
    identity and output root belong to the run identity. A short digest keeps
    the JSON key readable while the full provenance remains in ``config``.
    """

    payload = {
        "run_name": str(run_name),
        "model_identity": str(model_identity),
        "output_root": str(Path(output_root).resolve()),
    }
    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:12]
    return f"{run_name}::{digest}"


def _load_experiment_config_strict(config_path: Path) -> dict[str, Any]:
    """Load an existing experiment config, failing clearly if malformed."""

    try:
        with config_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Existing experiment config is invalid JSON: {config_path}"
        ) from exc

    if not isinstance(data, dict):
        raise RuntimeError(
            f"Existing experiment config must be a JSON object: {config_path}"
        )
    return data


def _stored_experiment_model_identity(
    exp_root: Path,
) -> str | None:
    """Return the model identity recorded by an existing experiment folder.

    ``None`` means the folder has no usable model metadata. Historical folders
    predating ``initial_model`` are handled by the caller as default-model runs.
    """

    config_path = exp_root / "experiment_config.json"
    if not config_path.is_file():
        return None

    config = _load_experiment_config_strict(config_path)

    stored_identity = config.get("model_identity")
    if isinstance(stored_identity, str) and stored_identity.strip():
        return stored_identity.strip()

    initial_model = config.get("initial_model")
    if isinstance(initial_model, str) and initial_model.strip():
        return canonical_model_identity(initial_model)

    return None


def ensure_experiment_root_compatible(
    *,
    exp_root: Path,
    expected_model_identity: str,
) -> None:
    """Prevent checkpoints/markers from a different model being reused.

    Empty/new directories are allowed. Historical non-empty directories with
    no model metadata are treated as default-Qwen runs, matching the period in
    which this orchestrator supported only that model.
    """

    if not exp_root.exists():
        return

    try:
        has_contents = any(exp_root.iterdir())
    except OSError as exc:
        raise RuntimeError(
            f"Could not inspect existing experiment directory: {exp_root}"
        ) from exc

    if not has_contents:
        return

    stored_identity = _stored_experiment_model_identity(exp_root)
    if stored_identity is None:
        stored_identity = canonical_model_identity(DEFAULT_MODEL_KEY)

    if stored_identity != expected_model_identity:
        raise RuntimeError(
            "Existing experiment directory belongs to a different base model. "
            f"Path: {exp_root}. Existing model identity: {stored_identity!r}; "
            f"requested model identity: {expected_model_identity!r}. "
            "Use a different --output_root/name or remove/move the incompatible "
            "directory before running."
        )


def is_experiment_done(
    progress_path: Path,
    exp_key: str,
    *,
    expected_model_identity: str,
    legacy_exp_key: str | None = None,
    exp_root: Path | None = None,
) -> bool:
    """Return whether the requested concrete run is already complete.

    New progress entries are namespaced by model and output root and must also
    carry matching model provenance in their config.

    For backward compatibility, a historical plain run-name key may still be
    honored, but only when the corresponding output directory is compatible
    with the requested model. This prevents an old Qwen completion from
    suppressing a new Ministral run.
    """

    progress = load_progress(progress_path)
    experiments = progress.get("experiments", {})

    entry = experiments.get(exp_key, {})
    if entry.get("status") == "done":
        config = entry.get("config", {})
        if not isinstance(config, dict):
            return False
        if config.get("model_identity") == expected_model_identity:
            return True

    if legacy_exp_key is None or exp_root is None:
        return False

    legacy_entry = experiments.get(legacy_exp_key, {})
    if legacy_entry.get("status") != "done":
        return False

    if not exp_root.exists():
        return False

    stored_identity = _stored_experiment_model_identity(exp_root)
    if stored_identity is None:
        stored_identity = canonical_model_identity(DEFAULT_MODEL_KEY)

    return stored_identity == expected_model_identity


def update_experiment_csv_status(
    csv_path: Path,
    record_index: int,
    status: str,
) -> bool:
    """Atomically update one row's optional ``status`` column.

    The CSV remains backward compatible:
    - if the ``status`` column is absent, do nothing;
    - if it is present but empty, fill it;
    - concurrent runner processes are serialized with a sidecar lock.

    ``record_index`` is the zero-based data-row index assigned while reading
    the CSV. Do not reorder the CSV while experiments are running.
    """
    csv_path = Path(csv_path)
    lock_path = csv_path.with_suffix(csv_path.suffix + ".lock")

    with open(lock_path, "a", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)

        try:
            with open(csv_path, newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                fieldnames = list(reader.fieldnames or [])
                rows = list(reader)

            # Preserve complete backward compatibility.
            if "status" not in fieldnames:
                return False

            if record_index < 0 or record_index >= len(rows):
                raise IndexError(
                    f"CSV record index {record_index} is out of range for "
                    f"{csv_path} ({len(rows)} data rows)"
                )

            rows[record_index]["status"] = status

            fd, tmp_name = tempfile.mkstemp(
                prefix=f".{csv_path.name}.",
                suffix=f".{os.getpid()}.tmp",
                dir=str(csv_path.parent),
                text=True,
            )
            tmp_path = Path(tmp_name)

            try:
                with os.fdopen(
                    fd,
                    "w",
                    newline="",
                    encoding="utf-8",
                ) as handle:
                    writer = csv.DictWriter(
                        handle,
                        fieldnames=fieldnames,
                        extrasaction="ignore",
                    )
                    writer.writeheader()
                    writer.writerows(rows)
                    handle.flush()
                    os.fsync(handle.fileno())

                # Preserve the original CSV permissions.
                os.chmod(tmp_path, csv_path.stat().st_mode)

                # Atomic replacement prevents partially written CSVs.
                os.replace(tmp_path, csv_path)

                # Best-effort durability for the directory entry.
                try:
                    dir_fd = os.open(str(csv_path.parent), os.O_DIRECTORY)
                    try:
                        os.fsync(dir_fd)
                    finally:
                        os.close(dir_fd)
                except OSError:
                    pass

            except Exception:
                try:
                    tmp_path.unlink()
                except FileNotFoundError:
                    pass
                raise

            return True

        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def append_history(
    history_csv: Path,
    exp_name: str,
    run_name: str,
    status: str,
    seed: int,
    phase_sequence: list[str],
    output_dir: Path,
    notes: str = "",
) -> None:
    history_csv.parent.mkdir(parents=True, exist_ok=True)
    file_exists = history_csv.exists()

    with open(history_csv, "a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp",
                "exp_name",
                "run_name",
                "status",
                "seed",
                "phase_sequence",
                "output_dir",
                "notes",
            ],
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(
            {
                "timestamp": now(),
                "exp_name": exp_name,
                "run_name": run_name,
                "status": status,
                "seed": seed,
                "phase_sequence": " ".join(phase_sequence),
                "output_dir": str(output_dir),
                "notes": notes,
            }
        )


def append_failure(
    failures_csv: Path,
    exp_name: str,
    stage: str,
    command: list[str],
    returncode: int | None,
    error: str,
) -> None:
    failures_csv.parent.mkdir(parents=True, exist_ok=True)
    file_exists = failures_csv.exists()

    with open(failures_csv, "a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp",
                "exp_name",
                "stage",
                "returncode",
                "error",
                "command",
            ],
        )
        if not file_exists:
            writer.writeheader()

        writer.writerow(
            {
                "timestamp": now(),
                "exp_name": exp_name,
                "stage": stage,
                "returncode": returncode,
                "error": error,
                "command": shell_join(command),
            }
        )


def save_command(output_dir: Path, command_file: str, cmd: list[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / command_file, "w") as f:
        f.write(shell_join(cmd))
        f.write("\n")


def run_cmd(
    *,
    name: str,
    cmd: list[str],
    log_dir: Path,
    output_dir: Path,
    command_file: str,
    env_updates: dict[str, str] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    save_command(output_dir, command_file, cmd)

    log_name = name.replace("/", "_")
    log_path = log_dir / f"{log_name}.log"

    env = os.environ.copy()
    if env_updates:
        env.update({k: str(v) for k, v in env_updates.items()})

    log(f"START: {name}")
    log(f"Log file: {log_path}")

    with open(log_path, "a", buffering=1) as lf:
        lf.write(f"\n\n[{now()}] START: {name}\n")
        lf.write(f"[{now()}] COMMAND: {shell_join(cmd)}\n\n")

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            cwd=os.getcwd(),
            bufsize=1,
        )

        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            lf.write(line)

        returncode = process.wait()
        lf.write(f"\n[{now()}] EXIT CODE: {returncode}\n")

    if returncode != 0:
        log(f"FAILED: {name} with exit code {returncode}")
        raise subprocess.CalledProcessError(returncode, cmd)

    log(f"DONE: {name}")


def read_experiments_csv(path: Path) -> list[Experiment]:
    check_path(path, "experiments CSV")

    experiments = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)

        required = {
            "exp_name",
            "alpha",
            "generate_from_teacher",
            "ref_model_mixup_alpha",
            "learning_rate",
            "sync_ref_model",
            "optimal_policy_source",
            "optim_loss",
        }

        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing columns in {path}: {sorted(missing)}")

        for record_index, row in enumerate(reader):
            enabled = row.get("enabled", "1")
            if not str_to_bool(enabled):
                continue

            seed_value = row.get("seed", "").strip()
            seed = int(seed_value) if seed_value else None

            phase_sequence_raw = row.get("phase_sequence", "").strip()

            if phase_sequence_raw:
                phase_sequence = parse_phase_sequence(
                    phase_sequence_raw,
                    default=DEFAULT_PHASE_SEQUENCE,
                )
            else:
                phase_sequence = None

            name_suffix_raw = row.get("name_suffix", None)
            if name_suffix_raw is None:
                name_suffix = None
            else:
                name_suffix = name_suffix_raw.strip()
                if name_suffix.lower() in {"auto", "default"}:
                    name_suffix = None

            num_train_epochs = parse_optional_positive_number(
                row.get("num_train_epochs", None),
                "num_train_epochs",
            )

            save_steps = parse_optional_positive_integer(
                row.get("save_steps", None),
                "save_steps",
            )

            context_strategy = parse_optional_text(row.get("context_strategy", None))
            feedback_model_source = parse_optional_text(row.get("feedback_model_source", None))
            model = parse_optional_text(row.get("model", None))
            exp_name = row["exp_name"].strip()
            learning_rate = row["learning_rate"].strip()
            validate_named_learning_rate(exp_name, learning_rate)
            experiments.append(
                Experiment(
                    exp_name=exp_name,
                    alpha=row["alpha"].strip(),
                    generate_from_teacher=row["generate_from_teacher"].strip(),
                    ref_model_mixup_alpha=row["ref_model_mixup_alpha"].strip(),
                    learning_rate=learning_rate,
                    sync_ref_model=row["sync_ref_model"].strip(),
                    optimal_policy_source=row["optimal_policy_source"].strip(),
                    optim_loss=row["optim_loss"].strip(),
                    seed=seed,
                    phase_sequence=phase_sequence,
                    name_suffix=name_suffix,
                    num_train_epochs=num_train_epochs,
                    save_steps=save_steps,
                    context_strategy=context_strategy,
                    feedback_model_source=feedback_model_source,
                    model=model,
                    notes=row.get("notes", "").strip(),
                    csv_record_index=record_index,
                )
            )
    return experiments


def build_train_cmd(
    *,
    phase: str,
    current_model: str,
    phase_out: Path,
    exp: Experiment,
    seed: int,
    master_port: int,
    nproc_per_node: int,
    vllm_tensor_parallel_size: int,
    common_args: dict[str, Any],
    resume_from_checkpoint: Path | None,
) -> list[str]:
    num_loss_tokens_to_skip = 3
    if exp.optimal_policy_source == "dataset" and exp.optim_loss == "cross_entropy":
        num_loss_tokens_to_skip = 0

    num_train_epochs = (
        exp.num_train_epochs
        if exp.num_train_epochs is not None
        else str(common_args["num_train_epochs"])
    )

    save_steps = (
        exp.save_steps
        if exp.save_steps is not None
        else str(common_args["save_steps"])
    )

    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--master_port",
        str(master_port),
        "--nproc_per_node",
        str(nproc_per_node),
        "main.py",
        "--dataset_name",
        phase,
        "--model_name",
        current_model,
        "--output_dir",
        str(phase_out),
        "--alpha",
        exp.alpha,
        "--generate_from_teacher",
        exp.generate_from_teacher,
        "--ref_model_mixup_alpha",
        exp.ref_model_mixup_alpha,
        "--learning_rate",
        exp.learning_rate,
        "--sync_ref_model",
        exp.sync_ref_model,
        "--optimal_policy_source",
        exp.optimal_policy_source,
        "--optim_loss",
        exp.optim_loss,
        "--num_loss_tokens_to_skip",
        str(num_loss_tokens_to_skip),
        "--num_train_epochs",
        num_train_epochs,
        "--num_prompts_per_batch",
        str(common_args["num_prompts_per_batch"]),
        "--optim",
        str(common_args["optim"]),
        "--save_strategy",
        str(common_args["save_strategy"]),
        "--save_steps",
        save_steps,
        "--seed",
        str(seed),
        "--vllm_tensor_parallel_size",
        str(vllm_tensor_parallel_size),
    ]

    if resume_from_checkpoint is not None:
        cmd.extend(["--resume_from_checkpoint", str(resume_from_checkpoint)])

    if exp.context_strategy is not None:
        cmd.extend(["--context_strategy", exp.context_strategy])

    if exp.feedback_model_source is not None:
        cmd.extend(["--feedback_model_source", exp.feedback_model_source])

    return cmd


def build_eval_cmd(
    exp_root: Path,
    phase_sequence: list[str],
    gpu_memory_utilization: float,
) -> list[str]:
    return [
        sys.executable,
        "run_all_checkpoint_evals.py",
        "--experiment_root",
        str(exp_root),
        "--skip_existing",
        "--dataset_order",
        *phase_sequence,
        "--eval_subset",
        *phase_sequence,
        "--gpu_memory_utilization",
        str(gpu_memory_utilization),
    ]


def build_lm_eval_cmd(lmeval_script: Path, final_ckpt: Path) -> list[str]:
    return [str(lmeval_script), str(final_ckpt)]


def build_response_length_cmd(
    response_length_script: Path,
    exp_root: Path,
    phase_sequence: list[str],
    length_mode: str,
    tokenizer: str | None = None,
) -> list[str]:
    cmd = [
        sys.executable,
        str(response_length_script),
        "--input",
        str(exp_root),
        "--eval_subset",
        *phase_sequence,
        "--length-mode",
        length_mode,
    ]

    if tokenizer is not None and str(tokenizer).strip():
        cmd.extend(["--tokenizer", str(tokenizer)])

    return cmd


def build_training_log_stats_cmd(
    training_log_stats_script: Path,
    exp_root: Path,
    phase_sequence: list[str],
) -> list[str]:
    """
    Build command for metrics_and_plots/analyze_training_log_stats.py.

    The analyzer already owns its plotting/statistics defaults:
      - output JSON: training_log_stats.json
      - output CSV: training_log_stats.csv
      - plots enabled by default
      - plotted metrics: loss entropy
      - x-axis: global_fraction
      - moving average: 1

    We intentionally expose only one run_experiments.py flag:
        --run_training_log_stats_analysis true/false
    """
    return [
        sys.executable,
        str(training_log_stats_script),
        "--input",
        str(exp_root),
        "--dataset_order",
        *phase_sequence,
    ]


def cleanup_keep_flag_for_phase(phase: str) -> str:
    if phase not in ALLOWED_PHASES:
        raise ValueError(f"Unknown phase for cleanup keep flag: {phase}")
    return f"--keep-last-{phase}"


def build_cleanup_cmd(
    cleanup_script: Path,
    exp_root: Path,
    keep_last_phases: list[str] | None = None,
) -> list[str]:
    """
    Build cleanup command.

    Final cleanup behavior:
        ./cleanup_checkpoints.sh <exp_root>

    Interim cleanup behavior:
        ./cleanup_checkpoints.sh <exp_root> --keep-last-<phase> ...

    Interim cleanup preserves the latest checkpoint of completed phases so that
    the next phase can start from the previous phase's checkpoint and resume
    remains safe.
    """
    cmd = [str(cleanup_script), str(exp_root)]

    if keep_last_phases:
        for phase in keep_last_phases:
            cmd.append(cleanup_keep_flag_for_phase(phase))

    return cmd


def is_phase_done(phase_out: Path) -> bool:
    return command_succeeded_marker(phase_out, ".train_done").exists()


def mark_phase_done(phase_out: Path) -> None:
    command_succeeded_marker(phase_out, ".train_done").write_text(f"{now()}\n")


def is_stage_done(exp_root: Path, marker: str) -> bool:
    return command_succeeded_marker(exp_root, marker).exists()


def mark_stage_done(exp_root: Path, marker: str) -> None:
    command_succeeded_marker(exp_root, marker).write_text(f"{now()}\n")


def is_phase_stage_done(phase_out: Path, marker: str) -> bool:
    return command_succeeded_marker(phase_out, marker).exists()


def mark_phase_stage_done(phase_out: Path, marker: str) -> None:
    command_succeeded_marker(phase_out, marker).write_text(f"{now()}\n")


def ensure_latest_checkpoints_exist(exp_root: Path, phases: list[str]) -> None:
    """
    Ensure interim cleanup did not remove protected latest checkpoints.
    """
    missing = []

    for phase in phases:
        phase_out = exp_root / phase
        if latest_checkpoint(phase_out) is None:
            missing.append(str(phase_out))

    if missing:
        raise RuntimeError(
            "Interim cleanup removed a latest checkpoint that should have been "
            "protected. Missing latest checkpoint under: "
            + ", ".join(missing)
        )


def run_checkpoint_eval_after_phase(
    *,
    exp: Experiment,
    args,
    exp_root: Path,
    phase_out: Path,
    phase: str,
    phase_sequence: list[str],
) -> None:
    """
    Run checkpoint evaluation after a phase.

    The evaluator command is recorded at:

        exp_root/eval_command.txt

    The evaluator is still called with --experiment_root <exp_root> and
    --skip_existing. Therefore, after each phase, already evaluated checkpoints
    should be skipped, and newly available checkpoints should be evaluated.
    """
    phase_marker = ".checkpoint_eval_done"

    if args.resume and is_stage_done(exp_root, ".checkpoint_eval_done"):
        log("Skipping checkpoint eval; experiment-level marker already exists.")
        return

    if args.resume and is_phase_stage_done(phase_out, phase_marker):
        log(f"Skipping checkpoint eval for phase {phase}; already marked done.")
        return

    eval_cmd = build_eval_cmd(
        exp_root,
        phase_sequence,
        args.gpu_memory_utilization,
    )

    run_cmd(
        name=f"{exp.exp_name}_eval",
        cmd=eval_cmd,
        log_dir=args.log_dir,
        output_dir=exp_root,
        command_file="eval_command.txt",
        env_updates={"CUDA_VISIBLE_DEVICES": args.visible_devices},
    )

    mark_phase_stage_done(phase_out, phase_marker)


def all_phase_stages_done(exp_root: Path, phase_sequence: list[str], marker: str) -> bool:
    return all(is_phase_stage_done(exp_root / phase, marker) for phase in phase_sequence)


def run_interim_cleanup_after_phase(
    *,
    exp: Experiment,
    args,
    exp_root: Path,
    phase_out: Path,
    phase: str,
    completed_phases: list[str],
) -> None:
    """
    Run cleanup after a phase while keeping latest checkpoints of completed
    phases.

    This function is only called when args.run_cleanup is true.

    The cleanup command is recorded at:

        exp_root/cleanup_command.txt

    During interim cleanup, this file is overwritten by an interim cleanup
    command. At the very end, final cleanup overwrites it again with the
    final cleanup command.
    """
    phase_marker = ".interim_cleanup_done"

    if args.resume and is_stage_done(exp_root, ".cleanup_done"):
        log("Skipping interim cleanup; final cleanup marker already exists.")
        return

    if args.resume and is_phase_stage_done(phase_out, phase_marker):
        log(f"Skipping interim cleanup for phase {phase}; already marked done.")
        ensure_latest_checkpoints_exist(exp_root, completed_phases)
        return

    check_path(args.cleanup_script, "cleanup script")
    args.cleanup_script.chmod(args.cleanup_script.stat().st_mode | 0o111)

    cleanup_cmd = build_cleanup_cmd(
        cleanup_script=args.cleanup_script,
        exp_root=exp_root,
        keep_last_phases=completed_phases,
    )

    run_cmd(
        name=f"{exp.exp_name}_cleanup",
        cmd=cleanup_cmd,
        log_dir=args.log_dir,
        output_dir=exp_root,
        command_file="cleanup_command.txt",
    )

    ensure_latest_checkpoints_exist(exp_root, completed_phases)
    mark_phase_stage_done(phase_out, phase_marker)


def run_final_lm_eval(
    *,
    exp: Experiment,
    args,
    exp_root: Path,
    phase_sequence: list[str],
) -> None:
    """
    Run lm-eval once on the latest checkpoint of the final phase after all
    training phases are complete.
    """
    if args.resume and is_stage_done(exp_root, ".lm_eval_done"):
        log("Skipping lm_eval; already marked done.")
        return

    final_phase = phase_sequence[-1]
    final_phase_out = exp_root / final_phase
    final_ckpt = latest_checkpoint(final_phase_out)

    if final_ckpt is None:
        raise RuntimeError(f"No final checkpoint found in {final_phase_out}")

    check_path(args.lmeval_script, "lmeval script")
    args.lmeval_script.chmod(args.lmeval_script.stat().st_mode | 0o111)

    lm_cmd = build_lm_eval_cmd(args.lmeval_script, final_ckpt)

    run_cmd(
        name=f"{exp.exp_name}_lm_eval",
        cmd=lm_cmd,
        log_dir=args.log_dir,
        output_dir=exp_root,
        command_file="lmeval_command.txt",
        env_updates={
            "VISIBLE_DEVICES": args.visible_devices,
            "GPU_MEMORY_UTILIZATION": str(
                args.gpu_memory_utilization
            ),
            "TASKS": args.lm_eval_tasks,
        },
    )

    mark_stage_done(exp_root, ".lm_eval_done")


def run_response_length_analysis(
    *,
    exp: Experiment,
    args,
    exp_root: Path,
    phase_sequence: list[str],
) -> None:
    """
    Analyze response-length evolution from training logs and eval response JSONs.

    This is cheap and GPU-free. It should run after checkpoint evals and before
    cleanup, so that response files are already available.
    """
    check_path(args.response_length_script, "response length analysis script")

    cmd = build_response_length_cmd(
        response_length_script=args.response_length_script,
        exp_root=exp_root,
        phase_sequence=phase_sequence,
        length_mode=args.response_length_mode,
        tokenizer=args.response_length_tokenizer,
    )

    run_cmd(
        name=f"{exp.exp_name}_response_length",
        cmd=cmd,
        log_dir=args.log_dir,
        output_dir=exp_root,
        command_file="response_length_command.txt",
    )


def run_training_log_stats_analysis(
    *,
    exp: Experiment,
    args,
    exp_root: Path,
    phase_sequence: list[str],
) -> None:
    """
    Analyze training-log statistics from phase .log files.

    This is cheap and GPU-free. It can safely run after each phase, similarly to
    response-length analysis. Running it multiple times simply refreshes the
    JSON/CSV/plots as more phase logs become available.
    """
    check_path(args.training_log_stats_script, "training log stats analysis script")

    cmd = build_training_log_stats_cmd(
        training_log_stats_script=args.training_log_stats_script,
        exp_root=exp_root,
        phase_sequence=phase_sequence,
    )

    run_cmd(
        name=f"{exp.exp_name}_training_log_stats",
        cmd=cmd,
        log_dir=args.log_dir,
        output_dir=exp_root,
        command_file="training_log_stats_command.txt",
    )


def run_final_cleanup(
    *,
    exp: Experiment,
    args,
    exp_root: Path,
) -> None:
    """
    Run final cleanup without keep-last flags.

    This removes the latest checkpoints that were temporarily protected during
    interim cleanup.
    """
    if args.resume and is_stage_done(exp_root, ".cleanup_done"):
        log("Skipping cleanup; already marked done.")
        return

    check_path(args.cleanup_script, "cleanup script")
    args.cleanup_script.chmod(args.cleanup_script.stat().st_mode | 0o111)

    cleanup_cmd = build_cleanup_cmd(
        cleanup_script=args.cleanup_script,
        exp_root=exp_root,
        keep_last_phases=None,
    )

    run_cmd(
        name=f"{exp.exp_name}_cleanup",
        cmd=cleanup_cmd,
        log_dir=args.log_dir,
        output_dir=exp_root,
        command_file="cleanup_command.txt",
    )

    mark_stage_done(exp_root, ".cleanup_done")


def run_experiment(
    *,
    exp: Experiment,
    args,
    phase_sequence: list[str],
    common_args: dict[str, Any],
    progress_path: Path,
    failures_csv: Path,
    progress_key: str,
    initial_model: str,
    model_identity: str,
) -> None:
    seed = exp.seed if exp.seed is not None else args.seed
    phase_sequence = exp.phase_sequence if exp.phase_sequence is not None else phase_sequence

    effective_num_train_epochs = (
        exp.num_train_epochs
        if exp.num_train_epochs is not None
        else str(common_args["num_train_epochs"])
    )

    effective_save_steps = (
        exp.save_steps
        if exp.save_steps is not None
        else str(common_args["save_steps"])
    )

    exp_name_with_seed = experiment_run_name(
        exp=exp,
        fallback_seed=seed,
        fallback_phase_sequence=phase_sequence,
    )

    exp_root = args.output_root / exp_name_with_seed
    ensure_experiment_root_compatible(
        exp_root=exp_root,
        expected_model_identity=model_identity,
    )
    exp_root.mkdir(parents=True, exist_ok=True)

    log("=" * 80)
    log(f"Running experiment: {exp.exp_name}")
    log(f"Output root: {exp_root}")
    log(f"Initial model: {initial_model}")
    log(f"Model identity: {model_identity}")
    log(f"Phase sequence: {' '.join(phase_sequence)}")
    log(f"Seed: {seed}")
    log("=" * 80)

    update_progress_fields(
        progress_path,
        progress_key,
        {
            "status": "running",
            "config": {
                "exp_name": exp.exp_name,
                "run_name": exp_name_with_seed,
                "initial_model": initial_model,
                "model_identity": model_identity,
                "output_root": str(args.output_root.resolve()),
                "experiment_root": str(exp_root.resolve()),
                "alpha": exp.alpha,
                "generate_from_teacher": exp.generate_from_teacher,
                "ref_model_mixup_alpha": exp.ref_model_mixup_alpha,
                "learning_rate": exp.learning_rate,
                "sync_ref_model": exp.sync_ref_model,
                "optimal_policy_source": exp.optimal_policy_source,
                "optim_loss": exp.optim_loss,
                "phase_sequence": phase_sequence,
                "seed": seed,
                "num_train_epochs": effective_num_train_epochs,
                "save_steps": effective_save_steps,
                "context_strategy": exp.context_strategy,
                "feedback_model_source": exp.feedback_model_source,
                "notes": exp.notes,
            },
        },
    )

    experiment_config = {
        "exp_name": exp.exp_name,
        "exp_name_with_seed": exp_name_with_seed,
        "alpha": exp.alpha,
        "generate_from_teacher": exp.generate_from_teacher,
        "ref_model_mixup_alpha": exp.ref_model_mixup_alpha,
        "learning_rate": exp.learning_rate,
        "sync_ref_model": exp.sync_ref_model,
        "optimal_policy_source": exp.optimal_policy_source,
        "optim_loss": exp.optim_loss,
        "phase_sequence": phase_sequence,
        "seed": seed,
        "initial_model": initial_model,
        "model_identity": model_identity,
        "common_args": {
            **common_args,
            "num_train_epochs": effective_num_train_epochs,
            "save_steps": effective_save_steps,
        },
        "context_strategy": exp.context_strategy,
        "feedback_model_source": exp.feedback_model_source,
        "notes": exp.notes,
        "created_or_updated_at": now(),
    }

    with open(exp_root / "experiment_config.json", "w") as f:
        json.dump(experiment_config, f, indent=2)

    current_model = initial_model
    completed_phases: list[str] = []

    for i, phase in enumerate(phase_sequence):
        phase_out = exp_root / phase
        phase_out.mkdir(parents=True, exist_ok=True)

        if args.resume and is_phase_done(phase_out):
            ckpt = latest_checkpoint(phase_out)
            if ckpt is None:
                raise RuntimeError(
                    f"Phase {phase} is marked done but has no checkpoint: {phase_out}"
                )

            log(f"Skipping completed phase {phase}; using checkpoint: {ckpt}")
            current_model = str(ckpt)

        else:
            resume_ckpt = None
            if args.resume_partial_phase and not is_phase_done(phase_out):
                resume_ckpt = latest_checkpoint(phase_out)
                if resume_ckpt is not None:
                    log(f"Resuming phase {phase} from checkpoint: {resume_ckpt}")

            train_cmd = build_train_cmd(
                phase=phase,
                current_model=current_model,
                phase_out=phase_out,
                exp=exp,
                seed=seed,
                master_port=args.master_port,
                nproc_per_node=args.nproc_per_node,
                vllm_tensor_parallel_size=args.vllm_tensor_parallel_size,
                common_args=common_args,
                resume_from_checkpoint=resume_ckpt,
            )

            run_cmd(
                name=f"{exp.exp_name}_{phase}",
                cmd=train_cmd,
                #log_dir=args.log_dir,
                log_dir=phase_out,
                output_dir=phase_out,
                command_file="train_command.txt",
                env_updates={"CUDA_VISIBLE_DEVICES": args.visible_devices},
            )

            ckpt = latest_checkpoint(phase_out)
            if ckpt is None:
                raise RuntimeError(f"Training completed but no checkpoint was found in {phase_out}")

            mark_phase_done(phase_out)
            current_model = str(ckpt)

            update_progress(
                progress_path,
                progress_key,
                f"phase_{phase}",
                {
                    "status": "done",
                    "checkpoint": str(ckpt),
                    "finished_at": now(),
                },
            )

            log(f"Using checkpoint for next phase: {ckpt}")

        completed_phases.append(phase)

        # Space optimization step 1:
        # Run checkpoint evals as soon as each phase finishes.
        #
        # The evaluator uses the experiment-root interface with --skip_existing,
        # so completed checkpoint evaluations are not duplicated.
        if args.run_checkpoint_eval:
            run_checkpoint_eval_after_phase(
                exp=exp,
                args=args,
                exp_root=exp_root,
                phase_out=phase_out,
                phase=phase,
                phase_sequence=phase_sequence,
            )

        if args.run_response_length_analysis:
            run_response_length_analysis(
                exp=exp,
                args=args,
                exp_root=exp_root,
                phase_sequence=phase_sequence,
            )

        if args.run_training_log_stats_analysis:
            run_training_log_stats_analysis(
                exp=exp,
                args=args,
                exp_root=exp_root,
                phase_sequence=phase_sequence,
            )

        # Space optimization step 2:
        # Clean after each phase, but only if --run_cleanup is enabled.
        #
        # Latest checkpoints of completed phases are protected because:
        #   - the next phase needs the previous phase's latest checkpoint;
        #   - resume needs completed phases to keep at least one checkpoint.
        if args.run_cleanup:
            run_interim_cleanup_after_phase(
                exp=exp,
                args=args,
                exp_root=exp_root,
                phase_out=phase_out,
                phase=phase,
                completed_phases=completed_phases,
            )

            refreshed_ckpt = latest_checkpoint(phase_out)
            if refreshed_ckpt is None:
                raise RuntimeError(
                    f"No latest checkpoint found for phase {phase} after interim cleanup."
                )
            current_model = str(refreshed_ckpt)

    if args.run_checkpoint_eval:
        if all_phase_stages_done(exp_root, phase_sequence, ".checkpoint_eval_done"):
            mark_stage_done(exp_root, ".checkpoint_eval_done")

    # lm-eval is final-only: one call on the latest checkpoint of the final phase.
    if args.run_lm_eval:
        run_final_lm_eval(
            exp=exp,
            args=args,
            exp_root=exp_root,
            phase_sequence=phase_sequence,
        )

    # Final cleanup without keep-last flags. This intentionally mirrors your
    # original final cleanup call and removes the latest checkpoints that were
    # temporarily preserved by interim cleanup.
    if args.run_cleanup:
        run_final_cleanup(
            exp=exp,
            args=args,
            exp_root=exp_root,
        )

    update_progress_fields(
        progress_path,
        progress_key,
        {
            "status": "done",
            "finished_at": now(),
        },
    )

    log(f"Experiment completed successfully: {exp.exp_name}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run sequential SDFT experiments from a CSV file."
    )

    parser.add_argument("--base_dir", type=Path, default=None)
    parser.add_argument("--experiments_csv", type=Path, default=None)

    parser.add_argument(
        "--output_root",
        type=Path,
        default=None,
        help=(
            "Optional output root directory. "
            "If omitted, defaults to <base_dir>/outputs."
        ),
    )

    parser.add_argument(
        "--reverse_experiments",
        action="store_true",
        help=(
            "Iterate enabled experiments from the CSV in reverse order. "
            "Useful when launching two workers on the same CSV, one from each end."
        ),
    )

    parser.add_argument("--visible_devices", type=str, default=os.environ.get("VISIBLE_DEVICES", "0"))
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.6,
        help=(
            "Fraction of GPU memory vLLM may use for checkpoint evaluation "
            "and lm_eval. Default: 0.6."
        ),
    )
    parser.add_argument("--nproc_per_node", type=int, default=int(os.environ.get("NPROC_PER_NODE", "1")))
    parser.add_argument("--master_port", type=int, default=29607)

    parser.add_argument(
        "--vllm_tensor_parallel_size",
        type=int,
        default=int(os.environ.get("VLLM_TENSOR_PARALLEL_SIZE", "1")),
        help=(
            "Tensor-parallel size for the colocated vLLM generation engine used "
            "during training (forwarded to main.py's --vllm_tensor_parallel_size). "
            "Must evenly divide --nproc_per_node. Default 1 preserves the "
            "single-process default."
        ),
    )

    parser.add_argument("--seed", type=int, default=int(os.environ.get("SEED", "42")))
    parser.add_argument(
        "--initial_model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help=(
            "Base model for experiments whose CSV row doesn't set the optional "
            "'model' column. Accepts a registered short key (see "
            "model_registry.known_model_keys(), e.g. 'qwen3.5-4b', "
            "'ministral-3-3b'), a Hugging Face repo id, or a local checkpoint "
            "directory."
        ),
    )

    parser.add_argument(
        "--phase_sequence",
        nargs="+",
        default=DEFAULT_PHASE_SEQUENCE,
        choices=ALLOWED_PHASES,
    )

    parser.add_argument("--num_train_epochs", type=int, default=DEFAULT_COMMON_ARGS["num_train_epochs"])
    parser.add_argument("--num_prompts_per_batch", type=int, default=DEFAULT_COMMON_ARGS["num_prompts_per_batch"])
    parser.add_argument("--optim", type=str, default=DEFAULT_COMMON_ARGS["optim"])
    parser.add_argument("--save_strategy", type=str, default=DEFAULT_COMMON_ARGS["save_strategy"])
    parser.add_argument("--save_steps", type=int, default=DEFAULT_COMMON_ARGS["save_steps"])

    parser.add_argument(
        "--run_checkpoint_eval",
        type=str_to_bool,
        default=True,
        help="Run run_all_checkpoint_evals.py after training.",
    )
    parser.add_argument(
        "--run_lm_eval",
        type=str_to_bool,
        default=str_to_bool(os.environ.get("RUN_LM_EVAL", "1")),
    )
    parser.add_argument(
        "--run_cleanup",
        type=str_to_bool,
        default=str_to_bool(os.environ.get("RUN_CLEANUP", "1")),
    )

    parser.add_argument(
        "--lm_eval_tasks",
        type=str,
        default="hellaswag,mmlu,truthfulqa,winogrande,humaneval,ifeval",
    )

    parser.add_argument(
        "--run_response_length_analysis",
        type=str_to_bool,
        default=str_to_bool(os.environ.get("RUN_RESPONSE_LENGTH_ANALYSIS", "1")),
        help="Run response-length analysis after checkpoint evals.",
    )

    parser.add_argument(
        "--response_length_script",
        type=Path,
        default=None,
        help=(
            "Path to analyze_response_lengths.py. Defaults to "
            "<base_dir>/metrics_and_plots/analyze_response_lengths.py."
        ),
    )

    parser.add_argument(
        "--response_length_mode",
        type=str,
        default=os.environ.get("RESPONSE_LENGTH_MODE", "hf_tokens_if_available"),
        choices=["chars", "whitespace", "hf_tokens", "hf_tokens_if_available"],
        help="Length unit used for the main eval-response plot.",
    )

    parser.add_argument(
        "--response_length_tokenizer",
        type=str,
        default=os.environ.get("RESPONSE_LENGTH_TOKENIZER", ""),
        help="Optional tokenizer name/path for response token counts.",
    )

    parser.add_argument(
        "--run_training_log_stats_analysis",
        type=str_to_bool,
        default=str_to_bool(os.environ.get("RUN_TRAINING_LOG_STATS_ANALYSIS", "1")),
        help="Run training-log statistics analysis after checkpoint evals.",
    )

    parser.add_argument(
        "--continue_on_error",
        action="store_true",
        help="Continue with the next experiment if one experiment fails.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip phases/stages already marked as done.",
    )
    parser.add_argument(
        "--resume_partial_phase",
        action="store_true",
        help=(
            "If a phase failed but has checkpoints, pass the latest checkpoint to "
            "--resume_from_checkpoint. Requires main.py to support this argument."
        ),
    )

    parser.add_argument(
        "--max_retries",
        type=int,
        default=0,
        help=(
            "Number of times to retry a failed experiment immediately. "
            "Retries automatically enable resume and resume_partial_phase."
        ),
    )

    parser.add_argument(
        "--no_skip_done",
        dest="skip_done",
        action="store_false",
        help="Do not skip experiments already marked as done in progress JSON.",
    )
    parser.set_defaults(skip_done=True)

    args = parser.parse_args()

    if not 0 < args.gpu_memory_utilization <= 1:
        parser.error(
            "--gpu_memory_utilization must be in the interval (0, 1]."
        )

    if args.base_dir is None:
        args.base_dir = PROJECT_ROOT

    args.base_dir = args.base_dir.resolve()
    if args.output_root is None:
        args.output_root = args.base_dir / "outputs"
    else:
        args.output_root = args.output_root.resolve()

    scripts_dir = args.base_dir / "scripts"
    analysis_dir = args.base_dir / "metrics_and_plots"

    args.log_dir = args.base_dir / "logs"
    args.cleanup_script = scripts_dir / "cleanup_checkpoints.sh"
    args.lmeval_script = scripts_dir / "lmeval.sh"

    if args.experiments_csv is None:
        args.experiments_csv = args.base_dir / "experiments.csv"

    args.progress_path = args.log_dir / "experiment_progress.json"
    args.failures_csv = args.log_dir / "experiment_failures.csv"
    args.history_csv = args.log_dir / "experiment_history.csv"

    if args.response_length_script is None:
        args.response_length_script = analysis_dir / "analyze_response_lengths.py"
    else:
        args.response_length_script = args.response_length_script.resolve()

    if args.response_length_tokenizer is not None and not str(args.response_length_tokenizer).strip():
        args.response_length_tokenizer = None

    args.training_log_stats_script = analysis_dir / "analyze_training_log_stats.py"

    return args


def main():
    args = parse_args()

    if args.max_retries > 0:
        args.resume = True
        args.resume_partial_phase = True

    if args.nproc_per_node % args.vllm_tensor_parallel_size != 0:
        raise ValueError(
            f"--vllm_tensor_parallel_size ({args.vllm_tensor_parallel_size}) must evenly "
            f"divide --nproc_per_node ({args.nproc_per_node}). main.py enforces this again "
            "at training time, but checking here avoids launching torchrun at all."
        )

    os.chdir(args.base_dir)

    args.output_root.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)

    os.environ["WANDB_MODE"] = "offline"
    os.environ["HF_ALLOW_CODE_EVAL"] = "1"

    check_path(args.base_dir, "base dir")
    check_path(args.experiments_csv, "experiments CSV")
    check_path(Path("main.py"), "main.py") 
    check_path(Path("run_all_checkpoint_evals.py"), "run_all_checkpoint_evals.py")

    if args.run_cleanup:
        check_path(args.cleanup_script, "cleanup script")

    if args.run_lm_eval:
        check_path(args.lmeval_script, "lmeval script")

    experiments = read_experiments_csv(args.experiments_csv)

    if args.reverse_experiments:
        experiments = list(reversed(experiments))

    if not experiments:
        raise RuntimeError(f"No enabled experiments found in {args.experiments_csv}")

    common_args = {
        "num_train_epochs": args.num_train_epochs,
        "num_prompts_per_batch": args.num_prompts_per_batch,
        "optim": args.optim,
        "save_strategy": args.save_strategy,
        "save_steps": args.save_steps,
    }

    log(f"Loaded {len(experiments)} enabled experiments from {args.experiments_csv}")
    log(f"Experiment iteration order: {'reverse' if args.reverse_experiments else 'forward'}")
    log(f"Base dir: {args.base_dir}")
    log(f"Output root: {args.output_root}")
    log(f"Log dir: {args.log_dir}")
    log(f"Phase sequence: {' '.join(args.phase_sequence)}")

    num_failed = 0

    for exp in experiments:
        attempt = 0

        final_name = experiment_run_name(
            exp=exp,
            fallback_seed=args.seed,
            fallback_phase_sequence=args.phase_sequence,
        )
        initial_model = effective_initial_model(exp, args)
        model_identity = canonical_model_identity(initial_model)
        exp_root = args.output_root / final_name
        exp_key = progress_experiment_key(
            run_name=final_name,
            model_identity=model_identity,
            output_root=args.output_root,
        )

        while True:
            try:
                ensure_experiment_root_compatible(
                    exp_root=exp_root,
                    expected_model_identity=model_identity,
                )

                if args.skip_done and is_experiment_done(
                    args.progress_path,
                    exp_key,
                    expected_model_identity=model_identity,
                    legacy_exp_key=final_name,
                    exp_root=exp_root,
                ):
                    update_experiment_csv_status(
                        args.experiments_csv,
                        exp.csv_record_index,
                        "done",
                    )
                    log(
                        "Skipping already completed experiment: "
                        f"{final_name} (model={model_identity})"
                    )
                    break

                update_experiment_csv_status(
                    args.experiments_csv,
                    exp.csv_record_index,
                    "running",
                )

                run_experiment(
                    exp=exp,
                    args=args,
                    phase_sequence=args.phase_sequence,
                    common_args=common_args,
                    progress_path=args.progress_path,
                    failures_csv=args.failures_csv,
                    progress_key=exp_key,
                    initial_model=initial_model,
                    model_identity=model_identity,
                )

                seed = exp.seed if exp.seed is not None else args.seed
                phase_sequence = exp.phase_sequence if exp.phase_sequence is not None else args.phase_sequence

                append_history(
                    history_csv=args.history_csv,
                    exp_name=exp.exp_name,
                    run_name=final_name,
                    status="done",
                    seed=seed,
                    phase_sequence=phase_sequence,
                    output_dir=args.output_root / final_name,
                    notes=exp.notes,
                )
                update_experiment_csv_status(
                    args.experiments_csv,
                    exp.csv_record_index,
                    "done",
                )
                break

            except subprocess.CalledProcessError as e:
                attempt += 1

                append_failure(
                    failures_csv=args.failures_csv,
                    exp_name=exp.exp_name,
                    stage=f"subprocess_attempt_{attempt}",
                    command=list(map(str, e.cmd)),
                    returncode=e.returncode,
                    error=str(e),
                )

                if attempt <= args.max_retries:
                    log(
                        f"Retrying experiment {exp.exp_name} "
                        f"from latest available progress "
                        f"({attempt}/{args.max_retries})"
                    )
                    continue

                num_failed += 1

                update_progress(
                    args.progress_path,
                    exp_key,
                    "status",
                    "failed",
                )

                update_experiment_csv_status(
                    args.experiments_csv,
                    exp.csv_record_index,
                    "failed",
                )

                if not args.continue_on_error:
                    raise

                log(f"Continuing after failed experiment: {exp.exp_name}")
                break

            except Exception as e:
                attempt += 1

                append_failure(
                    failures_csv=args.failures_csv,
                    exp_name=exp.exp_name,
                    stage=f"orchestrator_attempt_{attempt}",
                    command=[],
                    returncode=None,
                    error=repr(e),
                )

                if attempt <= args.max_retries:
                    log(
                        f"Retrying experiment {exp.exp_name} "
                        f"from latest available progress "
                        f"({attempt}/{args.max_retries})"
                    )
                    continue

                num_failed += 1

                update_progress(
                    args.progress_path,
                    exp_key,
                    "status",
                    "failed",
                )

                update_experiment_csv_status(
                    args.experiments_csv,
                    exp.csv_record_index,
                    "failed",
                )

                seed = exp.seed if exp.seed is not None else args.seed
                phase_sequence = exp.phase_sequence if exp.phase_sequence is not None else args.phase_sequence

                append_history(
                    history_csv=args.history_csv,
                    exp_name=exp.exp_name,
                    run_name=final_name,
                    status="failed",
                    seed=seed,
                    phase_sequence=phase_sequence,
                    output_dir=args.output_root / final_name,
                    notes=exp.notes,
                )

                if not args.continue_on_error:
                    raise

                log(f"Continuing after failed experiment: {exp.exp_name}")
                break

    if num_failed:
        log(f"ALL DONE, but {num_failed} experiment(s) failed. See {args.failures_csv}")
    else:
        log("ALL JOBS COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
