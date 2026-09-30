#!/usr/bin/env bash
#
# Remove disposable files from an experiment while preserving report/evaluation
# outputs and, when requested, the newest checkpoint of selected phases.
#
# Recognized Hugging Face/Trainer metadata is removed only when it is a direct
# child of a numeric checkpoint directory (checkpoint-N). This allows normally
# preserved files such as tokenizer.json and vocab.txt to be removed without
# matching arbitrary JSON/TXT files elsewhere.
#
# Usage:
#   cleanup_checkpoints.sh <experiment_dir> [options]
#
# Options:
#   --dry-run                 Print commands without changing the filesystem.
#   --keep-last-<phase>       Protect the newest numeric checkpoint directly
#                             under <experiment_dir>/<phase>. May be repeated.
#
# A hyphenated phase flag can fall back to an underscore-named directory:
#   --keep-last-math-contradiction -> <experiment_dir>/math_contradiction
#
# Exit status:
#   0  Cleanup completed, or dry-run completed.
#   1  Invalid input, an unhonored protection request, or command failure.

set -Eeuo pipefail

readonly SCRIPT_NAME="${0##*/}"

usage() {
  cat <<EOF
Usage: $SCRIPT_NAME <experiment_dir> [--dry-run] [--keep-last-<phase>]...
EOF
}

die() {
  printf 'Error: %s\n' "$*" >&2
  exit 1
}

if (( $# < 1 )); then
  usage >&2
  exit 1
fi

LOCAL_FOLDER="$1"
shift

[[ -d "$LOCAL_FOLDER" ]] ||
  die "local folder does not exist or is not a directory: $LOCAL_FOLDER"

LOCAL_FOLDER="$(readlink -f -- "$LOCAL_FOLDER")"
[[ -n "$LOCAL_FOLDER" ]] || die "could not resolve the local folder"
[[ "$LOCAL_FOLDER" != "/" ]] || die "refusing to clean the filesystem root"
readonly LOCAL_FOLDER

DRY_RUN=false
declare -a KEEP_LAST_PHASES=()
declare -A SEEN_KEEP_LAST_PHASES=()

for arg in "$@"; do
  case "$arg" in
    --dry-run)
      DRY_RUN=true
      ;;
    --keep-last-*)
      phase="${arg#--keep-last-}"

      [[ "$phase" =~ ^[[:alnum:]][[:alnum:]_.-]*$ ]] ||
        die "invalid phase in argument: $arg"

      # Repeated flags are harmless, but storing each phase once keeps the
      # generated find expression and status output concise.
      if [[ -z "${SEEN_KEEP_LAST_PHASES[$phase]+present}" ]]; then
        KEEP_LAST_PHASES+=("$phase")
        SEEN_KEEP_LAST_PHASES["$phase"]=true
      fi
      ;;
    *)
      printf 'Error: unknown argument: %s\n' "$arg" >&2
      usage >&2
      exit 1
      ;;
  esac
done

run_cmd() {
  if [[ "$DRY_RUN" == true ]]; then
    printf '[dry-run]'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

latest_checkpoint() {
  local phase_dir="$1"
  local latest=""

  [[ -d "$phase_dir" ]] || return 1

  latest="$(
    find "$phase_dir" -maxdepth 1 -regextype posix-extended \
      -type d -regex '.*/checkpoint-[0-9]+' \
      | LC_ALL=C sort -V \
      | tail -n 1
  )" || return 1

  [[ -n "$latest" ]] || return 1
  printf '%s\n' "$latest"
}

resolve_phase_dir() {
  local phase="$1"
  local exact_dir="$LOCAL_FOLDER/$phase"
  local underscore_phase
  local underscore_dir

  # Prefer the exact phase notation produced by run_experiments.py.
  if [[ -d "$exact_dir" ]]; then
    printf '%s\n' "$exact_dir"
    return 0
  fi

  # Manual convenience: --keep-last-math-contradiction can resolve
  # math_contradiction when
  # the exact hyphenated directory does not exist.
  underscore_phase="${phase//-/_}"
  underscore_dir="$LOCAL_FOLDER/$underscore_phase"

  if [[ "$underscore_phase" != "$phase" && -d "$underscore_dir" ]]; then
    printf '%s\n' "$underscore_dir"
    return 0
  fi

  return 1
}

declare -a PROTECTED_DIRS=()

for phase in "${KEEP_LAST_PHASES[@]}"; do
  if ! phase_dir="$(resolve_phase_dir "$phase")"; then
    die "cannot honor --keep-last-${phase}; phase directory not found: $LOCAL_FOLDER/$phase"
  fi

  if ! latest="$(latest_checkpoint "$phase_dir")"; then
    die "cannot honor --keep-last-${phase}; no numeric checkpoint found under $phase_dir"
  fi

  PROTECTED_DIRS+=("$latest")
  printf 'Protecting latest %s checkpoint: %s\n' "$phase" "$latest"
done

# Build a find prefix that prunes every protected checkpoint at its directory
# entry. Pruning the directory itself also excludes all descendants.
declare -a FIND_PRUNE_ARGS=()
if (( ${#PROTECTED_DIRS[@]} > 0 )); then
  FIND_PRUNE_ARGS+=( "(" )
  for protected in "${PROTECTED_DIRS[@]}"; do
    FIND_PRUNE_ARGS+=( -path "$protected" -o )
  done
  unset 'FIND_PRUNE_ARGS[${#FIND_PRUNE_ARGS[@]}-1]'
  FIND_PRUNE_ARGS+=( ")" -prune -o )
fi

# Exact direct-child artifacts produced by Hugging Face Transformers,
# Tokenizers, PEFT, Trainer, or common processors. Evaluation outputs and
# arbitrary user metadata are deliberately absent.
readonly -a CHECKPOINT_ARTIFACT_NAMES=(
  "added_tokens.json"
  "adapter_config.json"
  "adapter_model.bin.index.json"
  "adapter_model.safetensors.index.json"
  "audio_processor_config.json"
  "chat_template.jinja"
  "chat_template.json"
  "config.json"
  "feature_extractor_config.json"
  "generation_config.json"
  "image_processor_config.json"
  "merges.txt"
  "model.safetensors.index.json"
  "optimizer.pt"
  "preprocessor_config.json"
  "processor_config.json"
  "pytorch_model.bin.index.json"
  "rng_state.pth"
  "scaler.pt"
  "scheduler.pt"
  "sentencepiece.bpe.model"
  "special_tokens_map.json"
  "spiece.model"
  "tokenizer.json"
  "tokenizer.model"
  "tokenizer_config.json"
  "trainer_state.json"
  "training_args.bin"
  "video_processor_config.json"
  "video_preprocessor_config.json"
  "vocab.json"
  "vocab.txt"
  "zero_to_fp32.py"
  "tekken.json"
)

# Rank-specific Trainer state files. These patterns are also constrained to a
# direct child of checkpoint-N by the enclosing find expression.
readonly -a CHECKPOINT_ARTIFACT_PATTERNS=(
  "optimizer_*.pt"
  "rng_state_*.pth"
  "scaler_*.pt"
  "scheduler_*.pt"
)

declare -a CHECKPOINT_ARTIFACT_FIND_ARGS=( "(" )

for name in "${CHECKPOINT_ARTIFACT_NAMES[@]}"; do
  CHECKPOINT_ARTIFACT_FIND_ARGS+=( -name "$name" -o )
done

for pattern in "${CHECKPOINT_ARTIFACT_PATTERNS[@]}"; do
  CHECKPOINT_ARTIFACT_FIND_ARGS+=( -name "$pattern" -o )
done

# Remove the final, otherwise dangling, -o.
unset 'CHECKPOINT_ARTIFACT_FIND_ARGS[${#CHECKPOINT_ARTIFACT_FIND_ARGS[@]}-1]'
CHECKPOINT_ARTIFACT_FIND_ARGS+=( ")" )

if [[ "$DRY_RUN" == true ]]; then
  echo "Dry run: showing disposable files and recognized checkpoint artifacts ..."
else
  echo "Removing disposable files and recognized checkpoint artifacts ..."
fi

# Preserve the original general cleanup policy. Additionally delete allowlisted
# metadata even when its extension (.json or .txt) is normally preserved, but
# only when it is directly inside a numeric checkpoint directory.
find "$LOCAL_FOLDER" -regextype posix-extended \
  "${FIND_PRUNE_ARGS[@]}" \
  -type f \( \
    \( \
      -regex '.*/checkpoint-[0-9]+/[^/]+' \
      "${CHECKPOINT_ARTIFACT_FIND_ARGS[@]}" \
    \) -o \
    ! \( \
      -name "*.txt" -o \
      -name "*.json" -o \
      -name "*.jsonl" -o \
      -name "*.csv" -o \
      -name "*.png" -o \
      -name "*.pdf" -o \
      -name "*.log" -o \
      -name ".*_done" -o \
      -name ".train_done" \
    \) \
  \) -print0 |
  while IFS= read -r -d '' file; do
    run_cmd rm -f -- "$file"
  done

if [[ "$DRY_RUN" == true ]]; then
  echo "Dry run: showing currently empty directories ..."
else
  echo "Cleaning up empty directories ..."
fi

find "$LOCAL_FOLDER" \
  "${FIND_PRUNE_ARGS[@]}" \
  -type d -empty -print0 |
  LC_ALL=C sort -z -r |
  while IFS= read -r -d '' dir; do
    if [[ "$dir" != "$LOCAL_FOLDER" ]]; then
      run_cmd rmdir -- "$dir" 2>/dev/null || true
    fi
  done

if [[ "$DRY_RUN" == true ]]; then
  echo "Dry run complete. No files or directories were changed."
else
  echo "Cleanup complete."
fi
