#!/usr/bin/env bash
set -Eeuo pipefail

# Thin compatibility wrapper for run_experiments.py.
#
# Historical interface is preserved:
#   ./scripts/lmeval.sh /path/to/checkpoint
#
# Optional direct smoke test:
#   ./scripts/lmeval.sh /path/to/checkpoint --dry-run
#
# The wrapper deliberately does NOT activate a virtualenv. Instead it invokes
# the repository's Python explicitly, preventing accidental fallback to an
# unrelated environment elsewhere on the machine.

SCRIPT_DIR="$(
  cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1
  pwd
)"
REPO_ROOT="$(
  cd -- "$SCRIPT_DIR/.." >/dev/null 2>&1
  pwd
)"

PYTHON_BIN="${SDFT_PYTHON:-$REPO_ROOT/distillation-vlm/bin/python}"
RUNNER="$SCRIPT_DIR/run_lmeval.py"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: Python interpreter is missing or not executable: $PYTHON_BIN" >&2
  echo "Set SDFT_PYTHON=/path/to/python to override it intentionally." >&2
  exit 1
fi

if [[ ! -f "$RUNNER" ]]; then
  echo "ERROR: Missing lm-eval runner: $RUNNER" >&2
  exit 1
fi

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 /path/to/checkpoint [--dry-run]" >&2
  exit 2
fi

exec "$PYTHON_BIN" "$RUNNER" "$@"
