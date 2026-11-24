#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$ROOT_DIR/.env" ]]; then
  override_names=(SYSTEM_ROOT RUNPOD_ROOT OUTPUT_ROOT VENV_DIR LORA_PATH EXPECTED_LORA_SHA256 REFERENCE_ROOT REFERENCE_IMAGE MEMORY_PROFILE PYTORCH_CUDA_ALLOC_CONF)
  declare -A caller_overrides=()
  for name in "${override_names[@]}"; do
    if [[ -v "$name" ]]; then
      caller_overrides["$name"]="${!name}"
    fi
  done
  set -a
  # shellcheck disable=SC1091
  source "$ROOT_DIR/.env"
  set +a
  for name in "${!caller_overrides[@]}"; do
    export "$name=${caller_overrides[$name]}"
  done
fi

SYSTEM_ROOT="${SYSTEM_ROOT:-${RUNPOD_ROOT:-/workspace}}"
VENV_DIR="${VENV_DIR:-$SYSTEM_ROOT/venvs/wan22-babaji}"
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  echo "Python environment is missing at $VENV_DIR. Run setup_linux.sh first." >&2
  exit 1
fi

exec "$VENV_DIR/bin/python" "$ROOT_DIR/compare_checkpoints.py" "$@"
