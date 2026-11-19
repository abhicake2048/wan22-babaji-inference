#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$ROOT_DIR/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT_DIR/.env"
  set +a
fi

RUNPOD_ROOT="${RUNPOD_ROOT:-/workspace}"
VENV_DIR="${VENV_DIR:-$RUNPOD_ROOT/venvs/wan22-babaji}"
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  echo "Python environment is missing at $VENV_DIR. Run setup_runpod.sh first." >&2
  exit 1
fi

exec "$VENV_DIR/bin/python" "$ROOT_DIR/inference.py" "$@"

