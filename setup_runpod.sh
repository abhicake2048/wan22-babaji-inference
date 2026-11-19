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
MUSUBI_DIR="${MUSUBI_DIR:-$RUNPOD_ROOT/tools/musubi-tuner}"
VENV_DIR="${VENV_DIR:-$RUNPOD_ROOT/venvs/wan22-babaji}"
MUSUBI_COMMIT="${MUSUBI_COMMIT:-f8a1b03794a49239a3539015075f5123d6c07d66}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"

missing_commands=()
for command_name in git ffmpeg ffprobe python3; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    missing_commands+=("$command_name")
  fi
done

if (( ${#missing_commands[@]} > 0 )); then
  if [[ "$(id -u)" != "0" ]]; then
    echo "Missing commands: ${missing_commands[*]}. Re-run as root on the RunPod template." >&2
    exit 1
  fi
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y git ffmpeg python3 python3-pip python3-venv
fi

mkdir -p "$(dirname "$MUSUBI_DIR")" "$(dirname "$VENV_DIR")" "$RUNPOD_ROOT/.cache"

if [[ ! -d "$MUSUBI_DIR/.git" ]]; then
  git clone https://github.com/kohya-ss/musubi-tuner.git "$MUSUBI_DIR"
fi
git -C "$MUSUBI_DIR" fetch --depth 1 origin "$MUSUBI_COMMIT"
git -C "$MUSUBI_DIR" checkout --detach "$MUSUBI_COMMIT"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  python3 -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip setuptools wheel

if ! python - <<'PY'
try:
    import torch
except Exception:
    raise SystemExit(1)
parts = tuple(int(part) for part in torch.__version__.split("+", 1)[0].split(".")[:2])
raise SystemExit(0 if parts >= (2, 6) else 1)
PY
then
  python -m pip install --upgrade torch torchvision --index-url "$TORCH_INDEX_URL"
fi

python -m pip install -e "$MUSUBI_DIR"
python -m pip install -r "$ROOT_DIR/requirements.txt"

python - <<'PY'
import sys
import torch
print(f"Python: {sys.version.split()[0]}")
print(f"PyTorch: {torch.__version__}; CUDA runtime: {torch.version.cuda}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name}; VRAM: {props.total_memory / 1024**3:.1f} GiB")
PY

echo "Setup complete. Next: bash $ROOT_DIR/run_inference.sh inspect"

