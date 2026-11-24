#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$ROOT_DIR/.env" ]]; then
  override_names=(SYSTEM_ROOT RUNPOD_ROOT MUSUBI_DIR VENV_DIR HF_HOME HF_TOKEN MUSUBI_COMMIT TORCH_INDEX_URL)
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
MUSUBI_DIR="${MUSUBI_DIR:-$SYSTEM_ROOT/tools/musubi-tuner}"
VENV_DIR="${VENV_DIR:-$SYSTEM_ROOT/venvs/wan22-babaji}"
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
    echo "Missing commands: ${missing_commands[*]}. Install them with sudo, then rerun this script." >&2
    exit 1
  fi
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y git ffmpeg python3 python3-pip python3-venv
fi

mkdir -p "$(dirname "$MUSUBI_DIR")" "$(dirname "$VENV_DIR")" "$SYSTEM_ROOT/.cache"

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

# Blackwell (RTX 50 series) first receives toolkit support in CUDA 12.8. An
# older torch/CUDA wheel may import but cannot execute kernels on an RTX 5090.
if ! python - <<'PY'
try:
    import torch
    torch_version = tuple(int(part) for part in torch.__version__.split("+", 1)[0].split(".")[:2])
    cuda_version = tuple(int(part) for part in (torch.version.cuda or "0.0").split(".")[:2])
except Exception:
    raise SystemExit(1)
raise SystemExit(0 if torch_version >= (2, 7) and cuda_version >= (12, 8) else 1)
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
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available. Check the NVIDIA driver and GPU access.")
props = torch.cuda.get_device_properties(0)
capability = torch.cuda.get_device_capability(0)
print(f"GPU: {props.name}; VRAM: {props.total_memory / 1024**3:.1f} GiB; capability: {capability}")
if capability[0] >= 12 and tuple(int(p) for p in (torch.version.cuda or "0.0").split(".")[:2]) < (12, 8):
    raise SystemExit("Blackwell requires a CUDA 12.8+ PyTorch build.")
x = torch.randn((1024, 1024), device="cuda", dtype=torch.float16)
_ = x @ x
torch.cuda.synchronize()
print("CUDA matrix-multiplication smoke test: passed")
PY

echo "Setup complete. Next: bash $ROOT_DIR/run_inference.sh inspect"
