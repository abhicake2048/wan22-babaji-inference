#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -f "$ROOT_DIR/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT_DIR/.env"
  set +a
fi

SYSTEM_ROOT="${SYSTEM_ROOT:-${RUNPOD_ROOT:-/workspace}}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$SYSTEM_ROOT/nagharsubject_high_checkpoints/high}"
HASH_MANIFEST="${HASH_MANIFEST:-$CHECKPOINT_DIR/checkpoints.sha256}"

checkpoint_step=200
image_path=""
prompt=""
save_to=""
profile="${MEMORY_PROFILE:-rtx5090_quality}"
seed="${SEED:-42}"
lora_strength="${LORA_STRENGTH:-1.0}"
height=576
width=768
frames=81
fps=16
infer_steps=30
guidance=5.0
dry_run=0

usage() {
  cat <<'EOF'
Usage:
  bash start_inference.sh \
    --checkpoint-step 400 \
    --image /mnt/d/babaji_inputs/my_image.jpg \
    --save-to /mnt/d/lip_sync/generated \
    --prompt "nagharsubject, your prompt"

Required:
  --image PATH             First-frame image (.png, .jpg, .jpeg, .webp, .bmp)
  --save-to DIRECTORY      Parent directory for timestamped video runs
  --prompt TEXT            Generation prompt; trigger token is added if omitted

Optional:
  --checkpoint-step N      200, 400, or 600 (default: 200)
  --profile NAME           rtx5090_quality, rtx5090_fast, or rtx5090_safe
  --seed N                 Default: 42
  --lora-strength NUMBER   Default: 1.0
  --height N               Default: 576
  --width N                Default: 768
  --frames N               Default: 81
  --fps N                  Default: 16
  --steps N                Default: 30
  --guidance NUMBER        Default: 5.0
  --dry-run                Validate and print the command without generating
  -h, --help               Show this help
EOF
}

while (( $# > 0 )); do
  case "$1" in
    --checkpoint-step) checkpoint_step="${2:?Missing value for --checkpoint-step}"; shift 2 ;;
    --image) image_path="${2:?Missing value for --image}"; shift 2 ;;
    --save-to) save_to="${2:?Missing value for --save-to}"; shift 2 ;;
    --prompt) prompt="${2:?Missing value for --prompt}"; shift 2 ;;
    --profile) profile="${2:?Missing value for --profile}"; shift 2 ;;
    --seed) seed="${2:?Missing value for --seed}"; shift 2 ;;
    --lora-strength) lora_strength="${2:?Missing value for --lora-strength}"; shift 2 ;;
    --height) height="${2:?Missing value for --height}"; shift 2 ;;
    --width) width="${2:?Missing value for --width}"; shift 2 ;;
    --frames) frames="${2:?Missing value for --frames}"; shift 2 ;;
    --fps) fps="${2:?Missing value for --fps}"; shift 2 ;;
    --steps) infer_steps="${2:?Missing value for --steps}"; shift 2 ;;
    --guidance) guidance="${2:?Missing value for --guidance}"; shift 2 ;;
    --dry-run) dry_run=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$image_path" || -z "$save_to" || -z "$prompt" ]]; then
  echo "--image, --save-to, and --prompt are required." >&2
  usage >&2
  exit 2
fi

case "$checkpoint_step" in
  200|400|600) ;;
  *) echo "--checkpoint-step must be 200, 400, or 600; received: $checkpoint_step" >&2; exit 2 ;;
esac

case "$profile" in
  rtx5090_quality|rtx5090_fast|rtx5090_safe) ;;
  *) echo "Unsupported RTX 5090 profile: $profile" >&2; exit 2 ;;
esac

if [[ "$image_path" =~ ^[A-Za-z]:\\ ]]; then
  echo "Use a Linux/WSL path, not a Windows path. Example: D:\\images\\photo.jpg -> /mnt/d/images/photo.jpg" >&2
  exit 2
fi

image_path="$(realpath "$image_path")"
if [[ ! -f "$image_path" ]]; then
  echo "Input image not found: $image_path" >&2
  exit 1
fi

case "${image_path,,}" in
  *.png|*.jpg|*.jpeg|*.webp|*.bmp) ;;
  *) echo "Unsupported image extension: $image_path" >&2; exit 2 ;;
esac

printf -v padded_step '%08d' "$checkpoint_step"
checkpoint="$CHECKPOINT_DIR/nagharsubject_high-step${padded_step}.safetensors"
if [[ ! -f "$checkpoint" ]]; then
  echo "Checkpoint not found: $checkpoint" >&2
  exit 1
fi

expected_hash=""
if [[ -f "$HASH_MANIFEST" ]]; then
  expected_hash="$(awk -v filename="$(basename "$checkpoint")" '$2 == filename {print $1}' "$HASH_MANIFEST")"
fi
if [[ -z "$expected_hash" ]]; then
  echo "WARNING: no source hash found for $(basename "$checkpoint"); using the current file's SHA-256." >&2
  expected_hash="$(sha256sum "$checkpoint" | awk '{print $1}')"
fi

mkdir -p "$save_to"
save_to="$(realpath "$save_to")"
if [[ ! -w "$save_to" ]]; then
  echo "Output directory is not writable: $save_to" >&2
  exit 1
fi

echo "Checkpoint: $checkpoint"
echo "Checkpoint SHA-256: $expected_hash"
echo "Reference image: $image_path"
echo "Output root: $save_to"
echo "Profile: $profile"

command=(
  bash "$ROOT_DIR/run_inference.sh" generate
  --reference "$image_path"
  --output-root "$save_to"
  --prompt "$prompt"
  --seed "$seed"
  --lora-strength "$lora_strength"
  --height "$height"
  --width "$width"
  --frames "$frames"
  --fps "$fps"
  --steps "$infer_steps"
  --guidance "$guidance"
)
if (( dry_run )); then
  command+=(--dry-run)
fi

LORA_PATH="$checkpoint" \
EXPECTED_LORA_SHA256="$expected_hash" \
REFERENCE_IMAGE="$image_path" \
OUTPUT_ROOT="$save_to" \
MEMORY_PROFILE="$profile" \
exec "${command[@]}"
