#!/usr/bin/env python3
"""Reproducible high-noise-LoRA inference for Babaji Wan 2.2 I2V snippets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import struct
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


MUSUBI_COMMIT_DEFAULT = "f8a1b03794a49239a3539015075f5123d6c07d66"
LORA_SHA256 = "a05589023b0128de81a599e70632bec1a2e5cc7c172718182d25602f6e8311b4"
LORA_FILENAME = "nagharsubject_high-step00000200.safetensors"
TRIGGER_TOKEN = "nagharsubject"

DEFAULT_PROMPT = (
    "nagharsubject, a middle-aged man wearing a cream knitted cap, a light beige kurta, and a cream shawl, "
    "seated outdoors in a pine forest in a static medium close-up. He raises his right hand with an open palm "
    "and makes natural speaking mouth movements. Locked camera, natural daylight, realistic motion."
)
DEFAULT_NEGATIVE = (
    "low quality, blurry, deformed face, deformed hands, extra fingers, missing fingers, cropped hands, "
    "camera movement, another person, duplicate person, text, subtitles, watermark"
)

MODEL_MANIFEST = (
    {
        "name": "high_noise_dit",
        "repo_id": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
        "revision": "f97505f0d38bea4897c970db66cb5f97f73676de",
        "filename": "split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors",
        "sha256": "c21c21efa368d529a982a7d89dd11d86608b5937cae9a125d0ffe5491918a100",
        "approx_bytes": 28_600_000_000,
    },
    {
        "name": "low_noise_dit",
        "repo_id": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
        "revision": "f97505f0d38bea4897c970db66cb5f97f73676de",
        "filename": "split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors",
        "sha256": "edb89340c8a6fbf1a70e76a839ae01eaf7d289f05ea1ebd6b1a3fc6f533826e9",
        "approx_bytes": 28_600_000_000,
    },
    {
        "name": "vae",
        "repo_id": "Wan-AI/Wan2.1-I2V-14B-720P",
        "revision": "0df88eef5752db1f14be8a9ec52f029ad78efd0c",
        "filename": "Wan2.1_VAE.pth",
        "sha256": "38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981",
        "approx_bytes": 508_000_000,
    },
    {
        "name": "t5",
        "repo_id": "Wan-AI/Wan2.1-I2V-14B-720P",
        "revision": "0df88eef5752db1f14be8a9ec52f029ad78efd0c",
        "filename": "models_t5_umt5-xxl-enc-bf16.pth",
        "sha256": "7cace0da2b446bbbbc57d031ab6cf163a3d59b366da94e5afe36745b746fd81d",
        "approx_bytes": 11_400_000_000,
    },
)

PROFILE_ARGS = {
    "quality": ("--lazy_loading", "--vae_cache_cpu"),
    "balanced": ("--fp8", "--fp8_scaled", "--fp8_t5", "--lazy_loading", "--vae_cache_cpu"),
    "low_vram": ("--fp8", "--fp8_t5", "--blocks_to_swap", "20", "--lazy_loading", "--vae_cache_cpu"),
}
PROFILE_MIN_VRAM_GIB = {"quality": 70.0, "balanced": 40.0, "low_vram": 22.0}


def load_dotenv(path: Path) -> None:
    """Load a simple KEY=VALUE file without executing it."""
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if value[:1] == value[-1:] and value.startswith(("'", '"')):
            value = value[1:-1]
        os.environ.setdefault(key, value)


def env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safetensors_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        length_bytes = handle.read(8)
        if len(length_bytes) != 8:
            raise RuntimeError(f"Not a valid safetensors file (missing header length): {path}")
        header_length = struct.unpack("<Q", length_bytes)[0]
        if header_length <= 2 or header_length > 100 * 1024 * 1024:
            raise RuntimeError(f"Implausible safetensors header length {header_length}: {path}")
        header = handle.read(header_length)
        if len(header) != header_length:
            raise RuntimeError(f"Truncated safetensors header: {path}")
    return json.loads(header.decode("utf-8"))


def checkpoint_report(path: Path, verify_hash: bool) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"LoRA checkpoint not found: {path}")
    header = safetensors_header(path)
    metadata = header.get("__metadata__", {})
    tensors = [key for key in header if key != "__metadata__"]
    actual_hash = sha256_file(path) if verify_hash else None
    expected = {
        "ss_steps": "200",
        "ss_network_dim": "32",
        "ss_network_alpha": "32.0",
        "modelspec.timestep_range": "900,1000",
    }
    mismatches = {
        key: {"expected": expected_value, "actual": str(metadata.get(key))}
        for key, expected_value in expected.items()
        if str(metadata.get(key)) != expected_value
    }
    base_name = str(metadata.get("ss_sd_model_name", "")).lower()
    if "high_noise" not in base_name or "i2v" not in base_name:
        mismatches["ss_sd_model_name"] = {
            "expected": "Wan 2.2 I2V high-noise model",
            "actual": metadata.get("ss_sd_model_name"),
        }
    if verify_hash and actual_hash != LORA_SHA256:
        mismatches["sha256"] = {"expected": LORA_SHA256, "actual": actual_hash}
    if mismatches:
        raise RuntimeError(f"Checkpoint validation failed: {json.dumps(mismatches, indent=2)}")
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": actual_hash or "not checked",
        "tensor_count": len(tensors),
        "output_name": metadata.get("ss_output_name"),
        "steps": int(metadata["ss_steps"]),
        "epoch": int(metadata.get("ss_epoch", 0)),
        "rank": int(float(metadata["ss_network_dim"])),
        "alpha": float(metadata["ss_network_alpha"]),
        "base_model": metadata.get("ss_sd_model_name"),
        "timestep_range": metadata.get("modelspec.timestep_range"),
        "training_resolution": json.loads(metadata["ss_datasets"])[0]["resolution"],
        "training_frames": json.loads(metadata["ss_datasets"])[0]["target_frames"],
        "training_fps": json.loads(metadata["ss_datasets"])[0]["source_fps"],
        "trigger_token": TRIGGER_TOKEN,
    }


def first_existing(candidates: Iterable[Path]) -> Path:
    values = list(candidates)
    for candidate in values:
        if candidate.exists():
            return candidate.resolve()
    return values[0].expanduser().resolve()


@dataclass(frozen=True)
class Config:
    project_root: Path
    workspace_root: Path
    runpod_root: Path
    model_root: Path
    output_root: Path
    musubi_dir: Path
    lora_path: Path
    reference_root: Path
    reference_image: Path
    profile: str
    lora_strength: float
    seed: int
    verify_model_hashes: bool
    verify_lora_hash: bool

    @classmethod
    def from_env(cls) -> "Config":
        project_root = Path(__file__).resolve().parent
        workspace_root = project_root.parent
        runpod_root = Path(os.environ.get("RUNPOD_ROOT", "/workspace")).expanduser()
        local_mode = (workspace_root / "nagharsubject_high_checkpoints").exists()
        default_output = workspace_root / "wan22_babaji_outputs" if local_mode else runpod_root / "outputs/babaji_wan22_i2v"
        default_lora = first_existing(
            (
                workspace_root / "nagharsubject_high_checkpoints/high" / LORA_FILENAME,
                runpod_root / "nagharsubject_high_checkpoints/high" / LORA_FILENAME,
            )
        )
        default_references = first_existing(
            (
                workspace_root / "prepared_wan22_i2v_curated/reference_images",
                runpod_root / "prepared_wan22_i2v_curated/reference_images",
            )
        )
        reference_root = Path(os.environ.get("REFERENCE_ROOT", str(default_references))).expanduser().resolve()
        return cls(
            project_root=project_root,
            workspace_root=workspace_root,
            runpod_root=runpod_root.resolve(),
            model_root=Path(os.environ.get("MODEL_ROOT", str(runpod_root / "models/wan22_i2v_a14b"))).expanduser().resolve(),
            output_root=Path(os.environ.get("OUTPUT_ROOT", str(default_output))).expanduser().resolve(),
            musubi_dir=Path(os.environ.get("MUSUBI_DIR", str(runpod_root / "tools/musubi-tuner"))).expanduser().resolve(),
            lora_path=Path(os.environ.get("LORA_PATH", str(default_lora))).expanduser().resolve(),
            reference_root=reference_root,
            reference_image=Path(os.environ.get("REFERENCE_IMAGE", str(reference_root / "open_palm.png"))).expanduser().resolve(),
            profile=os.environ.get("MEMORY_PROFILE", "quality").strip().lower(),
            lora_strength=float(os.environ.get("LORA_STRENGTH", "1.0")),
            seed=int(os.environ.get("SEED", "42")),
            verify_model_hashes=env_bool("VERIFY_MODEL_HASHES", True),
            verify_lora_hash=env_bool("VERIFY_LORA_HASH", True),
        )

    def model_path(self, name: str) -> Path:
        item = next(entry for entry in MODEL_MANIFEST if entry["name"] == name)
        return self.model_root / str(item["filename"])


class InferencePipeline:
    def __init__(self, config: Config, dry_run: bool = False) -> None:
        self.cfg = config
        self.dry_run = dry_run
        if self.cfg.profile not in PROFILE_ARGS:
            raise RuntimeError(f"Unknown MEMORY_PROFILE={self.cfg.profile!r}; choose {', '.join(PROFILE_ARGS)}")

    def log(self, message: str) -> None:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)

    def inspect(self) -> dict[str, Any]:
        report = checkpoint_report(self.cfg.lora_path, self.cfg.verify_lora_hash)
        print(json.dumps(report, indent=2), flush=True)
        return report

    def download(self) -> None:
        self.cfg.model_root.mkdir(parents=True, exist_ok=True)
        missing_bytes = sum(
            int(item["approx_bytes"])
            for item in MODEL_MANIFEST
            if not (self.cfg.model_root / str(item["filename"])).is_file()
        )
        free_bytes = shutil.disk_usage(self.cfg.model_root).free
        required_bytes = missing_bytes + (12 * 1024**3 if missing_bytes else 0)
        if free_bytes < required_bytes:
            raise RuntimeError(
                f"Insufficient storage: {free_bytes / 1024**3:.1f} GiB free, "
                f"approximately {required_bytes / 1024**3:.1f} GiB required."
            )
        if self.dry_run:
            for item in MODEL_MANIFEST:
                self.log(f"would download {item['repo_id']}@{item['revision']}::{item['filename']}")
            return
        from huggingface_hub import hf_hub_download

        token = os.environ.get("HF_TOKEN") or None
        verified: dict[str, Any] = {}
        for item in MODEL_MANIFEST:
            path = self.cfg.model_root / str(item["filename"])
            if not path.is_file():
                self.log(f"downloading {item['name']}")
                hf_hub_download(
                    repo_id=str(item["repo_id"]),
                    filename=str(item["filename"]),
                    revision=str(item["revision"]),
                    local_dir=self.cfg.model_root,
                    token=token,
                )
            if not path.is_file():
                raise RuntimeError(f"Download did not create {path}")
            digest = "not checked"
            if self.cfg.verify_model_hashes:
                self.log(f"verifying {item['name']} SHA-256")
                digest = sha256_file(path)
                if digest != item["sha256"]:
                    raise RuntimeError(f"SHA-256 mismatch for {path}: expected {item['sha256']}, found {digest}")
            verified[str(item["name"])] = {"path": str(path), "bytes": path.stat().st_size, "sha256": digest}
        atomic_json(self.cfg.model_root / ".verified_models.json", verified)

    def _assert_install(self) -> None:
        script = self.cfg.musubi_dir / "src/musubi_tuner/wan_generate_video.py"
        if not script.is_file():
            raise RuntimeError(f"Musubi Tuner is missing at {self.cfg.musubi_dir}; run setup_runpod.sh")
        expected = os.environ.get("MUSUBI_COMMIT", MUSUBI_COMMIT_DEFAULT)
        result = subprocess.run(
            ["git", "-C", str(self.cfg.musubi_dir), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        if result.stdout.strip() != expected:
            raise RuntimeError(f"Musubi commit mismatch: expected {expected}, found {result.stdout.strip()}")

    def _check_gpu(self) -> dict[str, Any]:
        try:
            import torch
        except ImportError as error:
            raise RuntimeError("PyTorch is unavailable; run setup_runpod.sh") from error
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Wan 2.2 A14B inference requires an NVIDIA CUDA GPU.")
        properties = torch.cuda.get_device_properties(0)
        vram_gib = properties.total_memory / 1024**3
        minimum = PROFILE_MIN_VRAM_GIB[self.cfg.profile]
        if vram_gib < minimum:
            raise RuntimeError(
                f"Profile {self.cfg.profile!r} needs at least {minimum:.0f} GiB-class VRAM; "
                f"detected {properties.name} with {vram_gib:.1f} GiB."
            )
        return {"name": properties.name, "vram_gib": round(vram_gib, 2), "torch": torch.__version__}

    def preflight(self, require_runtime: bool = True) -> dict[str, Any]:
        checkpoint = checkpoint_report(self.cfg.lora_path, self.cfg.verify_lora_hash)
        if not self.cfg.reference_image.is_file():
            raise RuntimeError(f"Reference image not found: {self.cfg.reference_image}")
        missing_models = [
            str(self.cfg.model_path(str(item["name"])))
            for item in MODEL_MANIFEST
            if not self.cfg.model_path(str(item["name"])).is_file()
        ]
        if missing_models and not self.dry_run:
            raise RuntimeError("Base model files are missing; run the download stage:\n" + "\n".join(missing_models))
        if not missing_models and self.cfg.verify_model_hashes and not self.dry_run:
            marker_path = self.cfg.model_root / ".verified_models.json"
            if not marker_path.is_file():
                raise RuntimeError(
                    f"Model verification marker is missing: {marker_path}. Run the download stage to hash the model files."
                )
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            for item in MODEL_MANIFEST:
                name = str(item["name"])
                path = self.cfg.model_path(name)
                record = marker.get(name, {})
                if record.get("sha256") != item["sha256"] or record.get("bytes") != path.stat().st_size:
                    raise RuntimeError(
                        f"Model verification record is stale or invalid for {path}. Re-run the download stage."
                    )
        gpu: dict[str, Any] | str = "skipped in dry-run"
        if require_runtime and not self.dry_run:
            self._assert_install()
            gpu = self._check_gpu()
        report = {
            "status": "ok",
            "profile": self.cfg.profile,
            "checkpoint": checkpoint,
            "reference_image": str(self.cfg.reference_image),
            "models": "present" if not missing_models else "missing (allowed in dry-run)",
            "gpu": gpu,
        }
        print(json.dumps(report, indent=2), flush=True)
        return report

    @staticmethod
    def _validate_shape(height: int, width: int, frames: int) -> None:
        if height % 8 or width % 8:
            raise RuntimeError(f"Height and width must be divisible by 8; received {height}x{width}")
        if frames < 1 or (frames - 1) % 4:
            raise RuntimeError(f"Frame count must be 4n+1 for the Wan VAE; received {frames}")

    @staticmethod
    def _with_trigger(prompt: str) -> str:
        cleaned = " ".join(prompt.strip().split())
        if TRIGGER_TOKEN.lower() not in cleaned.lower():
            return f"{TRIGGER_TOKEN}, {cleaned}"
        return cleaned

    def build_command(
        self,
        *,
        save_path: Path,
        prompt: str | None,
        reference: Path | None,
        prompt_file: Path | None,
        negative_prompt: str,
        seed: int,
        height: int,
        width: int,
        frames: int,
        fps: int,
        steps: int,
        guidance: float,
        lora_strength: float,
    ) -> list[str]:
        self._validate_shape(height, width, frames)
        command = [
            sys.executable,
            "src/musubi_tuner/wan_generate_video.py",
            "--task", "i2v-A14B",
            "--video_size", str(height), str(width),
            "--video_length", str(frames),
            "--fps", str(fps),
            "--infer_steps", str(steps),
            "--flow_shift", "5.0",
            "--timestep_boundary", "900",
            "--guidance_scale", str(guidance),
            "--guidance_scale_high_noise", str(guidance),
            "--negative_prompt", negative_prompt,
            "--save_path", str(save_path),
            "--output_type", "video",
            "--dit", str(self.cfg.model_path("low_noise_dit")),
            "--dit_high_noise", str(self.cfg.model_path("high_noise_dit")),
            "--vae", str(self.cfg.model_path("vae")),
            "--t5", str(self.cfg.model_path("t5")),
            "--attn_mode", "sdpa",
            "--lora_weight_high_noise", str(self.cfg.lora_path),
            "--lora_multiplier_high_noise", str(lora_strength),
            "--seed", str(seed),
        ]
        if prompt_file is not None:
            command.extend(("--from_file", str(prompt_file)))
        else:
            if prompt is None or reference is None:
                raise RuntimeError("Single generation requires both a prompt and reference image")
            command.extend(("--prompt", self._with_trigger(prompt), "--image_path", str(reference)))
        command.extend(PROFILE_ARGS[self.cfg.profile])
        return command

    def _run(self, command: Sequence[str], run_dir: Path) -> None:
        printable = shlex.join(str(value) for value in command)
        self.log(printable)
        if self.dry_run:
            return
        log_path = run_dir / "inference.log"
        with log_path.open("w", encoding="utf-8", buffering=1) as log_handle:
            process = subprocess.Popen(
                [str(value) for value in command],
                cwd=self.cfg.musubi_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                log_handle.write(line)
            return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f"Inference failed with exit code {return_code}. See {log_path}")

    def _new_run_dir(self, label: str, seed: int) -> Path:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.cfg.output_root / f"{stamp}_{label}_seed{seed}"
        run_dir.mkdir(parents=True, exist_ok=False)
        return run_dir

    def generate(self, args: argparse.Namespace) -> Path:
        self.preflight(require_runtime=True)
        reference = Path(args.reference).expanduser().resolve() if args.reference else self.cfg.reference_image
        if not reference.is_file():
            raise RuntimeError(f"Reference image not found: {reference}")
        prompt = self._with_trigger(args.prompt or DEFAULT_PROMPT)
        seed = self.cfg.seed if args.seed is None else args.seed
        strength = self.cfg.lora_strength if args.lora_strength is None else args.lora_strength
        run_dir = self._new_run_dir("single", seed)
        command = self.build_command(
            save_path=run_dir,
            prompt=prompt,
            reference=reference,
            prompt_file=None,
            negative_prompt=args.negative_prompt,
            seed=seed,
            height=args.height,
            width=args.width,
            frames=args.frames,
            fps=args.fps,
            steps=args.steps,
            guidance=args.guidance,
            lora_strength=strength,
        )
        request = {
            "created_at": utc_now(),
            "mode": "single",
            "prompt": prompt,
            "negative_prompt": args.negative_prompt,
            "reference": str(reference),
            "seed": seed,
            "profile": self.cfg.profile,
            "lora_strength": strength,
            "lora_sha256": LORA_SHA256,
            "command": [str(value) for value in command],
        }
        atomic_json(run_dir / "request.json", request)
        self._run(command, run_dir)
        outputs = sorted(str(path) for path in run_dir.iterdir() if path.suffix.lower() in {".mp4", ".webm", ".mov"})
        atomic_json(run_dir / "result.json", {"finished_at": utc_now(), "outputs": outputs, "dry_run": self.dry_run})
        return run_dir

    def _resolve_reference(self, value: str | None) -> Path:
        if not value:
            return self.cfg.reference_image
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = self.cfg.reference_root / path
        return path.resolve()

    def prepare_batch_file(self, source: Path, destination: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        lines: list[str] = []
        for line_number, raw_line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
            if not raw_line.strip() or raw_line.lstrip().startswith("#"):
                continue
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as error:
                raise RuntimeError(f"Invalid JSON on {source}:{line_number}: {error}") from error
            prompt = self._with_trigger(str(record["prompt"]))
            reference = self._resolve_reference(record.get("reference"))
            if not reference.is_file():
                raise RuntimeError(f"Batch reference not found on line {line_number}: {reference}")
            seed = int(record.get("seed", self.cfg.seed + len(records)))
            negative = str(record.get("negative_prompt", args.negative_prompt))
            if " --" in prompt or " --" in negative:
                raise RuntimeError("Batch prompts and negative prompts cannot contain the literal delimiter ' --'")
            lines.append(
                f"{prompt} --w {args.width} --h {args.height} --f {args.frames} --d {seed} "
                f"--s {args.steps} --g {args.guidance} --fs 5.0 --i {reference} --n {negative}"
            )
            records.append({**record, "prompt": prompt, "reference": str(reference), "seed": seed})
        if not records:
            raise RuntimeError(f"Batch file contains no generation records: {source}")
        destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return records

    def batch(self, args: argparse.Namespace) -> Path:
        self.preflight(require_runtime=True)
        self._validate_shape(args.height, args.width, args.frames)
        source = Path(args.batch_file).expanduser().resolve()
        if not source.is_file():
            raise RuntimeError(f"Batch JSONL file not found: {source}")
        seed = self.cfg.seed if args.seed is None else args.seed
        strength = self.cfg.lora_strength if args.lora_strength is None else args.lora_strength
        run_dir = self._new_run_dir("batch", seed)
        resolved_prompts = run_dir / "resolved_prompts.txt"
        records = self.prepare_batch_file(source, resolved_prompts, args)
        command = self.build_command(
            save_path=run_dir,
            prompt=None,
            reference=None,
            prompt_file=resolved_prompts,
            negative_prompt=args.negative_prompt,
            seed=seed,
            height=args.height,
            width=args.width,
            frames=args.frames,
            fps=args.fps,
            steps=args.steps,
            guidance=args.guidance,
            lora_strength=strength,
        )
        atomic_json(
            run_dir / "request.json",
            {
                "created_at": utc_now(),
                "mode": "batch",
                "source": str(source),
                "records": records,
                "profile": self.cfg.profile,
                "lora_strength": strength,
                "lora_sha256": LORA_SHA256,
                "command": [str(value) for value in command],
            },
        )
        self._run(command, run_dir)
        outputs = sorted(str(path) for path in run_dir.iterdir() if path.suffix.lower() in {".mp4", ".webm", ".mov"})
        atomic_json(run_dir / "result.json", {"finished_at": utc_now(), "outputs": outputs, "dry_run": self.dry_run})
        return run_dir


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("inspect", "download", "preflight", "generate", "batch", "all"), nargs="?", default="generate")
    parser.add_argument("--prompt", help="single-generation prompt; the trigger token is added automatically")
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE)
    parser.add_argument("--reference", help="single-generation first-frame image")
    parser.add_argument("--batch-file", default=str(Path(__file__).resolve().parent / "prompts/babaji_snippets.jsonl"))
    parser.add_argument("--height", type=int, default=576)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--frames", type=int, default=81)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance", type=float, default=5.0)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--lora-strength", type=float)
    parser.add_argument("--dry-run", action="store_true", help="validate and print commands without running inference")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    project_root = Path(__file__).resolve().parent
    load_dotenv(project_root / ".env")
    args = parse_args(argv)
    pipeline = InferencePipeline(Config.from_env(), dry_run=args.dry_run)
    if args.stage == "inspect":
        pipeline.inspect()
    elif args.stage == "download":
        pipeline.download()
    elif args.stage == "preflight":
        pipeline.preflight(require_runtime=True)
    elif args.stage == "generate":
        print(f"Output directory: {pipeline.generate(args)}")
    elif args.stage == "batch":
        print(f"Output directory: {pipeline.batch(args)}")
    elif args.stage == "all":
        pipeline.download()
        print(f"Output directory: {pipeline.generate(args)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        raise SystemExit(130)
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)

