#!/usr/bin/env python3
"""Run an apples-to-apples Wan 2.2 LoRA checkpoint comparison."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Sequence

import inference


def load_hash_manifest(path: Path | None) -> dict[str, str]:
    """Read a sha256sum-compatible manifest keyed by checkpoint basename."""
    if path is None:
        return {}
    if not path.is_file():
        raise RuntimeError(f"SHA-256 manifest not found: {path}")
    hashes: dict[str, str] = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split(maxsplit=1)
        if len(fields) != 2 or len(fields[0]) != 64:
            raise RuntimeError(f"Invalid sha256sum line {line_number} in {path}")
        filename = fields[1].lstrip("* ")
        hashes[Path(filename).name] = fields[0].lower()
    return hashes


def checkpoint_path(checkpoint_dir: Path, step: int) -> Path:
    exact = checkpoint_dir / f"nagharsubject_high-step{step:08d}.safetensors"
    if exact.is_file():
        return exact.resolve()
    matches = sorted(checkpoint_dir.glob(f"*-step{step:08d}.safetensors"))
    if len(matches) == 1:
        return matches[0].resolve()
    if len(matches) > 1:
        raise RuntimeError(f"More than one checkpoint matches step {step}: {matches}")
    return exact.resolve()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--checkpoint-steps", type=int, nargs="+", default=(200, 400, 600))
    parser.add_argument("--hash-manifest", type=Path, help="sha256sum file created before upload")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--profile", choices=("rtx5090_quality", "rtx5090_fast", "rtx5090_safe"), default="rtx5090_quality")
    parser.add_argument("--prompt", default=inference.DEFAULT_PROMPT)
    parser.add_argument("--negative-prompt", default=inference.DEFAULT_NEGATIVE)
    parser.add_argument("--reference")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora-strength", type=float, default=1.0)
    parser.add_argument("--height", type=int, default=576)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--frames", type=int, default=81)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--infer-steps", type=int, default=30)
    parser.add_argument("--guidance", type=float, default=5.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    project_root = Path(__file__).resolve().parent
    inference.load_dotenv(project_root / ".env")
    args = parse_args(argv)
    system_root = Path(os.environ.get("SYSTEM_ROOT", os.environ.get("RUNPOD_ROOT", "/workspace"))).expanduser().resolve()
    checkpoint_dir = (args.checkpoint_dir or system_root / "nagharsubject_high_checkpoints/high").expanduser().resolve()
    comparison_parent = (args.output_root or system_root / "outputs/babaji_wan22_i2v/comparisons").expanduser().resolve()
    comparison_root = comparison_parent / datetime.now().strftime("%Y%m%d_%H%M%S_checkpoint_comparison")
    comparison_root.mkdir(parents=True, exist_ok=False)

    hashes = load_hash_manifest(args.hash_manifest.expanduser().resolve() if args.hash_manifest else None)
    summary: dict[str, object] = {
        "created_at": inference.utc_now(),
        "comparison_root": str(comparison_root),
        "profile": args.profile,
        "fixed_settings": {
            "prompt": args.prompt,
            "negative_prompt": args.negative_prompt,
            "reference": args.reference or os.environ.get("REFERENCE_IMAGE"),
            "seed": args.seed,
            "lora_strength": args.lora_strength,
            "height": args.height,
            "width": args.width,
            "frames": args.frames,
            "fps": args.fps,
            "infer_steps": args.infer_steps,
            "guidance": args.guidance,
        },
        "runs": [],
    }
    inference.atomic_json(comparison_root / "comparison_summary.json", summary)

    failed = False
    for step in args.checkpoint_steps:
        checkpoint = checkpoint_path(checkpoint_dir, step)
        run_record: dict[str, object] = {"checkpoint_step": step, "checkpoint": str(checkpoint)}
        summary["runs"].append(run_record)  # type: ignore[union-attr]
        if not checkpoint.is_file():
            run_record.update(status="missing", error=f"Checkpoint not found: {checkpoint}")
            inference.atomic_json(comparison_root / "comparison_summary.json", summary)
            failed = True
            continue

        expected_hash = hashes.get(checkpoint.name)
        try:
            report = inference.checkpoint_report(checkpoint, verify_hash=True, expected_hash=expected_hash)
        except Exception as error:
            run_record.update(status="invalid", error=str(error))
            inference.atomic_json(comparison_root / "comparison_summary.json", summary)
            failed = True
            continue

        step_output_root = comparison_root / f"checkpoint_{step:08d}"
        env = os.environ.copy()
        env.update(
            {
                "LORA_PATH": str(checkpoint),
                "MEMORY_PROFILE": args.profile,
                "OUTPUT_ROOT": str(step_output_root),
                "VERIFY_LORA_HASH": "1",
            }
        )
        if expected_hash:
            env["EXPECTED_LORA_SHA256"] = expected_hash
        else:
            env.pop("EXPECTED_LORA_SHA256", None)

        command = [
            sys.executable,
            str(project_root / "inference.py"),
            "generate",
            "--prompt", args.prompt,
            "--negative-prompt", args.negative_prompt,
            "--seed", str(args.seed),
            "--lora-strength", str(args.lora_strength),
            "--height", str(args.height),
            "--width", str(args.width),
            "--frames", str(args.frames),
            "--fps", str(args.fps),
            "--steps", str(args.infer_steps),
            "--guidance", str(args.guidance),
        ]
        if args.reference:
            command.extend(("--reference", args.reference))
        if args.dry_run:
            command.append("--dry-run")

        run_record.update(status="running", checkpoint_report=report, command=command)
        inference.atomic_json(comparison_root / "comparison_summary.json", summary)
        started = time.monotonic()
        driver_log = comparison_root / f"checkpoint_{step:08d}.log"
        with driver_log.open("w", encoding="utf-8", buffering=1) as log_handle:
            process = subprocess.Popen(
                command,
                cwd=project_root,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                print(f"[step {step}] {line}", end="", flush=True)
                log_handle.write(line)
            return_code = process.wait()

        outputs = sorted(
            str(path)
            for path in step_output_root.rglob("*")
            if path.suffix.lower() in {".mp4", ".webm", ".mov"}
        ) if step_output_root.exists() else []
        run_record.update(
            status="ok" if return_code == 0 else "failed",
            return_code=return_code,
            duration_seconds=round(time.monotonic() - started, 2),
            outputs=outputs,
            log=str(driver_log),
        )
        if return_code != 0:
            failed = True
        inference.atomic_json(comparison_root / "comparison_summary.json", summary)

    summary["finished_at"] = inference.utc_now()
    summary["status"] = "failed" if failed else "ok"
    inference.atomic_json(comparison_root / "comparison_summary.json", summary)
    print(f"Comparison summary: {comparison_root / 'comparison_summary.json'}")
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        raise SystemExit(130)
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
