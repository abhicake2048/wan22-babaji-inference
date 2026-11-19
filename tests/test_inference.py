import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import inference


class InferenceTests(unittest.TestCase):
    def make_config(self, root: Path, profile: str = "quality") -> inference.Config:
        reference_root = root / "references"
        reference_root.mkdir(parents=True, exist_ok=True)
        (reference_root / "open_palm.png").write_bytes(b"image")
        return inference.Config(
            project_root=root,
            workspace_root=root,
            runpod_root=root,
            model_root=root / "models",
            output_root=root / "outputs",
            musubi_dir=root / "musubi",
            lora_path=root / inference.LORA_FILENAME,
            reference_root=reference_root,
            reference_image=reference_root / "open_palm.png",
            profile=profile,
            lora_strength=1.0,
            seed=42,
            verify_model_hashes=False,
            verify_lora_hash=False,
        )

    def test_command_maps_lora_only_to_high_noise_expert(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = self.make_config(Path(directory))
            pipeline = inference.InferencePipeline(cfg, dry_run=True)
            command = pipeline.build_command(
                save_path=cfg.output_root,
                prompt="a calm teaching gesture",
                reference=cfg.reference_image,
                prompt_file=None,
                negative_prompt=inference.DEFAULT_NEGATIVE,
                seed=42,
                height=576,
                width=768,
                frames=81,
                fps=16,
                steps=30,
                guidance=5.0,
                lora_strength=1.0,
            )
            self.assertIn("--lora_weight_high_noise", command)
            self.assertIn("--lora_multiplier_high_noise", command)
            self.assertNotIn("--lora_weight", command)
            self.assertEqual(command[command.index("--timestep_boundary") + 1], "900")
            self.assertEqual(command[command.index("--video_size") + 1:command.index("--video_size") + 3], ["576", "768"])
            self.assertIn(inference.TRIGGER_TOKEN, command[command.index("--prompt") + 1])

    def test_balanced_profile_uses_scaled_fp8(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = self.make_config(Path(directory), profile="balanced")
            pipeline = inference.InferencePipeline(cfg, dry_run=True)
            command = pipeline.build_command(
                save_path=cfg.output_root,
                prompt=inference.DEFAULT_PROMPT,
                reference=cfg.reference_image,
                prompt_file=None,
                negative_prompt=inference.DEFAULT_NEGATIVE,
                seed=42,
                height=576,
                width=768,
                frames=81,
                fps=16,
                steps=30,
                guidance=5.0,
                lora_strength=1.0,
            )
            self.assertIn("--fp8", command)
            self.assertIn("--fp8_scaled", command)
            self.assertIn("--fp8_t5", command)

    def test_invalid_frame_count_is_rejected(self):
        with self.assertRaises(RuntimeError):
            inference.InferencePipeline._validate_shape(576, 768, 80)

    def test_batch_resolves_reference_and_trigger(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = self.make_config(root)
            source = root / "batch.jsonl"
            source.write_text(json.dumps({"id": "one", "reference": "open_palm.png", "prompt": "a calm address"}) + "\n", encoding="utf-8")
            destination = root / "resolved.txt"
            args = argparse.Namespace(
                width=768,
                height=576,
                frames=81,
                steps=30,
                guidance=5.0,
                negative_prompt=inference.DEFAULT_NEGATIVE,
            )
            records = inference.InferencePipeline(cfg, dry_run=True).prepare_batch_file(source, destination, args)
            self.assertEqual(len(records), 1)
            text = destination.read_text(encoding="utf-8")
            self.assertIn(inference.TRIGGER_TOKEN, text)
            self.assertIn(str(cfg.reference_image.resolve()), text)
            self.assertIn("--w 768 --h 576 --f 81", text)


if __name__ == "__main__":
    unittest.main()

