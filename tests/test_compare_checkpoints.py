import tempfile
import unittest
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import compare_checkpoints


class CompareCheckpointTests(unittest.TestCase):
    def test_sha256sum_manifest_is_loaded_by_basename(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "checkpoints.sha256"
            digest = "a" * 64
            manifest.write_text(f"{digest}  nested/model.safetensors\n", encoding="utf-8")
            self.assertEqual(compare_checkpoints.load_hash_manifest(manifest), {"model.safetensors": digest})

    def test_exact_checkpoint_name_is_preferred(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = root / "nagharsubject_high-step00000400.safetensors"
            expected.write_bytes(b"test")
            self.assertEqual(compare_checkpoints.checkpoint_path(root, 400), expected.resolve())


if __name__ == "__main__":
    unittest.main()
