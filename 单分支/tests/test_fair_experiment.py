import hashlib
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from fair_experiment import assert_finite, count_images, file_sha256, set_seed, validate_weight_files


class FairExperimentUtilityTests(unittest.TestCase):
    def test_set_seed_repeats_python_numpy_and_torch_sequences(self):
        set_seed(12345)
        first = (random.random(), np.random.rand(4).tolist(), torch.rand(4).tolist())

        set_seed(12345)
        second = (random.random(), np.random.rand(4).tolist(), torch.rand(4).tolist())

        self.assertEqual(first, second)

    def test_file_sha256_hashes_ot_transcolor_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "payload.bin"
            path.write_bytes(b"ot-transcolor")

            self.assertEqual(
                file_sha256(path),
                "26b90959c0a22f13cbe756744e0f32dc89a1e5421eefc1ac8c9ad0b9bf2a7450",
            )

    def test_validate_weight_files_lists_missing_name_and_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "missing.pth"

            with self.assertRaises(FileNotFoundError) as ctx:
                validate_weight_files({"missing": missing})
            self.assertIn("missing", str(ctx.exception))
            self.assertIn(str(missing), str(ctx.exception))

    def test_count_images_ignores_txt_nested_and_unsupported_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("one.jpg", "two.JPEG", "three.png", "notes.txt", "archive.gif"):
                (root / name).write_bytes(b"x")
            nested = root / "nested"
            nested.mkdir()
            (nested / "also.png").write_bytes(b"x")

            self.assertEqual(count_images(root), 3)

    def test_assert_finite_rejects_nan_and_inf(self):
        with self.assertRaises(FloatingPointError):
            assert_finite("nan_loss", torch.tensor(float("nan")))
        with self.assertRaises(FloatingPointError):
            assert_finite("inf_loss", torch.tensor(float("inf")))
        assert_finite("finite_loss", torch.tensor([1.0, -2.0]))


if __name__ == "__main__":
    unittest.main()
