"""CPU-only checks of the published input-only metadata contract."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.build_rellisur_language_metadata import collect_rows, list_images


class ConditionTests(unittest.TestCase):
    def test_lq_only_needs_no_gt_and_preserves_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            for idx, value in enumerate([4, 35, 110]):
                image = np.full((32, 32, 3), value, dtype=np.uint8)
                Image.fromarray(image).save(folder / f"scene-{idx}.png")
            before = {p.name: p.read_bytes() for p in list_images(folder)}
            rows = collect_rows(folder, ".", None, scale=2, stats_size=32)
            self.assertEqual(len(rows), 3)
            for row in rows:
                self.assertIsNone(row["hq"])
                self.assertEqual(len(row["attribute_vector"]), 7)
                self.assertTrue(all(0 <= x <= 1 for x in row["attribute_vector"]))
                self.assertEqual(row["attribute_vector"][-1], 0.5)
                self.assertIn("2x super-resolution", row["prompt"])
            self.assertEqual(before, {p.name: p.read_bytes() for p in list_images(folder)})

    def test_scale_is_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            Image.new("RGB", (32, 32), (20, 25, 30)).save(Path(tmp) / "input.png")
            rows = collect_rows(Path(tmp), ".", None, scale=4, stats_size=32)
            self.assertEqual(rows[0]["attribute_vector"][-1], 1.0)
            self.assertIn("4x super-resolution", rows[0]["prompt"])

    def test_empty_input_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                list_images(Path(tmp))


if __name__ == "__main__":
    unittest.main()
