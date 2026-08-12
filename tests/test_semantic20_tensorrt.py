from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from adom.runtime.semantic20_tensorrt import (
    compare_masks,
    percentile_summary,
    reference_pairs,
    valid_region_mask,
)


class Semantic20TensorRTTests(unittest.TestCase):
    def test_reference_pairs_require_corresponding_onnx_masks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            np.save(root / "000_frame_input.npy", np.zeros((1, 3, 2, 3)))
            with self.assertRaisesRegex(FileNotFoundError, "reference mask"):
                reference_pairs(root)
            Image.fromarray(np.zeros((2, 3), dtype=np.uint8)).save(
                root / "000_frame_onnx_mask.png"
            )
            self.assertEqual(len(reference_pairs(root)), 1)

    def test_valid_region_rejects_out_of_bounds_padding(self) -> None:
        valid = valid_region_mask((4, 6), (0, 0, 3, 6))
        self.assertEqual(int(valid.sum()), 18)
        with self.assertRaisesRegex(ValueError, "Invalid valid region"):
            valid_region_mask((4, 6), (0, 0, 5, 6))

    def test_mask_comparison_uses_valid_region_and_all_classes(self) -> None:
        onnx = np.zeros((4, 6), dtype=np.uint8)
        tensorrt = onnx.copy()
        tensorrt[3, :] = 1
        report = compare_masks(
            onnx,
            tensorrt,
            valid_region=(0, 0, 3, 6),
            num_classes=19,
        )
        self.assertEqual(report["valid_pixel_argmax_agreement"], 1.0)
        self.assertEqual(report["full_pixel_argmax_agreement"], 0.75)
        self.assertEqual(len(report["class_area_ratios"]), 19)
        self.assertEqual(
            report["maximum_class_area_difference_percentage_points"], 0.0
        )

    def test_percentile_summary_is_machine_readable(self) -> None:
        report = percentile_summary([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(report["mean_ms"], 2.5)
        self.assertEqual(report["p50_ms"], 2.5)
        self.assertGreater(report["p95_ms"], report["p50_ms"])


if __name__ == "__main__":
    unittest.main()
