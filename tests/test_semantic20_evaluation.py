from __future__ import annotations

import unittest

import numpy as np

from adom.evaluation_semantic20 import (
    SEMANTIC20_CLASSES,
    select_constrained_checkpoint,
    semantic20_metrics_from_confusion,
)


class Semantic20CleanMetricTests(unittest.TestCase):
    def test_absent_gt_false_positive_does_not_change_fixed_denominator(self) -> None:
        count = len(SEMANTIC20_CLASSES)
        confusion = np.zeros((count, count), dtype=np.int64)
        # Populate every TestSupported11 class with a perfect prediction.
        for index in (1, 2, 3, 5, 10, 13, 14, 15, 16, 17, 18):
            confusion[index, index] = 100
        # Dirt has no GT, but receives false positives from grass.
        confusion[1, 0] = 25
        artifact, flat = semantic20_metrics_from_confusion(
            confusion,
            evaluation_split="test",
            image_count=2,
            absent_fp_image_count=np.array([1] + [0] * (count - 1)),
        )
        self.assertEqual(flat["Denominator/TestSupported11"], 11.0)
        self.assertAlmostEqual(flat["mIoU/TestSupported11"], (80.0 + 1000.0) / 11)
        self.assertEqual(flat["AbsentClassFP/pixels/dirt"], 25.0)
        self.assertEqual(artifact["absent_class_fp"][0]["name"], "dirt")
        self.assertEqual(
            artifact["absent_class_fp"][0]["fp_source_classes"][0]["source_name"],
            "grass",
        )

    def test_validation_panels_and_class_metrics(self) -> None:
        count = len(SEMANTIC20_CLASSES)
        confusion = np.zeros((count, count), dtype=np.int64)
        for index in (1, 2, 3, 5, 6, 10, 11, 13, 14, 15, 16, 17, 18):
            confusion[index, index] = 10
        artifact, flat = semantic20_metrics_from_confusion(
            confusion,
            evaluation_split="val",
        )
        self.assertEqual(flat["Denominator/ValSupported13"], 13.0)
        self.assertEqual(flat["Denominator/RareRisk4"], 4.0)
        self.assertEqual(flat["Denominator/AugmentedRisk2"], 2.0)
        self.assertEqual(flat["mIoU/ValSupported13"], 100.0)
        self.assertEqual(flat["mPrecision/RareRisk4"], 100.0)
        self.assertEqual(artifact["panels"]["TerrainHazard"]["denominator"], 2)

    def test_canonical_support_regression_fails(self) -> None:
        count = len(SEMANTIC20_CLASSES)
        confusion = np.zeros((count, count), dtype=np.int64)
        for index in (1, 2, 3, 5, 10, 13, 14, 15, 16, 17):
            confusion[index, index] = 10
        with self.assertRaisesRegex(RuntimeError, "GT support changed"):
            semantic20_metrics_from_confusion(
                confusion,
                evaluation_split="test",
            )

    def test_macro_precision_counts_unpredicted_supported_class_as_zero(self) -> None:
        count = len(SEMANTIC20_CLASSES)
        confusion = np.zeros((count, count), dtype=np.int64)
        supported = (1, 2, 3, 5, 6, 10, 11, 13, 14, 15, 16, 17, 18)
        for index in supported:
            confusion[index, index] = 10
        # Pole has GT but is never predicted; those pixels become vehicle FP.
        confusion[3, 3] = 0
        confusion[3, 6] = 10
        artifact, flat = semantic20_metrics_from_confusion(
            confusion,
            evaluation_split="val",
        )
        pole = next(row for row in artifact["classes"] if row["name"] == "pole")
        self.assertIsNone(pole["precision"])
        self.assertEqual(flat["Denominator/ValSupported13"], 13.0)
        self.assertAlmostEqual(
            flat["mPrecision/ValSupported13"],
            (11 * 100.0 + 50.0) / 13.0,
        )

    def test_constrained_checkpoint_selection(self) -> None:
        records = [
            {"iteration": 6000, "overall_miou": 60.0, "rare_risk_miou": 35.0},
            {"iteration": 7000, "overall_miou": 59.2, "rare_risk_miou": 42.0},
            {"iteration": 8000, "overall_miou": 58.9, "rare_risk_miou": 50.0},
        ]
        selected = select_constrained_checkpoint(records, tolerance_pp=1.0)
        self.assertEqual(selected["iteration"], 7000)

    def test_invalid_shape_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "19x19"):
            semantic20_metrics_from_confusion(
                np.zeros((4, 4)), evaluation_split="val"
            )


if __name__ == "__main__":
    unittest.main()
