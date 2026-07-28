from __future__ import annotations

import unittest

import numpy as np

from adom.evaluation import metrics_from_confusion


class EvaluationTests(unittest.TestCase):
    def test_cost4_metrics(self) -> None:
        confusion = np.array(
            [
                [8, 2, 0, 0],
                [0, 9, 1, 0],
                [0, 1, 8, 1],
                [0, 0, 2, 8],
            ]
        )
        metrics = metrics_from_confusion(confusion)
        self.assertAlmostEqual(metrics["Recall/high_cost_or_obstacle"], 0.8)
        self.assertAlmostEqual(metrics["high_cost_or_obstacle_recall"], 0.8)
        self.assertAlmostEqual(metrics["traversable_precision"], 29 / 31)
        self.assertIn("mIoU", metrics)

    def test_confusion_shape_is_strict(self) -> None:
        with self.assertRaises(ValueError):
            metrics_from_confusion(np.zeros((3, 3)))


if __name__ == "__main__":
    unittest.main()
