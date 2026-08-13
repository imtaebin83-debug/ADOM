from __future__ import annotations

import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from adom.data.transform_audit import Sample, audit_transforms
from adom.runtime.source_sampling import RareClassSourceSchedule


class TransformAuditTests(unittest.TestCase):
    def test_crop_and_nocrop_candidates_record_retention_and_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            mask_path = Path(directory) / "mask.png"
            mask = np.zeros((12, 16), dtype=np.uint8)
            mask[2:10, 2:4] = 3
            mask[7:11, 5:14] = 10
            mask[:, 14:] = 255
            Image.fromarray(mask, mode="L").save(mask_path)
            report = audit_transforms(
                [Sample("rellis3d/sample", "rellis3d", mask_path)],
                draws=20,
                seed=42,
            )

        self.assertEqual(report["schema_version"], "adom-ta0-transform-audit-v1")
        self.assertEqual(report["monte_carlo_draws_per_sample"], 20)
        rows = {
            (row["candidate"], row["class_id"]): row
            for row in report["class_retention"]
        }
        for candidate in ("i1_nocrop_640x384", "i2_nocrop_640x480"):
            self.assertEqual(rows[(candidate, 3)]["retention_probability"], 1.0)
            self.assertEqual(rows[(candidate, 10)]["crop_miss_rate"], 0.0)
        summary = {
            row["candidate"]: row for row in report["candidate_source_summary"]
        }
        self.assertGreater(summary["i1_nocrop_640x384"]["mean_pad_ratio"], 0.0)
        self.assertAlmostEqual(
            summary["i2_nocrop_640x480"]["mean_aspect_distortion"], 0.0, places=2
        )

    def test_monte_carlo_floor_is_enforced(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 20"):
            audit_transforms([], draws=19)


class RareClassSourceScheduleTests(unittest.TestCase):
    def test_rcs_preserves_source_quota_and_is_deterministic(self) -> None:
        groups = {"rellis3d": [0, 1, 2], "adom_zed2i": [3]}
        weights = {"rellis3d": 0.75, "adom_zed2i": 0.25}
        presence = {0: set(), 1: {3}, 2: set(), 3: {3}}
        first = RareClassSourceSchedule(
            groups,
            weights,
            presence,
            [3],
            seed=42,
            rare_probability=1.0,
        )
        second = RareClassSourceSchedule(
            groups,
            weights,
            presence,
            [3],
            seed=42,
            rare_probability=1.0,
        )
        import itertools

        draws = list(itertools.islice(iter(first), 40))
        self.assertEqual(first.source_counts(40), Counter({"rellis3d": 30, "adom_zed2i": 10}))
        self.assertEqual(draws, list(itertools.islice(iter(second), 40)))
        self.assertEqual(set(draws), {1, 3})


if __name__ == "__main__":
    unittest.main()
