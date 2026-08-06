from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from adom.runtime.semantic20_aggregate import aggregate_paired_runs


class Semantic20PairedAggregateTests(unittest.TestCase):
    def _summaries(
        self,
        root: Path,
        experiment: str,
        values: tuple[float, float, float],
    ) -> list[Path]:
        paths = []
        for seed, value in zip((42, 43, 44), values, strict=True):
            path = root / f"{experiment}-{seed}.json"
            path.write_text(
                json.dumps(
                    {
                        "experiment": experiment,
                        "seed": seed,
                        "gate": "full",
                        "dataset_contract": {
                            "dataset_content_sha256": f"{experiment}-digest"
                        },
                        "models": [
                            {
                                "model": model,
                                "validation_selection": {
                                    "metrics": {
                                        "semantic20/mIoU/AugmentedRisk2": value,
                                        "semantic20/mIoU/ValSupported13": 50.0 + value,
                                        "semantic20/IoU/pole": value,
                                    }
                                },
                            }
                            for model in ("b0", "b2")
                        ],
                    }
                ),
                encoding="utf-8",
            )
            paths.append(path)
        return paths

    def test_paired_mean_std_and_delta(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = self._summaries(root, "e1", (10.0, 11.0, 12.0))
            candidate = self._summaries(root, "e2", (13.0, 15.0, 17.0))
            result = aggregate_paired_runs(baseline, candidate)
            metric = result["models"]["b0"]["metrics"][
                "semantic20/mIoU/AugmentedRisk2"
            ]
            self.assertEqual(metric["baseline"]["mean"], 11.0)
            self.assertEqual(metric["candidate"]["mean"], 15.0)
            self.assertEqual(metric["paired_delta"]["mean"], 4.0)
            self.assertEqual(metric["paired_delta"]["std"], 1.0)
            self.assertTrue(
                result["models"]["b0"]["success_gates"]["primary_gates_passed"]
            )

    def test_requires_exact_paired_seeds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = self._summaries(root, "e1", (10.0, 11.0, 12.0))[:2]
            candidate = self._summaries(root, "e2", (13.0, 15.0, 17.0))
            with self.assertRaisesRegex(ValueError, "Expected paired seeds"):
                aggregate_paired_runs(baseline, candidate)

    def test_rejects_mixed_dataset_digest_within_condition(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = self._summaries(root, "e1", (10.0, 11.0, 12.0))
            changed = json.loads(baseline[1].read_text(encoding="utf-8"))
            changed["dataset_contract"]["dataset_content_sha256"] = "different"
            baseline[1].write_text(json.dumps(changed), encoding="utf-8")
            candidate = self._summaries(root, "e2", (13.0, 15.0, 17.0))
            with self.assertRaisesRegex(ValueError, "dataset content digest"):
                aggregate_paired_runs(baseline, candidate)


if __name__ == "__main__":
    unittest.main()
