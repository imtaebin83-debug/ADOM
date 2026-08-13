from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from adom.data import target_adaptation
from adom.runtime import semantic20_cycle
from adom.runtime.source_sampling import WeightedSourceSchedule, integer_source_slots


HAS_TORCH = importlib.util.find_spec("torch") is not None


def save_pair(root: Path, key: str, value: int = 1) -> tuple[str, str]:
    image_rel = f"images/{key}.jpg"
    mask_rel = f"masks/{key}.png"
    image_path = root / image_rel
    mask_path = root / mask_rel
    image_path.parent.mkdir(parents=True, exist_ok=True)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((2, 3, 3), dtype=np.uint8)).save(image_path)
    Image.fromarray(np.full((2, 3), value, dtype=np.uint8), mode="L").save(mask_path)
    return image_rel, mask_rel


def write_manifest(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


class TargetAdaptationPackageTests(unittest.TestCase):
    def test_build_and_validate_shared_superset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            e1 = root / "e1"
            standalone = root / "standalone"
            output = root / "ta"
            for package in (e1, standalone):
                (package / "splits").mkdir(parents=True)
                (package / "_SUCCESS").write_text("PASS\n", encoding="utf-8")

            e1_samples = [
                ("rellis3d/a", "rellis3d", "train"),
                ("rellis3d/b", "rellis3d", "val"),
                ("rellis3d/c", "rellis3d", "test"),
                ("rugd/d", "rugd", "train"),
                ("ycor/e", "ycor", "train"),
            ]
            e1_rows = []
            for index, (key, source, split) in enumerate(e1_samples, start=1):
                image, mask = save_pair(e1, key, index)
                e1_rows.append(
                    {
                        "sample_key": key,
                        "source": source,
                        "source_split": split,
                        "image_path": image,
                        "mask_path": mask,
                    }
                )
            write_manifest(
                e1 / "manifest.csv",
                e1_rows,
                ["sample_key", "source", "source_split", "image_path", "mask_path"],
            )
            (e1 / "splits" / "train.txt").write_text(
                "rellis3d/a\nrugd/d\nycor/e\n", encoding="utf-8"
            )
            (e1 / "splits" / "val.txt").write_text("rellis3d/b\n", encoding="utf-8")
            (e1 / "splits" / "test.txt").write_text("rellis3d/c\n", encoding="utf-8")

            standalone_rows = []
            for index, (key, split) in enumerate(
                (("date/train/s1", "train"), ("date/val/s2", "val"), ("date/test/s3", "test")),
                start=10,
            ):
                image, mask = save_pair(standalone, key, index)
                standalone_rows.append(
                    {
                        "sample_key": key,
                        "split": split,
                        "image_path": image,
                        "mask_path": mask,
                    }
                )
                (standalone / "splits" / f"{split}.txt").write_text(
                    f"{key}\n", encoding="utf-8"
                )
            write_manifest(
                standalone / "manifest.csv",
                standalone_rows,
                ["sample_key", "split", "image_path", "mask_path"],
            )

            summary = target_adaptation.build_package(e1, standalone, output)
            self.assertEqual(summary["manifest_count"], 8)
            expected_counts = {
                "ta0_train": 1,
                "ta1_train": 2,
                "ta2_train": 4,
                "val": 1,
                "test": 1,
                "adom_val_diagnostic": 1,
                "adom_test_diagnostic": 1,
            }
            expected_sources = {
                "ta0_train": {"rellis3d"},
                "ta1_train": {"rellis3d", "adom_zed2i"},
                "ta2_train": {"rellis3d", "rugd", "ycor", "adom_zed2i"},
                "val": {"rellis3d"},
                "test": {"rellis3d"},
                "adom_val_diagnostic": {"adom_zed2i"},
                "adom_test_diagnostic": {"adom_zed2i"},
            }
            with patch.object(
                target_adaptation, "EXPECTED_SPLIT_COUNTS", expected_counts
            ), patch.object(
                target_adaptation, "EXPECTED_SPLIT_SOURCES", expected_sources
            ):
                report = target_adaptation.validate_package(output, write_success=True)
            self.assertEqual(report["status"], "PASS")
            self.assertTrue((output / "_SUCCESS").is_file())
            self.assertEqual(report["all_ignore_train_masks"], 0)

            reference = root / "reference"
            reference.mkdir()
            for split, value in {"train": "a", "val": "b", "test": "c"}.items():
                (reference / f"{split}.txt").write_text(f"{value}\n", encoding="utf-8")
            with patch.object(
                semantic20_cycle, "REFERENCE_SPLITS", reference
            ), patch.dict(
                semantic20_cycle.EXPECTED_SPLIT_COUNTS,
                {
                    "eadom": {"train": 2, "val": 1, "test": 1},
                    "ta0": {"train": 1, "val": 1, "test": 1},
                    "ta1": {"train": 2, "val": 1, "test": 1},
                    "ta2": {"train": 4, "val": 1, "test": 1},
                },
                clear=False,
            ), patch.object(
                semantic20_cycle,
                "EXPECTED_TA_MANIFEST_SOURCE_COUNTS",
                Counter({"rellis3d": 3, "rugd": 1, "ycor": 1, "adom_zed2i": 3}),
            ), patch.object(
                semantic20_cycle,
                "EXPECTED_TA_MAIN_SOURCE_COUNTS",
                {
                    "ta0": Counter({"rellis3d": 3}),
                    "ta1": Counter({"rellis3d": 3, "adom_zed2i": 1}),
                    "ta2": Counter(
                        {"rellis3d": 3, "rugd": 1, "ycor": 1, "adom_zed2i": 1}
                    ),
                },
            ):
                runtime_report = semantic20_cycle.validate_semantic20_dataset(
                    output, "ta2"
                )
            self.assertEqual(runtime_report["split_counts"]["train"], 4)
            self.assertEqual(runtime_report["verified_pairs"], 8)
            with patch.object(
                semantic20_cycle, "REFERENCE_SPLITS", reference
            ), patch.dict(
                semantic20_cycle.EXPECTED_SPLIT_COUNTS,
                {"eadom": {"train": 2, "val": 1, "test": 1}},
                clear=False,
            ), patch.object(
                semantic20_cycle,
                "EXPECTED_TA_MANIFEST_SOURCE_COUNTS",
                Counter({"rellis3d": 3, "rugd": 1, "ycor": 1, "adom_zed2i": 3}),
            ), patch.object(
                semantic20_cycle,
                "EXPECTED_TA_MAIN_SOURCE_COUNTS",
                {"ta1": Counter({"rellis3d": 3, "adom_zed2i": 1})},
            ):
                eadom_report = semantic20_cycle.validate_semantic20_dataset(
                    output, "eadom"
                )
            self.assertEqual(eadom_report["split_counts"]["train"], 2)
            self.assertEqual(
                eadom_report["class_support"]["by_source_split"][
                    "adom_zed2i/train"
                ]["sample_count"],
                1,
            )


class SourceWeightedScheduleTests(unittest.TestCase):
    def test_exact_weights_and_determinism(self) -> None:
        weights = {"rellis3d": 0.4375, "rugd": 0.25, "ycor": 0.0625, "adom_zed2i": 0.25}
        self.assertEqual(Counter(integer_source_slots(weights)), Counter({
            "rellis3d": 7, "rugd": 4, "ycor": 1, "adom_zed2i": 4
        }))
        groups = {
            "rellis3d": [0, 1],
            "rugd": [2, 3],
            "ycor": [4],
            "adom_zed2i": [5],
        }
        first = WeightedSourceSchedule(groups, weights, seed=42)
        second = WeightedSourceSchedule(groups, weights, seed=42)
        self.assertEqual(first.source_counts(160), Counter({
            "rellis3d": 70, "rugd": 40, "ycor": 10, "adom_zed2i": 40
        }))
        import itertools

        self.assertEqual(
            list(itertools.islice(iter(first), 100)),
            list(itertools.islice(iter(second), 100)),
        )
        uninterrupted = list(itertools.islice(iter(first), 200))
        resumed = list(itertools.islice(iter(second), 100, 200))
        self.assertEqual(uninterrupted[100:], resumed)


@unittest.skipUnless(HAS_TORCH, "checkpoint contract requires torch")
class WarmStartCheckpointTests(unittest.TestCase):
    def test_b0_semantic20_checkpoint_contract(self) -> None:
        import torch

        from adom.runtime.semantic20_cycle import validate_ta_initial_checkpoint

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "e0.pth"
            state = {
                "decode_head.conv_seg.weight": torch.zeros(19, 256, 1, 1),
            }
            for stage in range(4):
                for block in range(2):
                    state[f"backbone.layers.{stage}.1.{block}.norm1.weight"] = torch.zeros(1)
            torch.save({"state_dict": state}, path)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            report = validate_ta_initial_checkpoint(path, digest)
            self.assertEqual(report["architecture"], "segformer_b0")
            self.assertEqual(report["num_classes"], 19)
            with self.assertRaisesRegex(RuntimeError, "SHA-256 mismatch"):
                validate_ta_initial_checkpoint(path, "0" * 64)


if __name__ == "__main__":
    unittest.main()
