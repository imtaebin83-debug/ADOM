from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image


PAPER_EVAL = Path(__file__).resolve().parents[1] / "tools" / "paper_eval"
sys.path.insert(0, str(PAPER_EVAL))

from _common import (  # noqa: E402
    CLASS_INDEX,
    confusion_from_arrays,
    metrics_from_confusion,
)
import bootstrap_ci  # noqa: E402
import build_manifests  # noqa: E402
import evaluate_checkpoint  # noqa: E402


class MetricContractTests(unittest.TestCase):
    def test_ignore_and_absent_class_false_positive_contract(self) -> None:
        gt = np.asarray([[1, 1], [255, 1]], dtype=np.uint8)
        pred = np.asarray([[1, 3], [3, 1]], dtype=np.uint8)
        confusion, ignored = confusion_from_arrays(gt, pred)
        summary, rows = metrics_from_confusion(
            confusion, ignored_pixels=ignored, common_classes=["grass"]
        )
        grass = rows[CLASS_INDEX["grass"]]
        pole = rows[CLASS_INDEX["pole"]]
        self.assertEqual(summary["total_evaluated_pixels"], 3)
        self.assertEqual(summary["ignored_pixels"], 1)
        self.assertAlmostEqual(summary["aAcc"], 200 / 3)
        self.assertAlmostEqual(grass["iou"], 200 / 3)
        self.assertIsNone(pole["iou"])
        self.assertEqual(pole["absent_class_false_positive"], 1)
        # The prediction on the ignored pixel is not a false positive.
        self.assertEqual(pole["prediction_pixel_count"], 1)

    def test_invalid_ground_truth_id_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "invalid Semantic20"):
            confusion_from_arrays(np.asarray([[19]]), np.asarray([[0]]))


class EvaluatorArgumentTests(unittest.TestCase):
    def test_manifest_split_defaults_to_test_and_accepts_train(self) -> None:
        required = [
            "--audit-dir",
            "audit",
            "--output-dir",
            "output",
            "--dataset",
            "korean",
            "--model",
            "b0_e0",
            "--manifest",
            "manifest.csv",
            "--config",
            "config.py",
            "--checkpoint",
            "checkpoint.pth",
        ]
        self.assertEqual(evaluate_checkpoint.parse_args(required).manifest_split, "test")
        self.assertEqual(
            evaluate_checkpoint.parse_args(
                [*required, "--manifest-split", "train"]
            ).manifest_split,
            "train",
        )


def _write_package(root: Path, samples: list[tuple[str, str, int]]) -> None:
    rows: list[dict[str, str]] = []
    split_values: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for sample_key, split, pixel in samples:
        image = root / "images" / f"{sample_key}.png"
        mask = root / "masks" / f"{sample_key}.png"
        image.parent.mkdir(parents=True, exist_ok=True)
        mask.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.full((2, 2, 3), pixel, dtype=np.uint8), mode="RGB").save(image)
        Image.fromarray(np.full((2, 2), 10, dtype=np.uint8), mode="L").save(mask)
        rows.append(
            {
                "sample_key": sample_key,
                "source": "adom_zed2i",
                "source_split": split,
                "image_path": image.relative_to(root).as_posix(),
                "mask_path": mask.relative_to(root).as_posix(),
            }
        )
        split_values[split].append(sample_key)
    with (root / "manifest.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("sample_key", "source", "source_split", "image_path", "mask_path"),
        )
        writer.writeheader()
        writer.writerows(rows)
    split_root = root / "splits"
    split_root.mkdir()
    (split_root / "ta1_train.txt").write_text("\n".join(split_values["train"]) + "\n")
    (split_root / "adom_val_diagnostic.txt").write_text(
        "\n".join(split_values["val"]) + "\n"
    )
    (split_root / "adom_test_diagnostic.txt").write_text(
        "\n".join(split_values["test"]) + "\n"
    )


def _write_rellis(root: Path) -> None:
    image = root / "images" / "00004" / "frame.jpg"
    mask = root / "masks" / "00004" / "frame.png"
    image.parent.mkdir(parents=True)
    mask.parent.mkdir(parents=True)
    Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8), mode="RGB").save(image)
    Image.fromarray(np.full((2, 2), 10, dtype=np.uint8), mode="L").save(mask)
    (root / "splits").mkdir()
    (root / "splits" / "test.txt").write_text("00004/frame\n")


def _manifest_args(rellis: Path, korean: Path, output: Path) -> argparse.Namespace:
    return argparse.Namespace(
        rellis_root=rellis,
        korean_root=korean,
        output_dir=output,
        rellis_manifest="manifest.csv",
        korean_manifest="manifest.csv",
        rellis_test_split="splits/test.txt",
        korean_train_split="splits/ta1_train.txt",
        korean_val_split="splits/adom_val_diagnostic.txt",
        korean_test_split="splits/adom_test_diagnostic.txt",
        korean_source="adom_zed2i",
        sequence_contract=None,
        allow_count_mismatch=True,
    )


class ManifestAuditTests(unittest.TestCase):
    def test_reused_basenames_in_separate_sequences_are_warnings(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rellis, korean, output = root / "rellis", root / "korean", root / "out"
            _write_rellis(rellis)
            _write_package(
                korean,
                [
                    ("adom_zed2i/train_seq/capture/frame_000001", "train", 1),
                    ("adom_zed2i/val_seq/capture/frame_000001", "val", 2),
                    ("adom_zed2i/test_seq/capture/frame_000001", "test", 3),
                ],
            )
            result = build_manifests.build(_manifest_args(rellis, korean, output))
            self.assertEqual(result["status"], "PASS")
            self.assertTrue(any("basename" in value for value in result["warnings"]))
            self.assertEqual(result["blockers"], [])

    def test_exact_train_test_image_is_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rellis, korean, output = root / "rellis", root / "korean", root / "out"
            _write_rellis(rellis)
            _write_package(
                korean,
                [
                    ("adom_zed2i/train_seq/capture/frame_000001", "train", 7),
                    ("adom_zed2i/val_seq/capture/frame_000002", "val", 8),
                    ("adom_zed2i/test_seq/capture/frame_000003", "test", 7),
                ],
            )
            result = build_manifests.build(_manifest_args(rellis, korean, output))
            self.assertEqual(result["status"], "BLOCKED")
            self.assertTrue(any("held-out leakage" in value for value in result["blockers"]))


class BootstrapTests(unittest.TestCase):
    def test_sequence_bootstrap_is_deterministic_and_paired(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline_path = root / "baseline.npz"
            eadom_path = root / "eadom.npz"
            sample_ids = np.asarray(["a1", "a2", "b1", "b2"])
            sequences = np.asarray(["a", "a", "b", "b"])
            baseline = np.zeros((4, 19, 19), dtype=np.int64)
            eadom = np.zeros_like(baseline)
            for index in range(4):
                baseline[index, CLASS_INDEX["log"], CLASS_INDEX["log"]] = 5
                baseline[index, CLASS_INDEX["log"], CLASS_INDEX["grass"]] = 5
                eadom[index, CLASS_INDEX["log"], CLASS_INDEX["log"]] = 8
                eadom[index, CLASS_INDEX["log"], CLASS_INDEX["grass"]] = 2
                baseline[index, CLASS_INDEX["rubble"], CLASS_INDEX["rubble"]] = 6
                baseline[index, CLASS_INDEX["rubble"], CLASS_INDEX["grass"]] = 4
                eadom[index, CLASS_INDEX["rubble"], CLASS_INDEX["rubble"]] = 7
                eadom[index, CLASS_INDEX["rubble"], CLASS_INDEX["grass"]] = 3
            common = np.asarray(["log", "rubble"])
            for path, confusions in ((baseline_path, baseline), (eadom_path, eadom)):
                np.savez_compressed(
                    path,
                    sample_ids=sample_ids,
                    sequences=sequences,
                    confusions=confusions,
                    ignored_pixels=np.zeros(4, dtype=np.int64),
                    common_classes=common,
                )
            args = argparse.Namespace(
                dataset="korean",
                baseline=baseline_path,
                eadom=eadom_path,
                output_dir=root / "first",
                samples=10_000,
                seed=42,
                batch_size=500,
            )
            first = bootstrap_ci.bootstrap(args)
            args.output_dir = root / "second"
            second = bootstrap_ci.bootstrap(args)
            self.assertEqual(first["resampling_unit"], "sequence")
            self.assertEqual(first["results"], second["results"])
            log = next(row for row in first["results"] if row["metric"] == "log/IoU")
            self.assertEqual(log["status"], "PASS")
            self.assertGreater(log["delta_eadom_minus_b0_e0"], 0)


if __name__ == "__main__":
    unittest.main()
