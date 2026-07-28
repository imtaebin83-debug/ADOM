from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from adom.data.io import sha256_file
from adom.data.models import DatasetError, ValidationReport
from adom.data.packaging import create_deterministic_tar, verify_archive_checksum
from adom.data.pipeline import inspect_dataset, prepare_dataset
from adom.data.schema import LabelSchema
from adom.data.splits import load_splits
from adom.data.validation import validate_package


REPO_ROOT = Path(__file__).resolve().parents[1]
MAPPING = REPO_ROOT / "configs" / "datasets" / "rellis3d" / "label_mapping.yaml"


class SyntheticRellis:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.sequence = root / "Rellis-3D" / "00000"
        self.images = self.sequence / "pylon_camera_node"
        self.masks = self.sequence / "pylon_camera_node_label_id"
        self.splits = root / "splits"
        self.images.mkdir(parents=True)
        self.masks.mkdir(parents=True)
        self.splits.mkdir()

    def add_pair(self, stem: str, mask: np.ndarray | None = None) -> str:
        if mask is None:
            mask = np.array([[10, 1, 19], [4, 0, 23]], dtype=np.uint8)
        rgb = np.full((*mask.shape, 3), 100, dtype=np.uint8)
        Image.fromarray(rgb, mode="RGB").save(self.images / f"{stem}.jpg")
        Image.fromarray(mask).save(self.masks / f"{stem}.png")
        return f"00000_{stem}"

    def write_splits(self, train: list[str], val: list[str], test: list[str]) -> None:
        for name, values in (("train", train), ("val", val), ("test", test)):
            (self.splits / f"{name}.txt").write_text(
                "".join(f"{value}\n" for value in values),
                encoding="utf-8",
            )


class DataPipelineTests(unittest.TestCase):
    def test_indexed_remap_preserves_ignore(self) -> None:
        schema = LabelSchema.from_path(MAPPING)
        source = np.array([[10, 1, 19, 4, 0]], dtype=np.uint8)
        target = schema.remap(source)
        np.testing.assert_array_equal(
            target,
            np.array([[0, 1, 2, 3, 255]], dtype=np.uint8),
        )

    def test_prepare_validate_and_deterministic_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = SyntheticRellis(root / "raw")
            ids = [fixture.add_pair(f"frame{index:06d}") for index in range(3)]
            fixture.write_splits([ids[0]], [ids[1]], [ids[2]])
            first = root / "prepared-a"
            second = root / "prepared-b"
            for output in (first, second):
                report = prepare_dataset(
                    dataset="rellis3d",
                    input_root=fixture.root,
                    output_root=output,
                    mapping_path=MAPPING,
                    split_root=fixture.splits,
                    version="v2.0-test",
                )
                self.assertTrue(report.passed, report.to_dict())
                self.assertTrue(validate_package(output).passed)

            self.assertEqual(
                (first / "SHA256SUMS.txt").read_bytes(),
                (second / "SHA256SUMS.txt").read_bytes(),
            )
            with Image.open(
                first / "annotations" / "train" / f"{ids[0]}.png"
            ) as mask:
                array = np.asarray(mask)
                self.assertEqual(mask.mode, "L")
            self.assertEqual(array.dtype, np.uint8)
            self.assertEqual(set(np.unique(array)), {0, 1, 2, 3, 255})
            manifest = (
                first / "metadata" / "manifest_train.csv"
            ).read_text(encoding="utf-8")
            self.assertNotIn(str(root), manifest)
            self.assertNotIn("\\", manifest)
            archive, checksum = create_deterministic_tar(
                first,
                root / "rellis3d-cost4-v2-a.tar",
            )
            verify_archive_checksum(archive, checksum)
            second_archive, _ = create_deterministic_tar(
                second,
                root / "rellis3d-cost4-v2-b.tar",
            )
            self.assertEqual(
                sha256_file(archive),
                sha256_file(second_archive),
            )

    def test_unknown_source_id_fails_inspection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = SyntheticRellis(Path(directory))
            fixture.add_pair("frame000001", np.array([[250]], dtype=np.uint8))
            report = inspect_dataset("rellis3d", fixture.root, MAPPING)
            self.assertFalse(report.passed)
            self.assertIn("unknown_source_id", {issue.code for issue in report.errors})

    def test_missing_pair_is_not_silently_dropped(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = SyntheticRellis(Path(directory))
            Image.new("RGB", (2, 2)).save(fixture.images / "orphan.jpg")
            report = inspect_dataset("rellis3d", fixture.root, MAPPING)
            codes = {issue.code for issue in report.errors}
            self.assertIn("image_without_mask", codes)
            self.assertIn("no_samples", codes)

    def test_split_overlap_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample_id = "00000_frame000001"
            for split in ("train", "val", "test"):
                (root / f"{split}.txt").write_text(
                    sample_id + "\n", encoding="utf-8"
                )
            report = ValidationReport(dataset="rellis3d")
            load_splits(root, report)
            self.assertIn("split_overlap", {issue.code for issue in report.errors})

    def test_duplicate_within_split_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample_id = "00000_frame000001"
            (root / "train.txt").write_text(
                f"{sample_id}\n{sample_id}\n", encoding="utf-8"
            )
            (root / "val.txt").write_text(
                "00000_frame000002\n", encoding="utf-8"
            )
            (root / "test.txt").write_text(
                "00000_frame000003\n", encoding="utf-8"
            )
            report = ValidationReport(dataset="rellis3d")
            load_splits(root, report)
            self.assertIn("duplicate_split_id", {issue.code for issue in report.errors})

    def test_rgb_mask_size_mismatch_fails_prepare(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = SyntheticRellis(root / "raw")
            ids = [fixture.add_pair(f"frame{index:06d}") for index in range(3)]
            fixture.write_splits([ids[0]], [ids[1]], [ids[2]])
            Image.new("RGB", (8, 8)).save(
                fixture.images / "frame000000.jpg"
            )
            with self.assertRaises(DatasetError):
                prepare_dataset(
                    "rellis3d",
                    fixture.root,
                    root / "prepared",
                    MAPPING,
                    fixture.splits,
                    "v2.0-test",
                )

    def test_all_ignore_target_fails_prepare(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = SyntheticRellis(root / "raw")
            ids = [
                fixture.add_pair(
                    f"frame{index:06d}",
                    np.zeros((2, 2), dtype=np.uint8),
                )
                for index in range(3)
            ]
            fixture.write_splits([ids[0]], [ids[1]], [ids[2]])
            with self.assertRaises(DatasetError):
                prepare_dataset(
                    "rellis3d",
                    fixture.root,
                    root / "prepared",
                    MAPPING,
                    fixture.splits,
                    "v2.0-test",
                )

    def test_validation_rejects_absolute_manifest_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = SyntheticRellis(root / "raw")
            ids = [fixture.add_pair(f"frame{index:06d}") for index in range(3)]
            fixture.write_splits([ids[0]], [ids[1]], [ids[2]])
            output = root / "prepared"
            prepare_dataset(
                "rellis3d",
                fixture.root,
                output,
                MAPPING,
                fixture.splits,
                "v2.0-test",
            )
            manifest = output / "metadata" / "manifest_train.csv"
            text = manifest.read_text(encoding="utf-8")
            manifest.write_text(
                text.replace(
                    f"images/train/{ids[0]}.jpg",
                    "C:/private/rellis/image.jpg",
                ),
                encoding="utf-8",
            )
            report = validate_package(output, verify_checksums=False)
            self.assertIn(
                "nonportable_manifest_path",
                {issue.code for issue in report.errors},
            )

    def test_validation_requires_qc_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = SyntheticRellis(root / "raw")
            ids = [fixture.add_pair(f"frame{index:06d}") for index in range(3)]
            fixture.write_splits([ids[0]], [ids[1]], [ids[2]])
            output = root / "prepared"
            prepare_dataset(
                "rellis3d",
                fixture.root,
                output,
                MAPPING,
                fixture.splits,
                "v2.0-test",
            )
            (output / "reports" / "class_statistics.csv").write_text(
                "wrong\n", encoding="utf-8"
            )
            report = validate_package(output, verify_checksums=False)
            self.assertIn("qc_schema", {issue.code for issue in report.errors})

    def test_validation_rejects_multichannel_target_and_unlisted_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = SyntheticRellis(root / "raw")
            ids = [fixture.add_pair(f"frame{index:06d}") for index in range(3)]
            fixture.write_splits([ids[0]], [ids[1]], [ids[2]])
            output = root / "prepared"
            prepare_dataset(
                "rellis3d",
                fixture.root,
                output,
                MAPPING,
                fixture.splits,
                "v2.0-test",
            )
            target = output / "annotations" / "train" / f"{ids[0]}.png"
            with Image.open(target) as source:
                array = np.asarray(source)
            Image.fromarray(np.repeat(array[:, :, None], 3, axis=2)).save(target)
            (output / "reports" / "stale.txt").write_text(
                "stale", encoding="utf-8"
            )
            report = validate_package(output, verify_checksums=True)
            codes = {issue.code for issue in report.errors}
            self.assertIn("invalid_mask_channel", codes)
            self.assertIn("checksum", codes)


if __name__ == "__main__":
    unittest.main()
