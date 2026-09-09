from __future__ import annotations

import csv
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    REPO_ROOT
    / "scripts"
    / "data"
    / "github_dependent"
    / "prepare_adom_semantic20_v2.py"
)
SPEC = importlib.util.spec_from_file_location(
    "prepare_adom_semantic20_v2",
    SCRIPT_PATH,
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot load integration script: {SCRIPT_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_pair(root: Path, condition: str, frame: str, mask_id: int = 10) -> None:
    relative = Path(condition) / "seq01" / f"{frame}.png"
    image_path = root / "images" / relative
    mask_path = root / "masks" / relative
    image_path.parent.mkdir(parents=True, exist_ok=True)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 3), color=(1, 2, 3)).save(image_path)
    Image.fromarray(np.full((3, 4), mask_id, dtype=np.uint8), mode="L").save(
        mask_path
    )


def selection_row(condition: str, frame: str) -> dict[str, str]:
    sample_id = f"{condition}_seq01_{frame}"
    return {
        "sample_id": sample_id,
        "condition": condition,
        "sequence_id": "seq01",
        "frame": frame,
        "relative_path": f"images/{condition}/seq01/{frame}.png",
        "status": "OK",
    }


def create_release(root: Path, rows: list[dict[str, str]]) -> None:
    for row in rows:
        save_pair(root, row["condition"], row["frame"])
    write_csv(
        root / "metadata" / "selection.csv",
        MODULE.SELECTION_FIELDS,
        rows,
    )
    write_csv(
        root / "metadata" / "exclusions.csv",
        MODULE.EXCLUSION_FIELDS,
        [],
    )


def create_base_output(root: Path) -> None:
    root.mkdir(parents=True)
    write_csv(
        root / "manifest.csv",
        MODULE.MANIFEST_FIELDS,
        [
            {
                "sample_key": "rellis3d/train/base",
                "source": "rellis3d",
                "source_split": "train",
                "output_split": "train",
                "sample_id": "base",
                "image_path": "images/rellis3d/train/base.jpg",
                "mask_path": "masks/rellis3d/train/base.png",
                "non_ignore_ratio": "1.00000000",
            }
        ],
    )
    MODULE.write_split(root / "splits" / "train.txt", ["rellis3d/train/base"])
    MODULE.write_split(root / "splits" / "val.txt", [])
    MODULE.write_split(root / "splits" / "test.txt", [])


class AdomV2ReleaseImportTests(unittest.TestCase):
    def test_release_import_uses_relative_paths_for_duplicate_basenames(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            release = root / "release"
            rows = [
                selection_row("P1", "frame_000002"),
                selection_row("P3", "frame_000002"),
                selection_row("P2", "frame_000003"),
            ]
            create_release(release, rows)
            split_csv = root / "splits.csv"
            write_csv(
                split_csv,
                MODULE.SPLIT_ASSIGNMENT_FIELDS,
                [
                    {"sample_id": rows[0]["sample_id"], "split": "train"},
                    {"sample_id": rows[1]["sample_id"], "split": "val"},
                    {"sample_id": rows[2]["sample_id"], "split": "test"},
                ],
            )
            output = root / "output"
            create_base_output(output)

            samples = MODULE.load_adom_release_samples(release, split_csv)
            counts = MODULE.add_adom_v2_release(samples, output, "diagnostic")

            self.assertEqual(counts, {"train": 1, "val": 1, "test": 1})
            self.assertTrue(
                output.joinpath(
                    "images/adom_v2/train/P1/seq01/frame_000002.png"
                ).is_file()
            )
            self.assertTrue(
                output.joinpath(
                    "images/adom_v2/val/P3/seq01/frame_000002.png"
                ).is_file()
            )
            diagnostic = MODULE.read_split(
                output / "splits" / "adom_v2_val_diagnostic.txt"
            )
            self.assertEqual(diagnostic, [f"adom_v2/val/{rows[1]['sample_id']}"])

    def test_release_requires_explicit_split_csv(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            release = Path(directory) / "release"
            create_release(release, [selection_row("P1", "frame_000001")])
            with self.assertRaisesRegex(RuntimeError, "never infers|no published"):
                MODULE.load_adom_release_samples(release, None)

    def test_selection_and_exclusions_must_not_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            release = root / "release"
            rows = [
                selection_row("P1", "frame_000001"),
                selection_row("P2", "frame_000002"),
                selection_row("P3", "frame_000003"),
            ]
            create_release(release, rows)
            write_csv(
                release / "metadata" / "exclusions.csv",
                MODULE.EXCLUSION_FIELDS,
                [
                    {
                        **{field: rows[0].get(field, "") for field in MODULE.EXCLUSION_FIELDS},
                        "reason": "duplicate",
                        "notes": "test",
                    }
                ],
            )
            split_csv = root / "splits.csv"
            write_csv(
                split_csv,
                MODULE.SPLIT_ASSIGNMENT_FIELDS,
                [
                    {"sample_id": rows[0]["sample_id"], "split": "train"},
                    {"sample_id": rows[1]["sample_id"], "split": "val"},
                    {"sample_id": rows[2]["sample_id"], "split": "test"},
                ],
            )
            with self.assertRaisesRegex(RuntimeError, "selection/exclusions overlap"):
                MODULE.load_adom_release_samples(release, split_csv)

    def test_condition_sequence_cannot_leak_across_splits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            release = root / "release"
            rows = [
                selection_row("P1", "frame_000001"),
                selection_row("P1", "frame_000002"),
                selection_row("P2", "frame_000003"),
                selection_row("P3", "frame_000004"),
            ]
            create_release(release, rows)
            split_csv = root / "splits.csv"
            write_csv(
                split_csv,
                MODULE.SPLIT_ASSIGNMENT_FIELDS,
                [
                    {"sample_id": rows[0]["sample_id"], "split": "train"},
                    {"sample_id": rows[1]["sample_id"], "split": "val"},
                    {"sample_id": rows[2]["sample_id"], "split": "test"},
                    {"sample_id": rows[3]["sample_id"], "split": "train"},
                ],
            )
            with self.assertRaisesRegex(RuntimeError, "multiple splits"):
                MODULE.load_adom_release_samples(release, split_csv)

    def test_release_mask_ids_are_validated_before_import(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            release = root / "release"
            rows = [
                selection_row("P1", "frame_000001"),
                selection_row("P2", "frame_000002"),
                selection_row("P3", "frame_000003"),
            ]
            create_release(release, rows)
            save_pair(release, "P1", "frame_000001", mask_id=99)
            split_csv = root / "splits.csv"
            write_csv(
                split_csv,
                MODULE.SPLIT_ASSIGNMENT_FIELDS,
                [
                    {"sample_id": rows[0]["sample_id"], "split": "train"},
                    {"sample_id": rows[1]["sample_id"], "split": "val"},
                    {"sample_id": rows[2]["sample_id"], "split": "test"},
                ],
            )
            output = root / "output"
            create_base_output(output)
            samples = MODULE.load_adom_release_samples(release, split_csv)
            with self.assertRaisesRegex(RuntimeError, "Unexpected ADOM-v2 class IDs"):
                MODULE.add_adom_v2_release(samples, output, "diagnostic")


if __name__ == "__main__":
    unittest.main()
