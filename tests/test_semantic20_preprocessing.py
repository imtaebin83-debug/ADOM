from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from adom.data.semantic20 import resource_path
from adom.runtime import semantic20_cycle


converter = importlib.import_module(
    "data.semantic_20.scripts.01_convert_bridge_sources"
)


def save_rgb(path: Path, value: tuple[int, int, int], size: tuple[int, int] = (3, 2)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    array[:, :] = value
    Image.fromarray(array, mode="RGB").save(path)


def save_index(path: Path, value: int, size: tuple[int, int] = (3, 2)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((size[1], size[0]), value, dtype=np.uint8), mode="L").save(path)


class Semantic20PreprocessingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bridge_path = resource_path("semantic_20", "config", "bridge_mapping.yaml")
        cls.rugd_map_path = resource_path("rugd", "config", "label_mapping.json")
        cls.ycor_map_path = resource_path("ycor", "config", "label_mapping.json")
        cls.bridge = converter.load_bridge_config(cls.bridge_path)
        cls.indexed_mapping = converter.load_source_to_target(cls.bridge, "rugd")
        cls.rgb_mapping = converter.load_rugd_rgb_to_target(
            cls.rugd_map_path, cls.bridge
        )

    def test_official_rugd_rgb_composes_by_class_name(self) -> None:
        expected = {
            (0, 102, 0): 1,
            (0, 255, 0): 2,
            (0, 128, 255): 4,
            (0, 0, 255): 5,
            (64, 64, 64): 8,
            (255, 0, 0): 9,
            (204, 153, 255): 11,
            (255, 153, 204): 13,
            (102, 102, 0): 18,
            (108, 64, 20): 255,
        }
        for rgb, target in expected.items():
            key = (rgb[0] << 16) | (rgb[1] << 8) | rgb[2]
            self.assertEqual(self.rgb_mapping[key], target)

    def test_rgb_and_indexed_masks_are_supported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rgb_path = root / "rgb.png"
            indexed_path = root / "indexed.png"
            pixels = np.asarray([[(0, 102, 0), (102, 102, 0)]], dtype=np.uint8)
            Image.fromarray(pixels, mode="RGB").save(rgb_path)
            save_index(indexed_path, 3, (2, 1))
            rgb_target = converter.load_rugd_target_mask(
                rgb_path,
                mask_mode="auto",
                rgb_to_target=self.rgb_mapping,
                indexed_mapping=self.indexed_mapping,
            )
            indexed_target = converter.load_rugd_target_mask(
                indexed_path,
                mask_mode="auto",
                rgb_to_target=self.rgb_mapping,
                indexed_mapping=self.indexed_mapping,
            )
            np.testing.assert_array_equal(rgb_target, np.asarray([[1, 18]], dtype=np.uint8))
            np.testing.assert_array_equal(indexed_target, np.asarray([[1, 1]], dtype=np.uint8))
            self.assertEqual(rgb_target.dtype, np.uint8)

    def test_unknown_rugd_rgb_fails_fast(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unknown.png"
            save_rgb(path, (1, 2, 3))
            with self.assertRaisesRegex(ValueError, "Unknown RUGD RGB"):
                converter.load_rugd_target_mask(
                    path,
                    mask_mode="rgb",
                    rgb_to_target=self.rgb_mapping,
                    indexed_mapping=self.indexed_mapping,
                )

    def _rugd_fixture(self, root: Path, *, mismatch: bool = False) -> tuple[Path, Path, Path]:
        images = root / "images"
        masks = root / "masks"
        splits = root / "splits"
        splits.mkdir(parents=True)
        for split, sample in (("train", "a"), ("val", "b"), ("test", "c")):
            save_rgb(images / f"{sample}.png", (20, 30, 40))
            size = (4, 2) if mismatch and sample == "b" else (3, 2)
            save_rgb(masks / f"{sample}.png", (0, 102, 0), size)
            # Deliberately omit the final EOF newline for parser regression coverage.
            (splits / f"{split}.txt").write_text(sample, encoding="utf-8")
        return images, masks, splits

    def test_full_rugd_inventory_audit_and_eof_parsing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            images, masks, splits = self._rugd_fixture(Path(directory))
            _, _, values = converter.validate_rugd_inputs(
                image_root=images,
                mask_root=masks,
                split_root=splits,
                mask_mode="auto",
                rgb_to_target=self.rgb_mapping,
                indexed_mapping=self.indexed_mapping,
            )
            self.assertEqual(values, {"train": ["a"], "val": ["b"], "test": ["c"]})
            save_rgb(images / "unexpected.png", (20, 30, 40))
            with self.assertRaisesRegex(ValueError, "unexpected_images"):
                converter.validate_rugd_inputs(
                    image_root=images,
                    mask_root=masks,
                    split_root=splits,
                    mask_mode="auto",
                    rgb_to_target=self.rgb_mapping,
                    indexed_mapping=self.indexed_mapping,
                )

    def test_size_mismatch_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            images, masks, splits = self._rugd_fixture(Path(directory), mismatch=True)
            with self.assertRaisesRegex(ValueError, "size mismatch"):
                converter.validate_rugd_inputs(
                    image_root=images,
                    mask_root=masks,
                    split_root=splits,
                    mask_mode="auto",
                    rgb_to_target=self.rgb_mapping,
                    indexed_mapping=self.indexed_mapping,
                )

    def test_canonical_cli_is_importable(self) -> None:
        completed = subprocess.run(
            [sys.executable, "-m", "adom.data.semantic20", "--help"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--rugd-mask-mode", completed.stdout)

    def test_resume_preserves_completed_pair_and_publishes_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images, masks, splits = self._rugd_fixture(root / "rugd")
            ycor = root / "ycor"
            for split, sample in (("train", "y1"), ("valid", "y2")):
                sample_root = ycor / split / sample
                save_rgb(sample_root / "rgb.jpg", (20, 30, 40))
                save_rgb(sample_root / "labels.png", (128, 255, 0))

            published = root / "published"
            staging = root / "published.partial"
            completed_image = staging / "images" / "rugd" / "train" / "a.png"
            completed_mask = staging / "masks" / "rugd" / "train" / "a.png"
            completed_image.parent.mkdir(parents=True)
            completed_mask.parent.mkdir(parents=True)
            completed_image.write_bytes((images / "a.png").read_bytes())
            save_index(completed_mask, 1)
            old_time = 1_700_000_000
            os.utime(completed_image, (old_time, old_time))
            os.utime(completed_mask, (old_time, old_time))

            converter.main(
                [
                    "--rugd-image-root",
                    str(images),
                    "--rugd-mask-root",
                    str(masks),
                    "--rugd-split-root",
                    str(splits),
                    "--ycor-root",
                    str(ycor),
                    "--ycor-source-map",
                    str(self.ycor_map_path),
                    "--output-root",
                    str(published),
                    "--resume",
                ]
            )
            self.assertTrue((published / "_SUCCESS").is_file())
            self.assertFalse(staging.exists())
            self.assertEqual(
                int((published / "images" / "rugd" / "train" / "a.png").stat().st_mtime),
                old_time,
            )
            with Image.open(published / "masks" / "rugd" / "train" / "a.png") as mask:
                array = np.asarray(mask)
            self.assertEqual(array.dtype, np.uint8)
            self.assertEqual(array.ndim, 2)
            self.assertTrue(set(np.unique(array)).issubset(set(range(19)) | {255}))
            status = json.loads(
                (published / "results" / "status.json").read_text(encoding="utf-8")
            )
            self.assertEqual(status["status"], "PASS")

            with self.assertRaises(FileExistsError):
                converter.main(
                    [
                        "--rugd-image-root",
                        str(images),
                        "--rugd-mask-root",
                        str(masks),
                        "--ycor-root",
                        str(ycor),
                        "--ycor-source-map",
                        str(self.ycor_map_path),
                        "--output-root",
                        str(published),
                    ]
                )
            self.assertFalse(staging.exists())


class Semantic20DatasetContractTests(unittest.TestCase):
    def _write_pair(self, root: Path, key: str, *, value: int = 1) -> tuple[str, str]:
        image_rel = f"fixture/images/{key.replace('/', '_')}.jpg"
        mask_rel = f"fixture/masks/{key.replace('/', '_')}.png"
        save_rgb(root / image_rel, (10, 20, 30))
        save_index(root / mask_rel, value)
        return image_rel, mask_rel

    def test_e0_contract_reads_every_pair_and_rejects_invalid_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "e0"
            reference = Path(directory) / "reference"
            (root / "splits").mkdir(parents=True)
            reference.mkdir()
            (root / "_SUCCESS").write_text("PASS\n", encoding="utf-8")
            for split, key in (("train", "a"), ("val", "b"), ("test", "c")):
                (root / "splits" / f"{split}.txt").write_text(key, encoding="utf-8")
                (reference / f"{split}.txt").write_text(key, encoding="utf-8")
                save_rgb(root / "images" / f"{key}.jpg", (10, 20, 30))
                save_index(root / "masks" / f"{key}.png", 1)
            with patch.object(semantic20_cycle, "REFERENCE_SPLITS", reference), patch.dict(
                semantic20_cycle.EXPECTED_SPLIT_COUNTS,
                {"e0": {"train": 1, "val": 1, "test": 1}},
                clear=False,
            ):
                report = semantic20_cycle.validate_semantic20_dataset(root, "e0")
                self.assertEqual(report["verified_pairs"], 3)
                save_index(root / "masks" / "b.png", 99)
                with self.assertRaisesRegex(RuntimeError, "Invalid Semantic20 target"):
                    semantic20_cycle.validate_semantic20_dataset(root, "e0")

    def test_e1_contract_checks_manifest_final_check_and_source_policy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "e1"
            reference = Path(directory) / "reference"
            (root / "splits").mkdir(parents=True)
            (root / "results").mkdir()
            reference.mkdir()
            (root / "_SUCCESS").write_text("PASS\n", encoding="utf-8")
            (root / "results" / "final_check.json").write_text(
                json.dumps({"status": "PASS"}), encoding="utf-8"
            )
            splits = {
                "train": ["rellis3d/a", "rugd/b", "ycor/c"],
                "val": ["rellis3d/d"],
                "test": ["rellis3d/e"],
            }
            references = {"train": ["a"], "val": ["d"], "test": ["e"]}
            rows = []
            for split, keys in splits.items():
                (root / "splits" / f"{split}.txt").write_text(
                    "\n".join(keys), encoding="utf-8"
                )
                (reference / f"{split}.txt").write_text(
                    "\n".join(references[split]), encoding="utf-8"
                )
                for key in keys:
                    image_rel, mask_rel = self._write_pair(root, key)
                    rows.append((key, image_rel, mask_rel))
            manifest = root / "manifest.csv"
            manifest.write_text(
                "sample_key,image_path,mask_path\n"
                + "".join(f"{key},{image},{mask}\n" for key, image, mask in rows),
                encoding="utf-8",
            )
            with patch.object(semantic20_cycle, "REFERENCE_SPLITS", reference), patch.dict(
                semantic20_cycle.EXPECTED_SPLIT_COUNTS,
                {"e1": {"train": 3, "val": 1, "test": 1}},
                clear=False,
            ), patch.object(semantic20_cycle, "EXPECTED_E1_MANIFEST_COUNT", 5), patch.object(
                semantic20_cycle,
                "EXPECTED_E1_MAIN_SOURCE_COUNTS",
                Counter({"rellis3d": 3, "rugd": 1, "ycor": 1}),
            ), patch.object(
                semantic20_cycle,
                "EXPECTED_E1_MANIFEST_SOURCE_COUNTS",
                Counter({"rellis3d": 3, "rugd": 1, "ycor": 1}),
            ):
                report = semantic20_cycle.validate_semantic20_dataset(root, "e1")
                self.assertEqual(report["verified_pairs"], 5)
                (root / "results" / "final_check.json").write_text(
                    json.dumps({"status": "FAIL"}), encoding="utf-8"
                )
                with self.assertRaisesRegex(RuntimeError, "not PASS"):
                    semantic20_cycle.validate_semantic20_dataset(root, "e1")


if __name__ == "__main__":
    unittest.main()
