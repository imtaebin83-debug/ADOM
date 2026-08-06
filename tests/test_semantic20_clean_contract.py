from __future__ import annotations

import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from adom.runtime import semantic20_cycle


def _write_pair(root: Path, key: str, value: int) -> tuple[str, str]:
    stem = key.replace("/", "_")
    image_rel = f"fixture/images/{stem}.jpg"
    mask_rel = f"fixture/masks/{stem}.png"
    image_path = root / image_rel
    mask_path = root / mask_rel
    image_path.parent.mkdir(parents=True, exist_ok=True)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((2, 3, 3), dtype=np.uint8)).save(image_path)
    mask = np.full((2, 3), value, dtype=np.uint8)
    mask[0, 0] = 255
    Image.fromarray(mask, mode="L").save(mask_path)
    return image_rel, mask_rel


class Semantic20CleanContractTests(unittest.TestCase):
    def test_e2_support_and_digest_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "e2"
            reference = Path(directory) / "reference"
            (root / "splits").mkdir(parents=True)
            (root / "results").mkdir()
            reference.mkdir()
            (root / "_SUCCESS").write_text("PASS\n", encoding="utf-8")
            (root / "results" / "final_check.json").write_text(
                json.dumps({"status": "PASS"}), encoding="utf-8"
            )
            splits = {
                "train": ["rellis3d/a", "rugd/b", "ycor/c", "goose/d"],
                "val": ["rellis3d/e"],
                "test": ["rellis3d/f"],
            }
            references = {"train": ["a"], "val": ["e"], "test": ["f"]}
            rows = []
            for split, keys in splits.items():
                (root / "splits" / f"{split}.txt").write_text(
                    "\n".join(keys), encoding="utf-8"
                )
                (reference / f"{split}.txt").write_text(
                    "\n".join(references[split]), encoding="utf-8"
                )
                for index, key in enumerate(keys):
                    image_rel, mask_rel = _write_pair(root, key, index + 1)
                    rows.append((key, image_rel, mask_rel))
            (root / "manifest.csv").write_text(
                "sample_key,image_path,mask_path\n"
                + "".join(f"{key},{image},{mask}\n" for key, image, mask in rows),
                encoding="utf-8",
            )

            with patch.object(
                semantic20_cycle, "REFERENCE_SPLITS", reference
            ), patch.dict(
                semantic20_cycle.EXPECTED_SPLIT_COUNTS,
                {"e1": {"train": 3, "val": 1, "test": 1}},
                clear=False,
            ), patch.dict(
                semantic20_cycle.CANONICAL_EVAL_COUNTS,
                {"val": 1, "test": 1},
                clear=True,
            ), patch.object(
                semantic20_cycle,
                "EXPECTED_E1_MAIN_SOURCE_COUNTS",
                Counter({"rellis3d": 3, "rugd": 1, "ycor": 1}),
            ), patch.object(
                semantic20_cycle,
                "EXPECTED_E1_MANIFEST_SOURCE_COUNTS",
                Counter({"rellis3d": 3, "rugd": 1, "ycor": 1}),
            ):
                report = semantic20_cycle.validate_semantic20_dataset(root, "e2")

            self.assertEqual(report["split_counts"], {"train": 4, "val": 1, "test": 1})
            self.assertEqual(
                report["class_support"]["by_source_split"]["goose/train"][
                    "sample_count"
                ],
                1,
            )
            self.assertEqual(
                report["class_support"]["by_split"]["train"]["classes"][1][
                    "pixels"
                ],
                5,
            )
            self.assertEqual(len(report["split_contract_sha256"]), 64)
            self.assertIn("goose_direct_mapping.yaml", report["mapping_sha256"])
            self.assertEqual(len(report["dataset_images_sha256"]), 64)
            self.assertEqual(len(report["dataset_masks_sha256"]), 64)
            self.assertEqual(len(report["dataset_content_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
