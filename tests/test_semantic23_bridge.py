from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "src" / "data" / "semantic_23"
CONVERTER = PACKAGE_ROOT / "scripts" / "01_convert_dataset_bridge.py"
BUILDER = PACKAGE_ROOT / "scripts" / "02_build_combined_package.py"
VALIDATOR = PACKAGE_ROOT / "scripts" / "03_validate_combined_package.py"
MAPPING = PACKAGE_ROOT / "config" / "bridge_mapping.yaml"


def load_converter():
    spec = importlib.util.spec_from_file_location("semantic23_converter", CONVERTER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {CONVERTER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


converter = load_converter()


def save_rgb(path: Path, shape: tuple[int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((*shape, 3), 127, dtype=np.uint8), mode="RGB").save(path)


def save_mask(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array.astype(np.uint8), mode="L").save(path)


class Semantic23BridgeTests(unittest.TestCase):
    def test_mapping_contract_and_goose_artifact_union(self) -> None:
        config = converter.load_config(MAPPING)
        self.assertEqual(config["num_classes"], 23)
        self.assertEqual(set(config["target_classes"]), set(range(23)) | {255})
        self.assertEqual(config["target_classes"][19], "snow")
        self.assertEqual(config["target_classes"][20], "animal")
        self.assertEqual(config["target_classes"][21], "artifact")
        self.assertEqual(config["target_classes"][22], "cobble")
        goose = converter.load_mapping(config, "goose")
        self.assertEqual(set(goose), set(range(64)))
        self.assertEqual({goose[19], goose[46], goose[47]}, {21})
        self.assertEqual(goose[2], 19)
        self.assertEqual(goose[33], 20)
        self.assertEqual(goose[3], 22)
        self.assertEqual(goose[11], 8)
        self.assertEqual(goose[18], 0)
        self.assertEqual(goose[26], 8)
        self.assertEqual(goose[30], 13)
        self.assertEqual(goose[52], 255)
        self.assertEqual(goose[54], 255)

    def test_goose_integration_materializes_every_frame(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            native = root / "native"
            rows = []
            masks = {
                "keep_40": np.array([[31] + [50] * 40 + [0] * 59], dtype=np.uint8),
                "dominant_41": np.array([[31] + [50] * 41 + [0] * 58], dtype=np.uint8),
                "snow_only": np.array([[2] * 10 + [0] * 90], dtype=np.uint8),
                "artifact_and_dirt": np.array([[46, 31] + [0] * 98], dtype=np.uint8),
                "mapping_updates": np.array([[3, 52, 30, 26, 18, 11]], dtype=np.uint8),
                "all_ignore": np.zeros((1, 100), dtype=np.uint8),
            }
            for key, mask in masks.items():
                image = native / f"images/train/scene/{key}_windshield_vis.png"
                label = native / f"labels/train/scene/{key}_labelids.png"
                save_rgb(image, mask.shape)
                save_mask(label, mask)
                rows.append(
                    {
                        "split": "train",
                        "sample_key": f"scene/{key}",
                        "output_image": image.relative_to(native).as_posix(),
                        "output_label": label.relative_to(native).as_posix(),
                    }
                )
            metadata = native / "metadata"
            metadata.mkdir(parents=True)
            with (metadata / "pair_manifest.csv").open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

            output = root / "bridge"
            result = subprocess.run(
                [
                    sys.executable,
                    str(CONVERTER),
                    "--dataset",
                    "goose",
                    "--input-root",
                    str(native),
                    "--output-root",
                    str(output),
                    "--mapping",
                    str(MAPPING),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            with (output / "metadata/manifest.csv").open("r", encoding="utf-8", newline="") as file:
                kept = list(csv.DictReader(file))
            self.assertEqual(len(kept), len(masks))
            self.assertEqual(
                {row["sample_id"] for row in kept},
                {f"scene/{key}" for key in masks},
            )
            with (output / "metadata/per_image_distribution.csv").open(
                "r", encoding="utf-8", newline=""
            ) as file:
                audited = {row["sample_key"]: row for row in csv.DictReader(file)}
            self.assertEqual(len(audited), len(masks))
            self.assertTrue(all(row["materialized"] == "true" for row in audited.values()))
            self.assertTrue(
                all(row["processing_reason"] == "goose_full_dataset" for row in audited.values())
            )
            artifact_mask = np.asarray(Image.open(output / "masks/train/scene/artifact_and_dirt.png"))
            self.assertEqual(set(np.unique(artifact_mask)), {0, 21, 255})
            mapping_mask = np.asarray(Image.open(output / "masks/train/scene/mapping_updates.png"))
            np.testing.assert_array_equal(
                mapping_mask,
                np.array([[22, 255, 13, 8, 0, 8]], dtype=np.uint8),
            )
            summary = json.loads(
                (output / "metadata/conversion_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["input_samples"], len(masks))
            self.assertEqual(summary["materialized_samples"], len(masks))

    def test_combined_package_keeps_rellis_only_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_roots: dict[str, Path] = {}
            for source in ("rellis", "rugd", "ycor", "goose"):
                bridge = root / source
                source_roots[source] = bridge
                rows = []
                splits = ("train", "val", "test") if source == "rellis" else ("train", "val")
                for split in splits:
                    key = f"{source}_{split}"
                    image = bridge / f"images/{split}/{key}.png"
                    mask = bridge / f"masks/{split}/{key}.png"
                    save_rgb(image, (2, 2))
                    save_mask(mask, np.array([[0, 19], [21, 22]], dtype=np.uint8))
                    rows.append(
                        {
                            "sample_key": f"{source}/{split}/{key}",
                            "source": source,
                            "source_split": split,
                            "output_split": split,
                            "sample_id": key,
                            "image_path": image.relative_to(bridge).as_posix(),
                            "mask_path": mask.relative_to(bridge).as_posix(),
                            "non_ignore_ratio": "0.75",
                        }
                    )
                metadata = bridge / "metadata"
                metadata.mkdir(parents=True)
                with (metadata / "manifest.csv").open("w", encoding="utf-8", newline="") as file:
                    writer = csv.DictWriter(file, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)

            combined = root / "combined"
            command = [sys.executable, str(BUILDER)]
            for source, path in source_roots.items():
                command.extend([f"--{source}-root", str(path)])
            command.extend(["--output-root", str(combined)])
            built = subprocess.run(command, check=False, capture_output=True, text=True)
            self.assertEqual(built.returncode, 0, built.stderr)
            validated = subprocess.run(
                [sys.executable, str(VALIDATOR), "--input-root", str(combined)],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(validated.returncode, 0, validated.stderr)
            report = json.loads(
                (combined / "metadata/validation_report.json").read_text(encoding="utf-8")
            )
            self.assertEqual(report["status"], "PASS")
            val_keys = (combined / "splits/val.txt").read_text(encoding="utf-8").splitlines()
            self.assertTrue(val_keys)
            self.assertTrue(all(key.startswith("rellis/") for key in val_keys))


if __name__ == "__main__":
    unittest.main()
