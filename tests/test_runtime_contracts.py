from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from adom.data.io import sha256_file
from adom.data.models import ValidationReport
from adom.data.splits import load_splits
from adom.runtime.artifacts import write_export_metadata
from adom.runtime.checkpoints import resolve_single_best_checkpoint


REPO_ROOT = Path(__file__).resolve().parents[1]


class RuntimeContractTests(unittest.TestCase):
    def test_official_split_snapshot(self) -> None:
        split_root = REPO_ROOT / "data" / "splits" / "rellis3d" / "official"
        expected = {
            "train": (
                3302,
                "29d6c1d7cbf7d94e18a7ce83dd20d85bedd614ff0901d93fbd19cac6329397e9",
            ),
            "val": (
                983,
                "741f67a4f181f4494bbea044ae248d587f41c72033ae97bb8952a58d84934c8c",
            ),
            "test": (
                1672,
                "7a98484c9550b8d2825a7cc0b40c1789b452f30266047cb94bf757f99ad453d3",
            ),
        }
        report = ValidationReport(dataset="rellis3d")
        splits = load_splits(split_root, report)
        self.assertTrue(report.passed, report.to_dict())
        for name, (count, checksum) in expected.items():
            self.assertEqual(len(splits[name]), count)
            self.assertEqual(
                sha256_file(split_root / f"{name}.txt"),
                checksum,
            )

    def test_all_python_configs_compile(self) -> None:
        for path in sorted((REPO_ROOT / "configs").rglob("*.py")):
            compile(path.read_text(encoding="utf-8"), str(path), "exec")

    def test_model_contracts_are_explicit(self) -> None:
        for model in ("b0", "b2"):
            text = (
                REPO_ROOT
                / "configs"
                / "adom"
                / "_base_"
                / "models"
                / f"segformer_{model}_cost4.py"
            ).read_text(encoding="utf-8")
            self.assertIn("num_classes=4", text)
            self.assertIn("ignore_index=255", text)
        dataset = (
            REPO_ROOT
            / "configs"
            / "adom"
            / "_base_"
            / "datasets"
            / "rellis3d_cost4.py"
        ).read_text(encoding="utf-8")
        self.assertIn('reduce_zero_label=False', dataset)

    def test_checkpoint_resolution_rejects_ambiguity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(RuntimeError):
                resolve_single_best_checkpoint(root)
            first = root / "best_mIoU_iter_500.pth"
            first.touch()
            self.assertEqual(resolve_single_best_checkpoint(root), first.resolve())
            (root / "best_mIoU_iter_1000.pth").touch()
            with self.assertRaises(RuntimeError):
                resolve_single_best_checkpoint(root)

    def test_export_metadata_is_checksum_linked(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            dataset.mkdir()
            (dataset / "SHA256SUMS.txt").write_text(
                "synthetic\n", encoding="utf-8"
            )
            (dataset / "metadata").mkdir()
            (dataset / "metadata" / "dataset.json").write_text(
                json.dumps(
                    {
                        "dataset": "rellis3d",
                        "version": "v2.0-test",
                        "official_split_only": True,
                    }
                ),
                encoding="utf-8",
            )
            files = {}
            for name in ("best.pth", "end2end.onnx", "deploy.json"):
                path = root / name
                path.write_bytes(name.encode("utf-8"))
                files[name] = path
            output = root / "metadata.json"
            write_export_metadata(
                repo_root=REPO_ROOT,
                dataset_root=dataset,
                model_config=(
                    REPO_ROOT
                    / "configs"
                    / "adom"
                    / "export"
                    / "segformer_b0_384x384_rellis3d.py"
                ),
                checkpoint=files["best.pth"],
                onnx_path=files["end2end.onnx"],
                deploy_info=files["deploy.json"],
                mapping_path=(
                    REPO_ROOT
                    / "configs"
                    / "datasets"
                    / "rellis3d"
                    / "label_mapping.yaml"
                ),
                profile="384x384",
                width=384,
                height=384,
                output=output,
            )
            value = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(value["classes"][3]["name"], "high_cost_or_obstacle")
            self.assertEqual(value["input"]["shape_nchw"], [1, 3, 384, 384])
            self.assertFalse(value["tensorrt_engine_included"])


if __name__ == "__main__":
    unittest.main()
