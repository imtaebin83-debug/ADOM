from __future__ import annotations

import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from adom.data.io import sha256_file
from adom.data.models import ValidationReport
from adom.data.splits import load_splits
from adom.runtime.artifacts import _git_sha, write_export_metadata
from adom.runtime.checkpoints import resolve_single_best_checkpoint
from adom.runtime.cycle import (
    _bounded_wandb_id,
    _bounded_wandb_tag,
    _resumable_checkpoint,
    _tracking_env,
)
from adom.runtime.doctor import EXPECTED_VERSIONS
from adom.runtime.onnx_parity import (
    DEFAULT_EXPECTED_NUM_CLASSES,
    DEFAULT_MINIMUM_IMAGES,
    DEFAULT_VISUALIZATION_COUNT,
    keep_ratio_valid_region,
    normalized_polygon_mask,
    parse_normalized_polygon,
)


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

    def test_export_configs_are_semantic20_only(self) -> None:
        export_root = REPO_ROOT / "configs" / "adom" / "export"
        for model in ("b0", "b2"):
            for profile in ("384x384", "640x384"):
                text = (
                    export_root / f"segformer_{model}_{profile}_rellis3d.py"
                ).read_text(encoding="utf-8")
                self.assertIn(
                    f"segformer_{model}_stage2_e0_rellis.py",
                    text,
                )
                self.assertNotIn(
                    f'"../segformer_{model}_stage2_rellis3d.py"',
                    text,
                )
                self.assertIn('dict(type="Resize"', text)
                self.assertIn("keep_ratio=True", text)
                self.assertIn('dict(type="Pad"', text)
        cycle_text = (REPO_ROOT / "src" / "adom" / "runtime" / "cycle.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('/ "cost4"', cycle_text)

    def test_parity_defaults_and_roi_polygon_contract(self) -> None:
        self.assertEqual(DEFAULT_MINIMUM_IMAGES, 10)
        self.assertEqual(DEFAULT_EXPECTED_NUM_CLASSES, 19)
        self.assertEqual(DEFAULT_VISUALIZATION_COUNT, 3)
        valid_region = keep_ratio_valid_region((720, 1280), (384, 640))
        self.assertEqual(valid_region, (0, 0, 360, 640))
        polygon = parse_normalized_polygon("0,0;1,0;1,1;0,1")
        self.assertIsNotNone(polygon)
        mask = normalized_polygon_mask(
            (384, 640), polygon or [], valid_region=valid_region
        )
        self.assertEqual(mask.shape, (384, 640))
        self.assertGreater(int(mask.sum()), 0)
        self.assertFalse(mask[383].any())
        with self.assertRaises(ValueError):
            parse_normalized_polygon("0,0;1,0")
        with self.assertRaises(ValueError):
            parse_normalized_polygon("0,0;1.1,0;1,1")

    def test_checkpoint_resolution_rejects_ambiguity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(RuntimeError):
                resolve_single_best_checkpoint(root)
            first = root / "best_clean_selection_iter_500.pth"
            first.touch()
            self.assertEqual(resolve_single_best_checkpoint(root), first.resolve())
            (root / "best_clean_selection_iter_1000.pth").touch()
            with self.assertRaises(RuntimeError):
                resolve_single_best_checkpoint(root)

    def test_iteration_checkpoint_resume_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            work_dir = Path(directory)
            self.assertIsNone(_resumable_checkpoint(work_dir))
            checkpoint = work_dir / "iter_500.pth"
            checkpoint.touch()
            (work_dir / "last_checkpoint").write_text(
                checkpoint.name, encoding="utf-8"
            )
            self.assertEqual(_resumable_checkpoint(work_dir), checkpoint.resolve())
            checkpoint.unlink()
            with self.assertRaises(RuntimeError):
                _resumable_checkpoint(work_dir)

    def test_wandb_run_identity_is_stable_and_bounded(self) -> None:
        output_root = Path("/workspace/adom/runs/phase1-baseline")
        env = _tracking_env(
            {"WANDB_PROJECT": "adom", "WANDB_TAGS": "rellis,seed:42"},
            output_root=output_root,
            model="b2",
            phase="stage2",
            job_type="training",
        )
        self.assertEqual(env["WANDB_RUN_GROUP"], "phase1-baseline")
        self.assertEqual(env["WANDB_RUN_ID"], "phase1-baseline-b2-stage2")
        self.assertIn("model:b2", env["WANDB_TAGS"])
        self.assertLessEqual(len(_bounded_wandb_id("x" * 200)), 64)

    def test_wandb_extra_tag_is_stable_and_bounded(self) -> None:
        short = "extra:runpod+a100"
        self.assertEqual(_bounded_wandb_tag(short), short)

        long = "extra:" + "+".join(["phase1", "semantic20"] * 20)
        bounded = _bounded_wandb_tag(long)
        self.assertEqual(len(bounded), 64)
        self.assertEqual(bounded, _bounded_wandb_tag(long))
        self.assertNotEqual(bounded, _bounded_wandb_tag(long + "+e1"))

        env = _tracking_env(
            {
                "WANDB_TAGS": (
                    "runpod,a100,phase1,semantic20,e0,b0,experiment:e0,seed:42"
                )
            },
            output_root=Path(
                "/workspace/adom/runs/semantic20/e0/"
                "20260805T064732Z-4b3d33603c29-b0-smoke"
            ),
            model="b0",
            phase="e0-stage1-smoke",
            job_type="training",
        )
        self.assertLessEqual(len(env["WANDB_EXTRA_TAG"]), 64)
        self.assertTrue(env["WANDB_EXTRA_TAG"].startswith("extra:"))

    def test_image_revision_precedes_git_checkout(self) -> None:
        with patch.dict("os.environ", {"ADOM_GIT_SHA": "image-sha"}):
            self.assertEqual(_git_sha(REPO_ROOT), "image-sha")

    def test_mmseg_tokenizer_dependencies_are_pinned(self) -> None:
        requirements = (REPO_ROOT / "requirements" / "openmmlab.txt").read_text(
            encoding="utf-8"
        )
        for package, expected in (
            ("ftfy", "6.1.1"),
            ("regex", "2023.10.3"),
            ("prettytable", "3.9.0"),
        ):
            self.assertIn(f"{package}=={expected}", requirements)
            self.assertEqual(EXPECTED_VERSIONS[package], expected)

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
                    / "cost4"
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
