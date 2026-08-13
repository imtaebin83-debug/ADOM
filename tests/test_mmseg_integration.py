from __future__ import annotations

import csv
import importlib.util
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
HAS_MMSEG = importlib.util.find_spec("mmseg") is not None


@unittest.skipUnless(HAS_MMSEG, "MMSeg is validated in the training container")
class MMSegIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from mmseg.utils import register_all_modules

        register_all_modules(init_default_scope=True)
        import adom.mmseg  # noqa: F401

    def test_all_training_configs_load(self) -> None:
        from mmengine.config import Config

        for model in ("b0", "b2"):
            for stage in ("stage1", "stage2"):
                path = (
                    REPO_ROOT
                    / "configs"
                    / "adom"
                    / f"segformer_{model}_{stage}_rellis3d.py"
                )
                config = Config.fromfile(path)
                self.assertEqual(config.model.decode_head.num_classes, 4)
                self.assertEqual(config.model.decode_head.ignore_index, 255)

    def test_semantic20_training_configs_load(self) -> None:
        from mmengine.config import Config

        for experiment in ("e0_rellis", "e1_combined", "e2_combined_goose"):
            for model in ("b0", "b2"):
                for stage in ("stage1", "stage2"):
                    path = (
                        REPO_ROOT
                        / "configs"
                        / "adom"
                        / "phase1_semantic20"
                        / f"segformer_{model}_{stage}_{experiment}.py"
                    )
                    config = Config.fromfile(path)
                    self.assertEqual(config.model.decode_head.num_classes, 19)
                    self.assertEqual(config.model.decode_head.ignore_index, 255)
                    self.assertTrue(config.model.decode_head.loss_decode.avg_non_ignore)

    def test_semantic20_export_configs_load(self) -> None:
        from mmengine.config import Config

        expected_sizes = {
            "384x384": (384, 384),
            "640x384": (384, 640),
        }
        for model in ("b0", "b2"):
            for profile, expected_size in expected_sizes.items():
                path = (
                    REPO_ROOT
                    / "configs"
                    / "adom"
                    / "export"
                    / f"segformer_{model}_{profile}_rellis3d.py"
                )
                config = Config.fromfile(path)
                self.assertEqual(config.model.decode_head.num_classes, 19)
                self.assertEqual(config.model.decode_head.ignore_index, 255)
                self.assertEqual(tuple(config.model.data_preprocessor.size), expected_size)
                resize = config.test_pipeline[1]
                self.assertTrue(resize.keep_ratio)
                self.assertEqual(tuple(config.test_pipeline[2].size), expected_size)

    def test_semantic20_ros_runtime_config_preserves_padding_metadata(self) -> None:
        from mmengine.config import Config

        path = (
            REPO_ROOT
            / "configs"
            / "adom"
            / "runtime"
            / "segformer_b0_640x384_rellis3d.py"
        )
        config = Config.fromfile(path)
        self.assertEqual(config.model.decode_head.num_classes, 19)
        self.assertEqual(tuple(config.model.data_preprocessor.size), (384, 640))
        self.assertEqual(
            tuple(config.model.data_preprocessor.test_cfg.size), (384, 640)
        )
        self.assertEqual(
            [step.type for step in config.test_pipeline],
            ["LoadImageFromFile", "Resize", "PackSegInputs"],
        )

    def test_semantic20_wandb_backend_respects_mode(self) -> None:
        from mmengine.config import Config

        path = (
            REPO_ROOT
            / "configs"
            / "adom"
            / "phase1_semantic20"
            / "segformer_b0_stage1_e0_rellis.py"
        )
        with patch.dict("os.environ", {"WANDB_MODE": "disabled"}):
            disabled = Config.fromfile(path)
        disabled_backends = {
            backend["type"] for backend in disabled.visualizer.vis_backends
        }
        self.assertNotIn("WandbVisBackend", disabled_backends)
        self.assertIn("TensorboardVisBackend", disabled_backends)
        self.assertIn("LocalVisBackend", disabled_backends)

        with patch.dict("os.environ", {"WANDB_MODE": "online"}):
            online = Config.fromfile(path)
        online_backends = {backend["type"] for backend in online.visualizer.vis_backends}
        self.assertIn("WandbVisBackend", online_backends)

    def test_manifest_dataset_registration(self) -> None:
        import numpy as np
        from PIL import Image
        from mmseg.registry import DATASETS

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "images" / "train" / "sample.jpg"
            mask = root / "annotations" / "train" / "sample.png"
            image.parent.mkdir(parents=True)
            mask.parent.mkdir(parents=True)
            Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(image)
            Image.fromarray(np.zeros((4, 4), dtype=np.uint8)).save(mask)
            manifest = root / "metadata" / "manifest_train.csv"
            manifest.parent.mkdir()
            with manifest.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["sample_id", "image_relpath", "mask_relpath"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "sample_id": "sample",
                        "image_relpath": "images/train/sample.jpg",
                        "mask_relpath": "annotations/train/sample.png",
                    }
                )
            dataset = DATASETS.build(
                dict(
                    type="AdomCost4Dataset",
                    data_root=str(root),
                    manifest="metadata/manifest_train.csv",
                    pipeline=[],
                )
            )
            self.assertEqual(len(dataset), 1)

    def test_freeze_and_unfreeze_audits(self) -> None:
        import torch

        from adom.mmseg.hooks import (
            BackboneAuditHook,
            FiniteLossHook,
            FreezeBackboneHook,
        )

        class TinyModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.backbone = torch.nn.Linear(2, 2)

        with tempfile.TemporaryDirectory() as directory:
            runner = types.SimpleNamespace(
                model=TinyModel(),
                work_dir=directory,
                logger=types.SimpleNamespace(info=lambda *args: None),
            )
            freeze = FreezeBackboneHook()
            freeze.before_train(runner)
            self.assertTrue(
                all(not parameter.requires_grad for parameter in runner.model.backbone.parameters())
            )
            freeze.after_train(runner)

            audit = BackboneAuditHook()
            audit.before_train(runner)
            self.assertTrue(
                all(parameter.requires_grad for parameter in runner.model.backbone.parameters())
            )
            with torch.no_grad():
                runner.model.backbone.weight.add_(1)
            audit.after_train(runner)
            runner.iter = 1
            finite = FiniteLossHook()
            finite.after_train_iter(runner, 0, outputs={"loss": torch.tensor(1.0)})
            with self.assertRaises(FloatingPointError):
                finite.after_train_iter(
                    runner,
                    1,
                    outputs={"loss": torch.tensor(float("nan"))},
                )

    def test_cpu_b0_one_batch_forward(self) -> None:
        import torch
        from mmengine.config import Config
        from mmseg.registry import MODELS

        config = Config.fromfile(
            REPO_ROOT
            / "configs"
            / "adom"
            / "segformer_b0_stage1_rellis3d.py"
        )
        config.model.backbone.init_cfg = None
        config.model.data_preprocessor.size = None
        model = MODELS.build(config.model)
        model.eval()
        with torch.no_grad():
            output = model(torch.randn(1, 3, 64, 64), mode="tensor")
        self.assertEqual(tuple(output.shape[:2]), (1, 4))
        self.assertTrue(torch.isfinite(output).all())


if __name__ == "__main__":
    unittest.main()
