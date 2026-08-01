from __future__ import annotations

import csv
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from adom.data.io import sha256_file
from adom.data.models import ValidationReport
from adom.data.splits import load_splits
from adom.runtime.artifacts import write_export_metadata
from adom.runtime.checkpoints import resolve_single_best_checkpoint
from adom.runtime.cycle import CycleState, _probe_batch, _select_parity_images


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

    def test_docker_integration_tests_run_on_pull_requests_without_push(self) -> None:
        workflow = (
            REPO_ROOT / ".github" / "workflows" / "docker-build.yml"
        ).read_text(encoding="utf-8")
        self.assertIn("pull_request:", workflow)
        self.assertIn("if: github.event_name != 'pull_request'", workflow)
        self.assertIn("python -m unittest discover", workflow)

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
            for name in (
                "best.pth",
                "end2end.onnx",
                "deploy.json",
                "detail.json",
                "pipeline.json",
                "test_metrics.json",
            ):
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
                detail_info=files["detail.json"],
                pipeline_info=files["pipeline.json"],
                test_metrics=files["test_metrics.json"],
                mapping_path=(
                    REPO_ROOT
                    / "configs"
                    / "datasets"
                    / "rellis3d"
                    / "label_mapping.yaml"
                ),
                model_variant="b0",
                profile="384x384",
                width=384,
                height=384,
                output=output,
            )
            value = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(value["classes"][3]["name"], "high_cost_or_obstacle")
            self.assertEqual(value["input"]["shape_nchw"], [1, 3, 384, 384])
            self.assertFalse(value["tensorrt_engine_included"])
            self.assertEqual(value["model_variant"], "b0")
            self.assertEqual(
                value["artifacts"]["pipeline_info"]["sha256"],
                sha256_file(files["pipeline.json"]),
            )

    def test_batch_probe_keeps_separate_stage_plans(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with patch(
                "adom.runtime.cycle.subprocess.run",
                return_value=subprocess.CompletedProcess([], 0),
            ) as run_mock:
                stage1 = _probe_batch(
                    model="b0",
                    stage="stage1",
                    config=(
                        REPO_ROOT
                        / "configs"
                        / "adom"
                        / "segformer_b0_stage1_rellis3d.py"
                    ),
                    output_root=output,
                    env={},
                    train_tool=REPO_ROOT / "fake-train.py",
                    resume=False,
                )
                stage2 = _probe_batch(
                    model="b0",
                    stage="stage2",
                    config=(
                        REPO_ROOT
                        / "configs"
                        / "adom"
                        / "segformer_b0_stage2_rellis3d.py"
                    ),
                    output_root=output,
                    env={},
                    train_tool=REPO_ROOT / "fake-train.py",
                    resume=False,
                )
            self.assertEqual(stage1, (16, 1))
            self.assertEqual(stage2, (16, 1))
            probe_command = run_mock.call_args_list[0].args[0]
            self.assertIn("val_cfg=None", probe_command)
            self.assertIn("val_dataloader=None", probe_command)
            self.assertIn("val_evaluator=None", probe_command)
            plan = json.loads(
                (output / "b0" / "batch_plan.json").read_text(encoding="utf-8")
            )
            self.assertEqual(set(plan["stages"]), {"stage1", "stage2"})
            self.assertIn("stage1", plan["stages"]["stage1"]["config"])
            self.assertIn("stage2", plan["stages"]["stage2"]["config"])

    def test_cycle_state_requires_matching_context_and_artifact_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = CycleState(root / "status.json", resume=False)
            context = {"git_sha": "abc", "gpu": {"name": "test"}}
            state.bind_run_context(context, resume=False)
            state.bind_run_context(context, resume=True)
            with self.assertRaises(RuntimeError):
                state.bind_run_context(
                    {"git_sha": "changed", "gpu": {"name": "test"}},
                    resume=True,
                )

            artifact = root / "artifact.json"
            artifact.write_text("first", encoding="utf-8")
            state.start("phase", ["test"])
            state.finish("phase", [artifact])
            self.assertTrue(state.completed("phase", [artifact]))
            artifact.write_text("changed", encoding="utf-8")
            self.assertFalse(state.completed("phase", [artifact]))

    def test_parity_selection_covers_cost_classes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "metadata" / "manifest_test.csv"
            manifest.parent.mkdir()
            rows = []
            for class_id in (0, 1, 2, 3, 255):
                sample_id = f"0000{class_id % 2}_class{class_id}"
                image_path = root / "images" / f"{sample_id}.jpg"
                mask_path = root / "annotations" / f"{sample_id}.png"
                image_path.parent.mkdir(exist_ok=True)
                mask_path.parent.mkdir(exist_ok=True)
                Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8)).save(image_path)
                Image.fromarray(
                    np.full((2, 2), class_id, dtype=np.uint8)
                ).save(mask_path)
                rows.append(
                    {
                        "sample_id": sample_id,
                        "sequence": f"0000{class_id % 2}",
                        "image_relpath": image_path.relative_to(root).as_posix(),
                        "mask_relpath": mask_path.relative_to(root).as_posix(),
                    }
                )
            with manifest.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            selected = _select_parity_images(root, limit=5)
            self.assertEqual(len(selected), 5)


if __name__ == "__main__":
    unittest.main()
