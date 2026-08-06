from __future__ import annotations

import importlib.util
import inspect
import io
import os
import re
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

from adom.data.semantic20 import resource_path
from adom.runtime import semantic20_cycle


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = REPO_ROOT / "configs" / "adom" / "phase1_semantic20"
HAS_MMENGINE = importlib.util.find_spec("mmengine") is not None


def _execute_env_config(path: Path) -> dict:
    """Execute MMEngine {{$VAR:default}} substitution without MMEngine."""
    source = path.read_text(encoding="utf-8")
    pattern = r"\{\{['\"]?\s*\$(\w+)\s*:\s*(\S*?)\s*['\"]?\}\}"

    def replace(match: re.Match[str]) -> str:
        return os.getenv(match.group(1), match.group(2))

    namespace: dict = {"__file__": str(path)}
    exec(compile(re.sub(pattern, replace, source), str(path), "exec"), namespace)
    return namespace


class OptimizerUpdateScalingTests(unittest.TestCase):
    def _load_schedule(self, filename: str, accumulative: int) -> dict:
        with patch.dict(
            os.environ,
            {"ADOM_ACCUMULATIVE_COUNTS": str(accumulative)},
            clear=False,
        ):
            return _execute_env_config(
                CONFIG_ROOT / "_base_" / "schedules" / filename
            )

    def test_stage1_all_targets_scale_with_accumulation(self) -> None:
        for accumulative in (1, 2, 4):
            config = self._load_schedule(
                "stage1_head_4k_updates.py", accumulative
            )
            self.assertEqual(config["optim_wrapper"]["accumulative_counts"], accumulative)
            self.assertEqual(config["train_cfg"]["max_iters"], 4000 * accumulative)
            self.assertEqual(config["train_cfg"]["val_interval"], 500 * accumulative)
            self.assertEqual(config["param_scheduler"][0]["end"], 200 * accumulative)
            self.assertEqual(config["param_scheduler"][1]["end"], 4000 * accumulative)

    def test_stage2_all_targets_scale_with_accumulation(self) -> None:
        for accumulative in (1, 2, 4):
            config = self._load_schedule(
                "stage2_full_40k_updates.py", accumulative
            )
            self.assertEqual(config["train_cfg"]["max_iters"], 40000 * accumulative)
            self.assertEqual(config["train_cfg"]["val_interval"], 1000 * accumulative)
            self.assertEqual(config["param_scheduler"][0]["end"], 500 * accumulative)
            self.assertEqual(config["param_scheduler"][1]["end"], 40000 * accumulative)
            custom = config["optim_wrapper"]["paramwise_cfg"]["custom_keys"]
            self.assertEqual(custom["decode_head"]["lr_mult"], 10.0)

    def test_checkpoint_interval_scales_with_accumulation(self) -> None:
        runtime = CONFIG_ROOT / "_base_" / "semantic_default_runtime.py"
        for accumulative in (1, 2, 4):
            with patch.dict(
                os.environ,
                {"ADOM_ACCUMULATIVE_COUNTS": str(accumulative)},
                clear=False,
            ):
                config = _execute_env_config(runtime)
            self.assertEqual(
                config["default_hooks"]["checkpoint"]["interval"],
                500 * accumulative,
            )

    def test_clean_v1_loss_seed_and_determinism_contract(self) -> None:
        model = _execute_env_config(
            CONFIG_ROOT / "_base_" / "models" / "segformer_b0.py"
        )
        self.assertTrue(model["model"]["decode_head"]["loss_decode"]["avg_non_ignore"])
        with patch.dict(
            os.environ,
            {"ADOM_SEED": "43", "ADOM_DETERMINISTIC": "true"},
            clear=False,
        ):
            runtime = _execute_env_config(
                CONFIG_ROOT / "_base_" / "semantic_default_runtime.py"
            )
        self.assertEqual(runtime["randomness"], {"seed": 43, "deterministic": True})
        self.assertFalse(runtime["env_cfg"]["cudnn_benchmark"])
        self.assertIsNone(runtime["default_hooks"]["checkpoint"]["save_best"])

    def test_e2_configs_preserve_b0_b2_stage_matrix(self) -> None:
        for model in ("b0", "b2"):
            for stage in ("stage1", "stage2"):
                path = CONFIG_ROOT / f"segformer_{model}_{stage}_e2_combined_goose.py"
                self.assertTrue(path.is_file())
                source = path.read_text(encoding="utf-8")
                self.assertIn("e2_combined_goose.py", source)
                self.assertIn(f"segformer_{model}.py", source)

    def test_canonical_test_requires_explicit_unlock_and_final_model(self) -> None:
        cycle_source = inspect.getsource(semantic20_cycle)
        hooks_source = (REPO_ROOT / "src" / "adom" / "mmseg" / "hooks.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("ADOM_CANONICAL_TEST_UNLOCK", cycle_source)
        self.assertIn("class CanonicalTestLockHook", hooks_source)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            semantic20_cycle.main(
                [
                    "--experiment",
                    "e0",
                    "--output",
                    "unused",
                    "--models",
                    "b0",
                    "--run-test",
                ]
            )
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            semantic20_cycle.main(
                [
                    "--experiment",
                    "e0",
                    "--output",
                    "unused",
                    "--models",
                    "b0",
                    "--final-test-model",
                    "b0",
                ]
            )

    def test_short_smoke_schedule_is_valid(self) -> None:
        with patch.dict(
            os.environ,
            {
                "ADOM_ACCUMULATIVE_COUNTS": "4",
                "ADOM_MAX_OPTIMIZER_UPDATES": "50",
                "ADOM_VAL_INTERVAL_OPTIMIZER_UPDATES": "51",
            },
            clear=False,
        ):
            config = _execute_env_config(
                CONFIG_ROOT
                / "_base_"
                / "schedules"
                / "stage1_head_4k_updates.py"
            )
        self.assertEqual(config["train_cfg"]["max_iters"], 200)
        self.assertEqual(config["train_cfg"]["val_interval"], 204)
        self.assertEqual(len(config["param_scheduler"]), 1)
        self.assertEqual(config["param_scheduler"][0]["end"], 200)

    def test_cost4_reference_configs_remain_present(self) -> None:
        for model in ("b0", "b2"):
            self.assertTrue(
                (
                    REPO_ROOT
                    / "configs"
                    / "adom"
                    / "_base_"
                    / "models"
                    / f"segformer_{model}_cost4.py"
                ).is_file()
            )

    def test_canonical_semantic20_resources_are_packaged(self) -> None:
        expected_counts = {"train": 4435, "val": 900, "test": 899}
        for split, expected in expected_counts.items():
            values = [
                line.strip()
                for line in resource_path("rellis", "splits", f"{split}.txt")
                .read_text(encoding="utf-8-sig")
                .splitlines()
                if line.strip()
            ]
            self.assertEqual(len(values), expected)
            self.assertEqual(len(values), len(set(values)))
        self.assertTrue(resource_path("rugd", "config", "label_mapping.json").is_file())
        self.assertTrue(
            resource_path("semantic_20", "config", "bridge_mapping.yaml").is_file()
        )

    def test_runtime_has_no_legacy_study_dependency(self) -> None:
        source = (REPO_ROOT / "src" / "adom" / "runtime" / "semantic20_cycle.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn(' / "study"', source)


@unittest.skipUnless(HAS_MMENGINE, "MMEngine config import runs in training image")
class Semantic20ConfigImportTests(unittest.TestCase):
    def test_all_e0_e1_e2_b0_b2_stage_configs_import(self) -> None:
        from mmengine.config import Config

        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ,
            {
                # MMEngine 0.10.7 uses regex replacement on Windows; forward
                # slashes avoid interpreting C:\Users as a replacement escape.
                "ADOM_DATA_ROOT": Path(directory).as_posix(),
                "ADOM_ACCUMULATIVE_COUNTS": "2",
            },
            clear=False,
        ):
            for experiment in ("e0_rellis", "e1_combined", "e2_combined_goose"):
                for model in ("b0", "b2"):
                    for stage in ("stage1", "stage2"):
                        config = Config.fromfile(
                            CONFIG_ROOT
                            / f"segformer_{model}_{stage}_{experiment}.py",
                            import_custom_modules=False,
                        )
                        self.assertEqual(config.model.decode_head.num_classes, 19)
                        self.assertEqual(config.model.decode_head.ignore_index, 255)
                        self.assertTrue(config.model.decode_head.loss_decode.avg_non_ignore)
                        self.assertTrue(config.randomness.deterministic)
                        self.assertFalse(config.train_dataloader.dataset.get("reduce_zero_label", False))
                        if experiment in {"e1_combined", "e2_combined_goose"}:
                            self.assertEqual(
                                config.train_dataloader.dataset.manifest,
                                "manifest.csv",
                            )
                            self.assertEqual(
                                config.val_dataloader.dataset.manifest,
                                "manifest.csv",
                            )
                        expected_updates = 4000 if stage == "stage1" else 40000
                        self.assertEqual(config.train_cfg.max_iters, expected_updates * 2)
                        self.assertEqual(config.default_hooks.checkpoint.interval, 1000)


if __name__ == "__main__":
    unittest.main()
