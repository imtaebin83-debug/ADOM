from __future__ import annotations

import csv
import importlib.util
import os
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


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

    def test_e1_manifest_covers_all_main_splits_and_mixed_image_suffixes(self) -> None:
        package = (
            REPO_ROOT
            / "study"
            / "gahyung"
            / "Datasets_Repo"
            / "ADOM-Semantic20"
            / "adom_semantic20_rellis_rugd_ycor_v1"
        )
        with (package / "manifest.csv").open(
            "r", encoding="utf-8-sig", newline=""
        ) as handle:
            rows = list(csv.DictReader(handle))
        manifest = {row["sample_key"]: row for row in rows}
        self.assertEqual(len(manifest), len(rows))
        for split in ("train", "val", "test"):
            keys = [
                line.strip()
                for line in (package / "splits" / f"{split}.txt")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]
            self.assertTrue(set(keys).issubset(manifest))
        suffixes = {Path(row["image_path"]).suffix for row in rows}
        self.assertEqual(suffixes, {".jpg", ".png"})


@unittest.skipUnless(HAS_MMENGINE, "MMEngine config import runs in training image")
class Semantic20ConfigImportTests(unittest.TestCase):
    def test_all_e0_e1_b0_b2_stage_configs_import(self) -> None:
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
            for experiment in ("e0_rellis", "e1_combined"):
                for model in ("b0", "b2"):
                    for stage in ("stage1", "stage2"):
                        config = Config.fromfile(
                            CONFIG_ROOT
                            / f"segformer_{model}_{stage}_{experiment}.py",
                            import_custom_modules=False,
                        )
                        self.assertEqual(config.model.decode_head.num_classes, 19)
                        self.assertEqual(config.model.decode_head.ignore_index, 255)
                        self.assertFalse(config.train_dataloader.dataset.get("reduce_zero_label", False))
                        if experiment == "e1_combined":
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
