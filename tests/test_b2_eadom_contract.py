from __future__ import annotations

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from adom.runtime import b2_eadom_contract, semantic20_cycle


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = REPO_ROOT / "configs" / "adom" / "phase1_semantic20"
HAS_MMENGINE = importlib.util.find_spec("mmengine") is not None


class B2EadomSourceContractTests(unittest.TestCase):
    def test_stage_configs_change_only_model_base(self) -> None:
        for stage in ("stage1", "stage2"):
            b0 = (CONFIG_ROOT / f"segformer_b0_{stage}_eadom.py").read_text(
                encoding="utf-8"
            )
            b2 = (CONFIG_ROOT / f"segformer_b2_{stage}_eadom.py").read_text(
                encoding="utf-8"
            )
            self.assertEqual(
                b0.replace("models/segformer_b0.py", "models/segformer_b2.py"),
                b2,
            )

    def test_runtime_accepts_one_b2_eadom_model(self) -> None:
        self.assertEqual(semantic20_cycle._requested_models("b2", "eadom"), ["b2"])
        self.assertEqual(semantic20_cycle._requested_models("b0", "eadom"), ["b0"])
        with self.assertRaisesRegex(RuntimeError, "one architecture"):
            semantic20_cycle._requested_models("b0,b2", "eadom")

    def test_b2_probe_order_preserves_effective_batch_candidates(self) -> None:
        self.assertEqual(semantic20_cycle._batch_candidates("b2"), [16, 8, 4])
        self.assertEqual([16 // value for value in (16, 8, 4)], [1, 2, 4])

    def test_expected_architecture_difference_allowlist(self) -> None:
        self.assertEqual(
            set(b2_eadom_contract.ARCHITECTURE_DIFFS),
            {
                "checkpoint",
                "model.backbone.embed_dims",
                "model.backbone.init_cfg.checkpoint",
                "model.backbone.num_layers",
                "model.decode_head.channels",
                "model.decode_head.in_channels",
            },
        )


@unittest.skipUnless(HAS_MMENGINE, "MMEngine config import runs in training image")
class B2EadomResolvedConfigTests(unittest.TestCase):
    def test_resolved_configs_are_architecture_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ,
            {
                "ADOM_DATA_ROOT": Path(directory).as_posix(),
                "ADOM_MICRO_BATCH": "8",
                "ADOM_ACCUMULATIVE_COUNTS": "2",
                "ADOM_SEED": "42",
                "ADOM_DETERMINISTIC": "true",
            },
            clear=False,
        ):
            for stage, updates in (("stage1", 4000), ("stage2", 40000)):
                report = b2_eadom_contract._stage_contract(stage)
                self.assertEqual(report["effective_batch"], 16)
                self.assertEqual(report["checks"]["optimizer_updates"], updates)
                self.assertEqual(
                    {item["path"] for item in report["architecture_differences"]},
                    set(b2_eadom_contract.ARCHITECTURE_DIFFS),
                )


if __name__ == "__main__":
    unittest.main()
