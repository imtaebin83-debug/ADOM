from __future__ import annotations

import copy
import unittest
from pathlib import Path

from adom.runtime.semantic20_handoff import (
    EXPECTED_INPUT,
    EXPECTED_OUTPUT,
    validate_parity,
)


def passing_report() -> dict:
    return {
        "status": "PASS",
        "num_images": 12,
        "onnx_contract": {
            "inputs": [EXPECTED_INPUT],
            "outputs": [EXPECTED_OUTPUT],
            "opsets": [{"domain": "ai.onnx", "version": 13}],
        },
        "thresholds": {
            "max_absolute_error": 0.001,
            "pixel_argmax_agreement": 0.999,
        },
        "summary": {
            "all_finite_logits": True,
            "maximum_absolute_error": 0.0001035,
            "minimum_per_image_argmax_agreement": 1.0,
            "overall_pixel_argmax_agreement": 1.0,
            "reported_class_ids": list(range(19)),
        },
    }


class Semantic20HandoffTests(unittest.TestCase):
    def test_accepts_verified_cpu_parity(self) -> None:
        validate_parity(passing_report())

    def test_rejects_failed_or_wrong_shape_report(self) -> None:
        report = passing_report()
        report["status"] = "FAIL"
        with self.assertRaisesRegex(ValueError, "passing parity"):
            validate_parity(report)
        report = copy.deepcopy(passing_report())
        report["onnx_contract"]["inputs"][0]["shape"] = [1, 3, 640, 640]
        with self.assertRaisesRegex(ValueError, "input contract"):
            validate_parity(report)

    def test_rejects_incomplete_class_coverage(self) -> None:
        report = passing_report()
        report["summary"]["reported_class_ids"] = list(range(18))
        with self.assertRaisesRegex(ValueError, "IDs 0..18"):
            validate_parity(report)

    def test_tensorrt_builder_uses_mib_number_without_suffix(self) -> None:
        root = Path(__file__).resolve().parents[1]
        source = (root / "scripts" / "build_semantic20_tensorrt.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn('ADOM_TRT_WORKSPACE_MIB:-2048', source)
        self.assertIn('--memPoolSize="workspace:${WORKSPACE_MIB}"', source)
        self.assertNotIn("workspace:1024MiB", source)
        self.assertIn('test -s "${ENGINE_PATH}"', source)

    def test_export_contract_separates_hw_and_wh(self) -> None:
        root = Path(__file__).resolve().parents[1]
        model = (
            root / "configs" / "adom" / "export" / "segformer_b0_640x384_rellis3d.py"
        ).read_text(encoding="utf-8")
        deploy = (
            root / "configs" / "deployment" / "mmseg_onnxruntime_640x384.py"
        ).read_text(encoding="utf-8")
        self.assertIn("model_input_size_hw = (384, 640)", model)
        self.assertIn("pipeline_size_wh = (640, 384)", model)
        self.assertIn("opset_version=13", deploy)
        self.assertIn("input_shape=None", deploy)


if __name__ == "__main__":
    unittest.main()
