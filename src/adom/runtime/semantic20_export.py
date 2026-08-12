from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


EXPECTED_INPUT_SHAPE = [1, 3, 384, 640]
EXPECTED_OUTPUT_SHAPE = [1, 19, 384, 640]
EXPECTED_OPSET = 13


def validate_export_configs(model_config: Path, deploy_config: Path) -> None:
    try:
        from mmengine.config import Config
    except ImportError as error:
        raise RuntimeError("MMEngine is required for Semantic20 export") from error

    model = Config.fromfile(str(model_config))
    deploy = Config.fromfile(str(deploy_config))
    if model.model.decode_head.num_classes != 19:
        raise ValueError("Semantic20 export requires decode_head.num_classes=19")
    if model.model.decode_head.ignore_index != 255:
        raise ValueError("Semantic20 export requires ignore_index=255")
    if tuple(model.model.data_preprocessor.size) != (384, 640):
        raise ValueError("Expected model data_preprocessor size in H,W order")
    if tuple(model.test_pipeline[1].scale) != (640, 384):
        raise ValueError("Expected Resize.scale in W,H order")
    if tuple(model.test_pipeline[2].size) != (640, 384):
        raise ValueError("Expected Pad.size in W,H order")
    if not model.test_pipeline[1].keep_ratio:
        raise ValueError("Semantic20 deployment resize must preserve aspect ratio")
    if deploy.onnx_config.opset_version != EXPECTED_OPSET:
        raise ValueError("Semantic20 export requires ONNX opset 13")
    if deploy.onnx_config.input_shape is not None:
        raise ValueError(
            "MMDeploy input_shape must be None to preserve pipeline resize/pad"
        )
    if deploy.codebase_config.with_argmax:
        raise ValueError("Semantic20 export must preserve raw logits")


def validate_onnx_contract(path: Path) -> dict[str, Any]:
    try:
        import onnx
        import onnxruntime as ort
    except ImportError as error:
        raise RuntimeError("ONNX and ONNX Runtime are required after export") from error

    model = onnx.load(str(path), load_external_data=False)
    onnx.checker.check_model(model)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    opsets = [
        {"domain": item.domain or "ai.onnx", "version": int(item.version)}
        for item in model.opset_import
    ]
    if len(inputs) != 1 or inputs[0].name != "input":
        raise ValueError("Expected exactly one ONNX input named input")
    if list(inputs[0].shape) != EXPECTED_INPUT_SHAPE:
        raise ValueError(f"Unexpected ONNX input shape: {inputs[0].shape}")
    if inputs[0].type != "tensor(float)":
        raise ValueError(f"Unexpected ONNX input type: {inputs[0].type}")
    if len(outputs) != 1 or outputs[0].name != "output":
        raise ValueError("Expected exactly one ONNX output named output")
    if list(outputs[0].shape) != EXPECTED_OUTPUT_SHAPE:
        raise ValueError(f"Unexpected ONNX output shape: {outputs[0].shape}")
    if outputs[0].type != "tensor(float)":
        raise ValueError(f"Unexpected ONNX output type: {outputs[0].type}")
    if [item for item in opsets if item["domain"] == "ai.onnx"] != [
        {"domain": "ai.onnx", "version": EXPECTED_OPSET}
    ]:
        raise ValueError(f"Unexpected ONNX opsets: {opsets}")
    return {
        "opsets": opsets,
        "inputs": [
            {
                "name": inputs[0].name,
                "shape": list(inputs[0].shape),
                "dtype": inputs[0].type,
            }
        ],
        "outputs": [
            {
                "name": outputs[0].name,
                "shape": list(outputs[0].shape),
                "dtype": outputs[0].type,
            }
        ],
        "providers": session.get_providers(),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Export the Semantic20 B0 E0 checkpoint through the MMDeploy API"
    )
    parser.add_argument("--deploy-config", required=True, type=Path)
    parser.add_argument("--model-config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--save-file", default="end2end.onnx")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)

    validate_export_configs(args.model_config, args.deploy_config)
    for path in (args.checkpoint, args.image):
        if not path.is_file():
            raise FileNotFoundError(path)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    try:
        from mmdeploy.apis import torch2onnx
    except ImportError as error:
        raise RuntimeError("MMDeploy is required for Semantic20 export") from error
    torch2onnx(
        str(args.image),
        str(args.work_dir),
        args.save_file,
        str(args.deploy_config),
        str(args.model_config),
        str(args.checkpoint),
        args.device,
    )
    contract = validate_onnx_contract(args.work_dir / args.save_file)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps({"status": "PASS", "onnx_contract": contract}, indent=2)
            + "\n",
            encoding="utf-8",
        )
    print(json.dumps({"status": "PASS", "onnx_contract": contract}, indent=2))


if __name__ == "__main__":
    main()
