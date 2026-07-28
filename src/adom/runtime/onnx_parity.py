from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _as_input_tensor(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise RuntimeError(f"Expected one input tensor, got {len(value)}")
        value = value[0]
    return value


def compare(
    deploy_config: Path,
    model_config: Path,
    checkpoint: Path,
    onnx_path: Path,
    image: Path,
    device: str,
) -> dict[str, Any]:
    import onnxruntime as ort
    import torch
    from mmdeploy.apis.utils import build_task_processor
    from mmdeploy.utils import load_config

    deploy_cfg, model_cfg = load_config(str(deploy_config), str(model_config))
    task_processor = build_task_processor(model_cfg, deploy_cfg, device)
    pytorch_model = task_processor.build_pytorch_model(str(checkpoint))
    pytorch_model.eval()
    model_inputs, _ = task_processor.create_input(
        str(image),
        input_shape=deploy_cfg.onnx_config.input_shape,
    )
    processed = pytorch_model.data_preprocessor(model_inputs, training=False)
    tensor = _as_input_tensor(processed["inputs"])
    with torch.no_grad():
        torch_logits = pytorch_model(
            tensor,
            data_samples=processed.get("data_samples"),
            mode="tensor",
        )
    if isinstance(torch_logits, (tuple, list)):
        torch_logits = torch_logits[0]
    torch_array = torch_logits.detach().cpu().numpy()

    available_providers = set(ort.get_available_providers())
    providers = []
    if device.startswith("cuda") and "CUDAExecutionProvider" in available_providers:
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")
    session = ort.InferenceSession(str(onnx_path), providers=providers)
    input_name = session.get_inputs()[0].name
    onnx_array = session.run(None, {input_name: tensor.detach().cpu().numpy()})[0]
    if (
        torch_array.ndim == 4
        and onnx_array.ndim == 4
        and torch_array.shape[:2] == onnx_array.shape[:2]
        and torch_array.shape[2:] != onnx_array.shape[2:]
    ):
        # MMDeploy's segmentation rewrite may include the standard bilinear
        # resize performed by EncoderDecoder.predict, while mode="tensor"
        # returns decode-head resolution.
        torch_array = (
            torch.nn.functional.interpolate(
                torch.from_numpy(torch_array),
                size=onnx_array.shape[2:],
                mode="bilinear",
                align_corners=False,
            )
            .numpy()
        )
    if torch_array.shape != onnx_array.shape:
        raise RuntimeError(
            f"Output shape mismatch: torch={torch_array.shape}, onnx={onnx_array.shape}"
        )
    if not np.isfinite(torch_array).all() or not np.isfinite(onnx_array).all():
        raise RuntimeError("Non-finite logits detected during ONNX parity")

    max_absolute_error = float(np.max(np.abs(torch_array - onnx_array)))
    torch_prediction = np.argmax(torch_array, axis=1)
    onnx_prediction = np.argmax(onnx_array, axis=1)
    agreement = float(np.mean(torch_prediction == onnx_prediction))
    passed = max_absolute_error <= 1e-3 and agreement >= 0.999
    return {
        "status": "PASS" if passed else "FAIL",
        "torch_output_shape": list(torch_array.shape),
        "onnx_output_shape": list(onnx_array.shape),
        "finite_logits": True,
        "max_absolute_error": max_absolute_error,
        "pixel_argmax_agreement": agreement,
        "thresholds": {
            "max_absolute_error": 1e-3,
            "pixel_argmax_agreement": 0.999,
        },
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compare PyTorch and ONNX logits")
    parser.add_argument("--deploy-config", required=True, type=Path)
    parser.add_argument("--model-config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--onnx", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    report = compare(
        args.deploy_config,
        args.model_config,
        args.checkpoint,
        args.onnx,
        args.image,
        args.device,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
