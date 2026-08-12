from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import tempfile
from typing import Any


EXPECTED_INPUT = {"name": "input", "shape": [1, 3, 384, 640], "dtype": "tensor(float)"}
EXPECTED_OUTPUT = {
    "name": "output",
    "shape": [1, 19, 384, 640],
    "dtype": "tensor(float)",
}
CHECKPOINT_SHA256 = "d76229ff623eb382fd48011decf54c342d88a113bcbe650fb58cc20e42cabe73"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_parity(report: dict[str, Any]) -> None:
    summary = report.get("summary", {})
    contract = report.get("onnx_contract", {})
    thresholds = report.get("thresholds", {})
    if report.get("status") != "PASS" or report.get("num_images", 0) < 10:
        raise ValueError(
            "Hand-off requires a passing parity report with at least 10 images"
        )
    if summary.get("all_finite_logits") is not True:
        raise ValueError("Parity report contains non-finite logits")
    if summary.get("reported_class_ids") != list(range(19)):
        raise ValueError("Parity report must cover all Semantic20 IDs 0..18")
    if summary.get("maximum_absolute_error", float("inf")) > thresholds.get(
        "max_absolute_error", 0.001
    ):
        raise ValueError("Parity maximum absolute error exceeds its threshold")
    agreement = thresholds.get("pixel_argmax_agreement", 0.999)
    if summary.get("minimum_per_image_argmax_agreement", 0.0) < agreement:
        raise ValueError("Per-image argmax agreement is below its threshold")
    if summary.get("overall_pixel_argmax_agreement", 0.0) < agreement:
        raise ValueError("Overall argmax agreement is below its threshold")
    if contract.get("inputs") != [EXPECTED_INPUT]:
        raise ValueError("Unexpected ONNX input contract")
    if contract.get("outputs") != [EXPECTED_OUTPUT]:
        raise ValueError("Unexpected ONNX output contract")
    if contract.get("opsets") != [{"domain": "ai.onnx", "version": 13}]:
        raise ValueError("Semantic20 hand-off requires ai.onnx opset 13")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_checksums(root: Path) -> None:
    files = sorted(path for path in root.rglob("*") if path.is_file())
    (root / "SHA256SUMS").write_text(
        "".join(
            f"{sha256_file(path)}  {path.relative_to(root).as_posix()}\n"
            for path in files
        ),
        encoding="utf-8",
    )


def build_package(args: argparse.Namespace) -> Path:
    from mmengine.config import Config

    from adom.evaluation_semantic20 import SEMANTIC20_CLASSES
    from adom.mmseg.dataset import SEMANTIC20_PALETTE

    required_files = (
        args.onnx,
        args.checkpoint,
        args.model_config,
        args.deploy_config,
        args.parity_report,
        args.engine_script,
    )
    for path in required_files:
        if not path.is_file():
            raise FileNotFoundError(path)
    if not args.reference_io.is_dir():
        raise FileNotFoundError(args.reference_io)
    parity = json.loads(args.parity_report.read_text(encoding="utf-8"))
    validate_parity(parity)
    if sha256_file(args.checkpoint) != args.expected_checkpoint_sha256:
        raise ValueError("Checkpoint SHA256 does not match the selected E0 B0 artifact")
    reference_counts = {
        suffix: len(list(args.reference_io.glob(f"*_{suffix}")))
        for suffix in ("input.npy", "pytorch_mask.png", "onnx_mask.png")
    }
    if any(count != parity["num_images"] for count in reference_counts.values()):
        raise ValueError(
            f"Reference I/O counts do not match parity: {reference_counts}"
        )
    if not list(args.reference_io.glob("*_onnx_overlay.png")):
        raise ValueError("Reference I/O must contain at least one overlay")

    final_dir = args.output_root / args.package_name
    archive = args.output_root / f"{args.package_name}.tar.gz"
    if final_dir.exists() or archive.exists():
        raise FileExistsError("Refusing to overwrite an existing hand-off artifact")
    args.output_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output_root) as temporary:
        root = Path(temporary) / args.package_name
        root.mkdir()
        shutil.copy2(args.onnx, root / "model_static_1x3x384x640_fp32.onnx")
        Config.fromfile(str(args.model_config)).dump(
            str(root / "resolved_mmseg_config.py")
        )
        Config.fromfile(str(args.deploy_config)).dump(
            str(root / "resolved_mmdeploy_config.py")
        )
        shutil.copy2(args.parity_report, root / "pytorch_onnx_parity.json")
        shutil.copytree(args.reference_io, root / "reference_io")
        shutil.copy2(args.engine_script, root / "build_engine.sh")
        _write_json(
            root / "labels.json",
            {
                "num_trainable_classes": 19,
                "ignore_index": 255,
                "labels": [
                    {"id": index, "name": name}
                    for index, name in enumerate(SEMANTIC20_CLASSES)
                ],
            },
        )
        _write_json(
            root / "palette.json",
            {
                "color_space": "RGB",
                "entries": [
                    {"id": index, "name": name, "rgb": color}
                    for index, (name, color) in enumerate(
                        zip(SEMANTIC20_CLASSES, SEMANTIC20_PALETTE)
                    )
                ],
            },
        )
        _write_json(
            root / "preprocess.json",
            {
                "source": {"layout": "HWC", "dtype": "uint8", "channel_order": "RGB"},
                "resize": {
                    "keep_ratio": True,
                    "maximum_size_wh": [640, 384],
                    "interpolation": "bilinear",
                },
                "pad": {
                    "output_size_hw": [384, 640],
                    "placement": "right_and_bottom",
                    "raw_rgb_value": [0, 0, 0],
                    "performed_before_normalization": True,
                },
                "normalize": {
                    "mean": [123.675, 116.28, 103.53],
                    "std": [58.395, 57.12, 57.375],
                    "divide_by_255": False,
                },
                "tensor": {"layout": "NCHW", "shape": [1, 3, 384, 640]},
                "output": {
                    "name": "output",
                    "shape": [1, 19, 384, 640],
                    "semantic": "raw_logits",
                    "embedded_argmax": False,
                },
            },
        )
        _write_json(
            root / "export_report.json",
            {
                "status": "PASS",
                "model": "SegFormer-B0 E0 RELLIS-only Semantic20",
                "checkpoint_sha256": args.expected_checkpoint_sha256,
                "onnx_sha256": sha256_file(args.onnx),
                "parity": parity["summary"],
                "target_class": None,
                "roi": None,
            },
        )
        _write_checksums(root)
        root.replace(final_dir)
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(final_dir, arcname=args.package_name)
    (archive.parent / f"{archive.name}.sha256").write_text(
        f"{sha256_file(archive)}  {archive.name}\n", encoding="utf-8"
    )
    return archive


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build a verified Semantic20 Jetson hand-off"
    )
    parser.add_argument("--onnx", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--expected-checkpoint-sha256", default=CHECKPOINT_SHA256)
    parser.add_argument("--model-config", required=True, type=Path)
    parser.add_argument("--deploy-config", required=True, type=Path)
    parser.add_argument("--parity-report", required=True, type=Path)
    parser.add_argument("--reference-io", required=True, type=Path)
    parser.add_argument("--engine-script", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--package-name", required=True)
    args = parser.parse_args(argv)
    print(build_package(args))


if __name__ == "__main__":
    main()
