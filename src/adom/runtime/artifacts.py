from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from adom.data.io import sha256_file, write_json
from adom.data.schema import LabelSchema


def _git_sha(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def write_export_metadata(
    *,
    repo_root: Path,
    dataset_root: Path,
    model_config: Path,
    checkpoint: Path,
    onnx_path: Path,
    deploy_info: Path,
    detail_info: Path,
    pipeline_info: Path,
    test_metrics: Path,
    mapping_path: Path,
    model_variant: str,
    profile: str,
    width: int,
    height: int,
    output: Path,
) -> None:
    schema = LabelSchema.from_path(mapping_path)
    dataset_metadata = json.loads(
        (dataset_root / "metadata" / "dataset.json").read_text(encoding="utf-8")
    )
    metadata = {
        "format_version": 1,
        "model": "SegFormer",
        "model_variant": model_variant,
        "profile": profile,
        "input": {
            "shape_nchw": [1, 3, height, width],
            "dtype": "float32",
            "resize": "keep_ratio",
            "padding": "right_and_bottom",
            "padding_rgb": [0, 0, 0],
            "normalization": {
                "mean": [123.675, 116.28, 103.53],
                "std": [58.395, 57.12, 57.375],
                "source_channel_order": "BGR",
                "model_channel_order": "RGB",
            },
        },
        "classes": [
            {"id": class_id, "name": schema.target_classes[class_id]}
            for class_id in range(4)
        ],
        "ignore_index": schema.ignore_index,
        "dataset": {
            "name": dataset_metadata.get("dataset"),
            "version": dataset_metadata.get("version"),
            "checksum_manifest_sha256": sha256_file(
                dataset_root / "SHA256SUMS.txt"
            ),
            "official_split_only": dataset_metadata.get("official_split_only"),
            "temporal_correlation_caveat": True,
        },
        "artifacts": {
            "onnx": {
                "path": onnx_path.name,
                "sha256": sha256_file(onnx_path),
            },
            "checkpoint": {
                "path": checkpoint.name,
                "sha256": sha256_file(checkpoint),
            },
            "model_config": {
                "path": model_config.resolve()
                .relative_to(repo_root.resolve())
                .as_posix(),
                "sha256": sha256_file(model_config),
            },
            "deploy_info": {
                "path": deploy_info.name,
                "sha256": sha256_file(deploy_info),
            },
            "detail_info": {
                "path": detail_info.name,
                "sha256": sha256_file(detail_info),
            },
            "pipeline_info": {
                "path": pipeline_info.name,
                "sha256": sha256_file(pipeline_info),
            },
            "test_metrics": {
                "path": test_metrics.name,
                "sha256": sha256_file(test_metrics),
            },
            "label_mapping": {
                "path": "config/label_mapping.yaml",
                "sha256": sha256_file(mapping_path),
            },
            "dataset_checksums": {
                "path": "SHA256SUMS.txt",
                "sha256": sha256_file(dataset_root / "SHA256SUMS.txt"),
            },
        },
        "git_sha": _git_sha(repo_root),
        "deployment_target": "NVIDIA Jetson Orin Nano 8GB",
        "tensorrt_engine_included": False,
    }
    write_json(output, metadata)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--model-config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--onnx", required=True, type=Path)
    parser.add_argument("--deploy-info", required=True, type=Path)
    parser.add_argument("--detail-info", required=True, type=Path)
    parser.add_argument("--pipeline-info", required=True, type=Path)
    parser.add_argument("--test-metrics", required=True, type=Path)
    parser.add_argument("--mapping", required=True, type=Path)
    parser.add_argument("--model-variant", required=True, choices=["b0", "b2"])
    parser.add_argument("--profile", required=True)
    parser.add_argument("--width", required=True, type=int)
    parser.add_argument("--height", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    write_export_metadata(
        repo_root=args.repo_root,
        dataset_root=args.dataset_root,
        model_config=args.model_config,
        checkpoint=args.checkpoint,
        onnx_path=args.onnx,
        deploy_info=args.deploy_info,
        detail_info=args.detail_info,
        pipeline_info=args.pipeline_info,
        test_metrics=args.test_metrics,
        mapping_path=args.mapping,
        model_variant=args.model_variant,
        profile=args.profile,
        width=args.width,
        height=args.height,
        output=args.output,
    )
    print(json.dumps({"metadata": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
