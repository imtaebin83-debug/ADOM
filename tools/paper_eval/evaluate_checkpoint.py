from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np
from PIL import Image

from _common import (
    FOCUS_CLASSES,
    SEMANTIC20_CLASSES,
    canonical_json_sha256,
    confusion_from_arrays,
    load_mask,
    manifest_sha256,
    metrics_from_confusion,
    read_json,
    read_manifest,
    safe_prediction_name,
    sha256_file,
    write_dict_csv,
    write_json,
)


EXPECTED_CHECKPOINTS = {
    "b0_e0": "d76229ff623eb382fd48011decf54c342d88a113bcbe650fb58cc20e42cabe73",
    "eadom": "f4cc41fd91e9df8e7aa3f726498e80636b736dfadf0e1baf338fe7c82a83399c",
}


def _audit_gate(
    audit_dir: Path,
    *,
    dataset: str,
    manifest_split: str,
    model_name: str,
    manifest: Path,
    checkpoint: Path,
    expected_sha256: str,
) -> tuple[dict[str, Any], list[str]]:
    required = (
        "audit_report.md",
        "environment.json",
        "checkpoint_manifest.json",
        "dataset_manifest_summary.json",
    )
    missing = [name for name in required if not (audit_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"Audit is incomplete; missing {missing}")
    checkpoint_audit = read_json(audit_dir / "checkpoint_manifest.json")
    dataset_audit = read_json(audit_dir / "dataset_manifest_summary.json")
    if checkpoint_audit.get("status") != "PASS" or dataset_audit.get("status") != "PASS":
        raise RuntimeError("Audit gate is not PASS; inference is forbidden")
    model_audit = checkpoint_audit[model_name]
    actual_sha256 = sha256_file(checkpoint)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"Checkpoint SHA-256 mismatch: {actual_sha256} != {expected_sha256}"
        )
    if model_audit.get("actual_sha256") != actual_sha256:
        raise RuntimeError("Checkpoint differs from the audited selected artifact")
    split_name = f"{dataset}_{manifest_split}"
    if split_name not in dataset_audit["splits"]:
        raise RuntimeError(f"Audited split is missing: {split_name}")
    split_audit = dataset_audit["splits"][split_name]
    if sha256_file(manifest) != split_audit["manifest_csv_sha256"]:
        raise RuntimeError("Manifest CSV differs from the audited frozen manifest")
    return dataset_audit, list(dataset_audit["common_supported_classes"])


def _verify_manifest_files(records: list[Any]) -> None:
    for row in records:
        if not row.image_path.is_file() or not row.annotation_path.is_file():
            raise FileNotFoundError(
                f"Missing evaluated pair for {row.sample_id}: "
                f"{row.image_path}, {row.annotation_path}"
            )
        image_sha = sha256_file(row.image_path)
        annotation_sha = sha256_file(row.annotation_path)
        if image_sha != row.image_sha256 or annotation_sha != row.annotation_sha256:
            raise RuntimeError(f"Evaluated bytes changed after audit: {row.sample_id}")


def _configure_determinism(seed: int) -> Any:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except Exception as error:
        raise RuntimeError(f"PyTorch is required for checkpoint inference: {error}") from error
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    return torch


def _load_model(config_path: Path, checkpoint: Path, device: str) -> tuple[Any, str, dict[str, Any]]:
    try:
        from mmengine.config import Config
        from mmseg.apis import init_model
    except Exception as error:
        raise RuntimeError(f"MMSegmentation evaluation stack is unavailable: {error}") from error
    config = Config.fromfile(str(config_path))
    pipeline = config.get("test_pipeline")
    if pipeline is None:
        pipeline = config.get("test_dataloader", {}).get("dataset", {}).get("pipeline")
    if pipeline is None:
        raise RuntimeError("Config has no test pipeline")
    inference_pipeline = [
        dict(value)
        for value in pipeline
        if value.get("type") != "LoadAnnotations"
    ]
    if any(value.get("type") in {"RandomFlip", "MultiScaleFlipAug"} for value in inference_pipeline):
        raise RuntimeError("TTA/random augmentation is forbidden in paper evaluation")
    config.test_pipeline = inference_pipeline
    if config.model.get("test_cfg", {}).get("mode") != "whole":
        raise RuntimeError("Canonical paper evaluation requires whole-image inference")
    decode_classes = int(config.model.decode_head.num_classes)
    if decode_classes != len(SEMANTIC20_CLASSES):
        raise RuntimeError(f"Config decode head has {decode_classes} classes, expected 19")
    contract = {
        "resolved_model": config.model.to_dict(),
        "inference_pipeline": inference_pipeline,
        "classes": list(SEMANTIC20_CLASSES),
        "ignore_index": 255,
        "inference_mode": "torch.inference_mode",
        "tta": False,
        "prediction_resize_policy": "MMSeg postprocess to original shape; shape mismatch is fatal",
    }
    model = init_model(config, str(checkpoint), device=device)
    model.eval()
    return model, canonical_json_sha256(contract), contract


def _prediction(result: Any) -> np.ndarray:
    value = result.pred_sem_seg.data
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    output = np.asarray(value)
    if output.ndim == 3 and output.shape[0] == 1:
        output = output[0]
    if output.ndim != 2:
        raise ValueError(f"Unexpected MMSeg prediction shape: {output.shape}")
    return output.astype(np.int64, copy=False)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    audit_dir = args.audit_dir.resolve()
    manifest_path = args.manifest.resolve()
    checkpoint = args.checkpoint.resolve()
    config_path = args.config.resolve()
    expected_sha256 = args.expected_checkpoint_sha256 or EXPECTED_CHECKPOINTS[args.model]
    dataset_audit, common_classes = _audit_gate(
        audit_dir,
        dataset=args.dataset,
        manifest_split=args.manifest_split,
        model_name=args.model,
        manifest=manifest_path,
        checkpoint=checkpoint,
        expected_sha256=expected_sha256,
    )
    records = read_manifest(manifest_path)
    if any(
        row.dataset != args.dataset or row.split != args.manifest_split
        for row in records
    ):
        raise ValueError(
            "Manifest dataset/split does not match --dataset/--manifest-split"
        )
    expected_manifest_digest = dataset_audit["splits"][
        f"{args.dataset}_{args.manifest_split}"
    ]["manifest_sha256"]
    if manifest_sha256(records) != expected_manifest_digest:
        raise RuntimeError("Ordered manifest content differs from dataset audit")
    _verify_manifest_files(records)

    torch = _configure_determinism(args.seed)
    model, evaluation_contract_sha256, evaluation_contract = _load_model(
        config_path, checkpoint, args.device
    )
    try:
        from mmseg.apis import inference_model
    except Exception as error:  # pragma: no cover - covered by _load_model in practice
        raise RuntimeError(str(error)) from error

    metrics_dir = args.output_dir.resolve() / "metrics"
    split_suffix = "" if args.manifest_split == "test" else f"_{args.manifest_split}"
    prefix = f"{args.dataset}{split_suffix}__{args.model}"
    output_paths = {
        "summary": metrics_dir / f"{prefix}__summary.json",
        "per_class": metrics_dir / f"{prefix}__per_class.csv",
        "confusion": metrics_dir / f"{prefix}__confusion_matrix.npy",
        "per_image": metrics_dir / f"{prefix}__per_image.csv",
        "per_image_confusions": metrics_dir / f"{prefix}__per_image_confusions.npz",
    }
    if any(path.exists() for path in output_paths.values()):
        raise FileExistsError(f"Refusing to overwrite existing evaluation outputs for {prefix}")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir = args.output_dir.resolve() / "predictions" / prefix
    if prediction_dir.exists() and any(prediction_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite predictions: {prediction_dir}")
    prediction_dir.mkdir(parents=True, exist_ok=True)

    confusion = np.zeros((len(SEMANTIC20_CLASSES), len(SEMANTIC20_CLASSES)), dtype=np.int64)
    ignored_pixels = 0
    image_confusions: list[np.ndarray] = []
    image_ignored: list[int] = []
    per_image: list[dict[str, Any]] = []
    durations: list[float] = []
    started = time.perf_counter()
    for order, record in enumerate(records):
        ground_truth = load_mask(record.annotation_path)
        before = time.perf_counter()
        with torch.inference_mode():
            result = inference_model(model, str(record.image_path))
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        durations.append(time.perf_counter() - before)
        prediction = _prediction(result)
        image_confusion, image_ignore = confusion_from_arrays(ground_truth, prediction)
        confusion += image_confusion
        ignored_pixels += image_ignore
        image_confusions.append(image_confusion)
        image_ignored.append(image_ignore)
        image_summary, image_rows = metrics_from_confusion(
            image_confusion,
            ignored_pixels=image_ignore,
            common_classes=[
                name
                for name in common_classes
                if image_confusion[SEMANTIC20_CLASSES.index(name)].sum() > 0
            ],
        )
        prediction_path = prediction_dir / safe_prediction_name(record.sample_id)
        Image.fromarray(prediction.astype(np.uint8), mode="L").save(prediction_path)
        row: dict[str, Any] = {
            "dataset": args.dataset,
            "split": args.manifest_split,
            "model": args.model,
            "sample_id": record.sample_id,
            "sequence": record.sequence,
            "manifest_order": order,
            "supported_mIoU": image_summary["dataset_native_supported_mIoU"],
            "common_supported_mIoU": image_summary["common_supported_mIoU"],
            "evaluated_pixels": image_summary["total_evaluated_pixels"],
            "ignored_pixels": image_ignore,
            "prediction_path": str(prediction_path.resolve()),
        }
        for class_name in FOCUS_CLASSES:
            class_row = image_rows[SEMANTIC20_CLASSES.index(class_name)]
            row[f"{class_name}_gt_pixels"] = class_row["gt_pixel_count"]
            row[f"{class_name}_iou"] = class_row["iou"]
            row[f"{class_name}_recall"] = class_row["recall"]
        per_image.append(row)
        if args.progress_every and (order + 1) % args.progress_every == 0:
            print(f"{prefix}: {order + 1}/{len(records)}", flush=True)

    elapsed = time.perf_counter() - started
    evaluated_common_classes = [
        name
        for name in common_classes
        if confusion[SEMANTIC20_CLASSES.index(name)].sum() > 0
    ]
    aggregate, per_class = metrics_from_confusion(
        confusion,
        ignored_pixels=ignored_pixels,
        common_classes=evaluated_common_classes,
    )
    summary = {
        "schema_version": "adom-paper-eval-metrics-v1",
        "dataset": args.dataset,
        "split": args.manifest_split,
        "model": args.model,
        "metrics": aggregate,
        "provenance": {
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": expected_sha256,
            "config_path": str(config_path),
            "config_file_sha256": sha256_file(config_path),
            "evaluation_contract_sha256": evaluation_contract_sha256,
            "evaluation_contract": evaluation_contract,
            "manifest_path": str(manifest_path),
            "manifest_csv_sha256": sha256_file(manifest_path),
            "ordered_manifest_sha256": manifest_sha256(records),
            "audit_dir": str(audit_dir),
            "seed": args.seed,
            "device": args.device,
            "inference_mode": "torch.inference_mode",
            "tta": False,
        },
        "runtime": {
            "image_count": len(records),
            "wall_seconds": elapsed,
            "mean_inference_seconds": float(np.mean(durations)),
            "p95_inference_seconds": float(np.percentile(durations, 95)),
            "peak_cuda_memory_bytes": (
                int(torch.cuda.max_memory_allocated())
                if args.device.startswith("cuda")
                else None
            ),
        },
        "artifacts": {name: str(path.resolve()) for name, path in output_paths.items()},
    }
    np.save(output_paths["confusion"], confusion, allow_pickle=False)
    np.savez_compressed(
        output_paths["per_image_confusions"],
        sample_ids=np.asarray([row.sample_id for row in records], dtype=str),
        sequences=np.asarray([row.sequence for row in records], dtype=str),
        confusions=np.stack(image_confusions),
        ignored_pixels=np.asarray(image_ignored, dtype=np.int64),
        common_classes=np.asarray(evaluated_common_classes, dtype=str),
    )
    per_class_fields = (
        "class_id",
        "class_name",
        "gt_supported",
        "gt_pixel_count",
        "prediction_pixel_count",
        "true_positive",
        "false_positive",
        "false_negative",
        "iou",
        "precision",
        "recall",
        "f1",
        "absent_class_false_positive",
    )
    write_dict_csv(output_paths["per_class"], per_class, per_class_fields)
    per_image_fields = [
        "dataset",
        "split",
        "model",
        "sample_id",
        "sequence",
        "manifest_order",
        "supported_mIoU",
        "common_supported_mIoU",
        "evaluated_pixels",
        "ignored_pixels",
    ]
    for class_name in FOCUS_CLASSES:
        per_image_fields.extend(
            (f"{class_name}_gt_pixels", f"{class_name}_iou", f"{class_name}_recall")
        )
    per_image_fields.append("prediction_path")
    write_dict_csv(output_paths["per_image"], per_image, per_image_fields)
    write_json(output_paths["summary"], summary)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run direct MMSeg inference and accumulate a fresh Semantic20 confusion matrix"
    )
    parser.add_argument("--audit-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dataset", required=True, choices=("rellis", "korean"))
    parser.add_argument(
        "--manifest-split",
        default="test",
        help="Audited split name. Defaults to test to preserve paper-evaluation names.",
    )
    parser.add_argument("--model", required=True, choices=("b0_e0", "eadom"))
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--expected-checkpoint-sha256")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress-every", type=int, default=50)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    result = evaluate(parse_args(argv))
    print(json.dumps(result["metrics"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
