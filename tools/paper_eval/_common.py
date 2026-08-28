from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Sequence

import numpy as np


SEMANTIC20_CLASSES = (
    "dirt",
    "grass",
    "tree",
    "pole",
    "water",
    "sky",
    "vehicle",
    "object",
    "asphalt",
    "building",
    "log",
    "person",
    "fence",
    "bush",
    "concrete",
    "barrier",
    "puddle",
    "mud",
    "rubble",
)
SEMANTIC20_PALETTE = (
    (108, 64, 20),
    (0, 102, 0),
    (0, 255, 0),
    (0, 153, 153),
    (0, 128, 255),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 127),
    (64, 64, 64),
    (255, 0, 0),
    (102, 0, 0),
    (204, 153, 255),
    (102, 0, 204),
    (255, 153, 204),
    (170, 170, 170),
    (41, 121, 255),
    (134, 255, 239),
    (99, 66, 34),
    (110, 22, 138),
)
FOCUS_CLASSES = ("log", "rubble", "pole", "barrier", "mud", "puddle")
IGNORE_INDEX = 255
CLASS_INDEX = {name: index for index, name in enumerate(SEMANTIC20_CLASSES)}


@dataclass(frozen=True)
class ManifestRecord:
    dataset: str
    split: str
    sample_id: str
    sequence: str
    image_path: Path
    annotation_path: Path
    image_sha256: str
    annotation_sha256: str
    width: int
    height: int
    source: str = ""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_json_sha256(value: Any) -> str:
    serialized = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return sha256_text(serialized)


def manifest_sha256(records: Sequence[ManifestRecord]) -> str:
    rows = [
        {
            "sample_id": row.sample_id,
            "sequence": row.sequence,
            "image_sha256": row.image_sha256,
            "annotation_sha256": row.annotation_sha256,
            "width": row.width,
            "height": row.height,
        }
        for row in records
    ]
    return canonical_json_sha256(rows)


def read_manifest(path: Path) -> list[ManifestRecord]:
    output: list[ManifestRecord] = []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {
            "dataset",
            "split",
            "sample_id",
            "sequence",
            "image_path",
            "annotation_path",
            "image_sha256",
            "annotation_sha256",
            "width",
            "height",
        }
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Manifest {path} is missing fields: {sorted(missing)}")
        for row in reader:
            output.append(
                ManifestRecord(
                    dataset=row["dataset"],
                    split=row["split"],
                    sample_id=row["sample_id"],
                    sequence=row["sequence"],
                    image_path=Path(row["image_path"]),
                    annotation_path=Path(row["annotation_path"]),
                    image_sha256=row["image_sha256"],
                    annotation_sha256=row["annotation_sha256"],
                    width=int(row["width"] or 0),
                    height=int(row["height"] or 0),
                    source=row.get("source", ""),
                )
            )
    if not output:
        raise ValueError(f"Manifest is empty: {path}")
    sample_ids = [row.sample_id for row in output]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError(f"Manifest contains duplicate sample IDs: {path}")
    return output


def write_manifest(path: Path, records: Sequence[ManifestRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "dataset",
        "split",
        "sample_id",
        "sequence",
        "source",
        "image_path",
        "annotation_path",
        "image_sha256",
        "annotation_sha256",
        "width",
        "height",
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in records:
            writer.writerow(
                {
                    "dataset": row.dataset,
                    "split": row.split,
                    "sample_id": row.sample_id,
                    "sequence": row.sequence,
                    "source": row.source,
                    "image_path": str(row.image_path.resolve()),
                    "annotation_path": str(row.annotation_path.resolve()),
                    "image_sha256": row.image_sha256,
                    "annotation_sha256": row.annotation_sha256,
                    "width": row.width,
                    "height": row.height,
                }
            )


def load_mask(path: Path) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as image:
        value = np.asarray(image)
    if value.ndim != 2:
        raise ValueError(f"Expected a single-channel train-ID mask: {path}")
    return value.astype(np.int64, copy=False)


def confusion_from_arrays(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    *,
    ignore_index: int = IGNORE_INDEX,
) -> tuple[np.ndarray, int]:
    ground_truth = np.asarray(ground_truth, dtype=np.int64)
    prediction = np.asarray(prediction, dtype=np.int64)
    if ground_truth.shape != prediction.shape:
        raise ValueError(
            f"Prediction/GT shape mismatch: {prediction.shape} != {ground_truth.shape}"
        )
    valid_gt = ((ground_truth >= 0) & (ground_truth < len(SEMANTIC20_CLASSES))) | (
        ground_truth == ignore_index
    )
    if not np.all(valid_gt):
        values = sorted(int(value) for value in np.unique(ground_truth[~valid_gt]))
        raise ValueError(f"GT contains invalid Semantic20 IDs: {values}")
    valid = ground_truth != ignore_index
    invalid_predictions = valid & (
        (prediction < 0) | (prediction >= len(SEMANTIC20_CLASSES))
    )
    if np.any(invalid_predictions):
        values = sorted(int(value) for value in np.unique(prediction[invalid_predictions]))
        raise ValueError(f"Prediction contains invalid Semantic20 IDs: {values}")
    encoded = (
        len(SEMANTIC20_CLASSES) * ground_truth[valid] + prediction[valid]
    )
    confusion = np.bincount(
        encoded, minlength=len(SEMANTIC20_CLASSES) ** 2
    ).reshape(len(SEMANTIC20_CLASSES), len(SEMANTIC20_CLASSES))
    return confusion.astype(np.int64, copy=False), int(np.count_nonzero(~valid))


def _ratio(numerator: float, denominator: float) -> float | None:
    if denominator <= 0:
        return None
    return float(numerator / denominator)


def _percent(value: float | None) -> float | None:
    return None if value is None else float(value * 100.0)


def metrics_from_confusion(
    confusion: np.ndarray,
    *,
    ignored_pixels: int = 0,
    common_classes: Sequence[str] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    confusion = np.asarray(confusion, dtype=np.int64)
    class_count = len(SEMANTIC20_CLASSES)
    if confusion.shape != (class_count, class_count):
        raise ValueError(f"Expected {class_count}x{class_count}, got {confusion.shape}")
    if np.any(confusion < 0):
        raise ValueError("Confusion matrix cannot contain negative counts")

    true_positive = np.diag(confusion)
    gt_pixels = confusion.sum(axis=1)
    pred_pixels = confusion.sum(axis=0)
    false_positive = pred_pixels - true_positive
    false_negative = gt_pixels - true_positive
    supported_ids = [index for index, count in enumerate(gt_pixels) if count > 0]

    rows: list[dict[str, Any]] = []
    ious: dict[str, float | None] = {}
    for index, name in enumerate(SEMANTIC20_CLASSES):
        gt = int(gt_pixels[index])
        pred = int(pred_pixels[index])
        tp = int(true_positive[index])
        fp = int(false_positive[index])
        fn = int(false_negative[index])
        union = gt + pred - tp
        # Research IoU is undefined when the dataset has no GT for the class,
        # even if predictions create absent-class false positives.
        iou = _ratio(tp, union) if gt > 0 else None
        precision = _ratio(tp, pred)
        recall = _ratio(tp, gt)
        f1 = _ratio(2 * tp, 2 * tp + fp + fn)
        ious[name] = iou
        rows.append(
            {
                "class_id": index,
                "class_name": name,
                "gt_supported": gt > 0,
                "gt_pixel_count": gt,
                "prediction_pixel_count": pred,
                "true_positive": tp,
                "false_positive": fp,
                "false_negative": fn,
                "iou": _percent(iou),
                "precision": _percent(precision),
                "recall": _percent(recall),
                "f1": _percent(f1),
                "absent_class_false_positive": pred if gt == 0 else 0,
            }
        )

    native_values = [ious[SEMANTIC20_CLASSES[index]] for index in supported_ids]
    native_miou = (
        float(np.mean([value for value in native_values if value is not None]))
        if native_values
        else None
    )
    common_names = list(common_classes or ())
    invalid_common = [name for name in common_names if name not in CLASS_INDEX]
    if invalid_common:
        raise ValueError(f"Unknown common Semantic20 classes: {invalid_common}")
    unsupported_common = [name for name in common_names if gt_pixels[CLASS_INDEX[name]] == 0]
    if unsupported_common:
        raise ValueError(
            f"Common class set contains GT-absent classes: {unsupported_common}"
        )
    common_values = [ious[name] for name in common_names]
    common_miou = (
        float(np.mean([value for value in common_values if value is not None]))
        if common_values
        else None
    )
    evaluated_pixels = int(confusion.sum())
    accuracy = _ratio(int(true_positive.sum()), evaluated_pixels)
    summary = {
        "metric_unit": "percent",
        "aAcc": _percent(accuracy),
        "dataset_native_supported_mIoU": _percent(native_miou),
        "common_supported_mIoU": _percent(common_miou),
        "supported_class_count": len(supported_ids),
        "supported_classes": [SEMANTIC20_CLASSES[index] for index in supported_ids],
        "common_class_count": len(common_names),
        "common_classes": common_names,
        "total_evaluated_pixels": evaluated_pixels,
        "ignored_pixels": int(ignored_pixels),
        "absent_class_false_positive_pixels": {
            row["class_name"]: row["absent_class_false_positive"]
            for row in rows
            if not row["gt_supported"]
        },
    }
    return summary, rows


def metric_value(
    confusion: np.ndarray,
    metric: str,
    *,
    common_classes: Sequence[str],
) -> float | None:
    summary, rows = metrics_from_confusion(
        confusion, common_classes=common_classes
    )
    if metric == "common_supported_mIoU":
        return summary["common_supported_mIoU"]
    match = re.fullmatch(r"([^/]+)/(IoU|Recall)", metric)
    if not match:
        raise ValueError(f"Unsupported paired metric: {metric}")
    class_name, metric_name = match.groups()
    row = rows[CLASS_INDEX[class_name]]
    return row[metric_name.lower()]


def csv_value(value: Any) -> Any:
    if value is None:
        return "N/A"
    if isinstance(value, float) and not math.isfinite(value):
        return "N/A"
    return value


def write_dict_csv(path: Path, rows: Sequence[dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: csv_value(row.get(field)) for field in fields})


def markdown_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    output = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        output.append("| " + " | ".join(str(csv_value(value)) for value in row) + " |")
    return "\n".join(output) + "\n"


def safe_prediction_name(sample_id: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", sample_id).strip("._") or "sample"
    digest = sha256_text(sample_id)[:12]
    return f"{digest}__{stem[-100:]}.png"
