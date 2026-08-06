from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

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

VAL_SUPPORTED13 = (
    "grass",
    "tree",
    "pole",
    "sky",
    "vehicle",
    "log",
    "person",
    "bush",
    "concrete",
    "barrier",
    "puddle",
    "mud",
    "rubble",
)
TEST_SUPPORTED11 = (
    "grass",
    "tree",
    "pole",
    "sky",
    "log",
    "bush",
    "concrete",
    "barrier",
    "puddle",
    "mud",
    "rubble",
)
CORE11 = TEST_SUPPORTED11
RARE_RISK4 = ("pole", "log", "barrier", "rubble")
AUGMENTED_RISK2 = ("pole", "rubble")
TERRAIN_HAZARD = ("water", "puddle", "mud")

PANEL_CLASSES = {
    "ValSupported13": VAL_SUPPORTED13,
    "TestSupported11": TEST_SUPPORTED11,
    "Core11": CORE11,
    "RareRisk4": RARE_RISK4,
    "AugmentedRisk2": AUGMENTED_RISK2,
    "TerrainHazard": TERRAIN_HAZARD,
}


def _divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan, dtype=np.float64),
        where=denominator > 0,
    )


def _json_number(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _indices(names: Iterable[str]) -> list[int]:
    index_by_name = {name: index for index, name in enumerate(SEMANTIC20_CLASSES)}
    return [index_by_name[name] for name in names]


def _macro(values: np.ndarray, indices: Sequence[int]) -> float | None:
    selected = values[list(indices)]
    finite = selected[np.isfinite(selected)]
    if finite.size == 0:
        return None
    return float(finite.mean() * 100.0)


def semantic20_metrics_from_confusion(
    confusion: np.ndarray,
    *,
    evaluation_split: str,
    gt_image_count: np.ndarray | None = None,
    pred_image_count: np.ndarray | None = None,
    absent_fp_image_count: np.ndarray | None = None,
    image_count: int | None = None,
) -> tuple[dict[str, Any], dict[str, float]]:
    """Compute Clean v1 Semantic20 metrics from a GT-row confusion matrix.

    Overall averages use predeclared GT-supported class sets. A class without
    GT is never converted into an IoU zero for those averages; its predictions
    are reported by the AbsentClassFP panel instead.
    """

    confusion = np.asarray(confusion, dtype=np.int64)
    class_count = len(SEMANTIC20_CLASSES)
    if confusion.shape != (class_count, class_count):
        raise ValueError(
            f"Expected a {class_count}x{class_count} confusion matrix, "
            f"got {confusion.shape}"
        )
    if evaluation_split not in {"val", "test", "diagnostic"}:
        raise ValueError(f"Unsupported evaluation split: {evaluation_split}")

    true_positive = np.diag(confusion).astype(np.float64)
    gt_total = confusion.sum(axis=1, dtype=np.float64)
    pred_total = confusion.sum(axis=0, dtype=np.float64)
    union = gt_total + pred_total - true_positive

    raw_iou = _divide(true_positive, union)
    recall = _divide(true_positive, gt_total)
    precision = _divide(true_positive, pred_total)
    dice = _divide(2.0 * true_positive, gt_total + pred_total)

    # Keep undefined per-class precision explicit in the artifact, but count a
    # supported class with no predictions as zero in macro panels. Otherwise
    # the macro denominator would silently shrink and make the panel look
    # better precisely when a class is never predicted.
    macro_precision = precision.copy()
    macro_precision[(gt_total > 0) & ~np.isfinite(macro_precision)] = 0.0

    # For research claims, IoU is defined only when the split contains GT.
    supported_iou = raw_iou.copy()
    supported_iou[gt_total == 0] = np.nan

    if evaluation_split in {"val", "test"}:
        expected_names = (
            VAL_SUPPORTED13 if evaluation_split == "val" else TEST_SUPPORTED11
        )
        expected_indices = set(_indices(expected_names))
        actual_indices = {
            index for index, count in enumerate(gt_total) if count > 0
        }
        if actual_indices != expected_indices:
            missing = [
                SEMANTIC20_CLASSES[index]
                for index in sorted(expected_indices - actual_indices)
            ]
            unexpected = [
                SEMANTIC20_CLASSES[index]
                for index in sorted(actual_indices - expected_indices)
            ]
            raise RuntimeError(
                f"Canonical {evaluation_split} GT support changed: "
                f"missing={missing}, unexpected={unexpected}"
            )

    if gt_image_count is None:
        gt_image_count = (gt_total > 0).astype(np.int64)
    if pred_image_count is None:
        pred_image_count = (pred_total > 0).astype(np.int64)
    if absent_fp_image_count is None:
        absent_fp_image_count = np.zeros(class_count, dtype=np.int64)
    gt_image_count = np.asarray(gt_image_count, dtype=np.int64)
    pred_image_count = np.asarray(pred_image_count, dtype=np.int64)
    absent_fp_image_count = np.asarray(absent_fp_image_count, dtype=np.int64)
    for name, values in (
        ("gt_image_count", gt_image_count),
        ("pred_image_count", pred_image_count),
        ("absent_fp_image_count", absent_fp_image_count),
    ):
        if values.shape != (class_count,):
            raise ValueError(f"{name} must have shape ({class_count},)")

    non_ignore_pixels = int(confusion.sum())
    class_rows: list[dict[str, Any]] = []
    absent_rows: list[dict[str, Any]] = []
    flat: dict[str, float] = {}

    for index, name in enumerate(SEMANTIC20_CLASSES):
        gt_supported = bool(gt_total[index] > 0)
        row = {
            "id": index,
            "name": name,
            "gt_supported": gt_supported,
            "gt_pixels": int(gt_total[index]),
            "pred_pixels": int(pred_total[index]),
            "gt_images": int(gt_image_count[index]),
            "pred_images": int(pred_image_count[index]),
            "iou": _json_number(supported_iou[index] * 100.0),
            "raw_iou": _json_number(raw_iou[index] * 100.0),
            "recall": _json_number(recall[index] * 100.0),
            "precision": _json_number(precision[index] * 100.0),
            "f1_dice": _json_number(dice[index] * 100.0),
        }
        class_rows.append(row)
        for metric_name, values in (
            ("IoU", supported_iou),
            ("Recall", recall),
            ("Precision", precision),
            ("F1Dice", dice),
        ):
            value = values[index]
            if np.isfinite(value):
                flat[f"{metric_name}/{name}"] = float(value * 100.0)

        if not gt_supported:
            fp_pixels = int(pred_total[index])
            fp_images = int(absent_fp_image_count[index])
            source_pixels = confusion[:, index].astype(np.int64)
            source_rows = [
                {
                    "source_id": source_index,
                    "source_name": SEMANTIC20_CLASSES[source_index],
                    "pixels": int(count),
                    "share_of_class_fp": float(count / fp_pixels),
                }
                for source_index, count in sorted(
                    enumerate(source_pixels),
                    key=lambda item: (-int(item[1]), int(item[0])),
                )
                if count > 0
            ]
            absent_rows.append(
                {
                    "id": index,
                    "name": name,
                    "fp_pixels": fp_pixels,
                    "predicted_area_share": (
                        float(fp_pixels / non_ignore_pixels) if non_ignore_pixels else 0.0
                    ),
                    "fp_images": fp_images,
                    "fp_image_rate": (
                        float(fp_images / image_count) if image_count else None
                    ),
                    "fp_source_classes": source_rows,
                }
            )
            flat[f"AbsentClassFP/pixels/{name}"] = float(fp_pixels)
            flat[f"AbsentClassFP/area_share/{name}"] = (
                float(fp_pixels / non_ignore_pixels) if non_ignore_pixels else 0.0
            )
            if image_count:
                flat[f"AbsentClassFP/image_rate/{name}"] = float(
                    fp_images / image_count
                )

    active_panel_names = ["Core11", "RareRisk4", "AugmentedRisk2", "TerrainHazard"]
    if evaluation_split == "val":
        active_panel_names.insert(0, "ValSupported13")
    elif evaluation_split == "test":
        active_panel_names.insert(0, "TestSupported11")

    panels: dict[str, Any] = {}
    for panel_name in active_panel_names:
        names = PANEL_CLASSES[panel_name]
        panel_indices = _indices(names)
        supported_indices = [index for index in panel_indices if gt_total[index] > 0]
        metrics = {
            "mIoU": _macro(supported_iou, supported_indices),
            "mRecall": _macro(recall, supported_indices),
            "mPrecision": _macro(macro_precision, supported_indices),
            "mF1Dice": _macro(dice, supported_indices),
        }
        panels[panel_name] = {
            "classes": list(names),
            "supported_classes": [SEMANTIC20_CLASSES[index] for index in supported_indices],
            "denominator": len(supported_indices),
            **metrics,
        }
        flat[f"Denominator/{panel_name}"] = float(len(supported_indices))
        for metric_name, value in metrics.items():
            if value is not None:
                flat[f"{metric_name}/{panel_name}"] = float(value)

    aacc = float(true_positive.sum() / non_ignore_pixels * 100.0) if non_ignore_pixels else 0.0
    flat["aAcc"] = aacc
    flat["AbsentClassFP/total_pixels"] = float(
        sum(row["fp_pixels"] for row in absent_rows)
    )

    artifact = {
        "schema_version": "semantic20-clean-v1",
        "evaluation_split": evaluation_split,
        "classes": class_rows,
        "panels": panels,
        "absent_class_fp": absent_rows,
        "summary": {
            "image_count": image_count,
            "non_ignore_pixels": non_ignore_pixels,
            "aAcc": aacc,
            "macro_precision_zero_division": 0,
        },
    }
    return artifact, flat


def select_constrained_checkpoint(
    records: Sequence[dict[str, Any]],
    *,
    tolerance_pp: float = 1.0,
) -> dict[str, Any]:
    """Select highest RareRisk4 mIoU within tolerance of best ValSupported13."""

    if not records:
        raise ValueError("At least one validation record is required")
    normalized: list[dict[str, Any]] = []
    for record in records:
        overall = float(record["overall_miou"])
        rare = float(record["rare_risk_miou"])
        if not np.isfinite(overall) or not np.isfinite(rare):
            raise ValueError(f"Validation record contains non-finite metrics: {record}")
        normalized.append({**record, "overall_miou": overall, "rare_risk_miou": rare})

    best_overall = max(record["overall_miou"] for record in normalized)
    threshold = best_overall - float(tolerance_pp)
    eligible = [record for record in normalized if record["overall_miou"] >= threshold]
    return max(
        eligible,
        key=lambda record: (
            record["rare_risk_miou"],
            record["overall_miou"],
            -int(record["iteration"]),
        ),
    )
