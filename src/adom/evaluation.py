from __future__ import annotations

import numpy as np


CLASS_NAMES = (
    "paved_low_cost",
    "natural_low_cost",
    "medium_cost",
    "high_cost_or_obstacle",
)


def metrics_from_confusion(confusion: np.ndarray) -> dict[str, float]:
    """Compute Cost4 metrics from a ground-truth-row confusion matrix."""
    confusion = np.asarray(confusion, dtype=np.float64)
    if confusion.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 confusion matrix, got {confusion.shape}")
    true_positive = np.diag(confusion)
    gt_count = confusion.sum(axis=1)
    pred_count = confusion.sum(axis=0)
    union = gt_count + pred_count - true_positive
    iou = np.divide(
        true_positive,
        union,
        out=np.zeros_like(true_positive),
        where=union > 0,
    )
    precision = np.divide(
        true_positive,
        pred_count,
        out=np.zeros_like(true_positive),
        where=pred_count > 0,
    )
    recall = np.divide(
        true_positive,
        gt_count,
        out=np.zeros_like(true_positive),
        where=gt_count > 0,
    )
    output: dict[str, float] = {}
    for index, name in enumerate(CLASS_NAMES):
        output[f"IoU/{name}"] = float(iou[index])
        output[f"Precision/{name}"] = float(precision[index])
        output[f"Recall/{name}"] = float(recall[index])
    output["mIoU"] = float(iou.mean())
    output["high_cost_or_obstacle_recall"] = float(recall[3])

    # Positive means "the model says traversable". A correct positive may be
    # any ground-truth/prediction pair within classes 0, 1, and 2.
    traversable_tp = confusion[:3, :3].sum()
    traversable_predictions = confusion[:, :3].sum()
    output["traversable_precision"] = (
        float(traversable_tp / traversable_predictions)
        if traversable_predictions > 0
        else 0.0
    )
    return output
