from __future__ import annotations

from typing import Any

import numpy as np
from mmengine.evaluator import BaseMetric
from mmseg.registry import METRICS

from adom.evaluation import metrics_from_confusion
from adom.mmseg.dataset import SEMANTIC20_CLASSES


@METRICS.register_module()
class AdomSafetyMetric(BaseMetric):
    """Cost4 confusion metrics with class 3 recall and 0/1/2 precision."""

    default_prefix = "safety"

    def __init__(
        self,
        ignore_index: int = 255,
        collect_device: str = "cpu",
        prefix: str | None = None,
    ) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.ignore_index = ignore_index

    def process(self, data_batch: Any, data_samples: list[Any]) -> None:
        for sample in data_samples:
            if isinstance(sample, dict):
                prediction = sample["pred_sem_seg"]["data"]
                target = sample["gt_sem_seg"]["data"]
            else:
                prediction = sample.pred_sem_seg.data
                target = sample.gt_sem_seg.data
            pred = prediction.squeeze().detach().cpu().numpy().astype(np.int64)
            gt = target.squeeze().detach().cpu().numpy().astype(np.int64)
            valid = gt != self.ignore_index
            pred = pred[valid]
            gt = gt[valid]
            in_range = (gt >= 0) & (gt < 4) & (pred >= 0) & (pred < 4)
            encoded = gt[in_range] * 4 + pred[in_range]
            confusion = np.bincount(encoded, minlength=16).reshape(4, 4)
            self.results.append(confusion)

    def compute_metrics(self, results: list[np.ndarray]) -> dict[str, float]:
        if not results:
            raise RuntimeError("AdomSafetyMetric received no samples")
        confusion = np.sum(results, axis=0)
        return metrics_from_confusion(confusion)


@METRICS.register_module()
class AdomSemantic20Metric(BaseMetric):
    """Classwise IoU/recall plus a JSON confusion matrix for Semantic20."""

    default_prefix = "semantic20"

    def __init__(
        self,
        ignore_index: int = 255,
        output_dir: str | None = None,
        collect_device: str = "cpu",
        prefix: str | None = None,
    ) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.ignore_index = ignore_index
        self.output_dir = output_dir

    def process(self, data_batch: Any, data_samples: list[Any]) -> None:
        class_count = len(SEMANTIC20_CLASSES)
        for sample in data_samples:
            if isinstance(sample, dict):
                prediction = sample["pred_sem_seg"]["data"]
                target = sample["gt_sem_seg"]["data"]
            else:
                prediction = sample.pred_sem_seg.data
                target = sample.gt_sem_seg.data
            pred = prediction.squeeze().detach().cpu().numpy().astype(np.int64)
            gt = target.squeeze().detach().cpu().numpy().astype(np.int64)
            valid = gt != self.ignore_index
            pred = pred[valid]
            gt = gt[valid]
            in_range = (
                (gt >= 0)
                & (gt < class_count)
                & (pred >= 0)
                & (pred < class_count)
            )
            encoded = gt[in_range] * class_count + pred[in_range]
            self.results.append(
                np.bincount(encoded, minlength=class_count**2).reshape(
                    class_count, class_count
                )
            )

    def compute_metrics(self, results: list[np.ndarray]) -> dict[str, float]:
        if not results:
            raise RuntimeError("AdomSemantic20Metric received no samples")
        confusion = np.sum(results, axis=0, dtype=np.int64)
        true_positive = np.diag(confusion).astype(np.float64)
        gt_total = confusion.sum(axis=1, dtype=np.float64)
        pred_total = confusion.sum(axis=0, dtype=np.float64)
        union = gt_total + pred_total - true_positive
        iou = np.divide(
            true_positive,
            union,
            out=np.full_like(true_positive, np.nan),
            where=union > 0,
        )
        recall = np.divide(
            true_positive,
            gt_total,
            out=np.full_like(true_positive, np.nan),
            where=gt_total > 0,
        )
        if self.output_dir:
            import json
            from pathlib import Path

            output_path = Path(self.output_dir) / "confusion_matrix.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(
                    {
                        "classes": list(SEMANTIC20_CLASSES),
                        "ignore_index": self.ignore_index,
                        "matrix_convention": "rows=ground_truth, columns=prediction",
                        "matrix": confusion.tolist(),
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        metrics: dict[str, float] = {}
        for index, name in enumerate(SEMANTIC20_CLASSES):
            metrics[f"IoU/{name}"] = float(iou[index] * 100.0)
            metrics[f"Recall/{name}"] = float(recall[index] * 100.0)
        return metrics
