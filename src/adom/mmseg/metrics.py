from __future__ import annotations

from typing import Any

import numpy as np
from mmengine.evaluator import BaseMetric
from mmseg.registry import METRICS

from adom.evaluation import metrics_from_confusion


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
