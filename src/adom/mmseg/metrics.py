from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
from mmengine.evaluator import BaseMetric
from mmseg.registry import METRICS

from adom.evaluation import metrics_from_confusion
from adom.evaluation_semantic20 import (
    SEMANTIC20_CLASSES,
    semantic20_metrics_from_confusion,
)


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
    """Clean v1 fixed-panel metrics and permanent Semantic20 artifacts."""

    default_prefix = "semantic20"

    def __init__(
        self,
        ignore_index: int = 255,
        output_dir: str | None = None,
        evaluation_split: str = "val",
        collect_device: str = "cpu",
        prefix: str | None = None,
    ) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.ignore_index = ignore_index
        self.output_dir = output_dir
        self.evaluation_split = evaluation_split
        self.evaluation_count = 0

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
            confusion = np.bincount(
                encoded, minlength=class_count**2
            ).reshape(class_count, class_count)
            gt_presence = np.bincount(gt[in_range], minlength=class_count) > 0
            pred_presence = np.bincount(pred[in_range], minlength=class_count) > 0
            self.results.append(
                {
                    "confusion": confusion,
                    "gt_presence": gt_presence.astype(np.int64),
                    "pred_presence": pred_presence.astype(np.int64),
                    "absent_fp_presence": (
                        (~gt_presence) & pred_presence
                    ).astype(np.int64),
                    "image_count": 1,
                }
            )

    def compute_metrics(self, results: list[dict[str, Any]]) -> dict[str, float]:
        if not results:
            raise RuntimeError("AdomSemantic20Metric received no samples")
        confusion = np.sum(
            [result["confusion"] for result in results], axis=0, dtype=np.int64
        )
        gt_image_count = np.sum(
            [result["gt_presence"] for result in results], axis=0, dtype=np.int64
        )
        pred_image_count = np.sum(
            [result["pred_presence"] for result in results], axis=0, dtype=np.int64
        )
        absent_fp_image_count = np.sum(
            [result["absent_fp_presence"] for result in results],
            axis=0,
            dtype=np.int64,
        )
        image_count = int(sum(result["image_count"] for result in results))
        artifact, metrics = semantic20_metrics_from_confusion(
            confusion,
            evaluation_split=self.evaluation_split,
            gt_image_count=gt_image_count,
            pred_image_count=pred_image_count,
            absent_fp_image_count=absent_fp_image_count,
            image_count=image_count,
        )
        if self.output_dir:
            self.evaluation_count += 1
            root = Path(self.output_dir)
            root.mkdir(parents=True, exist_ok=True)
            confusion_sha = hashlib.sha256(confusion.tobytes()).hexdigest()
            suffix = (
                f"{self.evaluation_split}_{self.evaluation_count:04d}_"
                f"{confusion_sha[:12]}"
            )
            confusion_payload = {
                "schema_version": "semantic20-clean-v1",
                "classes": list(SEMANTIC20_CLASSES),
                "ignore_index": self.ignore_index,
                "evaluation_split": self.evaluation_split,
                "image_count": image_count,
                "confusion_sha256": confusion_sha,
                "matrix_convention": "rows=ground_truth, columns=prediction",
                "matrix": confusion.tolist(),
            }
            artifact["confusion_matrix_file"] = f"confusion_matrix_{suffix}.json"
            for filename, payload in (
                ("confusion_matrix.json", confusion_payload),
                (f"confusion_matrix_{suffix}.json", confusion_payload),
                ("semantic20_metrics.json", artifact),
                (f"semantic20_metrics_{suffix}.json", artifact),
            ):
                (root / filename).write_text(
                    json.dumps(payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
        return metrics
