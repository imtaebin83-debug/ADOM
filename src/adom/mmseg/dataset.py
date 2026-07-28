from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from mmseg.datasets import BaseSegDataset
from mmseg.registry import DATASETS


@DATASETS.register_module()
class AdomCost4Dataset(BaseSegDataset):
    """Manifest-backed ADOM Cost4 semantic segmentation dataset."""

    METAINFO = {
        "classes": (
            "paved_low_cost",
            "natural_low_cost",
            "medium_cost",
            "high_cost_or_obstacle",
        ),
        "palette": [
            [128, 128, 128],
            [60, 180, 75],
            [255, 165, 0],
            [220, 20, 60],
        ],
    }

    def __init__(
        self,
        manifest: str,
        data_root: str,
        pipeline: list[dict[str, Any]],
        **kwargs: Any,
    ) -> None:
        self.manifest = manifest
        super().__init__(
            ann_file="",
            img_suffix="",
            seg_map_suffix="",
            data_root=data_root,
            data_prefix={},
            pipeline=pipeline,
            reduce_zero_label=False,
            **kwargs,
        )

    def load_data_list(self) -> list[dict[str, Any]]:
        root = Path(self.data_root)
        manifest_path = root / self.manifest
        if not manifest_path.is_file():
            raise FileNotFoundError(f"ADOM manifest not found: {manifest_path}")
        output: list[dict[str, Any]] = []
        seen: set[str] = set()
        with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"sample_id", "image_relpath", "mask_relpath"}
            missing = required - set(reader.fieldnames or ())
            if missing:
                raise ValueError(
                    f"Manifest {manifest_path} is missing fields: {sorted(missing)}"
                )
            for row in reader:
                sample_id = row["sample_id"]
                if sample_id in seen:
                    raise ValueError(f"Duplicate manifest sample_id: {sample_id}")
                seen.add(sample_id)
                image_path = root / row["image_relpath"]
                mask_path = root / row["mask_relpath"]
                if not image_path.is_file() or not mask_path.is_file():
                    raise FileNotFoundError(
                        f"Missing manifest pair for {sample_id}: "
                        f"{image_path}, {mask_path}"
                    )
                output.append(
                    {
                        "sample_id": sample_id,
                        "img_path": str(image_path),
                        "seg_map_path": str(mask_path),
                        "label_map": self.label_map,
                        "reduce_zero_label": False,
                        "seg_fields": [],
                    }
                )
        if not output:
            raise ValueError(f"Manifest has no samples: {manifest_path}")
        return output
