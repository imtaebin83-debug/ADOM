from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from mmseg.datasets import BaseSegDataset
from mmseg.registry import DATASETS

from adom.evaluation_semantic20 import SEMANTIC20_CLASSES

SEMANTIC20_PALETTE = [
    [108, 64, 20],
    [0, 102, 0],
    [0, 255, 0],
    [0, 153, 153],
    [0, 128, 255],
    [0, 0, 255],
    [255, 255, 0],
    [255, 0, 127],
    [64, 64, 64],
    [255, 0, 0],
    [102, 0, 0],
    [204, 153, 255],
    [102, 0, 204],
    [255, 153, 204],
    [170, 170, 170],
    [41, 121, 255],
    [134, 255, 239],
    [99, 66, 34],
    [110, 22, 138],
]


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


@DATASETS.register_module()
class AdomSemantic20Dataset(BaseSegDataset):
    """Split-file dataset for the 19 trainable Semantic20 IDs.

    The source ontology contains 20 labels including void. Preprocessing maps
    void to 255, leaving contiguous train IDs 0..18. Both the RELLIS-only E0
    package and combined E1 package use ``images/<sample>.jpg`` and
    ``masks/<sample>.png`` with sample keys stored in ``splits/*.txt``.
    """

    METAINFO = {
        "classes": SEMANTIC20_CLASSES,
        "palette": SEMANTIC20_PALETTE,
    }

    def __init__(
        self,
        split: str,
        data_root: str,
        pipeline: list[dict[str, Any]],
        manifest: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.split = split
        self.manifest = manifest
        super().__init__(
            ann_file="",
            img_suffix=".jpg",
            seg_map_suffix=".png",
            data_root=data_root,
            data_prefix={},
            pipeline=pipeline,
            reduce_zero_label=False,
            **kwargs,
        )

    def load_data_list(self) -> list[dict[str, Any]]:
        root = Path(self.data_root)
        split_path = root / self.split
        if not split_path.is_file():
            raise FileNotFoundError(f"Semantic20 split not found: {split_path}")
        sample_keys = [
            line.strip()
            for line in split_path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        ]
        if not sample_keys:
            raise ValueError(f"Semantic20 split has no samples: {split_path}")
        if len(sample_keys) != len(set(sample_keys)):
            raise ValueError(f"Semantic20 split contains duplicate samples: {split_path}")

        manifest_rows: dict[str, tuple[str, str]] = {}
        if self.manifest:
            manifest_path = root / self.manifest
            if not manifest_path.is_file():
                raise FileNotFoundError(
                    f"Semantic20 manifest not found: {manifest_path}"
                )
            with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                required = {"sample_key", "image_path", "mask_path"}
                missing = required - set(reader.fieldnames or ())
                if missing:
                    raise ValueError(
                        f"Manifest {manifest_path} is missing fields: {sorted(missing)}"
                    )
                for row in reader:
                    sample_key = row["sample_key"]
                    if sample_key in manifest_rows:
                        raise ValueError(
                            f"Duplicate Semantic20 manifest sample: {sample_key}"
                        )
                    manifest_rows[sample_key] = (
                        row["image_path"],
                        row["mask_path"],
                    )

        output: list[dict[str, Any]] = []
        for sample_key in sample_keys:
            if self.manifest:
                if sample_key not in manifest_rows:
                    raise ValueError(
                        f"Split sample is absent from Semantic20 manifest: {sample_key}"
                    )
                image_relpath, mask_relpath = manifest_rows[sample_key]
                image_path = root / image_relpath
                mask_path = root / mask_relpath
            else:
                image_path = root / "images" / f"{sample_key}.jpg"
                mask_path = root / "masks" / f"{sample_key}.png"
            if not image_path.is_file() or not mask_path.is_file():
                raise FileNotFoundError(
                    f"Missing Semantic20 pair for {sample_key}: "
                    f"{image_path}, {mask_path}"
                )
            output.append(
                {
                    "sample_id": sample_key,
                    "img_path": str(image_path),
                    "seg_map_path": str(mask_path),
                    "label_map": self.label_map,
                    "reduce_zero_label": False,
                    "seg_fields": [],
                }
            )
        return output
