from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .io import load_structured_text
from .models import DatasetError


COST4_CLASSES = {
    0: "paved_low_cost",
    1: "natural_low_cost",
    2: "medium_cost",
    3: "high_cost_or_obstacle",
}
COST4_PALETTE = {
    0: (128, 128, 128),
    1: (60, 180, 75),
    2: (255, 165, 0),
    3: (220, 20, 60),
    255: (0, 0, 0),
}
IGNORE_INDEX = 255
ALLOWED_TARGET_IDS = frozenset({0, 1, 2, 3, IGNORE_INDEX})


@dataclass(frozen=True)
class LabelSchema:
    dataset: str
    version: str
    source_to_target: dict[int, int]
    target_classes: dict[int, str]
    ignore_index: int = IGNORE_INDEX

    @classmethod
    def from_path(cls, path: Path) -> "LabelSchema":
        raw = load_structured_text(path)
        if "source_to_target" in raw:
            mapping_raw = raw["source_to_target"]
            target_raw = raw.get("target", {}).get("classes", COST4_CLASSES)
            dataset = str(raw.get("dataset", ""))
            version = str(raw.get("schema_version", raw.get("version", "")))
            ignore_index = int(raw.get("target", {}).get("ignore_index", 255))
        else:
            mapping_raw = raw.get("rellis_to_target")
            target_raw = raw.get("target_classes", COST4_CLASSES)
            dataset = str(raw.get("dataset_name", "rellis3d"))
            version = str(raw.get("mapping_version", "legacy"))
            ignore_index = 255
        if not isinstance(mapping_raw, dict) or not mapping_raw:
            raise DatasetError(f"source_to_target mapping is missing in {path}")
        if not isinstance(target_raw, dict):
            raise DatasetError(f"target classes are invalid in {path}")
        mapping = {int(source): int(target) for source, target in mapping_raw.items()}
        target_classes = {
            int(class_id): str(name) for class_id, name in target_raw.items()
            if int(class_id) != ignore_index
        }
        if target_classes != COST4_CLASSES:
            raise DatasetError(
                f"Target classes must exactly match ADOM Cost4: {COST4_CLASSES}"
            )
        if ignore_index != IGNORE_INDEX:
            raise DatasetError("ADOM ignore_index must be 255")
        invalid_targets = set(mapping.values()) - ALLOWED_TARGET_IDS
        if invalid_targets:
            raise DatasetError(f"Invalid mapping target IDs: {sorted(invalid_targets)}")
        if any(source < 0 for source in mapping):
            raise DatasetError("Source IDs must be non-negative")
        return cls(
            dataset=dataset,
            version=version,
            source_to_target=mapping,
            target_classes=target_classes,
            ignore_index=ignore_index,
        )

    def remap(self, source: np.ndarray) -> np.ndarray:
        if source.ndim != 2 or source.size == 0:
            raise DatasetError(f"Source mask must be non-empty HxW, got {source.shape}")
        if not np.issubdtype(source.dtype, np.integer):
            raise DatasetError(f"Source mask must contain integer IDs, got {source.dtype}")
        observed = {int(value) for value in np.unique(source)}
        unknown = observed - set(self.source_to_target)
        if unknown:
            raise DatasetError(f"Unknown source label IDs: {sorted(unknown)}")
        maximum = max(max(self.source_to_target), max(observed))
        lookup = np.full(maximum + 1, self.ignore_index, dtype=np.uint8)
        for source_id, target_id in self.source_to_target.items():
            lookup[source_id] = target_id
        target = lookup[source.astype(np.int64)]
        invalid = {int(value) for value in np.unique(target)} - ALLOWED_TARGET_IDS
        if invalid:
            raise DatasetError(f"Invalid target label IDs: {sorted(invalid)}")
        return target

    def snapshot(self) -> dict[str, Any]:
        return {
            "schema_version": self.version,
            "dataset": self.dataset,
            "target": {
                "classes": {str(key): value for key, value in self.target_classes.items()},
                "ignore_index": self.ignore_index,
                "palette": {
                    str(key): list(COST4_PALETTE[key])
                    for key in [0, 1, 2, 3, 255]
                },
            },
            "source_to_target": {
                str(key): self.source_to_target[key]
                for key in sorted(self.source_to_target)
            },
        }
