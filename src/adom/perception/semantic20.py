from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


# RGB visualization contract shared with the Semantic20 MMSeg dataset.
SEMANTIC20_PALETTE_RGB = np.asarray(
    [
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
    ],
    dtype=np.uint8,
)
SEMANTIC20_PALETTE_BGR = SEMANTIC20_PALETTE_RGB[:, ::-1].copy()


@dataclass(frozen=True)
class Semantic20Ontology:
    dataset_name: str
    mapping_version: str
    classes: tuple[str, ...]
    ignore_index: int

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    def validate_mask(self, mask: np.ndarray) -> None:
        if mask.ndim != 2:
            raise ValueError(f"Semantic20 mask must be HxW, got {mask.shape}")
        values = np.unique(mask)
        invalid = values[
            ((values < 0) | (values >= self.num_classes))
            & (values != self.ignore_index)
        ]
        if invalid.size:
            raise ValueError(
                f"Semantic20 mask contains invalid IDs: {invalid.astype(int).tolist()}"
            )


def default_bridge_mapping_path() -> Path:
    from adom.data.semantic20 import resource_path

    return resource_path("semantic_20", "config", "bridge_mapping.yaml")


def load_semantic20_ontology(path: str | Path | None = None) -> Semantic20Ontology:
    mapping_path = Path(path) if path else default_bridge_mapping_path()
    mapping_path = mapping_path.expanduser().resolve()
    if not mapping_path.is_file():
        raise FileNotFoundError(f"Semantic20 bridge mapping not found: {mapping_path}")
    with mapping_path.open("r", encoding="utf-8-sig") as handle:
        payload: Any = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Semantic20 bridge mapping must be a YAML mapping")

    num_classes = int(payload.get("num_classes", -1))
    ignore_index = int(payload.get("ignore_index", -1))
    raw_classes = payload.get("target_classes")
    if num_classes != 19 or ignore_index != 255 or not isinstance(raw_classes, dict):
        raise ValueError(
            "Semantic20 bridge mapping must declare 19 classes and ignore_index 255"
        )
    normalized = {int(key): str(value) for key, value in raw_classes.items()}
    expected_ids = set(range(num_classes)) | {ignore_index}
    if set(normalized) != expected_ids or normalized[ignore_index] != "ignore":
        raise ValueError("Semantic20 target_classes must contain IDs 0..18 and 255 ignore")
    classes = tuple(normalized[index] for index in range(num_classes))
    if len(set(classes)) != num_classes:
        raise ValueError("Semantic20 class names must be unique")
    return Semantic20Ontology(
        dataset_name=str(payload.get("dataset_name", "")),
        mapping_version=str(payload.get("mapping_version", "")),
        classes=classes,
        ignore_index=ignore_index,
    )


def colorize_semantic20_mask(
    mask: np.ndarray, ontology: Semantic20Ontology
) -> np.ndarray:
    ontology.validate_mask(mask)
    output = np.zeros((*mask.shape, 3), dtype=np.uint8)
    valid = mask != ontology.ignore_index
    output[valid] = SEMANTIC20_PALETTE_BGR[mask[valid]]
    return output
