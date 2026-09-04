#!/usr/bin/env python3
"""
Self-contained ADOM Semantic20 preprocessing pipeline.

Purpose
-------
Build one unified semantic-segmentation package from:
  - RELLIS-3D
  - RUGD
  - YCOR
  - optional ADOM-v2

This file does NOT import or execute any preprocessing script, YAML, JSON,
or split TXT from the ADOM Git repository.  The project mappings and split
rules required by the pipeline are embedded below.

Runtime Python dependencies:
  - numpy
  - Pillow

Target ontology
---------------
0  dirt
1  grass
2  tree
3  pole
4  water
5  sky
6  vehicle
7  object
8  asphalt
9  building
10 log
11 person
12 fence
13 bush
14 concrete
15 barrier
16 puddle
17 mud
18 rubble
255 ignore

Important RUGD note
-------------------
The historical ADOM RUGD package used palette-indexed PNG masks whose
palette indices did not necessarily equal the official RUGD class IDs.
Therefore this standalone implementation prefers RGB palette decoding for
P/RGB/RGBA masks.  This keeps class meaning stable independent of palette
index assignment.

For truly single-channel RUGD masks, --rugd-index-scheme controls how IDs
are interpreted:
  auto     : detect legacy ADOM-v1 IDs when possible, otherwise use official
  legacy   : historical ADOM-v1 raw palette-index IDs
  official : official RUGD IDs with void=0, classes=1..24
  compact  : 0-based 24-class variant without void
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Target ontology: embedded SSOT
# ---------------------------------------------------------------------------

TARGET_CLASSES: dict[int, str] = {
    0: "dirt",
    1: "grass",
    2: "tree",
    3: "pole",
    4: "water",
    5: "sky",
    6: "vehicle",
    7: "object",
    8: "asphalt",
    9: "building",
    10: "log",
    11: "person",
    12: "fence",
    13: "bush",
    14: "concrete",
    15: "barrier",
    16: "puddle",
    17: "mud",
    18: "rubble",
    255: "ignore",
}

NUM_CLASSES = 19
IGNORE_INDEX = 255
ALLOWED_TARGET_IDS = set(range(NUM_CLASSES)) | {IGNORE_INDEX}

MANIFEST_FIELDS = [
    "sample_key",
    "source",
    "source_split",
    "output_split",
    "sample_id",
    "image_path",
    "mask_path",
    "non_ignore_ratio",
]


# ---------------------------------------------------------------------------
# RELLIS-3D -> ADOM Semantic20
# Embedded from the existing project class_mapping.yaml.
# Unlisted raw IDs are rejected, matching the previous strict converter.
# ---------------------------------------------------------------------------

RELLIS_TO_TARGET: dict[int, int] = {
    0: 255,   # void
    1: 0,     # dirt
    3: 1,     # grass
    4: 2,     # tree
    5: 3,     # pole
    6: 4,     # water
    7: 5,     # sky
    8: 6,     # vehicle
    9: 7,     # object
    10: 8,    # asphalt
    12: 9,    # building
    15: 10,   # log
    17: 11,   # person
    18: 12,   # fence
    19: 13,   # bush
    23: 14,   # concrete
    27: 15,   # barrier
    31: 16,   # puddle
    33: 17,   # mud
    34: 18,   # rubble
}

# Existing project split policy.
RELLIS_SEQUENCE_TO_SPLIT = {
    "00000": "train",
    "00001": "train",
    "00002": "train",
    "00003": "val",
    "00004": "test",
}


# ---------------------------------------------------------------------------
# RUGD -> ADOM Semantic20
#
# Project bridge policy:
#   use: grass/tree/water/sky/asphalt/building/person/bush/rock-bed
#   rock-bed -> rubble
#   ambiguous or unused source classes -> 255
#
# RGB decoding is preferred because palette index values can differ among
# RUGD distributions.
# ---------------------------------------------------------------------------

# Official RGB colors by semantic class name.
RUGD_RGB_TO_CLASS: dict[tuple[int, int, int], str] = {
    (0, 0, 0): "void",
    (108, 64, 20): "dirt",
    (255, 229, 204): "sand",
    (0, 102, 0): "grass",
    (0, 255, 0): "tree",
    (0, 153, 153): "pole",
    (0, 128, 255): "water",
    (0, 0, 255): "sky",
    (255, 255, 0): "vehicle",
    (255, 0, 127): "container/generic-object",
    (64, 64, 64): "asphalt",
    (255, 128, 0): "gravel",
    (255, 0, 0): "building",
    (153, 76, 0): "mulch",
    (102, 102, 0): "rock-bed",
    (102, 0, 0): "log",
    (0, 255, 128): "bicycle",
    (204, 153, 255): "person",
    (102, 0, 204): "fence",
    (255, 153, 204): "bush",
    (0, 102, 102): "sign",
    (153, 204, 255): "rock",
    (102, 255, 255): "bridge",
    (101, 101, 11): "concrete",
    (114, 85, 47): "picnic-table",
}

RUGD_CLASS_TO_TARGET: dict[str, int] = {
    # Used by historical ADOM Semantic20 bridge.
    "grass": 1,
    "tree": 2,
    "water": 4,
    "sky": 5,
    "asphalt": 8,
    "building": 9,
    "person": 11,
    "bush": 13,
    "rock-bed": 18,

    # Everything below is intentionally ignored in bridge v1.
    "void": 255,
    "dirt": 255,
    "sand": 255,
    "pole": 255,
    "vehicle": 255,
    "container/generic-object": 255,
    "gravel": 255,
    "mulch": 255,
    "log": 255,
    "bicycle": 255,
    "fence": 255,
    "sign": 255,
    "rock": 255,
    "bridge": 255,
    "concrete": 255,
    "picnic-table": 255,
}

RUGD_RGB_TO_TARGET = {
    rgb: RUGD_CLASS_TO_TARGET[name]
    for rgb, name in RUGD_RGB_TO_CLASS.items()
}

# Historical ADOM-v1 palette-index interpretation confirmed by the
# project's recorded raw class distribution.
RUGD_LEGACY_INDEX_TO_TARGET: dict[int, int] = {
    0: 255,   # mulch
    3: 1,     # grass
    4: 2,     # tree
    6: 255,   # gravel
    7: 9,     # building
    10: 8,    # asphalt
    11: 4,    # water
    12: 5,    # sky
    14: 255,  # sign
    17: 13,   # bush
    19: 11,   # person
    20: 18,   # rock-bed -> rubble
}

# Official RUGD index convention:
# 0 void, 1 dirt, ..., 24 picnic-table.
RUGD_OFFICIAL_ID_TO_CLASS: dict[int, str] = {
    0: "void",
    1: "dirt",
    2: "sand",
    3: "grass",
    4: "tree",
    5: "pole",
    6: "water",
    7: "sky",
    8: "vehicle",
    9: "container/generic-object",
    10: "asphalt",
    11: "gravel",
    12: "building",
    13: "mulch",
    14: "rock-bed",
    15: "log",
    16: "bicycle",
    17: "person",
    18: "fence",
    19: "bush",
    20: "sign",
    21: "rock",
    22: "bridge",
    23: "concrete",
    24: "picnic-table",
}

RUGD_OFFICIAL_ID_TO_TARGET = {
    source_id: RUGD_CLASS_TO_TARGET[class_name]
    for source_id, class_name in RUGD_OFFICIAL_ID_TO_CLASS.items()
}

# Some repackaged versions omit void and use 0..23 for the 24 semantic
# classes. Support it explicitly, but never silently prefer it over the
# official convention in ambiguous numeric-only inputs.
RUGD_COMPACT_ID_TO_CLASS: dict[int, str] = {
    0: "dirt",
    1: "sand",
    2: "grass",
    3: "tree",
    4: "pole",
    5: "water",
    6: "sky",
    7: "vehicle",
    8: "container/generic-object",
    9: "asphalt",
    10: "gravel",
    11: "building",
    12: "mulch",
    13: "rock-bed",
    14: "log",
    15: "bicycle",
    16: "person",
    17: "fence",
    18: "bush",
    19: "sign",
    20: "rock",
    21: "bridge",
    22: "concrete",
    23: "picnic-table",
}

RUGD_COMPACT_ID_TO_TARGET = {
    source_id: RUGD_CLASS_TO_TARGET[class_name]
    for source_id, class_name in RUGD_COMPACT_ID_TO_CLASS.items()
}

# RUGD benchmark video split.  This reproduces the source's sequence-level
# policy without requiring an external train.txt/val.txt/test.txt.
RUGD_TRAIN_SEQUENCES = {
    "park-2",
    "trail",
    "trail-3",
    "trail-4",
    "trail-6",
    "trail-9",
    "trail-10",
    "trail-11",
    "trail-12",
    "trail-14",
    "trail-15",
    "village",
}
RUGD_VAL_SEQUENCES = {
    "park-8",
    "trail-5",
}
RUGD_TEST_SEQUENCES = {
    "creek",
    "park-1",
    "trail-7",
    "trail-13",
}


# ---------------------------------------------------------------------------
# YCOR -> ADOM Semantic20
# RGB palette and bridge mapping embedded from the existing project configs.
# ---------------------------------------------------------------------------

YCOR_RGB_TO_SOURCE_ID: dict[tuple[int, int, int], int] = {
    (255, 255, 255): 0,  # background_or_unlabelled
    (40, 80, 0): 1,     # high_vegetation
    (128, 255, 0): 2,   # traversable_grass
    (178, 176, 153): 3, # smooth_trail
    (255, 0, 0): 4,     # obstacle
    (1, 88, 255): 5,    # sky
    (156, 76, 30): 6,   # rough_trail
    (255, 0, 128): 7,   # puddle
    (0, 160, 0): 8,     # non_traversable_low_vegetation
}

YCOR_SOURCE_ID_TO_TARGET: dict[int, int] = {
    0: 255,
    1: 255,
    2: 1,    # traversable_grass -> grass
    3: 255,
    4: 255,
    5: 255,
    6: 255,
    7: 16,   # puddle -> puddle
    8: 255,
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Self-contained unified preprocessing for "
            "RELLIS-3D + RUGD + YCOR + optional ADOM-v2."
        ),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help=(
            "Directory containing RELLIS-3D, RUGD, YCOR and optionally "
            "ADOM-v2. A nested <data-root>/raw layout is also detected."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Final unified dataset output directory.",
    )
    parser.add_argument(
        "--adom-v2-root",
        type=Path,
        default=None,
        help="Optional explicit ADOM-v2 root.",
    )
    parser.add_argument(
        "--skip-adom-v2",
        action="store_true",
        help="Do not process ADOM-v2 even if a directory is found.",
    )
    parser.add_argument(
        "--adom-eval-policy",
        choices=("diagnostic", "mixed"),
        default="diagnostic",
        help=(
            "diagnostic keeps main val/test RELLIS-only and writes "
            "ADOM-v2 val/test separately; mixed appends them to main val/test."
        ),
    )
    parser.add_argument(
        "--min-non-ignore-ratio",
        type=float,
        default=0.01,
        help=(
            "Minimum non-ignore ratio for YCOR. Puddle-containing samples "
            "are always retained."
        ),
    )
    parser.add_argument(
        "--rugd-index-scheme",
        choices=("auto", "legacy", "official", "compact"),
        default="auto",
        help=(
            "How numeric-only RUGD masks are interpreted. Palette/RGB masks "
            "always use RGB semantic decoding."
        ),
    )
    parser.add_argument(
        "--storage",
        choices=("auto", "copy", "hardlink"),
        default="auto",
        help="How source RGB images are placed in the output.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing non-empty output directory.",
    )
    parser.add_argument(
        "--no-full-validation",
        action="store_true",
        help="Skip the final second-pass full mask validation.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def require_dir(path: Path, label: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"{label} not found: {path}")


def prepare_output_root(output_root: Path, overwrite: bool) -> None:
    if output_root.exists() and any(output_root.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output root is not empty: {output_root}\n"
                "Use --overwrite to replace it."
            )
        shutil.rmtree(output_root)

    for relative in (
        "images/rellis3d",
        "images/rugd",
        "images/ycor",
        "images/adom_v2",
        "masks/rellis3d",
        "masks/rugd",
        "masks/ycor",
        "masks/adom_v2",
        "splits",
        "results",
    ):
        (output_root / relative).mkdir(parents=True, exist_ok=True)


def resolve_dataset_root(data_root: Path, dataset_name: str) -> Path:
    candidates = [
        data_root / dataset_name,
        data_root / "raw" / dataset_name,
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"{dataset_name} not found. Expected one of: "
        + ", ".join(str(p) for p in candidates)
    )


def resolve_optional_dataset_root(
    data_root: Path,
    dataset_name: str,
) -> Path | None:
    candidates = [
        data_root / dataset_name,
        data_root / "raw" / dataset_name,
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return None


def place_image(
    source: Path,
    destination: Path,
    storage: str,
) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists():
        destination.unlink()

    if storage == "copy":
        shutil.copy2(source, destination)
        return "copy"

    if storage in ("auto", "hardlink"):
        try:
            os.link(source, destination)
            return "hardlink"
        except OSError:
            if storage == "hardlink":
                raise

    shutil.copy2(source, destination)
    return "copy"


def save_mask(mask: np.ndarray, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    Image.fromarray(mask, mode="L").save(destination, format="PNG")


def relative_posix(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def non_ignore_ratio(mask: np.ndarray) -> float:
    return float(np.count_nonzero(mask != IGNORE_INDEX) / mask.size)


def add_pixel_counts(mask: np.ndarray, counter: Counter[int]) -> None:
    values, counts = np.unique(mask, return_counts=True)
    for value, count in zip(values, counts):
        counter[int(value)] += int(count)


def validate_target_mask_array(
    mask: np.ndarray,
    label: str,
) -> None:
    if mask.ndim != 2:
        raise ValueError(f"Target mask must be 2-D: {label}, {mask.shape}")
    ids = {int(v) for v in np.unique(mask)}
    invalid = sorted(ids - ALLOWED_TARGET_IDS)
    if invalid:
        raise ValueError(f"Invalid target IDs {invalid}: {label}")


def remap_index_mask_strict(
    source_mask: np.ndarray,
    mapping: dict[int, int],
    label: str,
) -> np.ndarray:
    if source_mask.ndim != 2:
        raise ValueError(f"Expected 2-D index mask: {label}")

    observed = {int(v) for v in np.unique(source_mask)}
    unknown = sorted(observed - set(mapping))
    if unknown:
        raise ValueError(f"Unknown source IDs {unknown}: {label}")

    max_source_id = max(mapping)
    table = np.full(max_source_id + 1, IGNORE_INDEX, dtype=np.uint8)
    for source_id, target_id in mapping.items():
        table[source_id] = target_id

    if int(source_mask.max()) > max_source_id:
        raise ValueError(f"Source ID out of mapping range: {label}")

    target = table[source_mask.astype(np.int64)]
    validate_target_mask_array(target, label)
    return target


def pack_rgb(rgb: np.ndarray) -> np.ndarray:
    return (
        (rgb[:, :, 0].astype(np.uint32) << 16)
        | (rgb[:, :, 1].astype(np.uint32) << 8)
        | rgb[:, :, 2].astype(np.uint32)
    )


def tuple_to_packed(rgb: tuple[int, int, int]) -> int:
    return (rgb[0] << 16) | (rgb[1] << 8) | rgb[2]


def remap_rgb_mask(
    rgb: np.ndarray,
    rgb_to_target: dict[tuple[int, int, int], int],
    label: str,
) -> np.ndarray:
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected RGB mask: {label}, {rgb.shape}")

    packed = pack_rgb(rgb)
    unique_keys, inverse = np.unique(packed, return_inverse=True)

    packed_mapping = {
        tuple_to_packed(color): target_id
        for color, target_id in rgb_to_target.items()
    }

    unknown = [
        int(key)
        for key in unique_keys
        if int(key) not in packed_mapping
    ]
    if unknown:
        unknown_rgb = [
            ((key >> 16) & 255, (key >> 8) & 255, key & 255)
            for key in unknown[:20]
        ]
        raise ValueError(
            f"Unknown RGB labels in {label}: {unknown_rgb}"
        )

    mapped_unique = np.asarray(
        [packed_mapping[int(key)] for key in unique_keys],
        dtype=np.uint8,
    )
    target = mapped_unique[inverse].reshape(packed.shape)
    validate_target_mask_array(target, label)
    return target


def collect_unique_by_stem(
    roots: Iterable[Path],
    suffixes: set[str],
) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for root in roots:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in suffixes:
                continue
            key = path.stem
            if key in found and found[key] != path:
                raise RuntimeError(
                    f"Duplicate stem '{key}' found at "
                    f"{found[key]} and {path}"
                )
            found[key] = path
    return found


def write_manifest(
    path: Path,
    rows: list[dict[str, str]],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_split(path: Path, entries: list[str]) -> None:
    text = "\n".join(entries)
    path.write_text((text + "\n") if text else "", encoding="utf-8")


# ---------------------------------------------------------------------------
# RELLIS
# ---------------------------------------------------------------------------

def load_rellis_mask(mask_path: Path) -> np.ndarray:
    with Image.open(mask_path) as image:
        image.load()
        array = np.asarray(image)

    if array.ndim == 3:
        channels_equal = all(
            np.array_equal(array[:, :, 0], array[:, :, i])
            for i in range(1, array.shape[2])
        )
        if not channels_equal:
            raise ValueError(
                f"RELLIS label-ID mask has differing channels: {mask_path}"
            )
        array = array[:, :, 0]

    if array.ndim != 2 or not np.issubdtype(array.dtype, np.integer):
        raise ValueError(
            f"Unsupported RELLIS mask: {mask_path}, "
            f"shape={array.shape}, dtype={array.dtype}"
        )

    return remap_index_mask_strict(
        array.astype(np.int32, copy=False),
        RELLIS_TO_TARGET,
        str(mask_path),
    )


def process_rellis(
    root: Path,
    output_root: Path,
    storage: str,
    manifest_rows: list[dict[str, str]],
    main_splits: dict[str, list[str]],
    pixel_counts: Counter[int],
    source_summary: dict,
    storage_counts: Counter[str],
) -> None:
    print("\n=== RELLIS-3D ===")
    require_dir(root, "RELLIS-3D root")

    total = 0
    for sequence_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        sequence = sequence_dir.name
        if sequence not in RELLIS_SEQUENCE_TO_SPLIT:
            continue

        source_split = RELLIS_SEQUENCE_TO_SPLIT[sequence]
        rgb_dir = sequence_dir / "pylon_camera_node"
        mask_dir = sequence_dir / "pylon_camera_node_label_id"

        if not rgb_dir.is_dir() and not mask_dir.is_dir():
            continue
        require_dir(rgb_dir, f"RELLIS RGB {sequence}")
        require_dir(mask_dir, f"RELLIS mask {sequence}")

        images = {
            p.stem: p
            for p in sorted(rgb_dir.iterdir())
            if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        }
        masks = {
            p.stem: p
            for p in sorted(mask_dir.iterdir())
            if p.is_file() and p.suffix.lower() == ".png"
        }

        mask_only = sorted(set(masks) - set(images))
        if mask_only:
            raise RuntimeError(
                f"RELLIS mask-only files in {sequence}: "
                f"{mask_only[:10]}"
            )

        paired = sorted(set(images) & set(masks))
        print(
            f"[RELLIS {sequence}] RGB={len(images)} "
            f"mask={len(masks)} paired={len(paired)} "
            f"RGB-only={len(set(images) - set(masks))}"
        )

        for stem in paired:
            image_src = images[stem]
            mask_src = masks[stem]

            with Image.open(image_src) as image:
                image_size = image.size

            target_mask = load_rellis_mask(mask_src)
            if (target_mask.shape[1], target_mask.shape[0]) != image_size:
                raise RuntimeError(
                    f"RELLIS image/mask size mismatch: "
                    f"{image_src}, {mask_src}"
                )

            image_dst = (
                output_root
                / "images"
                / "rellis3d"
                / source_split
                / sequence
                / image_src.name
            )
            mask_dst = (
                output_root
                / "masks"
                / "rellis3d"
                / source_split
                / sequence
                / f"{stem}.png"
            )

            storage_counts[place_image(image_src, image_dst, storage)] += 1
            save_mask(target_mask, mask_dst)

            sample_id = f"{sequence}/{stem}"
            sample_key = f"rellis3d/{source_split}/{sample_id}"
            ratio = non_ignore_ratio(target_mask)

            manifest_rows.append(
                {
                    "sample_key": sample_key,
                    "source": "rellis3d",
                    "source_split": source_split,
                    "output_split": source_split,
                    "sample_id": sample_id,
                    "image_path": relative_posix(image_dst, output_root),
                    "mask_path": relative_posix(mask_dst, output_root),
                    "non_ignore_ratio": f"{ratio:.8f}",
                }
            )
            main_splits[source_split].append(sample_key)
            add_pixel_counts(target_mask, pixel_counts)
            total += 1

    if total == 0:
        raise RuntimeError("No RELLIS-3D image/mask pairs were found.")

    source_summary["rellis3d"] = {"input": total, "kept": total}
    print(f"[PASS] RELLIS-3D converted: {total}")


# ---------------------------------------------------------------------------
# RUGD
# ---------------------------------------------------------------------------

def rugd_sequence_name(sample_id: str) -> str:
    if "_" not in sample_id:
        raise ValueError(f"Unexpected RUGD filename: {sample_id}")
    return sample_id.rsplit("_", 1)[0]


def rugd_split_for_sample(sample_id: str) -> str:
    sequence = rugd_sequence_name(sample_id)
    if sequence in RUGD_TRAIN_SEQUENCES:
        return "train"
    if sequence in RUGD_VAL_SEQUENCES:
        return "val"
    if sequence in RUGD_TEST_SEQUENCES:
        return "test"
    raise ValueError(
        f"Unknown RUGD sequence '{sequence}' from sample '{sample_id}'."
    )


def locate_rugd_roots(root: Path) -> tuple[list[Path], list[Path]]:
    # Preferred historical layout.
    image_dirs = sorted(
        p for p in root.rglob("*")
        if p.is_dir() and p.name.lower() == "image"
    )
    mask_dirs = sorted(
        p for p in root.rglob("*")
        if p.is_dir() and p.name.lower() == "indexlabel"
    )

    if image_dirs and mask_dirs:
        return image_dirs, mask_dirs

    # Common original/repackaged layout fallbacks.
    image_dirs = sorted(
        p for p in root.rglob("*")
        if p.is_dir()
        and p.name.lower() in {
            "rugd_frames-with-annotations",
            "frames-with-annotations",
            "images",
        }
    )
    mask_dirs = sorted(
        p for p in root.rglob("*")
        if p.is_dir()
        and p.name.lower() in {
            "rugd_annotations",
            "annotations",
            "labels",
        }
    )

    if not image_dirs or not mask_dirs:
        raise FileNotFoundError(
            "Could not locate RUGD image/mask roots. Expected directories "
            "named image + indexLabel, or a supported frames/annotations layout."
        )
    return image_dirs, mask_dirs


def inspect_rugd_mask_mode(mask_path: Path) -> str:
    with Image.open(mask_path) as image:
        if image.mode in ("P", "RGB", "RGBA"):
            return "rgb"
        array = np.asarray(image)
    if array.ndim == 2:
        return "numeric"
    if (
        array.ndim == 3
        and array.shape[2] >= 3
        and np.array_equal(array[:, :, 0], array[:, :, 1])
        and np.array_equal(array[:, :, 1], array[:, :, 2])
    ):
        return "numeric"
    raise ValueError(
        f"Unsupported RUGD mask format: {mask_path}"
    )


def infer_rugd_numeric_scheme(
    mask_paths: list[Path],
    requested: str,
) -> str:
    if requested != "auto":
        return requested

    observed: set[int] = set()
    # Enough to identify the historical 12-index palette in normal releases.
    for path in mask_paths[: min(250, len(mask_paths))]:
        with Image.open(path) as image:
            array = np.asarray(image)
        if array.ndim == 3:
            array = array[:, :, 0]
        observed.update(int(v) for v in np.unique(array))

        if 24 in observed:
            return "official"

    if observed and observed.issubset(set(RUGD_LEGACY_INDEX_TO_TARGET)):
        return "legacy"

    # The official 0..24 convention is preferred for ambiguous numeric-only
    # masks. Users of a compact 0..23 repack can explicitly select compact.
    if observed.issubset(set(RUGD_OFFICIAL_ID_TO_TARGET)):
        return "official"

    raise ValueError(
        "Could not infer numeric RUGD ID scheme from observed IDs "
        f"{sorted(observed)}. Use --rugd-index-scheme explicitly."
    )


def load_rugd_mask(
    mask_path: Path,
    numeric_scheme: str,
) -> np.ndarray:
    with Image.open(mask_path) as image:
        image.load()

        # Critical: P mode is decoded through its palette RGB, not through
        # raw palette index values.
        if image.mode in ("P", "RGB", "RGBA"):
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
            return remap_rgb_mask(
                rgb,
                RUGD_RGB_TO_TARGET,
                str(mask_path),
            )

        array = np.asarray(image)

    if array.ndim == 3:
        if (
            array.shape[2] >= 3
            and np.array_equal(array[:, :, 0], array[:, :, 1])
            and np.array_equal(array[:, :, 1], array[:, :, 2])
        ):
            array = array[:, :, 0]
        else:
            raise ValueError(f"Unsupported RUGD mask: {mask_path}")

    if array.ndim != 2:
        raise ValueError(f"Unsupported RUGD mask: {mask_path}")

    if numeric_scheme == "legacy":
        mapping = RUGD_LEGACY_INDEX_TO_TARGET
    elif numeric_scheme == "official":
        mapping = RUGD_OFFICIAL_ID_TO_TARGET
    elif numeric_scheme == "compact":
        mapping = RUGD_COMPACT_ID_TO_TARGET
    else:
        raise ValueError(f"Invalid RUGD numeric scheme: {numeric_scheme}")

    return remap_index_mask_strict(
        array.astype(np.int32, copy=False),
        mapping,
        str(mask_path),
    )


def process_rugd(
    root: Path,
    output_root: Path,
    storage: str,
    rugd_index_scheme: str,
    manifest_rows: list[dict[str, str]],
    main_splits: dict[str, list[str]],
    diagnostics: dict[str, list[str]],
    pixel_counts: Counter[int],
    source_summary: dict,
    storage_counts: Counter[str],
) -> None:
    print("\n=== RUGD ===")
    require_dir(root, "RUGD root")

    image_roots, mask_roots = locate_rugd_roots(root)
    images = collect_unique_by_stem(
        image_roots,
        {".png", ".jpg", ".jpeg"},
    )
    masks = collect_unique_by_stem(mask_roots, {".png"})

    only_images = sorted(set(images) - set(masks))
    only_masks = sorted(set(masks) - set(images))
    if only_images or only_masks:
        raise RuntimeError(
            "RUGD image/mask mismatch: "
            f"images-only={len(only_images)} {only_images[:10]}, "
            f"masks-only={len(only_masks)} {only_masks[:10]}"
        )

    sample_ids = sorted(images)
    if not sample_ids:
        raise RuntimeError("No RUGD image/mask pairs found.")

    mask_modes = {
        inspect_rugd_mask_mode(masks[sample_id])
        for sample_id in sample_ids[: min(30, len(sample_ids))]
    }

    numeric_scheme = rugd_index_scheme
    if mask_modes == {"numeric"}:
        numeric_scheme = infer_rugd_numeric_scheme(
            [masks[sample_id] for sample_id in sample_ids],
            rugd_index_scheme,
        )
        print(f"[RUGD] numeric ID scheme: {numeric_scheme}")
    else:
        print("[RUGD] palette/RGB semantic decoding enabled")

    split_input = Counter()
    for sample_id in sample_ids:
        split = rugd_split_for_sample(sample_id)
        split_input[split] += 1

        image_src = images[sample_id]
        mask_src = masks[sample_id]

        with Image.open(image_src) as image:
            image_size = image.size

        target_mask = load_rugd_mask(mask_src, numeric_scheme)
        if (target_mask.shape[1], target_mask.shape[0]) != image_size:
            raise RuntimeError(
                f"RUGD image/mask size mismatch: {image_src}, {mask_src}"
            )

        ratio = non_ignore_ratio(target_mask)

        image_dst = (
            output_root
            / "images"
            / "rugd"
            / split
            / image_src.name
        )
        mask_dst = (
            output_root
            / "masks"
            / "rugd"
            / split
            / f"{sample_id}.png"
        )

        storage_counts[place_image(image_src, image_dst, storage)] += 1
        save_mask(target_mask, mask_dst)

        sample_key = f"rugd/{split}/{sample_id}"
        manifest_rows.append(
            {
                "sample_key": sample_key,
                "source": "rugd",
                "source_split": split,
                "output_split": split,
                "sample_id": sample_id,
                "image_path": relative_posix(image_dst, output_root),
                "mask_path": relative_posix(mask_dst, output_root),
                "non_ignore_ratio": f"{ratio:.8f}",
            }
        )
        add_pixel_counts(target_mask, pixel_counts)

        if split == "train":
            main_splits["train"].append(sample_key)
        else:
            diagnostics[f"rugd_{split}"].append(sample_key)

    source_summary["rugd"] = {
        "input": len(sample_ids),
        "kept": len(sample_ids),
        "source_split_counts": dict(split_input),
        "mask_decode_mode": (
            "rgb_palette"
            if "rgb" in mask_modes
            else f"numeric_{numeric_scheme}"
        ),
    }
    print(
        "[PASS] RUGD converted: "
        f"{len(sample_ids)} "
        f"(train={split_input['train']}, "
        f"val={split_input['val']}, "
        f"test={split_input['test']})"
    )


# ---------------------------------------------------------------------------
# YCOR
# ---------------------------------------------------------------------------

def load_ycor_source_mask(mask_path: Path) -> np.ndarray:
    with Image.open(mask_path) as image:
        image.load()

        if image.mode in ("P", "RGB", "RGBA"):
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
            packed = pack_rgb(rgb)
            unique_keys, inverse = np.unique(packed, return_inverse=True)

            packed_map = {
                tuple_to_packed(color): source_id
                for color, source_id in YCOR_RGB_TO_SOURCE_ID.items()
            }
            unknown = [
                int(key)
                for key in unique_keys
                if int(key) not in packed_map
            ]
            if unknown:
                unknown_rgb = [
                    ((key >> 16) & 255, (key >> 8) & 255, key & 255)
                    for key in unknown[:20]
                ]
                raise ValueError(
                    f"Unknown YCOR RGB labels in {mask_path}: {unknown_rgb}"
                )

            mapped_unique = np.asarray(
                [packed_map[int(key)] for key in unique_keys],
                dtype=np.int32,
            )
            return mapped_unique[inverse].reshape(packed.shape)

        array = np.asarray(image)

    if array.ndim == 3:
        if (
            array.shape[2] >= 3
            and np.array_equal(array[:, :, 0], array[:, :, 1])
            and np.array_equal(array[:, :, 1], array[:, :, 2])
        ):
            array = array[:, :, 0]
        else:
            raise ValueError(f"Unsupported YCOR mask: {mask_path}")

    if array.ndim != 2:
        raise ValueError(f"Unsupported YCOR mask: {mask_path}")

    observed = {int(v) for v in np.unique(array)}
    unknown = sorted(observed - set(YCOR_SOURCE_ID_TO_TARGET))
    if unknown:
        raise ValueError(
            f"Unknown YCOR source IDs {unknown}: {mask_path}"
        )
    return array.astype(np.int32, copy=False)


def process_ycor(
    root: Path,
    output_root: Path,
    storage: str,
    min_non_ignore_ratio: float,
    manifest_rows: list[dict[str, str]],
    main_splits: dict[str, list[str]],
    diagnostics: dict[str, list[str]],
    pixel_counts: Counter[int],
    source_summary: dict,
    storage_counts: Counter[str],
) -> None:
    print("\n=== YCOR ===")
    require_dir(root, "YCOR root")

    source_to_output_split = {
        "train": "train",
        "valid": "val",
    }

    input_count = 0
    kept_count = 0
    skipped_count = 0
    puddle_images = 0
    puddle_pixels = 0
    split_kept = Counter()

    for source_split, output_split in source_to_output_split.items():
        split_root = root / source_split
        require_dir(split_root, f"YCOR {source_split}")

        sample_dirs = sorted(
            path for path in split_root.iterdir() if path.is_dir()
        )

        for sample_dir in sample_dirs:
            input_count += 1
            image_src = sample_dir / "rgb.jpg"
            mask_src = sample_dir / "labels.png"

            if not image_src.is_file():
                raise FileNotFoundError(f"YCOR image not found: {image_src}")
            if not mask_src.is_file():
                raise FileNotFoundError(f"YCOR mask not found: {mask_src}")

            with Image.open(image_src) as image:
                image_size = image.size

            source_mask = load_ycor_source_mask(mask_src)
            target_mask = remap_index_mask_strict(
                source_mask,
                YCOR_SOURCE_ID_TO_TARGET,
                str(mask_src),
            )

            if (target_mask.shape[1], target_mask.shape[0]) != image_size:
                raise RuntimeError(
                    f"YCOR image/mask size mismatch: {image_src}, {mask_src}"
                )

            ratio = non_ignore_ratio(target_mask)
            p_pixels = int(np.count_nonzero(target_mask == 16))

            keep = (
                p_pixels > 0
                or ratio >= min_non_ignore_ratio
            )
            if not keep:
                skipped_count += 1
                continue

            if p_pixels > 0:
                puddle_images += 1
                puddle_pixels += p_pixels

            sample_id = sample_dir.name
            image_dst = (
                output_root
                / "images"
                / "ycor"
                / output_split
                / f"{sample_id}.jpg"
            )
            mask_dst = (
                output_root
                / "masks"
                / "ycor"
                / output_split
                / f"{sample_id}.png"
            )

            storage_counts[place_image(image_src, image_dst, storage)] += 1
            save_mask(target_mask, mask_dst)

            sample_key = f"ycor/{output_split}/{sample_id}"
            manifest_rows.append(
                {
                    "sample_key": sample_key,
                    "source": "ycor",
                    "source_split": source_split,
                    "output_split": output_split,
                    "sample_id": sample_id,
                    "image_path": relative_posix(image_dst, output_root),
                    "mask_path": relative_posix(mask_dst, output_root),
                    "non_ignore_ratio": f"{ratio:.8f}",
                }
            )
            add_pixel_counts(target_mask, pixel_counts)

            if output_split == "train":
                main_splits["train"].append(sample_key)
            else:
                diagnostics["ycor_val"].append(sample_key)

            kept_count += 1
            split_kept[output_split] += 1

    if input_count == 0:
        raise RuntimeError("No YCOR samples found.")

    source_summary["ycor"] = {
        "input": input_count,
        "kept": kept_count,
        "skipped_low_non_ignore": skipped_count,
        "kept_split_counts": dict(split_kept),
        "puddle_image_count": puddle_images,
        "puddle_pixel_count": puddle_pixels,
    }

    print(
        f"[PASS] YCOR input={input_count}, "
        f"kept={kept_count}, skipped={skipped_count}, "
        f"puddle_images={puddle_images}"
    )


# ---------------------------------------------------------------------------
# ADOM-v2
# ---------------------------------------------------------------------------

def resolve_adom_split_dirs(
    root: Path,
    split: str,
) -> tuple[Path, Path] | None:
    candidates = [
        (root / "images" / split, root / "masks" / split),
        (root / split / "images", root / split / "masks"),
    ]
    for image_root, mask_root in candidates:
        if image_root.is_dir() and mask_root.is_dir():
            return image_root, mask_root
    return None


def load_adom_v2_target_mask(mask_path: Path) -> np.ndarray:
    with Image.open(mask_path) as image:
        image.load()
        array = np.asarray(image)

    if array.ndim == 3:
        if (
            array.shape[2] >= 3
            and np.array_equal(array[:, :, 0], array[:, :, 1])
            and np.array_equal(array[:, :, 1], array[:, :, 2])
        ):
            array = array[:, :, 0]
        else:
            raise ValueError(
                f"ADOM-v2 must use class-ID masks, not color masks: {mask_path}"
            )

    if array.ndim != 2 or not np.issubdtype(array.dtype, np.integer):
        raise ValueError(
            f"Invalid ADOM-v2 mask: {mask_path}, "
            f"shape={array.shape}, dtype={array.dtype}"
        )

    target = array.astype(np.uint8, copy=False)
    validate_target_mask_array(target, str(mask_path))
    return target


def process_adom_v2(
    root: Path,
    output_root: Path,
    storage: str,
    eval_policy: str,
    manifest_rows: list[dict[str, str]],
    main_splits: dict[str, list[str]],
    diagnostics: dict[str, list[str]],
    pixel_counts: Counter[int],
    source_summary: dict,
    storage_counts: Counter[str],
) -> None:
    print("\n=== ADOM-v2 ===")
    require_dir(root, "ADOM-v2 root")

    split_counts = Counter()
    total = 0

    for split in ("train", "val", "test"):
        resolved = resolve_adom_split_dirs(root, split)
        if resolved is None:
            raise FileNotFoundError(
                f"ADOM-v2 split '{split}' not found. Expected either "
                f"images/{split} + masks/{split}, or "
                f"{split}/images + {split}/masks."
            )

        image_root, mask_root = resolved
        images = collect_unique_by_stem(
            [image_root],
            {".png", ".jpg", ".jpeg"},
        )
        masks = collect_unique_by_stem([mask_root], {".png"})

        only_images = sorted(set(images) - set(masks))
        only_masks = sorted(set(masks) - set(images))
        if only_images or only_masks:
            raise RuntimeError(
                f"ADOM-v2 {split} image/mask mismatch: "
                f"images-only={only_images[:10]}, masks-only={only_masks[:10]}"
            )

        sample_ids = sorted(images)
        if not sample_ids:
            raise RuntimeError(f"No ADOM-v2 samples in split {split}.")

        for sample_id in sample_ids:
            image_src = images[sample_id]
            mask_src = masks[sample_id]

            with Image.open(image_src) as image:
                image_size = image.size

            target_mask = load_adom_v2_target_mask(mask_src)
            if (target_mask.shape[1], target_mask.shape[0]) != image_size:
                raise RuntimeError(
                    f"ADOM-v2 image/mask size mismatch: "
                    f"{image_src}, {mask_src}"
                )

            ratio = non_ignore_ratio(target_mask)

            image_dst = (
                output_root
                / "images"
                / "adom_v2"
                / split
                / f"{sample_id}{image_src.suffix.lower()}"
            )
            mask_dst = (
                output_root
                / "masks"
                / "adom_v2"
                / split
                / f"{sample_id}.png"
            )

            storage_counts[place_image(image_src, image_dst, storage)] += 1
            save_mask(target_mask, mask_dst)

            sample_key = f"adom_v2/{split}/{sample_id}"
            manifest_rows.append(
                {
                    "sample_key": sample_key,
                    "source": "adom_v2",
                    "source_split": split,
                    "output_split": split,
                    "sample_id": sample_id,
                    "image_path": relative_posix(image_dst, output_root),
                    "mask_path": relative_posix(mask_dst, output_root),
                    "non_ignore_ratio": f"{ratio:.8f}",
                }
            )
            add_pixel_counts(target_mask, pixel_counts)

            if split == "train":
                main_splits["train"].append(sample_key)
            elif eval_policy == "mixed":
                main_splits[split].append(sample_key)
            else:
                diagnostics[f"adom_v2_{split}"].append(sample_key)

            split_counts[split] += 1
            total += 1

    source_summary["adom_v2"] = {
        "input": total,
        "kept": total,
        "split_counts": dict(split_counts),
        "evaluation_policy": eval_policy,
    }
    print(
        f"[PASS] ADOM-v2 converted: {total} "
        f"(train={split_counts['train']}, "
        f"val={split_counts['val']}, test={split_counts['test']})"
    )


# ---------------------------------------------------------------------------
# Reports and validation
# ---------------------------------------------------------------------------

def write_target_statistics(
    path: Path,
    pixel_counts: Counter[int],
) -> None:
    total_pixels = sum(pixel_counts.values())
    trainable_pixels = total_pixels - pixel_counts.get(255, 0)

    with path.open("w", newline="", encoding="utf-8") as file:
        fieldnames = [
            "target_id",
            "target_class",
            "pixel_count",
            "pixel_percent_all",
            "pixel_percent_trainable",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for target_id in sorted(
            pixel_counts,
            key=lambda value: (value == 255, value),
        ):
            count = pixel_counts[target_id]
            writer.writerow(
                {
                    "target_id": target_id,
                    "target_class": TARGET_CLASSES.get(target_id, "unknown"),
                    "pixel_count": count,
                    "pixel_percent_all": (
                        100.0 * count / total_pixels
                        if total_pixels else 0.0
                    ),
                    "pixel_percent_trainable": (
                        ""
                        if target_id == 255
                        else (
                            100.0 * count / trainable_pixels
                            if trainable_pixels else 0.0
                        )
                    ),
                }
            )


def validate_split_sets(
    manifest_rows: list[dict[str, str]],
    main_splits: dict[str, list[str]],
    diagnostics: dict[str, list[str]],
) -> dict:
    keys = [row["sample_key"] for row in manifest_rows]
    if len(keys) != len(set(keys)):
        raise RuntimeError("Duplicate sample_key values in manifest.")

    known = set(keys)

    for name, entries in main_splits.items():
        if len(entries) != len(set(entries)):
            raise RuntimeError(f"Duplicate entries in main split {name}.")
        unknown = set(entries) - known
        if unknown:
            raise RuntimeError(
                f"Unknown manifest keys in main split {name}: "
                f"{sorted(unknown)[:10]}"
            )

    train_set = set(main_splits["train"])
    val_set = set(main_splits["val"])
    test_set = set(main_splits["test"])

    if train_set & val_set or train_set & test_set or val_set & test_set:
        raise RuntimeError("Overlap detected among main train/val/test.")

    for name, entries in diagnostics.items():
        if len(entries) != len(set(entries)):
            raise RuntimeError(f"Duplicate diagnostic entries: {name}")
        unknown = set(entries) - known
        if unknown:
            raise RuntimeError(
                f"Unknown manifest keys in diagnostic {name}: "
                f"{sorted(unknown)[:10]}"
            )

    return {
        "main_split_counts": {
            name: len(entries)
            for name, entries in main_splits.items()
        },
        "diagnostic_split_counts": {
            name: len(entries)
            for name, entries in diagnostics.items()
            if entries
        },
    }


def full_validate_output(
    output_root: Path,
    manifest_rows: list[dict[str, str]],
) -> dict:
    source_ids: dict[str, set[int]] = {}
    source_counts = Counter()

    for index, row in enumerate(manifest_rows, start=1):
        source = row["source"]
        source_counts[source] += 1

        image_path = output_root / row["image_path"]
        mask_path = output_root / row["mask_path"]

        if not image_path.is_file():
            raise FileNotFoundError(f"Missing output image: {image_path}")
        if not mask_path.is_file():
            raise FileNotFoundError(f"Missing output mask: {mask_path}")

        with Image.open(image_path) as image:
            image_size = image.size
        with Image.open(mask_path) as image:
            image.load()
            mask = np.asarray(image)
            mask_size = image.size

        if image_size != mask_size:
            raise RuntimeError(
                f"Output image/mask size mismatch: "
                f"{image_path}, {mask_path}"
            )

        validate_target_mask_array(mask, str(mask_path))
        ids = {int(v) for v in np.unique(mask)}
        source_ids.setdefault(source, set()).update(ids)

        ratio = non_ignore_ratio(mask)
        expected_ratio = float(row["non_ignore_ratio"])
        if abs(ratio - expected_ratio) > 1e-6:
            raise RuntimeError(
                f"non_ignore_ratio mismatch: {mask_path}"
            )

        if index % 1000 == 0:
            print(
                f"[validate] {index}/{len(manifest_rows)} pairs checked"
            )

    return {
        "status": "PASS",
        "manifest_count": len(manifest_rows),
        "source_counts": dict(source_counts),
        "observed_target_ids": {
            source: sorted(ids)
            for source, ids in source_ids.items()
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()

    if not 0.0 <= args.min_non_ignore_ratio <= 1.0:
        raise ValueError(
            "--min-non-ignore-ratio must be between 0 and 1."
        )

    data_root = args.data_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    require_dir(data_root, "data root")

    rellis_root = resolve_dataset_root(data_root, "RELLIS-3D")
    rugd_root = resolve_dataset_root(data_root, "RUGD")
    ycor_root = resolve_dataset_root(data_root, "YCOR")

    adom_root: Path | None = None
    if not args.skip_adom_v2:
        if args.adom_v2_root is not None:
            adom_root = args.adom_v2_root.expanduser().resolve()
        else:
            adom_root = resolve_optional_dataset_root(
                data_root,
                "ADOM-v2",
            )

    prepare_output_root(output_root, args.overwrite)

    manifest_rows: list[dict[str, str]] = []
    main_splits: dict[str, list[str]] = {
        "train": [],
        "val": [],
        "test": [],
    }
    diagnostics: dict[str, list[str]] = {
        "rugd_val": [],
        "rugd_test": [],
        "ycor_val": [],
        "adom_v2_val": [],
        "adom_v2_test": [],
    }

    pixel_counts: Counter[int] = Counter()
    storage_counts: Counter[str] = Counter()
    source_summary: dict = {}

    process_rellis(
        rellis_root,
        output_root,
        args.storage,
        manifest_rows,
        main_splits,
        pixel_counts,
        source_summary,
        storage_counts,
    )

    process_rugd(
        rugd_root,
        output_root,
        args.storage,
        args.rugd_index_scheme,
        manifest_rows,
        main_splits,
        diagnostics,
        pixel_counts,
        source_summary,
        storage_counts,
    )

    process_ycor(
        ycor_root,
        output_root,
        args.storage,
        args.min_non_ignore_ratio,
        manifest_rows,
        main_splits,
        diagnostics,
        pixel_counts,
        source_summary,
        storage_counts,
    )

    if adom_root is not None:
        process_adom_v2(
            adom_root,
            output_root,
            args.storage,
            args.adom_eval_policy,
            manifest_rows,
            main_splits,
            diagnostics,
            pixel_counts,
            source_summary,
            storage_counts,
        )
    else:
        print("\n=== ADOM-v2 ===")
        print("[SKIP] ADOM-v2 directory not provided/found.")

    # Deterministic ordering in generated metadata.
    manifest_rows.sort(key=lambda row: row["sample_key"])
    for entries in main_splits.values():
        entries.sort()
    for entries in diagnostics.values():
        entries.sort()

    split_check = validate_split_sets(
        manifest_rows,
        main_splits,
        diagnostics,
    )

    write_manifest(
        output_root / "manifest.csv",
        manifest_rows,
    )

    for split in ("train", "val", "test"):
        write_split(
            output_root / "splits" / f"{split}.txt",
            main_splits[split],
        )

    for name, entries in diagnostics.items():
        if entries:
            write_split(
                output_root
                / "splits"
                / f"{name}_diagnostic.txt",
                entries,
            )

    write_target_statistics(
        output_root / "results" / "target_class_statistics.csv",
        pixel_counts,
    )

    if args.no_full_validation:
        final_validation = {
            "status": "PASS_PARTIAL",
            "note": "Full second-pass output validation was skipped.",
        }
    else:
        print("\n=== Final full validation ===")
        final_validation = full_validate_output(
            output_root,
            manifest_rows,
        )

    summary = {
        "dataset_name": "adom_semantic20_standalone",
        "mapping_version": "standalone-v1",
        "num_classes": NUM_CLASSES,
        "ignore_index": IGNORE_INDEX,
        "reduce_zero_label": False,
        "external_adom_repo_dependency": False,
        "runtime_dependencies": ["numpy", "Pillow"],
        "min_non_ignore_ratio": args.min_non_ignore_ratio,
        "adom_eval_policy": args.adom_eval_policy,
        "source_summary": source_summary,
        "storage_mode_counts": dict(storage_counts),
        **split_check,
        "manifest_count": len(manifest_rows),
        "target_pixel_counts": {
            str(target_id): count
            for target_id, count in sorted(pixel_counts.items())
        },
        "final_validation": final_validation,
    }

    (output_root / "results" / "build_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_root / "results" / "final_check.json").write_text(
        json.dumps(final_validation, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n[PASS] Unified standalone preprocessing completed.")
    print(
        json.dumps(
            {
                "manifest_count": len(manifest_rows),
                **split_check,
                "sources": source_summary,
                "output_root": str(output_root),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
