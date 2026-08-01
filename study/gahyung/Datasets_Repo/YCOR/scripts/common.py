from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOWNLOAD_DIR = PROJECT_ROOT / "downloads"
RAW_SEARCH_ROOT = PROJECT_ROOT / "raw"

CONFIG_DIR = PROJECT_ROOT / "config"
MAPPING_FILE = CONFIG_DIR / "label_mapping.json"

WORK_DIR = PROJECT_ROOT / "work"
MANIFEST_DIR = WORK_DIR / "manifests"
REPORT_DIR = WORK_DIR / "reports"

PROCESSED_ROOT = PROJECT_ROOT / "processed" / "YCOR_ADOM"
IMAGES_ROOT = PROCESSED_ROOT / "images"
MASKS_ROOT = PROCESSED_ROOT / "masks"
METADATA_ROOT = PROCESSED_ROOT / "metadata"
QC_ROOT = PROCESSED_ROOT / "qc"

OUTPUT_SPLITS = ("train", "val")
SOURCE_SPLITS = {"train": "train", "val": "valid"}

EXPECTED_COUNTS = {"train": 931, "val": 145}

ALLOWED_TARGET_IDS = {0, 1, 2, 3, 255}

TARGET_CLASSES = {
    0: "paved_low_cost",
    1: "natural_low_cost",
    2: "medium_cost",
    3: "high_cost_or_obstacle",
    255: "ignore",
}

TARGET_PALETTE = {
    0: (128, 128, 128),
    1: (60, 180, 75),
    2: (255, 225, 25),
    3: (230, 25, 75),
    255: (0, 0, 0),
}


def ensure_directories() -> None:
    for path in (
        DOWNLOAD_DIR,
        RAW_SEARCH_ROOT,
        MANIFEST_DIR,
        REPORT_DIR,
        METADATA_ROOT,
        QC_ROOT,
    ):
        path.mkdir(parents=True, exist_ok=True)

    for split in OUTPUT_SPLITS:
        (IMAGES_ROOT / split).mkdir(parents=True, exist_ok=True)
        (MASKS_ROOT / split).mkdir(parents=True, exist_ok=True)


def is_dataset_root(path: Path) -> bool:
    if not path.is_dir():
        return False

    train_dir = path / "train"
    valid_dir = path / "valid"

    if not train_dir.is_dir() or not valid_dir.is_dir():
        return False

    train_samples = [
        d for d in train_dir.iterdir()
        if d.is_dir()
    ]
    valid_samples = [
        d for d in valid_dir.iterdir()
        if d.is_dir()
    ]

    return bool(train_samples) and bool(valid_samples)


def discover_dataset_root() -> Path:
    preferred = [
        RAW_SEARCH_ROOT / "yamaha",
        RAW_SEARCH_ROOT / "yamaha_v0" / "yamaha",
        RAW_SEARCH_ROOT / "yamaha_v0",
        RAW_SEARCH_ROOT / "YCOR" / "yamaha",
        RAW_SEARCH_ROOT / "YCOR",
    ]

    for candidate in preferred:
        if is_dataset_root(candidate):
            return candidate.resolve()

    candidates = []
    for train_dir in RAW_SEARCH_ROOT.rglob("train"):
        parent = train_dir.parent
        if is_dataset_root(parent):
            candidates.append(parent.resolve())

    unique_candidates = sorted(set(candidates), key=lambda p: (len(p.parts), str(p)))

    if not unique_candidates:
        raise FileNotFoundError(
            "YCOR dataset root was not found.\n"
            "Expected a folder containing both train/ and valid/ below:\n"
            f"  {RAW_SEARCH_ROOT}\n"
            "Each split must contain sample folders with rgb.jpg and labels.png."
        )

    if len(unique_candidates) > 1:
        formatted = "\n".join(f"  {path}" for path in unique_candidates)
        raise RuntimeError(
            "Multiple YCOR dataset roots were found. Keep only one copy or "
            "move the desired one to raw/yamaha:\n"
            + formatted
        )

    return unique_candidates[0]


def find_pair(sample_dir: Path) -> Tuple[Path, Path]:
    exact_image = sample_dir / "rgb.jpg"
    exact_mask = sample_dir / "labels.png"

    if exact_image.exists() and exact_mask.exists():
        return exact_image, exact_mask

    image_candidates = sorted(
        path for path in sample_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg"}
    )
    mask_candidates = sorted(
        path for path in sample_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".png"
    )

    if len(image_candidates) != 1 or len(mask_candidates) != 1:
        raise FileNotFoundError(
            f"Expected exactly one JPG image and one PNG mask in {sample_dir}, "
            f"but found images={len(image_candidates)}, masks={len(mask_candidates)}"
        )

    return image_candidates[0], mask_candidates[0]


def load_mapping_config() -> dict:
    if not MAPPING_FILE.exists():
        raise FileNotFoundError(f"Mapping file not found: {MAPPING_FILE}")

    with MAPPING_FILE.open("r", encoding="utf-8") as f:
        return json.load(f)


def rgb_to_key(rgb: Iterable[int]) -> int:
    r, g, b = (int(v) for v in rgb)
    return (r << 16) | (g << 8) | b


def load_rgb_mapping() -> Tuple[Dict[int, int], Dict[int, str]]:
    config = load_mapping_config()

    target_by_key: Dict[int, int] = {}
    name_by_key: Dict[int, str] = {}

    for entry in config["source_palette_rgb"]:
        key = rgb_to_key(entry["rgb"])
        target = int(entry["target_id"])

        if target not in ALLOWED_TARGET_IDS:
            raise ValueError(f"Invalid target ID in mapping: {target}")

        if key in target_by_key:
            raise ValueError(f"Duplicate RGB key in mapping: {entry['rgb']}")

        target_by_key[key] = target
        name_by_key[key] = entry["source_name"]

    return target_by_key, name_by_key


def load_index_mapping() -> Dict[int, int]:
    config = load_mapping_config()
    mapping = {
        int(entry["source_index_if_indexed"]): int(entry["target_id"])
        for entry in config["source_palette_rgb"]
    }
    return mapping


def load_source_mask(path: Path) -> Tuple[np.ndarray, str]:
    with Image.open(path) as image:
        image.load()
        mode = image.mode

        # RGB/RGBA masks are handled as exact color masks.
        if mode in ("RGB", "RGBA"):
            array = np.asarray(image.convert("RGB"), dtype=np.uint8)
            return array, "rgb"

        # Palette images are converted through their embedded palette.
        if mode == "P":
            array = np.asarray(image.convert("RGB"), dtype=np.uint8)
            return array, "rgb_palette"

        array = np.asarray(image)

    if array.ndim == 2:
        return array.astype(np.int32), "indexed"

    if array.ndim == 3 and array.shape[2] >= 3:
        rgb = array[:, :, :3].astype(np.uint8)
        if np.array_equal(rgb[:, :, 0], rgb[:, :, 1]) and np.array_equal(
            rgb[:, :, 1], rgb[:, :, 2]
        ):
            return rgb[:, :, 0].astype(np.int32), "indexed_repeated_channels"
        return rgb, "rgb"

    raise ValueError(f"Unsupported mask shape {array.shape}: {path}")


def remap_mask(path: Path) -> Tuple[np.ndarray, str]:
    source, encoding = load_source_mask(path)

    if encoding.startswith("rgb"):
        rgb_mapping, _ = load_rgb_mapping()
        keys = (
            (source[:, :, 0].astype(np.uint32) << 16)
            | (source[:, :, 1].astype(np.uint32) << 8)
            | source[:, :, 2].astype(np.uint32)
        )

        unique_keys, inverse = np.unique(keys, return_inverse=True)
        unknown_keys = [
            int(key) for key in unique_keys
            if int(key) not in rgb_mapping
        ]

        if unknown_keys:
            unknown_rgb = [
                ((key >> 16) & 255, (key >> 8) & 255, key & 255)
                for key in unknown_keys
            ]
            raise ValueError(
                f"Unknown RGB label colors in {path}: {unknown_rgb[:20]}"
            )

        remapped_unique = np.array(
            [rgb_mapping[int(key)] for key in unique_keys],
            dtype=np.uint8,
        )
        target = remapped_unique[inverse].reshape(keys.shape)
        return target, encoding

    index_mapping = load_index_mapping()
    unique_ids = [int(v) for v in np.unique(source)]
    unknown_ids = [v for v in unique_ids if v not in index_mapping]

    if unknown_ids:
        raise ValueError(
            f"Unknown indexed label IDs in {path}: {unknown_ids}"
        )

    target = np.full(source.shape, 255, dtype=np.uint8)
    for source_id, target_id in index_mapping.items():
        target[source == source_id] = target_id

    return target, encoding


def save_uint8_mask(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8), mode="L").save(
        path,
        format="PNG",
        optimize=True,
    )
