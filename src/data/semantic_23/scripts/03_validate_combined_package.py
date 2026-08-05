from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image


ALLOWED_IDS = set(range(23)) | {255}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate an ADOM Semantic23 package.")
    parser.add_argument("--input-root", type=Path, required=True)
    return parser.parse_args()


def resolve_relative(root: Path, value: str, field: str) -> Path:
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Non-portable {field}: {value}")
    resolved = root.joinpath(*relative.parts).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{field} escapes package root: {value}") from exc
    return resolved


def read_split(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Required split not found: {path}")
    rows = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if len(rows) != len(set(rows)):
        raise ValueError(f"Duplicate entries in split: {path}")
    return rows


def main() -> None:
    args = parse_args()
    root = args.input_root.expanduser().resolve()
    manifest_path = root / "metadata" / "manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError("Combined manifest is empty")
    required = {"sample_key", "source", "source_split", "package_role", "image_path", "mask_path"}
    if not required <= set(rows[0]):
        raise ValueError(f"Manifest requires {sorted(required)}")
    keys = [row["sample_key"] for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate sample_key in manifest")
    by_key = {row["sample_key"]: row for row in rows}

    main_splits = {
        split: read_split(root / "splits" / f"{split}.txt")
        for split in ("train", "val", "test")
    }
    split_sets = {name: set(values) for name, values in main_splits.items()}
    if split_sets["train"] & split_sets["val"] or split_sets["train"] & split_sets["test"] or split_sets["val"] & split_sets["test"]:
        raise ValueError("Main split overlap detected")
    listed = set().union(*split_sets.values())
    unknown_split_keys = listed - set(by_key)
    if unknown_split_keys:
        raise ValueError(f"Split contains unknown sample keys: {sorted(unknown_split_keys)[:20]}")
    for split in ("val", "test"):
        non_rellis = [key for key in main_splits[split] if by_key[key]["source"] != "rellis"]
        if non_rellis:
            raise ValueError(f"Main {split} must be RELLIS-only: {non_rellis[:20]}")

    class_pixels: Counter[int] = Counter()
    class_frames: Counter[int] = Counter()
    source_counts: Counter[str] = Counter()
    for index, row in enumerate(rows, start=1):
        image_path = resolve_relative(root, row["image_path"], "image_path")
        mask_path = resolve_relative(root, row["mask_path"], "mask_path")
        if not image_path.is_file() or not mask_path.is_file():
            raise FileNotFoundError(f"Missing pair: {row['sample_key']}")
        with Image.open(image_path) as image:
            image.load()
            image_size = image.size
        with Image.open(mask_path) as mask_image:
            mask_image.load()
            mask = np.asarray(mask_image)
            mode = mask_image.mode
        if mask.ndim != 2 or mode != "L" or mask.dtype != np.uint8:
            raise ValueError(f"Mask must be uint8 single-channel L: {mask_path}")
        if image_size != (mask.shape[1], mask.shape[0]):
            raise ValueError(f"Image/mask size mismatch: {row['sample_key']}")
        values, counts = np.unique(mask, return_counts=True)
        ids = {int(value) for value in values}
        invalid = sorted(ids - ALLOWED_IDS)
        if invalid:
            raise ValueError(f"Invalid target IDs in {mask_path}: {invalid}")
        for value, count in zip(values, counts):
            class_pixels[int(value)] += int(count)
            class_frames[int(value)] += 1
        source_counts[row["source"]] += 1
        if index % 500 == 0 or index == len(rows):
            print(f"Validated {index}/{len(rows)} pairs")

    report = {
        "status": "PASS",
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sample_count": len(rows),
        "source_counts": dict(sorted(source_counts.items())),
        "main_split_counts": {key: len(value) for key, value in main_splits.items()},
        "valid_target_ids": sorted(class_pixels),
        "class_pixel_counts": {str(key): class_pixels[key] for key in range(23)},
        "class_frame_counts": {str(key): class_frames[key] for key in range(23)},
        "evaluation_policy": "main val/test contain RELLIS only",
    }
    report_path = root / "metadata" / "validation_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Semantic23 validation PASS: {report_path}")


if __name__ == "__main__":
    main()
