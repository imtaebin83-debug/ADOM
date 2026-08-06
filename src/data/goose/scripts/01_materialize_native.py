from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
RGB_SUFFIX = "_windshield_vis"
LABEL_SUFFIX = "_labelids"


def parse_args() -> argparse.Namespace:
    tool_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess an extracted GOOSE directory into validated visible RGB "
            "and original 64-class label-ID pairs. No remapping or filtering is done."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="Extracted GOOSE root containing images/{split} and labels/{split}.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--classes-config",
        type=Path,
        default=tool_root / "config" / "goose64_classes.csv",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "val"),
        default=("train", "val"),
    )
    return parser.parse_args()


def normalize(value: str) -> str:
    return "_".join(value.strip().lower().replace("-", "_").split())


def load_classes(path: Path) -> dict[int, str]:
    if not path.is_file():
        raise FileNotFoundError(f"GOOSE class config not found: {path}")
    classes: dict[int, str] = {}
    names: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required = {"goose_raw_id", "goose_class"}
        if not required <= set(reader.fieldnames or []):
            raise ValueError("Class config requires goose_raw_id,goose_class")
        for row in reader:
            raw_id = int(row["goose_raw_id"])
            class_name = normalize(row["goose_class"])
            if raw_id in classes or class_name in names:
                raise ValueError(f"Duplicate GOOSE class: {raw_id}/{class_name}")
            classes[raw_id] = class_name
            names.add(class_name)
    if set(classes) != set(range(64)):
        raise ValueError(f"GOOSE raw IDs must be exactly 0..63: {sorted(classes)}")
    return classes


def require_input_root(path: Path) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"Extracted GOOSE input root not found: {path}")


def require_empty_or_absent(path: Path) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(
            f"Output root must be empty or absent; existing data is never overwritten: {path}"
        )


def locate_split_root(input_root: Path, section: str, split: str) -> Path:
    candidates = sorted(
        path.resolve()
        for path in input_root.rglob(split)
        if path.is_dir() and path.parent.name == section
    )
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected exactly one {section}/{split} below {input_root}; "
            f"found {len(candidates)}"
        )
    return candidates[0]


def sample_key(path: Path, split_root: Path, expected_suffix: str) -> str:
    relative = path.relative_to(split_root)
    if not path.stem.endswith(expected_suffix):
        raise ValueError(f"Unexpected GOOSE filename: {path}")
    stem = path.stem[: -len(expected_suffix)].rstrip("_")
    if not stem:
        raise ValueError(f"Empty GOOSE sample stem: {path}")
    return relative.with_name(stem).as_posix()


def index_raw_pairs(
    image_root: Path,
    label_root: Path,
) -> list[tuple[str, Path, Path]]:
    images: dict[str, Path] = {}
    labels: dict[str, Path] = {}
    for path in sorted(image_root.rglob("*")):
        if (
            path.is_file()
            and path.suffix.lower() in IMAGE_SUFFIXES
            and path.stem.endswith(RGB_SUFFIX)
        ):
            key = sample_key(path, image_root, RGB_SUFFIX)
            if key in images:
                raise ValueError(f"Duplicate visible RGB key: {key}")
            images[key] = path
    for path in sorted(label_root.rglob("*")):
        if (
            path.is_file()
            and path.suffix.lower() == ".png"
            and path.stem.endswith(LABEL_SUFFIX)
        ):
            key = sample_key(path, label_root, LABEL_SUFFIX)
            if key in labels:
                raise ValueError(f"Duplicate label-ID key: {key}")
            labels[key] = path
    if set(images) != set(labels):
        raise ValueError(
            "Pair mismatch: "
            f"missing_masks={sorted(set(images) - set(labels))[:20]}, "
            f"missing_images={sorted(set(labels) - set(images))[:20]}"
        )
    if not images:
        raise ValueError(
            f"No visible RGB/label-ID pairs found below {image_root} and {label_root}"
        )
    return [(key, images[key], labels[key]) for key in sorted(images)]


def load_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        image.load()
        array = np.asarray(image)
    if array.ndim != 2 or not np.issubdtype(array.dtype, np.integer):
        raise ValueError(
            f"Expected indexed 2D mask, got {array.shape}/{array.dtype}: {path}"
        )
    if array.size and int(array.max()) > 255:
        raise ValueError(f"Mask contains an ID above 255: {path}")
    return array.astype(np.uint8, copy=False)


def copy_file_new(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Destination already exists: {destination}")
    shutil.copy2(source, destination)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else fields
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if len(set(args.splits)) != len(args.splits):
        raise ValueError(f"Duplicate split arguments: {args.splits}")
    input_root = args.input_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    classes_config = args.classes_config.expanduser().resolve()
    require_input_root(input_root)
    require_empty_or_absent(output_root)
    try:
        output_root.relative_to(input_root)
    except ValueError:
        pass
    else:
        raise ValueError("Output root must not be inside the extracted input root")

    classes = load_classes(classes_config)
    indexed: dict[str, tuple[Path, Path, list[tuple[str, Path, Path]]]] = {}
    for split in args.splits:
        image_root = locate_split_root(input_root, "images", split)
        label_root = locate_split_root(input_root, "labels", split)
        if image_root.parent.parent != label_root.parent.parent:
            raise ValueError(
                f"images/{split} and labels/{split} do not share one dataset root"
            )
        indexed[split] = (
            image_root,
            label_root,
            index_raw_pairs(image_root, label_root),
        )

    output_root.mkdir(parents=True, exist_ok=True)
    metadata_root = output_root / "metadata"
    metadata_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(classes_config, metadata_root / "goose64_classes.csv")

    pair_rows: list[dict[str, Any]] = []
    distribution_rows: list[dict[str, Any]] = []
    class_frames: Counter[int] = Counter()
    class_pixels: Counter[int] = Counter()
    split_summary: dict[str, dict[str, int]] = {}

    for split, (image_root, label_root, pairs) in indexed.items():
        for index, (key, source_image, source_label) in enumerate(pairs, start=1):
            with Image.open(source_image) as image:
                image.load()
                image_size = image.size
            mask = load_mask(source_label)
            if image_size != (mask.shape[1], mask.shape[0]):
                raise ValueError(f"RGB/mask size mismatch for {split}/{key}")
            values, counts = np.unique(mask, return_counts=True)
            unknown = sorted(int(value) for value in values if int(value) not in classes)
            if unknown:
                raise ValueError(f"Unknown GOOSE IDs {unknown}: {split}/{key}")

            image_relative = source_image.relative_to(image_root)
            label_relative = source_label.relative_to(label_root)
            output_image = output_root / "images" / split / image_relative
            output_label = output_root / "labels" / split / label_relative
            copy_file_new(source_image, output_image)
            copy_file_new(source_label, output_label)

            total_pixels = int(mask.size)
            counts_by_id = {int(value): int(count) for value, count in zip(values, counts)}
            present_ids = sorted(counts_by_id)
            for raw_id in present_ids:
                class_frames[raw_id] += 1
                class_pixels[raw_id] += counts_by_id[raw_id]
            pair_rows.append(
                {
                    "split": split,
                    "sample_key": key,
                    "source_image": source_image.relative_to(input_root).as_posix(),
                    "source_label": source_label.relative_to(input_root).as_posix(),
                    "output_image": output_image.relative_to(output_root).as_posix(),
                    "output_label": output_label.relative_to(output_root).as_posix(),
                    "width": image_size[0],
                    "height": image_size[1],
                    "image_size_bytes": source_image.stat().st_size,
                    "label_size_bytes": source_label.stat().st_size,
                    "present_raw_ids": " ".join(map(str, present_ids)),
                    "present_classes": " ".join(classes[value] for value in present_ids),
                }
            )
            row: dict[str, Any] = {
                "split": split,
                "sample_key": key,
                "total_pixels": total_pixels,
            }
            for raw_id, class_name in classes.items():
                count = counts_by_id.get(raw_id, 0)
                row[f"raw_{raw_id:02d}_{class_name}_pixels"] = count
                row[f"raw_{raw_id:02d}_{class_name}_percent"] = (
                    f"{count * 100.0 / total_pixels:.10f}"
                )
            distribution_rows.append(row)
            if index % 100 == 0 or index == len(pairs):
                print(f"[{split}] {index}/{len(pairs)} GOOSE-native pairs")
        split_summary[split] = {"paired_and_materialized": len(pairs)}

    class_rows = [
        {
            "goose_raw_id": raw_id,
            "goose_class": class_name,
            "frame_count": class_frames[raw_id],
            "pixel_count": class_pixels[raw_id],
        }
        for raw_id, class_name in classes.items()
    ]
    write_csv(metadata_root / "pair_manifest.csv", pair_rows, ["split", "sample_key"])
    write_csv(
        metadata_root / "per_image_goose64_distribution.csv",
        distribution_rows,
        ["split", "sample_key", "total_pixels"],
    )
    write_csv(
        metadata_root / "goose64_class_summary.csv",
        class_rows,
        ["goose_raw_id", "goose_class", "frame_count", "pixel_count"],
    )
    summary = {
        "status": "PASS",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "GOOSE original 64-class fine semantic labels",
        "scope": "visible RGB plus original label-ID masks; no remapping or filtering",
        "goose_class_count": 64,
        "input_contract": "already extracted images/{split} and labels/{split}",
        "splits": list(args.splits),
        "split_summary": split_summary,
    }
    (metadata_root / "preprocess_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"GOOSE-native preprocessing completed: {output_root}")


if __name__ == "__main__":
    main()
