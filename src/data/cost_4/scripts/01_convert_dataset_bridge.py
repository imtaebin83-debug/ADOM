from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import numpy as np
import yaml
from PIL import Image


TARGET_IDS = {0, 1, 2, 3, 255}
EXPECTED_SOURCE_IDS = {
    "rellis": set(range(19)) | {255},
    "rugd": set(range(25)),
    "ycor": set(range(9)),
    "goose": set(range(64)),
}
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")


@dataclass(frozen=True)
class Sample:
    split: str
    key: str
    image_path: Path
    mask_path: Path


def parse_args() -> argparse.Namespace:
    tool_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Convert one validated dataset source into the ADOM Cost4 "
            "traversability ID space without filtering samples."
        )
    )
    parser.add_argument(
        "--dataset",
        choices=("rellis", "rugd", "ycor", "goose"),
        required=True,
    )
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--mapping",
        type=Path,
        default=tool_root / "config" / "bridge_mapping.yaml",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        help="RUGD original image root; defaults to INPUT_ROOT/images.",
    )
    parser.add_argument(
        "--mask-root",
        type=Path,
        help="RUGD original index-label root; defaults to INPUT_ROOT/indexLabel.",
    )
    parser.add_argument(
        "--split-root",
        type=Path,
        help="RELLIS/RUGD split root; defaults to INPUT_ROOT/splits.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Cost4 mapping not found: {path}")
    with path.open("r", encoding="utf-8-sig") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("Cost4 mapping root must be an object")
    if int(config.get("num_classes", -1)) != 4:
        raise ValueError("num_classes must be 4")
    if int(config.get("ignore_index", -1)) != 255:
        raise ValueError("ignore_index must be 255")
    classes = {int(key): str(value) for key, value in config.get("target_classes", {}).items()}
    expected_classes = {
        0: "paved",
        1: "natural_low",
        2: "medium",
        3: "high_obstacle",
        255: "ignore",
    }
    if classes != expected_classes:
        raise ValueError(f"Target classes must be exactly {expected_classes}")
    return config


def load_mapping(config: dict[str, Any], dataset: str) -> dict[int, int]:
    section = config.get(dataset)
    if not isinstance(section, dict) or not isinstance(section.get("source_to_target"), dict):
        raise KeyError(f"Missing source_to_target mapping: {dataset}")
    mapping: dict[int, int] = {}
    for source_key, entry in section["source_to_target"].items():
        if not isinstance(entry, dict):
            raise ValueError(f"Invalid mapping entry: {dataset}/{source_key}")
        source_id = int(source_key)
        target_id = int(entry["target_id"])
        if source_id in mapping:
            raise ValueError(f"Duplicate source ID: {dataset}/{source_id}")
        if target_id not in TARGET_IDS:
            raise ValueError(f"Invalid Cost4 target ID: {dataset}/{source_id} -> {target_id}")
        mapping[source_id] = target_id
    if set(mapping) != EXPECTED_SOURCE_IDS[dataset]:
        missing = sorted(EXPECTED_SOURCE_IDS[dataset] - set(mapping))
        extra = sorted(set(mapping) - EXPECTED_SOURCE_IDS[dataset])
        raise ValueError(f"Incomplete {dataset} source IDs: missing={missing}, extra={extra}")
    return mapping


def safe_relative_path(root: Path, stored_value: str, field_name: str) -> Path:
    relative = PurePosixPath(stored_value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Non-portable {field_name}: {stored_value}")
    resolved = root.joinpath(*relative.parts).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{field_name} escapes input root: {stored_value}") from exc
    return resolved


def read_split(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Split file not found: {path}")
    rows = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if len(rows) != len(set(rows)):
        raise ValueError(f"Duplicate split entries: {path}")
    return rows


def find_unique_file(root: Path, relative_stem: str, suffixes: Iterable[str]) -> Path:
    normalized = PurePosixPath(relative_stem)
    if normalized.is_absolute() or ".." in normalized.parts:
        raise ValueError(f"Unsafe sample ID: {relative_stem}")
    direct = [root.joinpath(*normalized.parts).with_suffix(suffix) for suffix in suffixes]
    matches = [path.resolve() for path in direct if path.is_file()]
    if not matches:
        basename = normalized.name
        matches = sorted(
            path.resolve()
            for path in root.rglob("*")
            if path.is_file() and path.stem == basename and path.suffix.lower() in suffixes
        )
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one file for {relative_stem} below {root}; found {len(matches)}"
        )
    return matches[0]


def discover_rellis(input_root: Path, split_root: Path) -> list[Sample]:
    image_root = input_root / "images"
    mask_root = input_root / "masks"
    samples: list[Sample] = []
    seen: set[str] = set()
    for split in ("train", "val", "test"):
        for key in read_split(split_root / f"{split}.txt"):
            if key in seen:
                raise ValueError(f"RELLIS sample appears in multiple splits: {key}")
            seen.add(key)
            samples.append(
                Sample(
                    split,
                    key,
                    find_unique_file(image_root, key, IMAGE_SUFFIXES),
                    find_unique_file(mask_root, key, (".png",)),
                )
            )
    return samples


def discover_rugd(image_root: Path, mask_root: Path, split_root: Path) -> list[Sample]:
    samples: list[Sample] = []
    seen: set[str] = set()
    for split in ("train", "val", "test"):
        for key in read_split(split_root / f"{split}.txt"):
            if key in seen:
                raise ValueError(f"RUGD sample appears in multiple splits: {key}")
            seen.add(key)
            samples.append(
                Sample(
                    split,
                    key,
                    find_unique_file(image_root, key, IMAGE_SUFFIXES),
                    find_unique_file(mask_root, key, (".png",)),
                )
            )
    return samples


def discover_ycor(input_root: Path) -> list[Sample]:
    samples: list[Sample] = []
    seen: set[str] = set()
    for source_split, output_split in (("train", "train"), ("valid", "val")):
        split_root = input_root / source_split
        if not split_root.is_dir():
            raise FileNotFoundError(f"YCOR split not found: {split_root}")
        for sample_dir in sorted(path for path in split_root.iterdir() if path.is_dir()):
            key = sample_dir.name
            if key in seen:
                raise ValueError(f"Duplicate YCOR sample ID: {key}")
            seen.add(key)
            image_path = sample_dir / "rgb.jpg"
            mask_path = sample_dir / "labels.png"
            if not image_path.is_file() or not mask_path.is_file():
                raise FileNotFoundError(f"YCOR pair is incomplete: {sample_dir}")
            samples.append(Sample(output_split, key, image_path, mask_path))
    return samples


def discover_goose(input_root: Path) -> list[Sample]:
    manifest_path = input_root / "metadata" / "pair_manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"GOOSE native manifest not found: {manifest_path}")
    samples: list[Sample] = []
    seen: set[tuple[str, str]] = set()
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required = {"split", "sample_key", "output_image", "output_label"}
        if not required <= set(reader.fieldnames or []):
            raise ValueError(f"GOOSE manifest requires {sorted(required)}")
        for row in reader:
            split = row["split"]
            key = row["sample_key"]
            if split not in {"train", "val"}:
                raise ValueError(f"Invalid GOOSE split: {split}")
            identity = (split, key)
            if identity in seen:
                raise ValueError(f"Duplicate GOOSE sample: {split}/{key}")
            seen.add(identity)
            samples.append(
                Sample(
                    split,
                    key,
                    safe_relative_path(input_root, row["output_image"], "output_image"),
                    safe_relative_path(input_root, row["output_label"], "output_label"),
                )
            )
    if not samples:
        raise ValueError("GOOSE native manifest is empty")
    return samples


def load_index_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        image.load()
        array = np.asarray(image)
    if array.ndim == 2 and np.issubdtype(array.dtype, np.integer):
        return array.astype(np.int32, copy=False)
    if array.ndim == 3 and array.shape[2] >= 3:
        rgb = array[:, :, :3]
        if np.array_equal(rgb[:, :, 0], rgb[:, :, 1]) and np.array_equal(rgb[:, :, 1], rgb[:, :, 2]):
            return rgb[:, :, 0].astype(np.int32, copy=False)
    raise ValueError(f"Expected an indexed mask, got {array.shape}/{array.dtype}: {path}")


def load_ycor_mask(path: Path, config: dict[str, Any]) -> np.ndarray:
    palette = config["ycor"]["source_palette_rgb"]
    rgb_to_source = {
        (int(rgb[0]) << 16) | (int(rgb[1]) << 8) | int(rgb[2]): int(source_id)
        for source_id, rgb in palette.items()
    }
    with Image.open(path) as image:
        image.load()
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    packed = (
        (rgb[:, :, 0].astype(np.uint32) << 16)
        | (rgb[:, :, 1].astype(np.uint32) << 8)
        | rgb[:, :, 2].astype(np.uint32)
    )
    unique, inverse = np.unique(packed, return_inverse=True)
    unknown = [int(value) for value in unique if int(value) not in rgb_to_source]
    if unknown:
        unknown_rgb = [((value >> 16) & 255, (value >> 8) & 255, value & 255) for value in unknown[:20]]
        raise ValueError(f"Unknown YCOR RGB labels in {path}: {unknown_rgb}")
    source_values = np.asarray([rgb_to_source[int(value)] for value in unique], dtype=np.int32)
    return source_values[inverse].reshape(packed.shape)


def remap_mask(
    source_mask: np.ndarray,
    mapping: dict[int, int],
    dataset: str,
    path: Path,
) -> np.ndarray:
    observed = {int(value) for value in np.unique(source_mask)}
    unknown = sorted(observed - set(mapping))
    if unknown:
        raise ValueError(f"Unknown {dataset} source IDs in {path}: {unknown}")
    target = np.full(source_mask.shape, 255, dtype=np.uint8)
    for source_id, target_id in mapping.items():
        target[source_mask == source_id] = target_id
    invalid = sorted({int(value) for value in np.unique(target)} - TARGET_IDS)
    if invalid:
        raise ValueError(f"Invalid Cost4 IDs after remapping {path}: {invalid}")
    return target


def require_empty_or_absent(path: Path) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f"Output root must be empty or absent: {path}")


def link_or_copy(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Destination already exists: {destination}")
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def safe_key_path(key: str) -> PurePosixPath:
    path = PurePosixPath(key)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"Unsafe sample key: {key}")
    return path


def append_suffix(path: PurePosixPath, suffix: str) -> Path:
    return Path(*path.parent.parts, f"{path.name}{suffix}")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    input_root = args.input_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    mapping_path = args.mapping.expanduser().resolve()
    if not input_root.is_dir():
        raise FileNotFoundError(f"Input root not found: {input_root}")
    require_empty_or_absent(output_root)
    config = load_config(mapping_path)
    mapping = load_mapping(config, args.dataset)

    if args.dataset == "rellis":
        split_root = (args.split_root or input_root / "splits").expanduser().resolve()
        samples = discover_rellis(input_root, split_root)
    elif args.dataset == "rugd":
        image_root = (args.image_root or input_root / "images").expanduser().resolve()
        mask_root = (args.mask_root or input_root / "indexLabel").expanduser().resolve()
        split_root = (args.split_root or input_root / "splits").expanduser().resolve()
        samples = discover_rugd(image_root, mask_root, split_root)
    elif args.dataset == "ycor":
        samples = discover_ycor(input_root)
    else:
        samples = discover_goose(input_root)

    output_root.mkdir(parents=True, exist_ok=True)
    metadata_root = output_root / "metadata"
    metadata_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(mapping_path, metadata_root / "bridge_mapping.yaml")

    manifest_rows: list[dict[str, Any]] = []
    distribution_rows: list[dict[str, Any]] = []
    class_pixels: Counter[int] = Counter()
    class_frames: Counter[int] = Counter()
    split_counts: Counter[str] = Counter()
    storage_counts: Counter[str] = Counter()

    for index, sample in enumerate(samples, start=1):
        if not sample.image_path.is_file() or not sample.mask_path.is_file():
            raise FileNotFoundError(f"Missing pair: {sample.image_path} / {sample.mask_path}")
        with Image.open(sample.image_path) as image:
            image.load()
            image_size = image.size
        source_mask = (
            load_ycor_mask(sample.mask_path, config)
            if args.dataset == "ycor"
            else load_index_mask(sample.mask_path)
        )
        if image_size != (source_mask.shape[1], source_mask.shape[0]):
            raise ValueError(f"RGB/mask size mismatch: {args.dataset}/{sample.key}")
        target_mask = remap_mask(source_mask, mapping, args.dataset, sample.mask_path)

        key_path = safe_key_path(sample.key)
        image_suffix = sample.image_path.suffix.lower()
        destination_image = output_root / "images" / sample.split / append_suffix(key_path, image_suffix)
        destination_mask = output_root / "masks" / sample.split / append_suffix(key_path, ".png")
        storage_counts[link_or_copy(sample.image_path, destination_image)] += 1
        destination_mask.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(target_mask, mode="L").save(destination_mask, format="PNG")

        values, counts = np.unique(target_mask, return_counts=True)
        counts_by_id = {int(value): int(count) for value, count in zip(values, counts)}
        for target_id, count in counts_by_id.items():
            class_pixels[target_id] += count
            class_frames[target_id] += 1
        total_pixels = int(target_mask.size)
        non_ignore_ratio = float(np.count_nonzero(target_mask != 255) / total_pixels)
        sample_key = f"{args.dataset}/{sample.split}/{key_path.as_posix()}"
        manifest_rows.append(
            {
                "sample_key": sample_key,
                "source": args.dataset,
                "source_split": sample.split,
                "output_split": sample.split,
                "sample_id": key_path.as_posix(),
                "image_path": destination_image.relative_to(output_root).as_posix(),
                "mask_path": destination_mask.relative_to(output_root).as_posix(),
                "non_ignore_ratio": f"{non_ignore_ratio:.8f}",
            }
        )
        distribution: dict[str, Any] = {
            "sample_key": sample_key,
            "total_pixels": total_pixels,
        }
        for target_id in (0, 1, 2, 3, 255):
            count = counts_by_id.get(target_id, 0)
            distribution[f"class_{target_id}_pixels"] = count
            distribution[f"class_{target_id}_percent"] = f"{count * 100.0 / total_pixels:.10f}"
        distribution_rows.append(distribution)
        split_counts[sample.split] += 1
        if index % 500 == 0 or index == len(samples):
            print(f"[{args.dataset}] converted {index}/{len(samples)} pairs")

    write_csv(metadata_root / "manifest.csv", manifest_rows)
    write_csv(metadata_root / "per_image_distribution.csv", distribution_rows)
    class_rows = [
        {
            "target_id": target_id,
            "target_class": config["target_classes"][target_id],
            "frame_count": class_frames[target_id],
            "pixel_count": class_pixels[target_id],
        }
        for target_id in (0, 1, 2, 3, 255)
    ]
    write_csv(metadata_root / "class_summary.csv", class_rows)
    summary = {
        "status": "PASS",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "target_ontology": "ADOM Cost4 traversability",
        "num_classes": 4,
        "ignore_index": 255,
        "input_samples": len(samples),
        "materialized_samples": len(manifest_rows),
        "filtering_policy": "none; every discovered pair is materialized",
        "split_counts": dict(sorted(split_counts.items())),
        "storage_counts": dict(sorted(storage_counts.items())),
    }
    (metadata_root / "conversion_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Cost4 {args.dataset} bridge completed: {output_root}")


if __name__ == "__main__":
    main()
