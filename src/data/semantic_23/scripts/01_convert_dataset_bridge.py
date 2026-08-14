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


TARGET_IDS = set(range(23)) | {255}
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
            "Bridge one validated dataset source into the ADOM Semantic23 ID space. "
            "Every GOOSE input pair is materialized after remapping."
        )
    )
    parser.add_argument("--dataset", choices=("rellis", "rugd", "ycor", "goose"), required=True)
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
    parser.add_argument("--min-non-ignore-ratio", type=float, default=0.01)
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Bridge mapping not found: {path}")
    with path.open("r", encoding="utf-8-sig") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid YAML root: {path}")
    if int(config.get("num_classes", -1)) != 23:
        raise ValueError("num_classes must be 23")
    if int(config.get("ignore_index", -1)) != 255:
        raise ValueError("ignore_index must be 255")
    target_classes = {int(key): str(value) for key, value in config["target_classes"].items()}
    if set(target_classes) != TARGET_IDS:
        raise ValueError("target_classes must contain IDs 0..22 and 255")
    expected_new = {19: "snow", 20: "animal", 21: "artifact", 22: "cobble"}
    if {key: target_classes[key] for key in expected_new} != expected_new:
        raise ValueError(
            "New target classes must be snow=19, animal=20, artifact=21, cobble=22"
        )
    return config


def target_classes(config: dict[str, Any]) -> dict[int, str]:
    return {int(key): str(value) for key, value in config["target_classes"].items()}


def load_mapping(config: dict[str, Any], dataset: str) -> dict[int, int]:
    section = config.get(dataset)
    if not isinstance(section, dict) or not isinstance(section.get("source_to_target"), dict):
        raise KeyError(f"Missing source_to_target for {dataset}")
    mapping: dict[int, int] = {}
    for source_key, entry in section["source_to_target"].items():
        if not isinstance(entry, dict):
            raise ValueError(f"Invalid mapping entry: {dataset}/{source_key}")
        source_id = int(source_key)
        target_id = int(entry["target_id"])
        use = bool(entry.get("use", target_id != 255))
        if target_id not in TARGET_IDS:
            raise ValueError(f"Invalid target ID {target_id}: {dataset}/{source_id}")
        if not use and target_id != 255:
            raise ValueError(f"use=false must map to 255: {dataset}/{source_id}")
        if source_id in mapping:
            raise ValueError(f"Duplicate source ID: {dataset}/{source_id}")
        mapping[source_id] = target_id
    if dataset == "goose" and set(mapping) != set(range(64)):
        raise ValueError("GOOSE mapping must explicitly cover raw IDs 0..63")
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
    rows = [line.strip().replace("\\", "/") for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if len(rows) != len(set(rows)):
        raise ValueError(f"Duplicate sample in split: {path}")
    for row in rows:
        relative = PurePosixPath(row)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe split entry: {row}")
    return rows


def find_unique_file(root: Path, relative_stem: str, suffixes: Iterable[str]) -> Path:
    stem_path = PurePosixPath(relative_stem)
    candidates = [
        root.joinpath(*stem_path.parts[:-1], stem_path.name + suffix)
        for suffix in suffixes
    ]
    matches = [path for path in candidates if path.is_file()]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one file for {relative_stem} below {root}; found {matches}"
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
                    split=split,
                    key=key,
                    image_path=find_unique_file(image_root, key, IMAGE_SUFFIXES),
                    mask_path=find_unique_file(mask_root, key, (".png",)),
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
                    split=split,
                    key=key,
                    image_path=find_unique_file(image_root, key, IMAGE_SUFFIXES),
                    mask_path=find_unique_file(mask_root, key, (".png",)),
                )
            )
    return samples


def discover_ycor(input_root: Path) -> list[Sample]:
    samples: list[Sample] = []
    for source_split, output_split in (("train", "train"), ("valid", "val")):
        split_root = input_root / source_split
        if not split_root.is_dir():
            raise FileNotFoundError(f"YCOR split not found: {split_root}")
        for sample_dir in sorted(path for path in split_root.iterdir() if path.is_dir()):
            image_path = sample_dir / "rgb.jpg"
            mask_path = sample_dir / "labels.png"
            if not image_path.is_file() or not mask_path.is_file():
                raise FileNotFoundError(f"YCOR pair is incomplete: {sample_dir}")
            samples.append(Sample(output_split, sample_dir.name, image_path, mask_path))
    return samples


def discover_goose(input_root: Path) -> list[Sample]:
    manifest_path = input_root / "metadata" / "pair_manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"GOOSE native manifest not found: {manifest_path}")
    samples: list[Sample] = []
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required = {"split", "sample_key", "output_image", "output_label"}
        if not required <= set(reader.fieldnames or []):
            raise ValueError(f"GOOSE manifest requires {sorted(required)}")
        for row in reader:
            split = row["split"]
            if split not in {"train", "val"}:
                raise ValueError(f"Invalid GOOSE split: {split}")
            samples.append(
                Sample(
                    split=split,
                    key=row["sample_key"],
                    image_path=safe_relative_path(input_root, row["output_image"], "output_image"),
                    mask_path=safe_relative_path(input_root, row["output_label"], "output_label"),
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
        raise ValueError(f"Unknown YCOR palette values {unknown_rgb}: {path}")
    values = np.asarray([rgb_to_source[int(value)] for value in unique], dtype=np.int32)
    return values[inverse].reshape(packed.shape)


def remap_mask(source_mask: np.ndarray, mapping: dict[int, int], dataset: str, path: Path) -> np.ndarray:
    source_ids = {int(value) for value in np.unique(source_mask)}
    unknown = sorted(source_ids - set(mapping))
    if unknown:
        raise ValueError(f"Unknown {dataset} source IDs in {path}: {unknown}")
    target = np.full(source_mask.shape, 255, dtype=np.uint8)
    for source_id, target_id in mapping.items():
        target[source_mask == source_id] = target_id
    output_ids = {int(value) for value in np.unique(target)}
    invalid = sorted(output_ids - TARGET_IDS)
    if invalid:
        raise ValueError(f"Invalid target IDs after remapping {path}: {invalid}")
    return target


def class_counts(mask: np.ndarray) -> dict[int, int]:
    values, counts = np.unique(mask, return_counts=True)
    return {int(value): int(count) for value, count in zip(values, counts)}


def prepare_output(path: Path) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f"Output root must be empty or absent: {path}")
    path.mkdir(parents=True, exist_ok=True)
    (path / "metadata").mkdir(parents=True, exist_ok=True)


def link_or_copy(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    fieldnames = list(rows[0]) if rows else fields
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def safe_key_path(key: str) -> PurePosixPath:
    path = PurePosixPath(key)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"Unsafe sample key: {key}")
    return path


def append_suffix(path: PurePosixPath, suffix: str) -> Path:
    return Path(*path.parts[:-1], path.name + suffix)


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.min_non_ignore_ratio <= 1.0:
        raise ValueError("--min-non-ignore-ratio must be between 0 and 1")
    input_root = args.input_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    mapping_path = args.mapping.expanduser().resolve()
    if not input_root.is_dir():
        raise FileNotFoundError(f"Input root not found: {input_root}")
    config = load_config(mapping_path)
    mapping = load_mapping(config, args.dataset)

    if args.dataset == "rellis":
        samples = discover_rellis(input_root, (args.split_root or input_root / "splits").resolve())
    elif args.dataset == "rugd":
        samples = discover_rugd(
            (args.image_root or input_root / "images").resolve(),
            (args.mask_root or input_root / "indexLabel").resolve(),
            (args.split_root or input_root / "splits").resolve(),
        )
    elif args.dataset == "ycor":
        samples = discover_ycor(input_root)
    else:
        samples = discover_goose(input_root)
    if not samples:
        raise ValueError(f"No {args.dataset} samples discovered")

    prepare_output(output_root)
    shutil.copy2(mapping_path, output_root / "metadata" / "bridge_mapping.yaml")
    manifest_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    target_pixels: Counter[int] = Counter()
    target_frames: Counter[int] = Counter()
    split_counts: Counter[str] = Counter()
    processing_counts: Counter[str] = Counter()
    storage_counts: Counter[str] = Counter()
    preserve_ids = {int(value) for value in config[args.dataset].get("preserve_if_target_present", [])}

    for index, sample in enumerate(samples, start=1):
        if not sample.image_path.is_file() or not sample.mask_path.is_file():
            raise FileNotFoundError(f"Incomplete pair: {sample}")
        source_mask = load_ycor_mask(sample.mask_path, config) if args.dataset == "ycor" else load_index_mask(sample.mask_path)
        with Image.open(sample.image_path) as image:
            image.load()
            image_size = image.size
        if image_size != (source_mask.shape[1], source_mask.shape[0]):
            raise ValueError(f"Image/mask size mismatch: {sample.image_path} / {sample.mask_path}")
        target_mask = remap_mask(source_mask, mapping, args.dataset, sample.mask_path)
        counts = class_counts(target_mask)
        total_pixels = int(target_mask.size)
        non_ignore_ratio = 1.0 - counts.get(255, 0) / total_pixels

        if args.dataset == "goose":
            keep = True
            reason = "goose_full_dataset"
        else:
            preserved = any(counts.get(target_id, 0) > 0 for target_id in preserve_ids)
            keep = non_ignore_ratio >= args.min_non_ignore_ratio or preserved
            reason = "kept" if keep else "low_non_ignore_ratio"
        processing_counts[reason] += 1

        row: dict[str, Any] = {
            "dataset": args.dataset,
            "split": sample.split,
            "sample_key": sample.key,
            "total_pixels": total_pixels,
            "non_ignore_ratio": f"{non_ignore_ratio:.10f}",
            "materialized": str(keep).lower(),
            "processing_reason": reason,
        }
        for target_id in range(23):
            row[f"target_{target_id:02d}_pixels"] = counts.get(target_id, 0)
            row[f"target_{target_id:02d}_percent"] = f"{counts.get(target_id, 0) * 100.0 / total_pixels:.10f}"
        row["ignore_255_pixels"] = counts.get(255, 0)
        row["ignore_255_percent"] = f"{counts.get(255, 0) * 100.0 / total_pixels:.10f}"
        audit_rows.append(row)
        if not keep:
            continue

        key_path = safe_key_path(sample.key)
        destination_image = (
            output_root
            / "images"
            / sample.split
            / append_suffix(key_path, sample.image_path.suffix.lower())
        )
        destination_mask = (
            output_root
            / "masks"
            / sample.split
            / append_suffix(key_path, ".png")
        )
        storage = link_or_copy(sample.image_path, destination_image)
        destination_mask.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(target_mask, mode="L").save(destination_mask)
        storage_counts[storage] += 1
        split_counts[sample.split] += 1
        for target_id, count in counts.items():
            target_pixels[target_id] += count
            if count:
                target_frames[target_id] += 1
        manifest_rows.append(
            {
                "sample_key": f"{args.dataset}/{sample.split}/{sample.key}",
                "source": args.dataset,
                "source_split": sample.split,
                "output_split": sample.split,
                "sample_id": sample.key,
                "image_path": destination_image.relative_to(output_root).as_posix(),
                "mask_path": destination_mask.relative_to(output_root).as_posix(),
                "non_ignore_ratio": f"{non_ignore_ratio:.10f}",
            }
        )
        if index % 100 == 0 or index == len(samples):
            print(f"[{args.dataset}] processed {index}/{len(samples)} samples")

    class_names = target_classes(config)
    class_rows = [
        {
            "target_id": target_id,
            "target_class": class_names[target_id],
            "frame_count": target_frames[target_id],
            "pixel_count": target_pixels[target_id],
        }
        for target_id in range(23)
    ]
    write_csv(output_root / "metadata" / "manifest.csv", manifest_rows, ["sample_key", "source", "output_split", "image_path", "mask_path"])
    write_csv(
        output_root / "metadata" / "per_image_distribution.csv",
        audit_rows,
        ["dataset", "split", "sample_key", "materialized", "processing_reason"],
    )
    write_csv(output_root / "metadata" / "target_class_summary.csv", class_rows, ["target_id", "target_class", "frame_count", "pixel_count"])
    summary = {
        "status": "PASS",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "mapping_version": config["mapping_version"],
        "num_classes": 23,
        "ignore_index": 255,
        "input_samples": len(samples),
        "materialized_samples": len(manifest_rows),
        "processing_reason_counts": dict(sorted(processing_counts.items())),
        "output_split_counts": dict(sorted(split_counts.items())),
        "image_storage_counts": dict(sorted(storage_counts.items())),
    }
    (output_root / "metadata" / "conversion_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Semantic23 bridge completed: {output_root}")


if __name__ == "__main__":
    main()
