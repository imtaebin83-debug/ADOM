#!/usr/bin/env python3
"""Convert the standalone ADOM CVAT export into a Semantic20 package."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError


SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class Sample:
    sample_key: str
    split: str
    source_date: str
    source_sequence: str
    relative_path: Path
    image_path: Path
    mask_path: Path


def parse_args() -> argparse.Namespace:
    script_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Convert RGB CVAT masks to single-channel Semantic20 train IDs "
            "and build split/manifest files."
        )
    )
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--mapping",
        type=Path,
        default=script_root / "config" / "label_mapping.json",
    )
    parser.add_argument(
        "--splits",
        type=Path,
        default=script_root / "config" / "split_sequences.json",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def configure_utf8_output() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def load_mapping(path: Path) -> tuple[dict[tuple[int, int, int], int], int]:
    config = load_json(path)
    if int(config.get("num_classes", -1)) != 19:
        raise ValueError("Semantic20 num_classes must be 19")
    ignore_index = int(config.get("ignore_index", -1))
    if ignore_index != 255:
        raise ValueError("Semantic20 ignore_index must be 255")
    if config.get("reduce_zero_label") is not False:
        raise ValueError("reduce_zero_label must be false")

    target_classes = {int(key): str(value) for key, value in config.get("target_classes", {}).items()}
    if set(target_classes) != set(range(19)) | {ignore_index}:
        raise ValueError("target_classes must contain IDs 0..18 and 255")

    mapping: dict[tuple[int, int, int], int] = {}
    for item in config.get("rgb_to_target", []):
        rgb_value = item.get("rgb")
        if not isinstance(rgb_value, list) or len(rgb_value) != 3:
            raise ValueError(f"Invalid RGB mapping entry: {item}")
        rgb = tuple(int(channel) for channel in rgb_value)
        if any(channel < 0 or channel > 255 for channel in rgb):
            raise ValueError(f"RGB channel outside uint8 range: {rgb}")
        target_id = int(item.get("target_id", -1))
        if target_id not in target_classes:
            raise ValueError(f"Unknown Semantic20 target ID: {target_id}")
        if rgb in mapping:
            raise ValueError(f"Duplicate RGB mapping: {rgb}")
        mapping[rgb] = target_id

    if mapping.get((0, 0, 0)) != ignore_index:
        raise ValueError("Black background must map to ignore_index 255")
    return mapping, ignore_index


def load_split_assignments(path: Path) -> dict[str, str]:
    config = load_json(path)
    raw_splits = config.get("splits")
    if not isinstance(raw_splits, dict) or set(raw_splits) != set(SPLITS):
        raise ValueError(f"Split config must contain exactly {SPLITS}")

    assignments: dict[str, str] = {}
    for split in SPLITS:
        sequences = raw_splits[split]
        if not isinstance(sequences, list) or not sequences:
            raise ValueError(f"Split must contain at least one sequence: {split}")
        for sequence in sequences:
            if not isinstance(sequence, str) or sequence.count("/") != 1:
                raise ValueError(f"Sequence must be DATE/SESSION: {sequence}")
            date, session = sequence.split("/", 1)
            if not date or not session or Path(sequence).is_absolute():
                raise ValueError(f"Invalid sequence key: {sequence}")
            if sequence in assignments:
                raise ValueError(f"Sequence appears in multiple splits: {sequence}")
            assignments[sequence] = split
    return assignments


def collect_pngs(root: Path) -> dict[Path, Path]:
    if not root.is_dir():
        raise FileNotFoundError(root)
    output: dict[Path, Path] = {}
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if not path.is_file() or path.suffix.lower() != ".png":
            continue
        relative_path = path.relative_to(root)
        if relative_path in output:
            raise ValueError(f"Duplicate relative path: {relative_path.as_posix()}")
        output[relative_path] = path
    return output


def discover_samples(input_root: Path, assignments: dict[str, str]) -> list[Sample]:
    samples: list[Sample] = []
    discovered_sequences: set[str] = set()
    date_roots = sorted(
        path for path in input_root.iterdir() if path.is_dir() and path.name != "upload"
    )
    for date_root in date_roots:
        raw = collect_pngs(date_root / "raw")
        masks = collect_pngs(date_root / "masks")
        if set(raw) != set(masks):
            raw_only = sorted(set(raw) - set(masks), key=lambda item: item.as_posix())
            mask_only = sorted(set(masks) - set(raw), key=lambda item: item.as_posix())
            raise ValueError(
                f"Raw/mask paths differ for {date_root.name}: "
                f"raw_only={[item.as_posix() for item in raw_only[:10]]}, "
                f"mask_only={[item.as_posix() for item in mask_only[:10]]}"
            )
        for relative_path in sorted(raw, key=lambda item: item.as_posix()):
            if len(relative_path.parts) < 2:
                raise ValueError(
                    f"Expected SESSION/frame.png below {date_root.name}: "
                    f"{relative_path.as_posix()}"
                )
            session = relative_path.parts[0]
            sequence_key = f"{date_root.name}/{session}"
            discovered_sequences.add(sequence_key)
            if sequence_key not in assignments:
                raise ValueError(f"Sequence is not assigned to a split: {sequence_key}")
            sample_key = f"{date_root.name}/{relative_path.with_suffix('').as_posix()}"
            samples.append(
                Sample(
                    sample_key=sample_key,
                    split=assignments[sequence_key],
                    source_date=date_root.name,
                    source_sequence=session,
                    relative_path=relative_path,
                    image_path=raw[relative_path],
                    mask_path=masks[relative_path],
                )
            )

    configured_sequences = set(assignments)
    if discovered_sequences != configured_sequences:
        missing = sorted(configured_sequences - discovered_sequences)
        extra = sorted(discovered_sequences - configured_sequences)
        raise ValueError(
            f"Configured/discovered sequences differ: missing={missing}, extra={extra}"
        )
    sample_keys = [sample.sample_key for sample in samples]
    if len(sample_keys) != len(set(sample_keys)):
        raise ValueError("Duplicate sample_key generated")
    if not samples:
        raise ValueError("No source pairs found")
    return samples


def mask_array(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        image.load()
        if image.mode != "RGB":
            raise ValueError(f"CVAT mask must be RGB: {path}, mode={image.mode}")
        array = np.asarray(image, dtype=np.uint8)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"CVAT mask must have shape HxWx3: {path}, shape={array.shape}")
    return array


def convert_mask(
    source: np.ndarray,
    mapping: dict[tuple[int, int, int], int],
    ignore_index: int,
    path: Path,
) -> tuple[np.ndarray, Counter[int]]:
    packed = (
        source[..., 0].astype(np.uint32) << 16
        | source[..., 1].astype(np.uint32) << 8
        | source[..., 2].astype(np.uint32)
    )
    packed_mapping = {
        (red << 16) | (green << 8) | blue: target_id
        for (red, green, blue), target_id in mapping.items()
    }
    observed, counts = np.unique(packed, return_counts=True)
    unknown = [
        (int(value >> 16), int((value >> 8) & 255), int(value & 255))
        for value in observed
        if int(value) not in packed_mapping
    ]
    if unknown:
        raise ValueError(f"Unknown RGB colors in {path}: {unknown}")

    target = np.full(packed.shape, ignore_index, dtype=np.uint8)
    distribution: Counter[int] = Counter()
    for value, count in zip(observed, counts, strict=True):
        target_id = packed_mapping[int(value)]
        target[packed == value] = target_id
        distribution[target_id] += int(count)
    return target, distribution


def ensure_safe_output(input_root: Path, output_root: Path) -> None:
    if output_root == input_root or input_root in output_root.parents:
        raise ValueError("OUTPUT_ROOT must not be inside INPUT_ROOT")
    if output_root.exists() and not output_root.is_dir():
        raise ValueError(f"OUTPUT_ROOT is not a directory: {output_root}")
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError(f"OUTPUT_ROOT must be new or empty: {output_root}")


def preflight(
    samples: list[Sample],
    mapping: dict[tuple[int, int, int], int],
    ignore_index: int,
) -> tuple[Counter[int], int]:
    distribution: Counter[int] = Counter()
    all_ignore_masks = 0
    for sample in samples:
        try:
            with Image.open(sample.image_path) as image:
                image.load()
                image_size = image.size
            source_mask = mask_array(sample.mask_path)
        except (OSError, UnidentifiedImageError) as exc:
            raise ValueError(f"Unreadable pair for {sample.sample_key}: {exc}") from exc
        mask_size = (source_mask.shape[1], source_mask.shape[0])
        if image_size != mask_size:
            raise ValueError(
                f"Size mismatch for {sample.sample_key}: "
                f"image={image_size}, mask={mask_size}"
            )
        _, sample_distribution = convert_mask(
            source_mask, mapping, ignore_index, sample.mask_path
        )
        distribution.update(sample_distribution)
        if sum(
            count
            for target_id, count in sample_distribution.items()
            if target_id != ignore_index
        ) == 0:
            all_ignore_masks += 1
    return distribution, all_ignore_masks


def write_package(
    samples: list[Sample],
    output_root: Path,
    mapping: dict[tuple[int, int, int], int],
    ignore_index: int,
    mapping_path: Path,
    splits_path: Path,
    distribution: Counter[int],
    all_ignore_masks: int,
) -> None:
    manifest_rows: list[dict[str, str]] = []
    split_keys: dict[str, list[str]] = {split: [] for split in SPLITS}
    for sample in samples:
        image_relative = Path("images") / sample.source_date / sample.relative_path
        mask_relative = Path("masks") / sample.source_date / sample.relative_path
        image_destination = output_root / image_relative
        mask_destination = output_root / mask_relative
        image_destination.parent.mkdir(parents=True, exist_ok=True)
        mask_destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(sample.image_path, image_destination)
        source_mask = mask_array(sample.mask_path)
        target_mask, _ = convert_mask(
            source_mask, mapping, ignore_index, sample.mask_path
        )
        Image.fromarray(target_mask, mode="L").save(mask_destination)
        manifest_rows.append(
            {
                "sample_key": sample.sample_key,
                "split": sample.split,
                "source_date": sample.source_date,
                "source_sequence": sample.source_sequence,
                "image_path": image_relative.as_posix(),
                "mask_path": mask_relative.as_posix(),
            }
        )
        split_keys[sample.split].append(sample.sample_key)

    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "manifest.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "sample_key",
                "split",
                "source_date",
                "source_sequence",
                "image_path",
                "mask_path",
            ),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    split_root = output_root / "splits"
    split_root.mkdir(parents=True, exist_ok=True)
    split_counts: dict[str, int] = {}
    for split in SPLITS:
        keys = sorted(split_keys[split])
        (split_root / f"{split}.txt").write_text(
            "".join(f"{key}\n" for key in keys), encoding="utf-8"
        )
        split_counts[split] = len(keys)

    metadata_root = output_root / "metadata"
    metadata_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(mapping_path, metadata_root / "label_mapping.json")
    shutil.copy2(splits_path, metadata_root / "split_sequences.json")
    summary = {
        "format_version": 1,
        "dataset": "adom_data_semantic20_partial_v1",
        "num_classes": 19,
        "ignore_index": ignore_index,
        "reduce_zero_label": False,
        "samples": len(samples),
        "split_counts": split_counts,
        "pixel_counts_by_target_id": {
            str(target_id): distribution[target_id]
            for target_id in sorted(distribution)
        },
        "all_ignore_masks": all_ignore_masks,
        "path_policy": "All manifest paths are relative to the package root.",
    }
    (metadata_root / "conversion_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    configure_utf8_output()
    args = parse_args()
    input_root = args.input_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    mapping_path = args.mapping.expanduser().resolve()
    splits_path = args.splits.expanduser().resolve()
    try:
        if not input_root.is_dir():
            raise ValueError(f"INPUT_ROOT is not a directory: {input_root}")
        if not args.dry_run:
            ensure_safe_output(input_root, output_root)
        mapping, ignore_index = load_mapping(mapping_path)
        assignments = load_split_assignments(splits_path)
        samples = discover_samples(input_root, assignments)
        distribution, all_ignore_masks = preflight(samples, mapping, ignore_index)
        split_counts = Counter(sample.split for sample in samples)
        print(f"Samples: {len(samples)}")
        for split in SPLITS:
            print(f"{split}: {split_counts[split]}")
        print(f"Observed target IDs: {sorted(distribution)}")
        print(f"All-ignore masks: {all_ignore_masks}")
        if args.dry_run:
            print("DRY RUN: no files were created or modified")
            return 0
        write_package(
            samples,
            output_root,
            mapping,
            ignore_index,
            mapping_path,
            splits_path,
            distribution,
            all_ignore_masks,
        )
        print(f"Output: {output_root}")
        print("✅ ADOM SEMANTIC20 PACKAGE CREATED")
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
