#!/usr/bin/env python3
"""Validate an ADOM standalone package against the Semantic20 loader contract."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError


SPLITS = ("train", "val", "test")
REQUIRED_FIELDS = {
    "sample_key",
    "split",
    "source_date",
    "source_sequence",
    "image_path",
    "mask_path",
}
ALLOWED_IDS = set(range(19)) | {255}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument(
        "--write-success-marker",
        action="store_true",
        help="Write _SUCCESS only after all package validation gates pass.",
    )
    return parser.parse_args()


def resolve_relative(root: Path, stored: str, field: str) -> Path:
    path = Path(stored)
    if path.is_absolute():
        raise ValueError(f"Absolute path in {field}: {stored}")
    resolved = (root / path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Path escapes package root in {field}: {stored}") from exc
    return resolved


def read_manifest(root: Path) -> list[dict[str, str]]:
    path = root / "manifest.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = REQUIRED_FIELDS - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Manifest fields missing: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError("Manifest is empty")
    keys = [row["sample_key"] for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Manifest contains duplicate sample_key values")
    return rows


def read_split_keys(root: Path) -> dict[str, list[str]]:
    output: dict[str, list[str]] = {}
    for split in SPLITS:
        path = root / "splits" / f"{split}.txt"
        if not path.is_file():
            raise FileNotFoundError(path)
        keys = [
            line.strip()
            for line in path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        ]
        if not keys or len(keys) != len(set(keys)):
            raise ValueError(f"Split is empty or has duplicates: {split}")
        output[split] = keys
    overlap = (
        set(output["train"]) & set(output["val"])
        | set(output["train"]) & set(output["test"])
        | set(output["val"]) & set(output["test"])
    )
    if overlap:
        raise ValueError(f"Split sample overlap: {sorted(overlap)[:10]}")
    return output


def collect_png_relative_paths(root: Path) -> set[str]:
    if not root.is_dir():
        raise FileNotFoundError(root)
    return {
        path.relative_to(root.parent).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() == ".png"
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def counter_payload(value: Counter[int]) -> dict[str, int]:
    return {str(target_id): value[target_id] for target_id in sorted(value)}


def validate(root: Path) -> dict[str, object]:
    rows = read_manifest(root)
    split_keys = read_split_keys(root)
    manifest_keys = {row["sample_key"] for row in rows}
    file_split_keys = set().union(*(set(values) for values in split_keys.values()))
    if manifest_keys != file_split_keys:
        raise ValueError("Manifest and split files contain different sample keys")

    expected_images = {row["image_path"] for row in rows}
    expected_masks = {row["mask_path"] for row in rows}
    actual_images = collect_png_relative_paths(root / "images")
    actual_masks = collect_png_relative_paths(root / "masks")
    if expected_images != actual_images:
        raise ValueError("Manifest and images directory differ")
    if expected_masks != actual_masks:
        raise ValueError("Manifest and masks directory differ")

    key_to_split = {
        sample_key: split
        for split, values in split_keys.items()
        for sample_key in values
    }
    sequence_splits: dict[str, set[str]] = {}
    distribution: Counter[int] = Counter()
    distribution_by_split = {split: Counter() for split in SPLITS}
    distribution_by_sequence: dict[str, Counter[int]] = {}
    image_presence_by_split = {split: Counter() for split in SPLITS}
    image_presence_by_sequence: dict[str, Counter[int]] = {}
    non_ignore_pixels_by_split: Counter[str] = Counter()
    non_ignore_pixels_by_sequence: Counter[str] = Counter()
    all_ignore_by_split = {split: [] for split in SPLITS}

    for row in rows:
        split = row["split"]
        if split not in SPLITS or key_to_split[row["sample_key"]] != split:
            raise ValueError(f"Manifest split mismatch: {row['sample_key']}")
        sequence = f"{row['source_date']}/{row['source_sequence']}"
        sequence_splits.setdefault(sequence, set()).add(split)
        image_path = resolve_relative(root, row["image_path"], "image_path")
        mask_path = resolve_relative(root, row["mask_path"], "mask_path")
        if not image_path.is_file() or not mask_path.is_file():
            raise FileNotFoundError(f"Missing pair: {row['sample_key']}")
        try:
            with Image.open(image_path) as image:
                image.load()
                image_size = image.size
            with Image.open(mask_path) as mask:
                mask.load()
                if mask.mode != "L":
                    raise ValueError(
                        f"Mask must be uint8 single-channel L: "
                        f"{row['mask_path']}, mode={mask.mode}"
                    )
                mask_size = mask.size
                mask_array = np.asarray(mask, dtype=np.uint8)
        except (OSError, UnidentifiedImageError) as exc:
            raise ValueError(f"Unreadable pair: {row['sample_key']}: {exc}") from exc
        if image_size != mask_size:
            raise ValueError(f"Pair size mismatch: {row['sample_key']}")
        observed, counts = np.unique(mask_array, return_counts=True)
        unexpected = {int(value) for value in observed} - ALLOWED_IDS
        if unexpected:
            raise ValueError(
                f"Unexpected Semantic20 IDs in {row['mask_path']}: {sorted(unexpected)}"
            )
        non_ignore = 0
        for value, count in zip(observed, counts, strict=True):
            target_id = int(value)
            distribution[target_id] += int(count)
            distribution_by_split[split][target_id] += int(count)
            distribution_by_sequence.setdefault(sequence, Counter())[target_id] += int(
                count
            )
            if target_id != 255:
                non_ignore += int(count)
                image_presence_by_split[split][target_id] += 1
                image_presence_by_sequence.setdefault(sequence, Counter())[target_id] += 1
        non_ignore_pixels_by_split[split] += non_ignore
        non_ignore_pixels_by_sequence[sequence] += non_ignore
        if non_ignore == 0:
            all_ignore_by_split[split].append(row["sample_key"])

    leaked_sequences = {
        sequence: sorted(splits)
        for sequence, splits in sequence_splits.items()
        if len(splits) != 1
    }
    if leaked_sequences:
        raise ValueError(f"Sequence split leakage: {leaked_sequences}")
    if all_ignore_by_split["train"]:
        raise ValueError(
            "Train contains all-ignore masks: "
            f"{all_ignore_by_split['train'][:10]}"
        )

    summary_path = root / "metadata" / "conversion_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8-sig"))
    if summary.get("num_classes") != 19 or summary.get("ignore_index") != 255:
        raise ValueError("Conversion summary has the wrong Semantic20 contract")
    if summary.get("reduce_zero_label") is not False:
        raise ValueError("Conversion summary reduce_zero_label must be false")

    actual_split_counts = {split: len(split_keys[split]) for split in SPLITS}
    actual_distribution = counter_payload(distribution)
    actual_distribution_by_split = {
        split: counter_payload(distribution_by_split[split]) for split in SPLITS
    }
    actual_distribution_by_sequence = {
        sequence: counter_payload(value)
        for sequence, value in sorted(distribution_by_sequence.items())
    }
    actual_image_presence_by_split = {
        split: counter_payload(image_presence_by_split[split]) for split in SPLITS
    }
    actual_image_presence_by_sequence = {
        sequence: counter_payload(value)
        for sequence, value in sorted(image_presence_by_sequence.items())
    }
    actual_non_ignore_by_split = {
        split: non_ignore_pixels_by_split[split] for split in SPLITS
    }
    actual_non_ignore_by_sequence = dict(sorted(non_ignore_pixels_by_sequence.items()))
    all_ignore_count = sum(len(values) for values in all_ignore_by_split.values())
    expected_summary_values = {
        "samples": len(rows),
        "split_counts": actual_split_counts,
        "pixel_counts_by_target_id": actual_distribution,
        "pixel_counts_by_split": actual_distribution_by_split,
        "pixel_counts_by_sequence": actual_distribution_by_sequence,
        "image_presence_by_split": actual_image_presence_by_split,
        "image_presence_by_sequence": actual_image_presence_by_sequence,
        "non_ignore_pixels_by_split": actual_non_ignore_by_split,
        "non_ignore_pixels_by_sequence": actual_non_ignore_by_sequence,
        "all_ignore_masks": all_ignore_count,
        "all_ignore_by_split": all_ignore_by_split,
    }
    for field, expected_value in expected_summary_values.items():
        if summary.get(field) != expected_value:
            raise ValueError(
                f"Conversion summary mismatch for {field}: "
                f"summary={summary.get(field)}, actual={expected_value}"
            )

    return {
        "schema_version": "adom-semantic20-validation-v1",
        "status": "PASS",
        "samples": len(rows),
        "split_counts": actual_split_counts,
        "sequences": len(sequence_splits),
        "observed_target_ids": sorted(distribution),
        "pixel_counts_by_target_id": actual_distribution,
        "pixel_counts_by_split": actual_distribution_by_split,
        "pixel_counts_by_sequence": actual_distribution_by_sequence,
        "image_presence_by_split": actual_image_presence_by_split,
        "image_presence_by_sequence": actual_image_presence_by_sequence,
        "non_ignore_pixels_by_split": actual_non_ignore_by_split,
        "non_ignore_pixels_by_sequence": actual_non_ignore_by_sequence,
        "all_ignore_masks": all_ignore_count,
        "all_ignore_by_split": all_ignore_by_split,
        "manifest_sha256": sha256(root / "manifest.csv"),
        "conversion_summary_sha256": sha256(summary_path),
    }


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    root = args.input_root.expanduser().resolve()
    try:
        if not root.is_dir():
            raise ValueError(f"INPUT_ROOT is not a directory: {root}")
        report = validate(root)
        results_root = root / "results"
        results_root.mkdir(parents=True, exist_ok=True)
        report_path = results_root / "validation_report.json"
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        if args.write_success_marker:
            (root / "_SUCCESS").write_text(
                json.dumps(
                    {
                        "status": "PASS",
                        "validation_report_sha256": sha256(report_path),
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        print(json.dumps(report, indent=2, ensure_ascii=False))
        print("✅ ADOM SEMANTIC20 PACKAGE VALID")
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
