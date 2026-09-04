#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from pathlib import PurePosixPath

import numpy as np
from PIL import Image

ALLOWED_IDS = set(range(19)) | {255}
SPLITS = ("train", "val", "test")
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
SELECTION_FIELDS = [
    "sample_id",
    "condition",
    "sequence_id",
    "frame",
    "relative_path",
    "status",
]
EXCLUSION_FIELDS = [
    "sample_id",
    "condition",
    "sequence_id",
    "frame",
    "relative_path",
    "reason",
    "notes",
]
SPLIT_ASSIGNMENT_FIELDS = ["sample_id", "split"]


@dataclass(frozen=True)
class AdomReleaseSample:
    sample_id: str
    condition: str
    sequence_id: str
    frame: str
    image_relative: PurePosixPath
    mask_relative: PurePosixPath
    image_path: Path
    mask_path: Path
    split: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Prepare the unified ADOM Semantic20 dataset from "
            "RELLIS-3D + RUGD + YCOR + optional ADOM-v2."
        )
    )
    p.add_argument(
        "--repo-root",
        type=Path,
        required=True,
        help="Root of the cloned ADOM Git repository.",
    )
    p.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help=(
            "Root containing raw/RELLIS-3D, raw/RUGD, raw/YCOR "
            "and optionally raw/ADOM-v2."
        ),
    )
    p.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Final unified package output directory.",
    )
    p.add_argument(
        "--adom-v2-root",
        type=Path,
        default=None,
        help=(
            "Optional ADOM-v2 root. If omitted, "
            "<data-root>/raw/ADOM-v2 is used when it exists."
        ),
    )
    p.add_argument(
        "--adom-eval-policy",
        choices=("diagnostic", "mixed"),
        default="diagnostic",
        help=(
            "diagnostic: keep legacy RELLIS-only main val/test and write "
            "ADOM-v2 val/test as diagnostic splits. "
            "mixed: append ADOM-v2 val/test to main val/test."
        ),
    )
    p.add_argument(
        "--adom-split-csv",
        type=Path,
        default=None,
        help=(
            "Explicit ADOM-v2 release split assignment CSV with columns "
            "sample_id,split. Required when metadata/selection.csv is used; "
            "the pipeline never infers an unpublished split policy."
        ),
    )
    p.add_argument(
        "--min-non-ignore-ratio",
        type=float,
        default=0.01,
        help="Legacy RUGD/YCOR bridge threshold. Default: 0.01.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove generated intermediate/final output when necessary.",
    )
    p.add_argument(
        "--skip-adom-v2",
        action="store_true",
        help="Build only RELLIS-3D + RUGD + YCOR.",
    )
    return p.parse_args()


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")


def require_dir(path: Path, label: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"{label} not found: {path}")


def run(cmd: list[str]) -> None:
    print("\n$", " ".join(str(x) for x in cmd), flush=True)
    subprocess.run(cmd, check=True)


def hardlink_or_copy(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def locate_unique_named_dirs(root: Path, name: str) -> list[Path]:
    return sorted(
        [p for p in root.rglob(name) if p.is_dir()],
        key=lambda p: p.as_posix().casefold(),
    )


def collect_pngs_under_dirs(dirs: list[Path]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    duplicates: dict[str, list[Path]] = {}
    for directory in dirs:
        for path in sorted(directory.rglob("*.png")):
            key = path.name.casefold()
            if key in result and result[key] != path:
                duplicates.setdefault(key, [result[key]]).append(path)
            else:
                result[key] = path

    if duplicates:
        examples = {
            key: [str(p) for p in paths[:3]]
            for key, paths in list(duplicates.items())[:10]
        }
        raise RuntimeError(
            "Duplicate RUGD PNG basenames were found. "
            "The legacy split files assume unique basenames. "
            f"Examples: {examples}"
        )
    return result


def stage_rugd(raw_root: Path, staging_root: Path, split_root: Path) -> tuple[Path, Path]:
    """
    Create flat RUGD image/index-mask staging folders expected by the
    existing bridge converter.

    Official/extracted RUGD layouts may contain one or more directories
    named 'image' and 'indexLabel'. We search recursively and only stage
    sample IDs referenced by the repository split files.
    """
    require_dir(raw_root, "RUGD raw root")
    image_dirs = locate_unique_named_dirs(raw_root, "image")
    mask_dirs = locate_unique_named_dirs(raw_root, "indexLabel")

    if not image_dirs:
        raise FileNotFoundError(
            f"No directory named 'image' found under RUGD root: {raw_root}"
        )
    if not mask_dirs:
        raise FileNotFoundError(
            f"No directory named 'indexLabel' found under RUGD root: {raw_root}"
        )

    images = collect_pngs_under_dirs(image_dirs)
    masks = collect_pngs_under_dirs(mask_dirs)

    sample_ids: list[str] = []
    for split in ("train", "val", "test"):
        split_path = split_root / f"{split}.txt"
        require_file(split_path, f"RUGD {split} split")
        for line in split_path.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if line:
                sample_ids.append(line)

    if len(sample_ids) != len(set(sample_ids)):
        raise RuntimeError("Duplicate sample IDs exist across RUGD split files.")

    image_out = staging_root / "images"
    mask_out = staging_root / "masks"
    if staging_root.exists():
        shutil.rmtree(staging_root)
    image_out.mkdir(parents=True)
    mask_out.mkdir(parents=True)

    missing_images: list[str] = []
    missing_masks: list[str] = []

    for sample_id in sample_ids:
        filename = f"{sample_id}.png".casefold()
        image_path = images.get(filename)
        mask_path = masks.get(filename)
        if image_path is None:
            missing_images.append(sample_id)
            continue
        if mask_path is None:
            missing_masks.append(sample_id)
            continue
        hardlink_or_copy(image_path, image_out / f"{sample_id}.png")
        hardlink_or_copy(mask_path, mask_out / f"{sample_id}.png")

    if missing_images or missing_masks:
        raise RuntimeError(
            "RUGD staging failed. "
            f"Missing images={len(missing_images)} examples={missing_images[:10]}, "
            f"missing masks={len(missing_masks)} examples={missing_masks[:10]}"
        )

    print(f"[PASS] RUGD staged pairs: {len(sample_ids)}")
    return image_out, mask_out


def copy_rellis_splits(repo_split_root: Path, processed_root: Path) -> None:
    dst = processed_root / "splits"
    dst.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        src = repo_split_root / f"{split}.txt"
        require_file(src, f"RELLIS {split} split")
        shutil.copy2(src, dst / src.name)


def read_manifest(path: Path) -> list[dict[str, str]]:
    require_file(path, "manifest.csv")
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"Empty manifest: {path}")
    return rows


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def read_split(path: Path) -> list[str]:
    require_file(path, f"split {path.name}")
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def write_split(path: Path, entries: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(entries)
    path.write_text((text + "\n") if text else "", encoding="utf-8")


def read_csv_contract(
    path: Path,
    expected_fields: list[str],
    label: str,
    *,
    allow_empty: bool = False,
) -> list[dict[str, str]]:
    require_file(path, label)
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != expected_fields:
            raise RuntimeError(
                f"Unexpected {label} fields in {path}: {reader.fieldnames}. "
                f"Expected {expected_fields}."
            )
        rows = list(reader)
    if not rows and not allow_empty:
        raise RuntimeError(f"Empty {label}: {path}")
    return rows


def find_release_metadata(adom_root: Path, name: str) -> Path:
    candidates = [adom_root / "metadata" / name, adom_root / name]
    existing = [path for path in candidates if path.is_file()]
    if len(existing) > 1:
        raise RuntimeError(
            f"Multiple ADOM-v2 {name} files found: "
            f"{[str(path) for path in existing]}"
        )
    if existing:
        return existing[0]
    raise FileNotFoundError(
        f"ADOM-v2 {name} not found. Expected {candidates[0]} or {candidates[1]}."
    )


def parse_release_relative_path(value: str, label: str) -> PurePosixPath:
    if not value or "\\" in value or (len(value) >= 2 and value[1] == ":"):
        raise RuntimeError(f"Invalid POSIX-relative {label}: {value!r}")
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts or "." in relative.parts:
        raise RuntimeError(f"Unsafe relative {label}: {value!r}")
    return relative


def resolve_release_file(
    adom_root: Path,
    relative: PurePosixPath,
    label: str,
) -> Path:
    root = adom_root.resolve()
    path = root.joinpath(*relative.parts).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(f"{label} escapes ADOM-v2 root: {relative}") from exc
    require_file(path, label)
    return path


def validate_unique_casefold(values: list[str], label: str) -> None:
    seen: dict[str, str] = {}
    for value in values:
        key = value.casefold()
        if key in seen:
            raise RuntimeError(
                f"Duplicate {label} (case-insensitive): {seen[key]!r} and {value!r}"
            )
        seen[key] = value


def load_adom_release_samples(
    adom_root: Path,
    split_csv: Path | None,
) -> list[AdomReleaseSample]:
    selection_path = find_release_metadata(adom_root, "selection.csv")
    exclusions_path = find_release_metadata(adom_root, "exclusions.csv")
    if split_csv is None:
        raise RuntimeError(
            "ADOM-v2 release has no published train/val/test policy. "
            "Provide --adom-split-csv with explicit sample_id,split assignments."
        )

    selection_rows = read_csv_contract(
        selection_path,
        SELECTION_FIELDS,
        "ADOM-v2 selection.csv",
    )
    exclusion_rows = read_csv_contract(
        exclusions_path,
        EXCLUSION_FIELDS,
        "ADOM-v2 exclusions.csv",
        allow_empty=True,
    )
    split_rows = read_csv_contract(
        split_csv,
        SPLIT_ASSIGNMENT_FIELDS,
        "ADOM-v2 split assignment CSV",
    )

    selection_ids = [row["sample_id"].strip() for row in selection_rows]
    selection_paths = [row["relative_path"].strip() for row in selection_rows]
    if any(not value for value in selection_ids):
        raise RuntimeError("ADOM-v2 selection.csv contains an empty sample_id.")
    validate_unique_casefold(selection_ids, "ADOM-v2 selection sample_id")
    validate_unique_casefold(selection_paths, "ADOM-v2 selection relative_path")

    exclusion_id_values = [row["sample_id"].strip() for row in exclusion_rows]
    exclusion_path_values = [
        row["relative_path"].strip() for row in exclusion_rows
    ]
    if any(not value for value in exclusion_id_values + exclusion_path_values):
        raise RuntimeError(
            "ADOM-v2 exclusions.csv rows require sample_id and relative_path."
        )
    validate_unique_casefold(exclusion_id_values, "ADOM-v2 exclusion sample_id")
    validate_unique_casefold(
        exclusion_path_values,
        "ADOM-v2 exclusion relative_path",
    )
    for value in exclusion_path_values:
        parse_release_relative_path(value, "exclusion relative_path")
    exclusion_ids = {value.casefold() for value in exclusion_id_values}
    exclusion_paths = {value.casefold() for value in exclusion_path_values}
    overlap_ids = sorted(
        sample_id
        for sample_id in selection_ids
        if sample_id.casefold() in exclusion_ids
    )
    overlap_paths = sorted(
        relative_path
        for relative_path in selection_paths
        if relative_path.casefold() in exclusion_paths
    )
    if overlap_ids or overlap_paths:
        raise RuntimeError(
            "ADOM-v2 selection/exclusions overlap: "
            f"sample_ids={overlap_ids[:10]}, relative_paths={overlap_paths[:10]}"
        )

    split_ids = [row["sample_id"].strip() for row in split_rows]
    validate_unique_casefold(split_ids, "ADOM-v2 split sample_id")
    split_by_id: dict[str, str] = {}
    for row in split_rows:
        sample_id = row["sample_id"].strip()
        split = row["split"].strip().lower()
        if not sample_id or split not in SPLITS:
            raise RuntimeError(
                f"Invalid ADOM-v2 split assignment: sample_id={sample_id!r}, "
                f"split={row['split']!r}. Expected one of {SPLITS}."
            )
        split_by_id[sample_id.casefold()] = split

    selected_casefold = {sample_id.casefold() for sample_id in selection_ids}
    split_casefold = set(split_by_id)
    missing_assignments = sorted(selected_casefold - split_casefold)
    extra_assignments = sorted(split_casefold - selected_casefold)
    if missing_assignments or extra_assignments:
        raise RuntimeError(
            "ADOM-v2 split assignments must cover selection.csv exactly once: "
            f"missing={missing_assignments[:10]}, extra={extra_assignments[:10]}"
        )

    split_counts = Counter(split_by_id.values())
    empty_splits = [split for split in SPLITS if split_counts[split] == 0]
    if empty_splits:
        raise RuntimeError(
            f"ADOM-v2 split assignment has empty splits: {empty_splits}"
        )

    samples: list[AdomReleaseSample] = []
    sequence_splits: dict[tuple[str, str], set[str]] = {}
    for row in selection_rows:
        sample_id = row["sample_id"].strip()
        condition = row["condition"].strip()
        sequence_id = row["sequence_id"].strip()
        frame = row["frame"].strip()
        status = row["status"].strip()
        if not all((condition, sequence_id, frame)):
            raise RuntimeError(f"Incomplete ADOM-v2 selection row: {row}")
        if status.casefold() != "ok":
            raise RuntimeError(
                f"Selected ADOM-v2 sample must have status OK: {sample_id}={status!r}"
            )
        expected_sample_id = f"{condition}_{sequence_id}_{frame}"
        if sample_id != expected_sample_id:
            raise RuntimeError(
                f"ADOM-v2 sample_id mismatch: {sample_id!r}, "
                f"expected {expected_sample_id!r}"
            )

        image_relative = parse_release_relative_path(
            row["relative_path"].strip(),
            "selection relative_path",
        )
        if (
            len(image_relative.parts) < 4
            or image_relative.parts[0] != "images"
            or image_relative.parts[1] != condition
            or image_relative.parts[2] != sequence_id
            or image_relative.stem != frame
            or image_relative.suffix.lower() != ".png"
        ):
            raise RuntimeError(
                f"ADOM-v2 metadata/path mismatch for {sample_id}: "
                f"condition={condition!r}, sequence_id={sequence_id!r}, "
                f"frame={frame!r}, relative_path={image_relative.as_posix()!r}"
            )

        mask_relative = PurePosixPath(
            "masks",
            *image_relative.parts[1:],
        )
        image_path = resolve_release_file(
            adom_root,
            image_relative,
            "ADOM-v2 selected image",
        )
        mask_path = resolve_release_file(
            adom_root,
            mask_relative,
            "ADOM-v2 selected mask",
        )
        split = split_by_id[sample_id.casefold()]
        sequence_splits.setdefault((condition, sequence_id), set()).add(split)
        samples.append(
            AdomReleaseSample(
                sample_id=sample_id,
                condition=condition,
                sequence_id=sequence_id,
                frame=frame,
                image_relative=image_relative,
                mask_relative=mask_relative,
                image_path=image_path,
                mask_path=mask_path,
                split=split,
            )
        )

    leaking_sequences = {
        f"{condition}/{sequence_id}": sorted(splits)
        for (condition, sequence_id), splits in sequence_splits.items()
        if len(splits) > 1
    }
    if leaking_sequences:
        raise RuntimeError(
            "ADOM-v2 condition/sequence appears in multiple splits: "
            f"{leaking_sequences}"
        )

    return sorted(
        samples,
        key=lambda sample: (
            SPLITS.index(sample.split),
            sample.image_relative.as_posix().casefold(),
        ),
    )


def resolve_adom_pair_roots(adom_root: Path, split: str) -> tuple[Path, Path]:
    """
    Supported ADOM-v2 contracts:
      A) images/<split>/..., masks/<split>/...
      B) <split>/images/..., <split>/masks/...
    """
    candidates = [
        (adom_root / "images" / split, adom_root / "masks" / split),
        (adom_root / split / "images", adom_root / split / "masks"),
    ]
    for image_root, mask_root in candidates:
        if image_root.is_dir() and mask_root.is_dir():
            return image_root, mask_root
    raise FileNotFoundError(
        f"ADOM-v2 split '{split}' not found. Expected either "
        f"{adom_root}/images/{split} + masks/{split}, or "
        f"{adom_root}/{split}/images + masks."
    )


def collect_by_stem(root: Path, suffixes: set[str]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        key = path.stem
        if key in out:
            raise RuntimeError(
                f"Duplicate basename/stem '{key}' under {root}: "
                f"{out[key]} and {path}"
            )
        out[key] = path
    return out


def inspect_semantic20_mask(mask_path: Path, image_path: Path) -> tuple[float, Counter[int]]:
    with Image.open(image_path) as image:
        image_size = image.size

    with Image.open(mask_path) as image:
        image.load()
        mask_size = image.size
        mask = np.asarray(image)

    if mask.ndim != 2:
        raise RuntimeError(
            f"ADOM-v2 mask must be single-channel class-ID PNG: "
            f"{mask_path}, shape={mask.shape}"
        )

    if image_size != mask_size:
        raise RuntimeError(
            f"ADOM-v2 image/mask size mismatch: "
            f"{image_path}={image_size}, {mask_path}={mask_size}"
        )

    ids = {int(v) for v in np.unique(mask)}
    unexpected = sorted(ids - ALLOWED_IDS)
    if unexpected:
        raise RuntimeError(
            f"Unexpected ADOM-v2 class IDs in {mask_path}: {unexpected}. "
            "Expected only 0..18 and 255."
        )

    values, counts = np.unique(mask, return_counts=True)
    counter = Counter({int(v): int(c) for v, c in zip(values, counts)})
    ratio = float(np.count_nonzero(mask != 255) / mask.size)
    return ratio, counter


def write_adom_statistics(
    output_root: Path,
    pixel_counts: Counter[int],
) -> None:
    stats_path = output_root / "results" / "adom_v2_class_statistics.csv"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    total = sum(pixel_counts.values())
    with stats_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class_id", "pixel_count", "pixel_percent"])
        for class_id in sorted(pixel_counts, key=lambda x: (x == 255, x)):
            count = pixel_counts[class_id]
            writer.writerow(
                [class_id, count, (100.0 * count / total if total else 0.0)]
            )


def add_presplit_adom_v2(
    adom_root: Path,
    output_root: Path,
    eval_policy: str,
) -> dict[str, int]:
    require_dir(adom_root, "ADOM-v2 root")
    manifest_path = output_root / "manifest.csv"
    rows = read_manifest(manifest_path)
    existing_keys = {row["sample_key"] for row in rows}

    split_root = output_root / "splits"
    main_splits = {
        split: read_split(split_root / f"{split}.txt")
        for split in ("train", "val", "test")
    }

    adom_counts: dict[str, int] = {}
    pixel_counts: Counter[int] = Counter()
    diagnostic_splits: dict[str, list[str]] = {"val": [], "test": []}

    for split in ("train", "val", "test"):
        image_root, mask_root = resolve_adom_pair_roots(adom_root, split)
        images = collect_by_stem(image_root, {".png", ".jpg", ".jpeg"})
        masks = collect_by_stem(mask_root, {".png"})

        only_images = sorted(set(images) - set(masks))
        only_masks = sorted(set(masks) - set(images))
        if only_images or only_masks:
            raise RuntimeError(
                f"ADOM-v2 {split} image-mask mismatch: "
                f"images-only={only_images[:10]}, masks-only={only_masks[:10]}"
            )

        sample_ids = sorted(images)
        if not sample_ids:
            raise RuntimeError(f"No ADOM-v2 samples found in split '{split}'.")

        entries: list[str] = []
        for sample_id in sample_ids:
            image_src = images[sample_id]
            mask_src = masks[sample_id]
            ratio, counts = inspect_semantic20_mask(mask_src, image_src)
            pixel_counts.update(counts)

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
            hardlink_or_copy(image_src, image_dst)
            hardlink_or_copy(mask_src, mask_dst)

            sample_key = f"adom_v2/{split}/{sample_id}"
            if sample_key in existing_keys:
                raise RuntimeError(f"Duplicate sample key: {sample_key}")
            existing_keys.add(sample_key)
            entries.append(sample_key)

            rows.append(
                {
                    "sample_key": sample_key,
                    "source": "adom_v2",
                    "source_split": split,
                    "output_split": split,
                    "sample_id": sample_id,
                    "image_path": image_dst.relative_to(output_root).as_posix(),
                    "mask_path": mask_dst.relative_to(output_root).as_posix(),
                    "non_ignore_ratio": f"{ratio:.8f}",
                }
            )

        adom_counts[split] = len(entries)

        if split == "train":
            main_splits["train"].extend(entries)
        elif eval_policy == "mixed":
            main_splits[split].extend(entries)
        else:
            diagnostic_splits[split] = entries

    write_manifest(manifest_path, rows)
    for split in ("train", "val", "test"):
        write_split(split_root / f"{split}.txt", main_splits[split])

    if eval_policy == "diagnostic":
        write_split(
            split_root / "adom_v2_val_diagnostic.txt",
            diagnostic_splits["val"],
        )
        write_split(
            split_root / "adom_v2_test_diagnostic.txt",
            diagnostic_splits["test"],
        )

    write_adom_statistics(output_root, pixel_counts)

    return adom_counts


def add_adom_v2_release(
    samples: list[AdomReleaseSample],
    output_root: Path,
    eval_policy: str,
) -> dict[str, int]:
    manifest_path = output_root / "manifest.csv"
    rows = read_manifest(manifest_path)
    existing_keys = {row["sample_key"] for row in rows}

    split_root = output_root / "splits"
    main_splits = {
        split: read_split(split_root / f"{split}.txt")
        for split in SPLITS
    }
    entries_by_split: dict[str, list[str]] = {split: [] for split in SPLITS}
    pixel_counts: Counter[int] = Counter()

    for sample in samples:
        ratio, counts = inspect_semantic20_mask(
            sample.mask_path,
            sample.image_path,
        )
        pixel_counts.update(counts)

        image_subpath = Path(*sample.image_relative.parts[1:])
        mask_subpath = Path(*sample.mask_relative.parts[1:])
        image_dst = (
            output_root
            / "images"
            / "adom_v2"
            / sample.split
            / image_subpath
        )
        mask_dst = (
            output_root
            / "masks"
            / "adom_v2"
            / sample.split
            / mask_subpath
        )
        hardlink_or_copy(sample.image_path, image_dst)
        hardlink_or_copy(sample.mask_path, mask_dst)

        sample_key = f"adom_v2/{sample.split}/{sample.sample_id}"
        if sample_key in existing_keys:
            raise RuntimeError(f"Duplicate sample key: {sample_key}")
        existing_keys.add(sample_key)
        entries_by_split[sample.split].append(sample_key)
        rows.append(
            {
                "sample_key": sample_key,
                "source": "adom_v2",
                "source_split": sample.split,
                "output_split": sample.split,
                "sample_id": sample.sample_id,
                "image_path": image_dst.relative_to(output_root).as_posix(),
                "mask_path": mask_dst.relative_to(output_root).as_posix(),
                "non_ignore_ratio": f"{ratio:.8f}",
            }
        )

    main_splits["train"].extend(entries_by_split["train"])
    if eval_policy == "mixed":
        main_splits["val"].extend(entries_by_split["val"])
        main_splits["test"].extend(entries_by_split["test"])

    write_manifest(manifest_path, rows)
    for split in SPLITS:
        write_split(split_root / f"{split}.txt", main_splits[split])
    if eval_policy == "diagnostic":
        write_split(
            split_root / "adom_v2_val_diagnostic.txt",
            entries_by_split["val"],
        )
        write_split(
            split_root / "adom_v2_test_diagnostic.txt",
            entries_by_split["test"],
        )

    write_adom_statistics(output_root, pixel_counts)
    return {split: len(entries_by_split[split]) for split in SPLITS}


def add_adom_v2(
    adom_root: Path,
    output_root: Path,
    eval_policy: str,
    split_csv: Path | None,
) -> dict[str, int]:
    selection_candidates = [
        adom_root / "metadata" / "selection.csv",
        adom_root / "selection.csv",
    ]
    if any(path.is_file() for path in selection_candidates):
        samples = load_adom_release_samples(adom_root, split_csv)
        return add_adom_v2_release(samples, output_root, eval_policy)
    if split_csv is not None:
        raise RuntimeError(
            "--adom-split-csv is only valid for an ADOM-v2 release "
            "containing selection.csv."
        )
    return add_presplit_adom_v2(adom_root, output_root, eval_policy)


def validate_final_package(output_root: Path) -> dict:
    rows = read_manifest(output_root / "manifest.csv")
    row_by_key = {row["sample_key"]: row for row in rows}

    if len(row_by_key) != len(rows):
        raise RuntimeError("Duplicate sample_key in final manifest.")

    source_counts = Counter()
    observed_ids: dict[str, set[int]] = {}
    split_counts = {}

    for idx, row in enumerate(rows, start=1):
        source = row["source"]
        source_counts[source] += 1
        image_path = output_root / row["image_path"]
        mask_path = output_root / row["mask_path"]
        require_file(image_path, "manifest image")
        require_file(mask_path, "manifest mask")

        with Image.open(image_path) as image:
            image_size = image.size
        with Image.open(mask_path) as image:
            image.load()
            mask_size = image.size
            mask = np.asarray(image)

        if image_size != mask_size:
            raise RuntimeError(
                f"Image/mask size mismatch: {image_path} vs {mask_path}"
            )
        if mask.ndim != 2:
            raise RuntimeError(
                f"Mask must be single-channel: {mask_path}, shape={mask.shape}"
            )

        ids = {int(v) for v in np.unique(mask)}
        unexpected = sorted(ids - ALLOWED_IDS)
        if unexpected:
            raise RuntimeError(
                f"Unexpected target IDs {unexpected} in {mask_path}"
            )
        observed_ids.setdefault(source, set()).update(ids)

        ratio = float(np.count_nonzero(mask != 255) / mask.size)
        if abs(ratio - float(row["non_ignore_ratio"])) > 1e-6:
            raise RuntimeError(f"non_ignore_ratio mismatch: {mask_path}")

        if idx % 1000 == 0:
            print(f"Validated pairs: {idx}/{len(rows)}")

    split_sets: dict[str, set[str]] = {}
    for split in ("train", "val", "test"):
        entries = read_split(output_root / "splits" / f"{split}.txt")
        if len(entries) != len(set(entries)):
            raise RuntimeError(f"Duplicate entries in {split}.txt")
        unknown = sorted(set(entries) - set(row_by_key))
        if unknown:
            raise RuntimeError(
                f"Unknown manifest keys in {split}.txt: {unknown[:10]}"
            )
        split_sets[split] = set(entries)
        split_counts[split] = len(entries)

    if (
        split_sets["train"] & split_sets["val"]
        or split_sets["train"] & split_sets["test"]
        or split_sets["val"] & split_sets["test"]
    ):
        raise RuntimeError("Overlap detected among main train/val/test splits.")

    summary = {
        "status": "PASS",
        "manifest_count": len(rows),
        "source_counts": dict(source_counts),
        "main_split_counts": split_counts,
        "observed_ids": {
            source: sorted(ids) for source, ids in observed_ids.items()
        },
    }

    summary_path = output_root / "results" / "final_unified_check.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def main() -> int:
    args = parse_args()

    repo_root = args.repo_root.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    adom_split_csv = (
        args.adom_split_csv.expanduser().resolve()
        if args.adom_split_csv is not None
        else None
    )

    require_dir(repo_root, "ADOM repository root")
    require_dir(data_root, "data root")

    datasets_repo = repo_root / "src" / "data"
    rellis_workflow = datasets_repo / "rellis"
    combined_workflow = datasets_repo / "semantic_20"
    rugd_repo = datasets_repo / "rugd"
    ycor_repo = datasets_repo / "ycor"

    # Existing scripts/configs: these are the SSOT for legacy mappings.
    rellis_convert = rellis_workflow / "scripts" / "02_convert_masks.py"
    rellis_mapping = rellis_workflow / "config" / "class_mapping.yaml"
    rellis_splits = rellis_workflow / "splits"

    bridge_convert = combined_workflow / "scripts" / "01_convert_bridge_sources.py"
    build_combined = combined_workflow / "scripts" / "03_build_combined_package.py"
    legacy_validate = combined_workflow / "scripts" / "04_validate_combined_package.py"
    bridge_mapping = combined_workflow / "config" / "bridge_mapping.yaml"

    rugd_splits = rugd_repo / "splits"
    ycor_source_mapping = ycor_repo / "config" / "label_mapping.json"

    for path, label in [
        (rellis_convert, "RELLIS converter"),
        (rellis_mapping, "RELLIS class mapping"),
        (bridge_convert, "bridge converter"),
        (build_combined, "combined package builder"),
        (legacy_validate, "legacy package validator"),
        (bridge_mapping, "bridge mapping"),
        (ycor_source_mapping, "YCOR source palette mapping"),
    ]:
        require_file(path, label)

    raw_root = data_root / "raw"
    rellis_raw = raw_root / "RELLIS-3D"
    rugd_raw = raw_root / "RUGD"
    ycor_raw = raw_root / "YCOR"

    adom_root = args.adom_v2_root
    if adom_root is None:
        candidate = raw_root / "ADOM-v2"
        if candidate.exists():
            adom_root = candidate
    if adom_root is not None:
        adom_root = adom_root.expanduser().resolve()

    if args.skip_adom_v2 and adom_split_csv is not None:
        raise RuntimeError(
            "--adom-split-csv cannot be combined with --skip-adom-v2."
        )
    if adom_split_csv is not None and adom_root is None:
        raise RuntimeError(
            "--adom-split-csv was provided but no ADOM-v2 root was found."
        )
    if not args.skip_adom_v2 and adom_root is not None:
        require_dir(adom_root, "ADOM-v2 root")
        selection_candidates = [
            adom_root / "metadata" / "selection.csv",
            adom_root / "selection.csv",
        ]
        if any(path.is_file() for path in selection_candidates):
            # Fail before processing the legacy sources when release metadata
            # or the explicit split assignment is incomplete or inconsistent.
            load_adom_release_samples(adom_root, adom_split_csv)

    require_dir(rellis_raw, "RELLIS-3D raw dataset")
    require_dir(rugd_raw, "RUGD raw dataset")
    require_dir(ycor_raw, "YCOR raw dataset")

    work_root = data_root / "_work"
    rellis_processed = work_root / "rellis3d_semantic20"
    rugd_staging = work_root / "rugd_staging"

    if args.overwrite:
        if rellis_processed.exists():
            shutil.rmtree(rellis_processed)
        if rugd_staging.exists():
            shutil.rmtree(rugd_staging)
        if output_root.exists():
            shutil.rmtree(output_root)

    work_root.mkdir(parents=True, exist_ok=True)

    print("\n=== 1/6 RELLIS-3D -> Semantic20 ===")
    run(
        [
            sys.executable,
            str(rellis_convert),
            "--input-root",
            str(rellis_raw),
            "--output-root",
            str(rellis_processed),
            "--mapping",
            str(rellis_mapping),
            "--overwrite",
        ]
    )
    copy_rellis_splits(rellis_splits, rellis_processed)

    print("\n=== 2/6 Stage RUGD raw image/indexLabel pairs ===")
    rugd_image_root, rugd_mask_root = stage_rugd(
        rugd_raw,
        rugd_staging,
        rugd_splits,
    )

    print("\n=== 3/6 RUGD + YCOR -> Semantic20 bridge ===")
    run(
        [
            sys.executable,
            str(bridge_convert),
            "--mapping",
            str(bridge_mapping),
            "--rugd-image-root",
            str(rugd_image_root),
            "--rugd-mask-root",
            str(rugd_mask_root),
            "--rugd-split-root",
            str(rugd_splits),
            "--ycor-root",
            str(ycor_raw),
            "--ycor-source-map",
            str(ycor_source_mapping),
            "--output-root",
            str(output_root),
            "--min-non-ignore-ratio",
            str(args.min_non_ignore_ratio),
            "--overwrite",
        ]
    )

    print("\n=== 4/6 Add RELLIS-3D and build legacy unified package ===")
    run(
        [
            sys.executable,
            str(build_combined),
            "--rellis-root",
            str(rellis_processed),
            "--output-root",
            str(output_root),
            "--overwrite-rellis",
        ]
    )

    print("\n=== 5/6 Validate legacy RELLIS + RUGD + YCOR package ===")
    run(
        [
            sys.executable,
            str(legacy_validate),
            "--output-root",
            str(output_root),
        ]
    )

    adom_counts = None
    if not args.skip_adom_v2 and adom_root is not None:
        print("\n=== 6/6 Add ADOM-v2 ===")
        adom_counts = add_adom_v2(
            adom_root,
            output_root,
            args.adom_eval_policy,
            adom_split_csv,
        )
    else:
        print("\n=== 6/6 ADOM-v2 skipped/not present ===")

    final_summary = validate_final_package(output_root)

    build_info = {
        "evaluation_policy": args.adom_eval_policy,
        "adom_v2_counts": adom_counts,
        "final": final_summary,
    }
    (output_root / "results" / "build_info.json").write_text(
        json.dumps(build_info, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n[PASS] Unified dataset build completed.")
    print(json.dumps(build_info, indent=2, ensure_ascii=False))
    print(f"\nOutput: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
