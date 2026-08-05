from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import shutil
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
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
            "Materialize visible RGB and original 64-class label-ID masks from "
            "the GOOSE train/val ZIP archives. No remapping or filtering is done."
        )
    )
    parser.add_argument(
        "--archive-root",
        type=Path,
        required=True,
        help="Directory containing goose_2d_train.zip and goose_2d_val.zip.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--classes-config",
        type=Path,
        default=tool_root / "config" / "goose64_classes.csv",
    )
    parser.add_argument(
        "--archive-checksums",
        type=Path,
        default=tool_root / "config" / "archive_checksums.csv",
    )
    parser.add_argument(
        "--skip-archive-verification",
        action="store_true",
        help="Skip known size/SHA-256 verification. Intended only for fixtures.",
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


def require_empty_or_absent(path: Path) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(
            f"Output root must be empty or absent; existing data is never overwritten: {path}"
        )


def find_archive(root: Path, split: str) -> Path:
    expected = f"goose_2d_{split}.zip"
    matches = sorted(path for path in root.rglob(expected) if path.is_file())
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one {expected} below {root}; found {len(matches)}"
        )
    return matches[0]


def load_archive_checksums(path: Path) -> dict[str, tuple[int, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Archive checksum config not found: {path}")
    expected: dict[str, tuple[int, str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required = {"filename", "size_bytes", "sha256"}
        if not required <= set(reader.fieldnames or []):
            raise ValueError(f"Archive checksum config requires {sorted(required)}")
        for row in reader:
            filename = row["filename"].strip()
            digest = row["sha256"].strip().lower()
            if filename in expected or len(digest) != 64:
                raise ValueError(f"Invalid archive checksum entry: {filename}")
            expected[filename] = (int(row["size_bytes"]), digest)
    return expected


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify_archives(
    archives: dict[str, Path],
    checksum_config: Path,
) -> None:
    expected = load_archive_checksums(checksum_config)
    for split, archive in archives.items():
        if archive.name not in expected:
            raise KeyError(f"No checksum entry for {archive.name}")
        expected_size, expected_sha256 = expected[archive.name]
        actual_size = archive.stat().st_size
        if actual_size != expected_size:
            raise ValueError(
                f"Archive size mismatch for {split}: "
                f"expected={expected_size}, actual={actual_size}"
            )
        print(f"[{split}] calculating SHA-256: {archive}")
        actual_sha256 = sha256_file(archive)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"Archive SHA-256 mismatch for {split}: "
                f"expected={expected_sha256}, actual={actual_sha256}"
            )
        print(f"[{split}] archive size/SHA-256 verified")


def member_sample_key(member_name: str, split: str, is_label: bool) -> str:
    path = PurePosixPath(member_name)
    expected_root = "labels" if is_label else "images"
    expected_suffix = LABEL_SUFFIX if is_label else RGB_SUFFIX
    if (
        path.is_absolute()
        or ".." in path.parts
        or len(path.parts) < 4
        or path.parts[0] != expected_root
        or path.parts[1] != split
        or not path.stem.endswith(expected_suffix)
    ):
        raise ValueError(f"Unexpected GOOSE archive member: {member_name}")
    stem = path.stem[: -len(expected_suffix)].rstrip("_")
    parent = PurePosixPath(*path.parts[2:-1]).as_posix()
    return f"{parent}/{stem}"


def index_archive_pairs(
    zip_file: zipfile.ZipFile,
    split: str,
) -> list[tuple[str, zipfile.ZipInfo, zipfile.ZipInfo]]:
    images: dict[str, zipfile.ZipInfo] = {}
    labels: dict[str, zipfile.ZipInfo] = {}
    for info in zip_file.infolist():
        if info.is_dir():
            continue
        path = PurePosixPath(info.filename)
        if path.is_absolute() or ".." in path.parts or len(path.parts) < 4:
            continue
        if (
            path.parts[:2] == ("images", split)
            and path.suffix.lower() in IMAGE_SUFFIXES
            and path.stem.endswith(RGB_SUFFIX)
        ):
            key = member_sample_key(info.filename, split, False)
            if key in images:
                raise ValueError(f"Duplicate visible RGB key: {key}")
            images[key] = info
        elif (
            path.parts[:2] == ("labels", split)
            and info.filename.lower().endswith(f"{LABEL_SUFFIX}.png")
        ):
            key = member_sample_key(info.filename, split, True)
            if key in labels:
                raise ValueError(f"Duplicate label-ID key: {key}")
            labels[key] = info
    if set(images) != set(labels):
        raise ValueError(
            f"Pair mismatch in {split}: "
            f"missing_masks={sorted(set(images) - set(labels))[:20]}, "
            f"missing_images={sorted(set(labels) - set(images))[:20]}"
        )
    if not images:
        raise ValueError(f"No visible RGB/label-ID pairs found for split: {split}")
    return [(key, images[key], labels[key]) for key in sorted(images)]


def load_mask(data: bytes, description: str) -> np.ndarray:
    with Image.open(io.BytesIO(data)) as image:
        image.load()
        array = np.asarray(image)
    if array.ndim != 2 or not np.issubdtype(array.dtype, np.integer):
        raise ValueError(
            f"Expected indexed 2D mask, got {array.shape}/{array.dtype}: {description}"
        )
    if array.size and int(array.max()) > 255:
        raise ValueError(f"Mask contains an ID above 255: {description}")
    return array.astype(np.uint8, copy=False)


def write_bytes_new(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as file:
        file.write(data)


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
    archive_root = args.archive_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    classes_config = args.classes_config.expanduser().resolve()
    archive_checksums = args.archive_checksums.expanduser().resolve()
    classes = load_classes(classes_config)
    require_empty_or_absent(output_root)

    archives = {split: find_archive(archive_root, split) for split in args.splits}
    if not args.skip_archive_verification:
        verify_archives(archives, archive_checksums)
    indexed: dict[str, list[tuple[str, zipfile.ZipInfo, zipfile.ZipInfo]]] = {}
    for split, archive in archives.items():
        with zipfile.ZipFile(archive) as zip_file:
            indexed[split] = index_archive_pairs(zip_file, split)

    output_root.mkdir(parents=True, exist_ok=True)
    metadata_root = output_root / "metadata"
    metadata_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(classes_config, metadata_root / "goose64_classes.csv")

    pair_rows: list[dict[str, Any]] = []
    distribution_rows: list[dict[str, Any]] = []
    class_frames: Counter[int] = Counter()
    class_pixels: Counter[int] = Counter()
    split_summary: dict[str, dict[str, int]] = {}

    for split, archive in archives.items():
        pairs = indexed[split]
        with zipfile.ZipFile(archive) as zip_file:
            for index, (key, image_info, label_info) in enumerate(pairs, start=1):
                # A full read validates each selected member's ZIP CRC.
                image_bytes = zip_file.read(image_info)
                label_bytes = zip_file.read(label_info)
                with Image.open(io.BytesIO(image_bytes)) as image:
                    image.load()
                    image_size = image.size
                mask = load_mask(label_bytes, f"{archive.name}!{label_info.filename}")
                if image_size != (mask.shape[1], mask.shape[0]):
                    raise ValueError(f"RGB/mask size mismatch for {split}/{key}")
                values, counts = np.unique(mask, return_counts=True)
                unknown = sorted(int(value) for value in values if int(value) not in classes)
                if unknown:
                    raise ValueError(f"Unknown GOOSE IDs {unknown}: {split}/{key}")

                image_member = PurePosixPath(image_info.filename)
                label_member = PurePosixPath(label_info.filename)
                output_image = output_root.joinpath(*image_member.parts)
                output_label = output_root.joinpath(*label_member.parts)
                write_bytes_new(output_image, image_bytes)
                write_bytes_new(output_label, label_bytes)

                total_pixels = int(mask.size)
                counts_by_id = {
                    int(value): int(count) for value, count in zip(values, counts)
                }
                present_ids = sorted(counts_by_id)
                for raw_id in present_ids:
                    class_frames[raw_id] += 1
                    class_pixels[raw_id] += counts_by_id[raw_id]
                pair_rows.append(
                    {
                        "split": split,
                        "sample_key": key,
                        "source_archive": archive.name,
                        "source_image_member": image_info.filename,
                        "source_label_member": label_info.filename,
                        "output_image": output_image.relative_to(output_root).as_posix(),
                        "output_label": output_label.relative_to(output_root).as_posix(),
                        "width": image_size[0],
                        "height": image_size[1],
                        "image_size_bytes": len(image_bytes),
                        "label_size_bytes": len(label_bytes),
                        "image_crc32": f"{image_info.CRC:08x}",
                        "label_crc32": f"{label_info.CRC:08x}",
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
        "archive_verification": (
            "SKIPPED" if args.skip_archive_verification else "PASS"
        ),
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
