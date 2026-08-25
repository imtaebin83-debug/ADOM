from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable

import numpy as np
from PIL import Image

from _common import (
    IGNORE_INDEX,
    ManifestRecord,
    SEMANTIC20_CLASSES,
    load_mask,
    manifest_sha256,
    sha256_file,
    write_json,
    write_manifest,
)


EXPECTED_COUNTS = {
    "rellis_test": 899,
    "korean_train": 133,
    "korean_val": 21,
    "korean_test": 61,
}
ADOM_SOURCE = "adom_zed2i"


def _read_split(root: Path, relative: str) -> list[str]:
    path = root / relative
    if not path.is_file():
        raise FileNotFoundError(f"Split file is missing: {path}")
    values = [
        line.strip()
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    if not values:
        raise ValueError(f"Split is empty: {path}")
    if len(values) != len(set(values)):
        raise ValueError(f"Split contains duplicate sample keys: {path}")
    return values


def _read_package_manifest(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    output: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or ())
        key_field = "sample_key" if "sample_key" in fields else "sample_id"
        image_field = "image_path" if "image_path" in fields else "image_relpath"
        mask_field = "mask_path" if "mask_path" in fields else "mask_relpath"
        missing = {
            name
            for name, present in (
                ("sample_key/sample_id", key_field in fields),
                ("image_path/image_relpath", image_field in fields),
                ("mask_path/mask_relpath", mask_field in fields),
            )
            if not present
        }
        if missing:
            raise ValueError(f"Package manifest {path} is missing: {sorted(missing)}")
        for row in reader:
            key = row[key_field]
            if key in output:
                raise ValueError(f"Duplicate package manifest key: {key}")
            output[key] = {**row, "image_path": row[image_field], "mask_path": row[mask_field]}
    return output


def _load_sequence_contract(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    output: dict[str, str] = {}
    for split, values in payload.get("splits", {}).items():
        for value in values:
            if value in output:
                raise ValueError(f"Sequence appears twice in contract: {value}")
            output[value] = split
    return output


def _infer_sequence(
    sample_key: str,
    row: dict[str, str],
    sequence_contract: dict[str, str],
) -> str:
    source_date = row.get("source_date", "").strip()
    source_sequence = row.get("source_sequence", "").strip()
    if source_sequence:
        return f"{source_date}/{source_sequence}" if source_date else source_sequence
    for field in ("sequence", "sequence_id", "capture_sequence"):
        if row.get(field, "").strip():
            return row[field].strip()
    haystack = "|".join(
        (sample_key, row.get("image_path", ""), row.get("mask_path", ""))
    ).replace("\\", "/")
    matches = [sequence for sequence in sequence_contract if sequence in haystack]
    if len(matches) == 1:
        return matches[0]
    normalized = sample_key.replace("\\", "/").strip("/")
    parts = normalized.split("/")
    if parts and parts[0] == ADOM_SOURCE:
        parts = parts[1:]
    if len(parts) >= 3:
        return "/".join(parts[:2])
    if len(parts) >= 2:
        return parts[0]
    return "unknown"


def _paths_for_key(
    root: Path,
    key: str,
    row: dict[str, str] | None,
) -> tuple[Path, Path, str]:
    if row is None:
        return root / "images" / f"{key}.jpg", root / "masks" / f"{key}.png", "rellis3d"
    source = row.get("source", "") or key.split("/", 1)[0]
    return root / row["image_path"], root / row["mask_path"], source


def _record(
    *,
    dataset: str,
    split: str,
    root: Path,
    key: str,
    row: dict[str, str] | None,
    sequence_contract: dict[str, str],
) -> tuple[ManifestRecord, list[str]]:
    image_path, annotation_path, source = _paths_for_key(root, key, row)
    issues: list[str] = []
    width = height = 0
    image_digest = annotation_digest = ""
    if image_path.is_file():
        image_digest = sha256_file(image_path)
        try:
            with Image.open(image_path) as image:
                width, height = image.size
        except Exception as error:  # pragma: no cover - exercised on damaged input
            issues.append(f"unreadable image {key}: {error}")
    else:
        issues.append(f"missing image {key}: {image_path}")
    if annotation_path.is_file():
        annotation_digest = sha256_file(annotation_path)
        try:
            with Image.open(annotation_path) as mask:
                if mask.size != (width, height) and width and height:
                    issues.append(
                        f"image/mask size mismatch {key}: {(width, height)} != {mask.size}"
                    )
        except Exception as error:  # pragma: no cover - exercised on damaged input
            issues.append(f"unreadable annotation {key}: {error}")
    else:
        issues.append(f"missing annotation {key}: {annotation_path}")
    metadata = row or {}
    return (
        ManifestRecord(
            dataset=dataset,
            split=split,
            sample_id=key,
            sequence=_infer_sequence(key, metadata, sequence_contract),
            source=source,
            image_path=image_path.resolve(),
            annotation_path=annotation_path.resolve(),
            image_sha256=image_digest,
            annotation_sha256=annotation_digest,
            width=width,
            height=height,
        ),
        issues,
    )


def _build_records(
    *,
    dataset: str,
    split: str,
    root: Path,
    split_file: str,
    package_rows: dict[str, dict[str, str]],
    sequence_contract: dict[str, str],
    source_filter: str | None = None,
) -> tuple[list[ManifestRecord], list[str]]:
    keys = _read_split(root, split_file)
    output: list[ManifestRecord] = []
    issues: list[str] = []
    for key in keys:
        row = package_rows.get(key) if package_rows else None
        if package_rows and row is None:
            issues.append(f"split key absent from package manifest: {key}")
            continue
        source = (row or {}).get("source", "") or key.split("/", 1)[0]
        if source_filter is not None and source != source_filter:
            continue
        record, record_issues = _record(
            dataset=dataset,
            split=split,
            root=root,
            key=key,
            row=row,
            sequence_contract=sequence_contract,
        )
        output.append(record)
        issues.extend(record_issues)
    return output, issues


def _duplicates(values: Iterable[str]) -> list[str]:
    counts = Counter(value for value in values if value)
    return sorted(value for value, count in counts.items() if count > 1)


def _gt_support(records: list[ManifestRecord]) -> dict[str, Any]:
    pixels = np.zeros(len(SEMANTIC20_CLASSES), dtype=np.int64)
    image_counts = np.zeros(len(SEMANTIC20_CLASSES), dtype=np.int64)
    ignored = 0
    invalid: dict[str, list[int]] = {}
    for row in records:
        if not row.annotation_path.is_file():
            continue
        mask = load_mask(row.annotation_path)
        valid_values = ((mask >= 0) & (mask < len(SEMANTIC20_CLASSES))) | (
            mask == IGNORE_INDEX
        )
        if not np.all(valid_values):
            invalid[row.sample_id] = [
                int(value) for value in sorted(np.unique(mask[~valid_values]))
            ]
            continue
        ignored += int(np.count_nonzero(mask == IGNORE_INDEX))
        valid = mask != IGNORE_INDEX
        counts = np.bincount(mask[valid], minlength=len(SEMANTIC20_CLASSES))
        pixels += counts
        image_counts += counts > 0
    return {
        "ignore_index": IGNORE_INDEX,
        "ignored_pixels": ignored,
        "class_mapping": [
            {"id": index, "name": name}
            for index, name in enumerate(SEMANTIC20_CLASSES)
        ],
        "classes": [
            {
                "id": index,
                "name": name,
                "gt_pixels": int(pixels[index]),
                "gt_images": int(image_counts[index]),
            }
            for index, name in enumerate(SEMANTIC20_CLASSES)
        ],
        "zero_gt_classes": [
            name
            for index, name in enumerate(SEMANTIC20_CLASSES)
            if pixels[index] == 0
        ],
        "invalid_label_ids": invalid,
    }


def _split_summary(records: list[ManifestRecord], issues: list[str]) -> dict[str, Any]:
    return {
        "count": len(records),
        "image_count": sum(row.image_path.is_file() for row in records),
        "annotation_count": sum(row.annotation_path.is_file() for row in records),
        "manifest_sha256": manifest_sha256(records),
        "issues": issues,
        "duplicate_sample_ids": _duplicates(row.sample_id for row in records),
        "duplicate_image_paths": _duplicates(str(row.image_path) for row in records),
        "duplicate_annotation_paths": _duplicates(
            str(row.annotation_path) for row in records
        ),
        "duplicate_image_basenames": _duplicates(
            row.image_path.name for row in records
        ),
        "duplicate_annotation_basenames": _duplicates(
            row.annotation_path.name for row in records
        ),
        "duplicate_image_hashes": _duplicates(row.image_sha256 for row in records),
        "duplicate_annotation_hashes": _duplicates(
            row.annotation_sha256 for row in records
        ),
        "sequence_counts": dict(sorted(Counter(row.sequence for row in records).items())),
        "gt_support": _gt_support(records),
    }


def _overlap(left: list[ManifestRecord], right: list[ManifestRecord]) -> dict[str, Any]:
    def values(rows: list[ManifestRecord], field: str) -> set[str]:
        return {str(getattr(row, field)) for row in rows if getattr(row, field)}

    left_by_sequence: defaultdict[str, list[ManifestRecord]] = defaultdict(list)
    right_by_sequence: defaultdict[str, list[ManifestRecord]] = defaultdict(list)
    for row in left:
        left_by_sequence[row.sequence].append(row)
    for row in right:
        right_by_sequence[row.sequence].append(row)
    sequence_overlap = sorted(
        (set(left_by_sequence) & set(right_by_sequence)) - {"unknown"}
    )

    adjacent: list[dict[str, Any]] = []
    frame_pattern = re.compile(r"^(.*?)(\d+)$")
    for sequence in sequence_overlap:
        left_frames: list[tuple[int, str, str]] = []
        right_frames: list[tuple[int, str, str]] = []
        for destination, rows in (
            (left_frames, left_by_sequence[sequence]),
            (right_frames, right_by_sequence[sequence]),
        ):
            for row in rows:
                match = frame_pattern.match(Path(row.sample_id).stem)
                if match:
                    destination.append((int(match.group(2)), match.group(1), row.sample_id))
        for left_index, left_prefix, left_id in left_frames:
            for right_index, right_prefix, right_id in right_frames:
                if left_prefix == right_prefix and abs(left_index - right_index) <= 1:
                    adjacent.append(
                        {
                            "sequence": sequence,
                            "left": left_id,
                            "right": right_id,
                            "frame_gap": abs(left_index - right_index),
                        }
                    )
    return {
        "sample_id": sorted(values(left, "sample_id") & values(right, "sample_id")),
        "image_basename": sorted(
            {row.image_path.name for row in left}
            & {row.image_path.name for row in right}
        ),
        "annotation_basename": sorted(
            {row.annotation_path.name for row in left}
            & {row.annotation_path.name for row in right}
        ),
        "image_sha256": sorted(
            values(left, "image_sha256") & values(right, "image_sha256")
        ),
        "annotation_sha256": sorted(
            values(left, "annotation_sha256") & values(right, "annotation_sha256")
        ),
        "sequence": sequence_overlap,
        "adjacent_frames_gap_le_1": adjacent,
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.resolve()
    manifest_dir = output_dir / "manifests"
    if output_dir.exists() and any(manifest_dir.glob("*.csv")):
        raise FileExistsError(
            f"Refusing to overwrite existing paper-eval manifests: {manifest_dir}"
        )
    rellis_root = args.rellis_root.resolve()
    korean_root = args.korean_root.resolve()
    sequence_contract = _load_sequence_contract(args.sequence_contract)
    rellis_rows = _read_package_manifest(rellis_root / args.rellis_manifest)
    korean_rows = _read_package_manifest(korean_root / args.korean_manifest)
    if not korean_rows:
        raise FileNotFoundError(
            f"Korean target-adaptation manifest is required: {korean_root / args.korean_manifest}"
        )

    rellis_test, rellis_issues = _build_records(
        dataset="rellis",
        split="test",
        root=rellis_root,
        split_file=args.rellis_test_split,
        package_rows=rellis_rows,
        sequence_contract=sequence_contract,
    )
    korean_train, train_issues = _build_records(
        dataset="korean",
        split="train",
        root=korean_root,
        split_file=args.korean_train_split,
        package_rows=korean_rows,
        sequence_contract=sequence_contract,
        source_filter=args.korean_source,
    )
    korean_val, val_issues = _build_records(
        dataset="korean",
        split="val",
        root=korean_root,
        split_file=args.korean_val_split,
        package_rows=korean_rows,
        sequence_contract=sequence_contract,
        source_filter=args.korean_source,
    )
    korean_test, test_issues = _build_records(
        dataset="korean",
        split="test",
        root=korean_root,
        split_file=args.korean_test_split,
        package_rows=korean_rows,
        sequence_contract=sequence_contract,
        source_filter=args.korean_source,
    )
    named = {
        "rellis_test": (rellis_test, rellis_issues, "rellis_test_manifest.csv"),
        "korean_train": (korean_train, train_issues, "korean_train_manifest.csv"),
        "korean_val": (korean_val, val_issues, "korean_val_manifest.csv"),
        "korean_test": (korean_test, test_issues, "korean_test_manifest.csv"),
    }
    summaries: dict[str, Any] = {}
    blockers: list[str] = []
    for name, (records, issues, filename) in named.items():
        write_manifest(manifest_dir / filename, records)
        summary = _split_summary(records, issues)
        summary["manifest_csv"] = str((manifest_dir / filename).resolve())
        summary["manifest_csv_sha256"] = sha256_file(manifest_dir / filename)
        summaries[name] = summary
        if issues:
            blockers.append(f"{name} has {len(issues)} pair/format issues")
        expected = EXPECTED_COUNTS[name]
        if len(records) != expected and not args.allow_count_mismatch:
            blockers.append(f"{name} count {len(records)} != frozen contract {expected}")
        if summary["gt_support"]["invalid_label_ids"]:
            blockers.append(f"{name} contains invalid Semantic20 label IDs")

    overlaps = {
        "korean_train_vs_val": _overlap(korean_train, korean_val),
        "korean_train_vs_test": _overlap(korean_train, korean_test),
        "korean_val_vs_test": _overlap(korean_val, korean_test),
    }
    warnings: list[str] = []
    for name, value in overlaps.items():
        for field in ("image_basename", "annotation_basename"):
            if value[field]:
                warnings.append(
                    f"{name} reuses {len(value[field])} {field} values across sequences"
                )
        if name == "korean_train_vs_val":
            for field in (
                "image_sha256",
                "annotation_sha256",
                "sequence",
                "adjacent_frames_gap_le_1",
            ):
                if value[field]:
                    warnings.append(
                        f"{name} has {len(value[field])} overlaps in {field}; "
                        "held-out test remains separate but train/val quality is limited"
                    )
        for field in ("sample_id",):
            if value[field]:
                blockers.append(f"{name} duplicate identity detected in {field}")
        if name.endswith("_vs_test"):
            for field in (
                "image_sha256",
                "sequence",
                "adjacent_frames_gap_le_1",
            ):
                if value[field]:
                    blockers.append(f"{name} held-out leakage detected in {field}")

    rellis_supported = {
        row["name"]
        for row in summaries["rellis_test"]["gt_support"]["classes"]
        if row["gt_pixels"] > 0
    }
    korean_supported = {
        row["name"]
        for row in summaries["korean_test"]["gt_support"]["classes"]
        if row["gt_pixels"] > 0
    }
    common_classes = [
        name
        for name in SEMANTIC20_CLASSES
        if name in rellis_supported and name in korean_supported
    ]
    if not common_classes:
        blockers.append("RELLIS and Korean tests have no common GT-supported classes")

    payload = {
        "schema_version": "adom-paper-eval-dataset-audit-v1",
        "status": "PASS" if not blockers else "BLOCKED",
        "roots": {
            "rellis": str(rellis_root),
            "korean_target_adaptation": str(korean_root),
        },
        "split_files": {
            "rellis_test": str((rellis_root / args.rellis_test_split).resolve()),
            "korean_train": str((korean_root / args.korean_train_split).resolve()),
            "korean_val": str((korean_root / args.korean_val_split).resolve()),
            "korean_test": str((korean_root / args.korean_test_split).resolve()),
        },
        "class_mapping": [
            {"id": index, "name": name}
            for index, name in enumerate(SEMANTIC20_CLASSES)
        ],
        "ignore_index": IGNORE_INDEX,
        "splits": summaries,
        "overlaps": overlaps,
        "common_supported_classes": common_classes,
        "warnings": warnings,
        "blockers": blockers,
    }
    write_json(output_dir / "dataset_manifest_summary.json", payload)
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build frozen RELLIS/Korean paper-evaluation manifests and leakage audit"
    )
    parser.add_argument("--rellis-root", required=True, type=Path)
    parser.add_argument("--korean-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--rellis-manifest", default="manifest.csv")
    parser.add_argument("--korean-manifest", default="manifest.csv")
    parser.add_argument("--rellis-test-split", default="splits/test.txt")
    parser.add_argument("--korean-train-split", default="splits/ta1_train.txt")
    parser.add_argument("--korean-val-split", default="splits/adom_val_diagnostic.txt")
    parser.add_argument("--korean-test-split", default="splits/adom_test_diagnostic.txt")
    parser.add_argument("--korean-source", default=ADOM_SOURCE)
    parser.add_argument(
        "--sequence-contract",
        type=Path,
        default=Path("src/data/adom_data/config/split_sequences.json"),
    )
    parser.add_argument(
        "--allow-count-mismatch",
        action="store_true",
        help="Record, but do not block on the frozen 899/133/21/61 sample counts.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    result = build(parse_args(argv))
    print(json.dumps({"status": result["status"], "blockers": result["blockers"]}, indent=2))
    if result["status"] != "PASS":
        sys.exit(2)


if __name__ == "__main__":
    main()
