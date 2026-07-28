from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from .io import is_portable_relative, sha256_file, verify_checksum_manifest
from .models import ValidationReport
from .schema import ALLOWED_TARGET_IDS, COST4_CLASSES, LabelSchema


MANIFEST_FIELDS = (
    "sample_id",
    "split",
    "sequence",
    "image_relpath",
    "mask_relpath",
    "width",
    "height",
    "source_image_relpath",
    "source_mask_relpath",
)
CLASS_STATISTICS_FIELDS = (
    "split",
    "class_id",
    "class_name",
    "pixel_count",
    "ratio_all_pixels",
    "ratio_valid_pixels",
    "image_count",
)
QC_FIELDS = ("severity", "code", "sample_id", "detail")


def validate_package(root: Path, verify_checksums: bool = True) -> ValidationReport:
    report = ValidationReport(dataset="rellis3d")
    metadata_path = root / "metadata" / "dataset.json"
    metadata: dict[str, object] = {}
    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            report.dataset = str(metadata.get("dataset", report.dataset))
            report.version = str(metadata.get("version", ""))
        except (json.JSONDecodeError, OSError) as error:
            report.error("invalid_dataset_metadata", str(error))
    else:
        report.error("missing_dataset_metadata", str(metadata_path))
    expected_metadata = {
        "num_classes": 4,
        "ignore_index": 255,
        "reduce_zero_label": False,
    }
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            report.error(
                "dataset_metadata_contract",
                f"{key}={metadata.get(key)!r}, expected={expected!r}",
            )

    mapping_path = root / "config" / "label_mapping.yaml"
    if not mapping_path.is_file():
        report.error("missing_label_mapping", str(mapping_path))
    else:
        try:
            schema = LabelSchema.from_path(mapping_path)
            if schema.target_classes != COST4_CLASSES:
                report.error("label_mapping_contract", str(mapping_path))
        except Exception as error:
            report.error("invalid_label_mapping", str(error))

    validation_path = root / "reports" / "validation.json"
    if not validation_path.is_file():
        report.error("missing_validation_report", str(validation_path))
    else:
        try:
            validation_value = json.loads(validation_path.read_text(encoding="utf-8"))
            required = {"dataset", "status", "error_count", "statistics"}
            missing = required - set(validation_value)
            if missing:
                report.error(
                    "validation_report_schema",
                    f"Missing fields: {sorted(missing)}",
                )
            if not validation_value.get("statistics"):
                report.error("empty_validation_statistics", str(validation_path))
        except (json.JSONDecodeError, OSError) as error:
            report.error("invalid_validation_report", str(error))

    for report_name, required_fields, require_rows in (
        ("class_statistics.csv", CLASS_STATISTICS_FIELDS, True),
        ("qc_errors.csv", QC_FIELDS, False),
    ):
        report_path = root / "reports" / report_name
        if not report_path.is_file():
            report.error("missing_qc_file", str(report_path))
            continue
        with report_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            missing = set(required_fields) - set(reader.fieldnames or ())
            rows = list(reader)
        if missing:
            report.error(
                "qc_schema",
                f"{report_name} missing fields: {sorted(missing)}",
            )
        if require_rows and not rows:
            report.error("empty_qc_file", str(report_path))
        if report_name == "class_statistics.csv" and rows:
            observed = {
                (row.get("split"), row.get("class_id"))
                for row in rows
            }
            expected = {
                (split, str(class_id))
                for split in ("train", "val", "test")
                for class_id in (0, 1, 2, 3, 255)
            }
            if observed != expected:
                report.error(
                    "class_statistics_coverage",
                    "observed="
                    f"{sorted(observed, key=lambda item: (str(item[0]), str(item[1])))}, "
                    f"expected={sorted(expected)}",
                )

    seen: dict[str, str] = {}
    split_counts: dict[str, int] = {}
    class_pixels: Counter[int] = Counter()
    for split in ("train", "val", "test"):
        manifest_path = root / "metadata" / f"manifest_{split}.csv"
        split_path = root / "splits" / f"{split}.txt"
        if not manifest_path.is_file():
            report.error("missing_manifest", str(manifest_path), split)
            continue
        if not split_path.is_file():
            report.error("missing_split_file", str(split_path), split)
            continue
        expected_split_hashes = metadata.get("official_split_sha256", {})
        if not isinstance(expected_split_hashes, dict):
            report.error(
                "dataset_metadata_schema",
                "official_split_sha256 must be a mapping",
            )
        else:
            expected_split_hash = expected_split_hashes.get(split)
            if not expected_split_hash:
                report.error("missing_split_checksum", split)
            elif sha256_file(split_path) != expected_split_hash:
                report.error("split_checksum_mismatch", str(split_path), split)
        with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = tuple(reader.fieldnames or ())
            missing_fields = set(MANIFEST_FIELDS) - set(fields)
            if missing_fields:
                report.error(
                    "manifest_schema",
                    f"Missing fields: {sorted(missing_fields)}",
                    split,
                )
            rows = list(reader)
        split_ids = [
            line.strip()
            for line in split_path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        ]
        row_ids = [row.get("sample_id", "") for row in rows]
        if row_ids != split_ids:
            report.error(
                "manifest_split_mismatch",
                "Manifest sample order/content differs from split file",
                split,
            )
        if not rows:
            report.error("empty_manifest", str(manifest_path), split)
        split_counts[split] = len(rows)
        for row in rows:
            sample_id = row.get("sample_id", "")
            if not sample_id:
                report.error("empty_sample_id", str(manifest_path), split)
                continue
            previous = seen.get(sample_id)
            if previous is not None:
                report.error("split_overlap", f"{previous} and {split}", sample_id)
            seen[sample_id] = split
            for field in (
                "image_relpath",
                "mask_relpath",
                "source_image_relpath",
                "source_mask_relpath",
            ):
                value = row.get(field, "")
                if not is_portable_relative(value):
                    report.error("nonportable_manifest_path", f"{field}={value}", sample_id)
            image_path = root / row.get("image_relpath", "")
            mask_path = root / row.get("mask_relpath", "")
            if not image_path.is_file():
                report.error("missing_image", str(image_path), sample_id)
                continue
            if not mask_path.is_file():
                report.error("missing_mask", str(mask_path), sample_id)
                continue
            try:
                with Image.open(image_path) as image:
                    image.load()
                    image_size = image.size
                with Image.open(mask_path) as mask_image:
                    mask_image.load()
                    mask_mode = mask_image.mode
                    mask = np.asarray(mask_image)
            except (OSError, UnidentifiedImageError) as error:
                report.error("unreadable_pair", str(error), sample_id)
                continue
            if mask.ndim != 2 or mask_mode not in {"L", "P"}:
                report.error(
                    "invalid_mask_channel",
                    f"mode={mask_mode}, shape={mask.shape}",
                    sample_id,
                )
                continue
            if mask.dtype != np.uint8:
                report.error("invalid_mask_dtype", str(mask.dtype), sample_id)
            if image_size != (mask.shape[1], mask.shape[0]):
                report.error(
                    "size_mismatch",
                    f"image={image_size}, mask={mask.shape}",
                    sample_id,
                )
            try:
                expected_size = (int(row["width"]), int(row["height"]))
                if expected_size != image_size:
                    report.error(
                        "manifest_size_mismatch",
                        f"manifest={expected_size}, actual={image_size}",
                        sample_id,
                    )
            except (KeyError, TypeError, ValueError):
                report.error("invalid_manifest_size", "width/height", sample_id)
            ids, counts = np.unique(mask, return_counts=True)
            observed = {int(value) for value in ids}
            invalid = observed - ALLOWED_TARGET_IDS
            if invalid:
                report.error("invalid_target_id", str(sorted(invalid)), sample_id)
            if observed == {255}:
                report.error("all_ignore_mask", str(mask_path), sample_id)
            for class_id, count in zip(ids, counts):
                class_pixels[int(class_id)] += int(count)

    if not seen:
        report.error("no_samples", f"No samples found in {root}")
    report.statistics.update(
        {
            "sample_count": len(seen),
            "split_counts": split_counts,
            "class_pixel_counts": {
                str(class_id): class_pixels[class_id]
                for class_id in [0, 1, 2, 3, 255]
            },
        }
    )
    if verify_checksums:
        for error in verify_checksum_manifest(root):
            report.error("checksum", error)
    return report
