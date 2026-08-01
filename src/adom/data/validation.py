from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from .io import (
    ensure_no_symlink_escape,
    is_portable_relative,
    sha256_file,
    verify_checksum_manifest,
)
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
            packaged_mapping_sha256 = metadata.get("packaged_mapping_sha256")
            if not packaged_mapping_sha256:
                report.error("missing_packaged_mapping_checksum", str(mapping_path))
            elif sha256_file(mapping_path) != packaged_mapping_sha256:
                report.error("packaged_mapping_checksum_mismatch", str(mapping_path))
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
            if validation_value.get("status") != "PASS":
                report.error(
                    "embedded_validation_failed",
                    f"status={validation_value.get('status')!r}",
                )
            if validation_value.get("error_count") != 0:
                report.error(
                    "embedded_validation_errors",
                    f"error_count={validation_value.get('error_count')!r}",
                )
        except (json.JSONDecodeError, OSError) as error:
            report.error("invalid_validation_report", str(error))

    class_statistics_rows: list[dict[str, str]] = []
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
            class_statistics_rows = rows
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
        if report_name == "qc_errors.csv":
            embedded_errors = [
                row for row in rows if row.get("severity", "").strip().lower() == "error"
            ]
            if embedded_errors:
                report.error(
                    "embedded_qc_errors",
                    f"qc_errors.csv contains {len(embedded_errors)} error row(s)",
                )

    seen: dict[str, str] = {}
    split_counts: dict[str, int] = {}
    class_pixels_by_split: dict[str, Counter[int]] = {
        split: Counter() for split in ("train", "val", "test")
    }
    image_counts_by_split: dict[str, Counter[int]] = {
        split: Counter() for split in ("train", "val", "test")
    }
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
            if row.get("split") != split:
                report.error(
                    "manifest_split_field_mismatch",
                    f"row split={row.get('split')!r}, expected={split!r}",
                    sample_id,
                )
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
                ensure_no_symlink_escape(root, image_path)
                ensure_no_symlink_escape(root, mask_path)
            except Exception as error:
                report.error("symlink_escape", str(error), sample_id)
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
                class_id_int = int(class_id)
                class_pixels_by_split[split][class_id_int] += int(count)
                image_counts_by_split[split][class_id_int] += 1

    if not seen:
        report.error("no_samples", f"No samples found in {root}")
    expected_split_counts = metadata.get("split_counts")
    if not isinstance(expected_split_counts, dict):
        report.error("dataset_metadata_schema", "split_counts must be a mapping")
    else:
        normalized_expected = {
            split: expected_split_counts.get(split)
            for split in ("train", "val", "test")
        }
        if normalized_expected != split_counts:
            report.error(
                "metadata_split_count_mismatch",
                f"metadata={normalized_expected}, actual={split_counts}",
            )

    for row in class_statistics_rows:
        split = row.get("split", "")
        try:
            class_id = int(row.get("class_id", ""))
            expected_pixels = class_pixels_by_split[split][class_id]
            expected_images = image_counts_by_split[split][class_id]
            if int(row.get("pixel_count", "")) != expected_pixels:
                report.error(
                    "class_statistics_pixel_mismatch",
                    f"{split}/{class_id}: expected={expected_pixels}, "
                    f"reported={row.get('pixel_count')!r}",
                )
            if int(row.get("image_count", "")) != expected_images:
                report.error(
                    "class_statistics_image_mismatch",
                    f"{split}/{class_id}: expected={expected_images}, "
                    f"reported={row.get('image_count')!r}",
                )
            expected_name = "ignore" if class_id == 255 else COST4_CLASSES[class_id]
            if row.get("class_name") != expected_name:
                report.error(
                    "class_statistics_name_mismatch",
                    f"{split}/{class_id}: expected={expected_name!r}, "
                    f"reported={row.get('class_name')!r}",
                )
            total_pixels = sum(class_pixels_by_split[split].values())
            valid_pixels = total_pixels - class_pixels_by_split[split][255]
            expected_ratio_all = (
                expected_pixels / total_pixels if total_pixels else 0.0
            )
            if not np.isclose(
                float(row.get("ratio_all_pixels", "")),
                expected_ratio_all,
                rtol=0.0,
                atol=1e-12,
            ):
                report.error(
                    "class_statistics_ratio_mismatch",
                    f"{split}/{class_id}: ratio_all_pixels",
                )
            ratio_valid = row.get("ratio_valid_pixels", "")
            if class_id == 255:
                if ratio_valid != "":
                    report.error(
                        "class_statistics_ratio_mismatch",
                        f"{split}/{class_id}: ignore ratio_valid_pixels must be empty",
                    )
            else:
                expected_ratio_valid = (
                    expected_pixels / valid_pixels if valid_pixels else 0.0
                )
                if not np.isclose(
                    float(ratio_valid),
                    expected_ratio_valid,
                    rtol=0.0,
                    atol=1e-12,
                ):
                    report.error(
                        "class_statistics_ratio_mismatch",
                        f"{split}/{class_id}: ratio_valid_pixels",
                    )
        except (KeyError, TypeError, ValueError):
            report.error("invalid_class_statistics_row", str(row))

    class_pixels: Counter[int] = Counter()
    for counts in class_pixels_by_split.values():
        class_pixels.update(counts)
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


def validate_manual_approval(root: Path) -> list[str]:
    approval_path = root / "reports" / "approval.json"
    if not approval_path.is_file():
        return ["reports/approval.json is missing"]
    try:
        approval = json.loads(approval_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        return [f"approval record is invalid: {error}"]

    errors: list[str] = []
    if approval.get("status") != "APPROVED":
        errors.append(f"approval status is {approval.get('status')!r}")
    if not str(approval.get("approver", "")).strip():
        errors.append("approval approver is empty")
    if not approval.get("approved_at"):
        errors.append("approval timestamp is missing")

    mapping_path = root / "config" / "label_mapping.yaml"
    if (
        not mapping_path.is_file()
        or approval.get("mapping_sha256") != sha256_file(mapping_path)
    ):
        errors.append("approval mapping checksum does not match")
    approved_splits = approval.get("split_sha256")
    if not isinstance(approved_splits, dict):
        errors.append("approval split_sha256 is invalid")
    else:
        for split in ("train", "val", "test"):
            split_path = root / "splits" / f"{split}.txt"
            if (
                not split_path.is_file()
                or approved_splits.get(split) != sha256_file(split_path)
            ):
                errors.append(f"approval {split} split checksum does not match")

    reviewed = approval.get("reviewed_previews")
    if not isinstance(reviewed, list) or not reviewed:
        errors.append("approval has no reviewed previews")
    else:
        for item in reviewed:
            if not isinstance(item, dict):
                errors.append(f"approved preview entry is invalid: {item!r}")
                continue
            relative = item.get("path")
            if (
                not isinstance(relative, str)
                or not is_portable_relative(relative)
                or not (root / relative).is_file()
            ):
                errors.append(f"approved preview is missing or unsafe: {relative!r}")
            elif item.get("sha256") != sha256_file(root / relative):
                errors.append(f"approved preview checksum does not match: {relative}")
    return errors
