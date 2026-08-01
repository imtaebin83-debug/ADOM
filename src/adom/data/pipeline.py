from __future__ import annotations

import csv
import json
import shutil
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from .adapters import Rellis3DAdapter
from .io import (
    atomic_replace_directory,
    sha256_file,
    write_checksum_manifest,
    write_csv,
    write_json,
)
from .models import DatasetError, SampleRecord, ValidationReport
from .preview import save_preview, write_legend
from .schema import COST4_CLASSES, LabelSchema
from .splits import load_splits, split_owner_map
from .validation import MANIFEST_FIELDS, validate_package


def _adapter(dataset: str, input_root: Path) -> Rellis3DAdapter:
    if dataset.lower() != "rellis3d":
        raise DatasetError(f"Unsupported dataset adapter: {dataset}")
    if not input_root.is_dir():
        raise DatasetError(f"Input root is not a directory: {input_root}")
    return Rellis3DAdapter(input_root)


def inspect_dataset(
    dataset: str,
    input_root: Path,
    mapping_path: Path,
) -> ValidationReport:
    schema = LabelSchema.from_path(mapping_path)
    report = ValidationReport(dataset=dataset, version=schema.version)
    adapter = _adapter(dataset, input_root)
    samples = adapter.discover(report)
    source_counts: Counter[int] = Counter()
    for sample in samples:
        try:
            source = adapter.read_source_mask(sample.source_mask_path)
            values, counts = np.unique(source, return_counts=True)
            unknown = {int(value) for value in values} - set(schema.source_to_target)
            if unknown:
                report.error("unknown_source_id", str(sorted(unknown)), sample.sample_id)
            for value, count in zip(values, counts):
                source_counts[int(value)] += int(count)
        except Exception as error:
            report.error("unreadable_source_mask", str(error), sample.sample_id)
    report.statistics.update(
        {
            "pair_count": len(samples),
            "source_pixel_counts": {
                str(key): source_counts[key] for key in sorted(source_counts)
            },
        }
    )
    return report


def _write_report_files(root: Path, report: ValidationReport) -> None:
    write_json(root / "reports" / "validation.json", report.to_dict())
    rows = [
        {
            "severity": issue.severity,
            "code": issue.code,
            "sample_id": issue.sample_id,
            "detail": issue.detail,
        }
        for issue in report.issues
    ]
    write_csv(
        root / "reports" / "qc_errors.csv",
        rows,
        ["severity", "code", "sample_id", "detail"],
    )


def _select_preview_ids(
    per_sample_counts: dict[str, dict[int, int]],
    limit: int = 8,
) -> list[str]:
    selected: list[str] = []
    for class_id in [0, 1, 2, 3]:
        candidates = sorted(
            per_sample_counts,
            key=lambda sample_id: (
                -per_sample_counts[sample_id].get(class_id, 0),
                sample_id,
            ),
        )
        if candidates and per_sample_counts[candidates[0]].get(class_id, 0) > 0:
            selected.append(candidates[0])
    ignore_order = sorted(
        per_sample_counts,
        key=lambda sample_id: (
            -per_sample_counts[sample_id].get(255, 0),
            sample_id,
        ),
    )
    selected.extend(ignore_order[:2])
    selected.extend(sorted(per_sample_counts))
    return list(dict.fromkeys(selected))[:limit]


def prepare_dataset(
    dataset: str,
    input_root: Path,
    output_root: Path,
    mapping_path: Path,
    split_root: Path,
    version: str,
    overwrite: bool = False,
) -> ValidationReport:
    if output_root.exists() and not overwrite:
        raise DatasetError(
            f"Output already exists: {output_root}. Pass --overwrite explicitly."
        )
    schema = LabelSchema.from_path(mapping_path)
    report = ValidationReport(dataset=dataset, version=version)
    splits = load_splits(split_root, report)
    owners = split_owner_map(splits) if report.passed else {}
    adapter = _adapter(dataset, input_root)
    samples = adapter.discover(report)
    discovered = {sample.sample_id: sample for sample in samples}
    for sample_id in sorted(set(owners) - set(discovered)):
        report.error("split_sample_missing", "No complete raw pair", sample_id)
    for sample_id in sorted(set(discovered) - set(owners)):
        report.warning("unassigned_sample", "Not present in official split", sample_id)
    report.require_success()

    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_root.name}.staging.",
            dir=output_root.parent,
        )
    )
    manifests: dict[str, list[dict[str, object]]] = {
        split: [] for split in ("train", "val", "test")
    }
    class_rows: list[dict[str, object]] = []
    pixel_counts: dict[str, Counter[int]] = {
        split: Counter() for split in manifests
    }
    image_counts: dict[str, Counter[int]] = {
        split: Counter() for split in manifests
    }
    per_sample_counts: dict[str, dict[int, int]] = {}
    output_paths: dict[str, tuple[Path, Path]] = {}
    try:
        for split in manifests:
            (staging / "images" / split).mkdir(parents=True, exist_ok=True)
            (staging / "annotations" / split).mkdir(parents=True, exist_ok=True)
            (staging / "splits").mkdir(parents=True, exist_ok=True)
        for split in ("train", "val", "test"):
            for sample_id in splits[split]:
                sample: SampleRecord = discovered[sample_id]
                try:
                    with Image.open(sample.image_path) as image:
                        image.load()
                        image_size = image.size
                    source = adapter.read_source_mask(sample.source_mask_path)
                    target = schema.remap(source)
                    mask_size = (target.shape[1], target.shape[0])
                    if image_size != mask_size:
                        raise DatasetError(
                            f"RGB/mask size mismatch: {image_size} != {mask_size}"
                        )
                    observed, counts = np.unique(target, return_counts=True)
                    if set(int(value) for value in observed) == {255}:
                        raise DatasetError("Target mask is entirely ignore_index")
                    image_name = sample.sample_id + sample.image_path.suffix.lower()
                    mask_name = sample.sample_id + ".png"
                    image_output = staging / "images" / split / image_name
                    mask_output = staging / "annotations" / split / mask_name
                    shutil.copy2(sample.image_path, image_output)
                    Image.fromarray(target, mode="L").save(
                        mask_output,
                        format="PNG",
                        compress_level=6,
                        optimize=False,
                    )
                    sample_counts = {
                        int(value): int(count)
                        for value, count in zip(observed, counts)
                    }
                    per_sample_counts[sample_id] = sample_counts
                    for class_id, count in sample_counts.items():
                        pixel_counts[split][class_id] += count
                        image_counts[split][class_id] += 1
                    row = {
                        "sample_id": sample.sample_id,
                        "split": split,
                        "sequence": sample.sequence,
                        "image_relpath": image_output.relative_to(staging).as_posix(),
                        "mask_relpath": mask_output.relative_to(staging).as_posix(),
                        "width": image_size[0],
                        "height": image_size[1],
                        "source_image_relpath": sample.image_path.relative_to(
                            input_root
                        ).as_posix(),
                        "source_mask_relpath": sample.source_mask_path.relative_to(
                            input_root
                        ).as_posix(),
                    }
                    manifests[split].append(row)
                    output_paths[sample_id] = (image_output, mask_output)
                except (DatasetError, OSError, ValueError, UnidentifiedImageError) as error:
                    report.error("conversion_failed", str(error), sample_id)
        report.require_success()

        for split, rows in manifests.items():
            write_csv(
                staging / "metadata" / f"manifest_{split}.csv",
                rows,
                list(MANIFEST_FIELDS),
            )
            (staging / "splits" / f"{split}.txt").write_text(
                "".join(f"{row['sample_id']}\n" for row in rows),
                encoding="utf-8",
            )
            total = sum(pixel_counts[split].values())
            valid = total - pixel_counts[split][255]
            for class_id in [0, 1, 2, 3, 255]:
                count = pixel_counts[split][class_id]
                class_rows.append(
                    {
                        "split": split,
                        "class_id": class_id,
                        "class_name": (
                            "ignore"
                            if class_id == 255
                            else COST4_CLASSES[class_id]
                        ),
                        "pixel_count": count,
                        "ratio_all_pixels": count / total if total else 0.0,
                        "ratio_valid_pixels": (
                            ""
                            if class_id == 255
                            else (count / valid if valid else 0.0)
                        ),
                        "image_count": image_counts[split][class_id],
                    }
                )
        write_csv(
            staging / "reports" / "class_statistics.csv",
            class_rows,
            [
                "split",
                "class_id",
                "class_name",
                "pixel_count",
                "ratio_all_pixels",
                "ratio_valid_pixels",
                "image_count",
            ],
        )
        packaged_mapping = staging / "config" / "label_mapping.yaml"
        write_json(packaged_mapping, schema.snapshot())
        dataset_metadata = {
            "dataset": dataset,
            "version": version,
            "label_schema": "ADOM Cost4",
            "num_classes": 4,
            "ignore_index": 255,
            "reduce_zero_label": False,
            "official_split_only": True,
            "known_limitations": [
                "RELLIS official splits share source sequences; temporal correlation "
                "may make generalization metrics optimistic."
            ],
            "mapping_sha256": sha256_file(mapping_path),
            "packaged_mapping_sha256": sha256_file(packaged_mapping),
            "source_mapping_file": mapping_path.name,
            "official_split_sha256": {
                split: sha256_file(split_root / f"{split}.txt")
                for split in ("train", "val", "test")
            },
            "split_counts": {
                split: len(rows) for split, rows in manifests.items()
            },
        }
        write_json(staging / "metadata" / "dataset.json", dataset_metadata)
        card = (
            "# RELLIS-3D ADOM Cost4\n\n"
            f"- Version: `{version}`\n"
            "- Trainable classes: `0, 1, 2, 3`\n"
            "- Ignore index: `255`\n"
            "- Split policy: official RELLIS split only\n\n"
            "## Known limitation\n\n"
            "The official split contains frames from shared source sequences. "
            "Temporal correlation can make held-out metrics optimistic.\n"
        )
        (staging / "DATASET_CARD.md").write_text(card, encoding="utf-8")

        preview_root = staging / "reports" / "previews"
        write_legend(preview_root / "legend.png")
        for sample_id in _select_preview_ids(per_sample_counts):
            image_output, mask_output = output_paths[sample_id]
            save_preview(
                image_output,
                mask_output,
                preview_root / f"{sample_id}.jpg",
                sample_id,
            )

        report.statistics.update(
            {
                "processed_samples": sum(len(rows) for rows in manifests.values()),
                "split_counts": {
                    split: len(rows) for split, rows in manifests.items()
                },
                "unassigned_samples": len(set(discovered) - set(owners)),
            }
        )
        _write_report_files(staging, report)
        write_checksum_manifest(staging)
        package_report = validate_package(staging, verify_checksums=True)
        if not package_report.passed:
            for issue in package_report.errors:
                report.error(issue.code, issue.detail, issue.sample_id)
            report.require_success()
        atomic_replace_directory(staging, output_root, overwrite)
        return validate_package(output_root, verify_checksums=True)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
