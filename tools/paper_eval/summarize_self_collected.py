from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from _common import (
    SEMANTIC20_CLASSES,
    metrics_from_confusion,
    read_json,
    read_manifest,
    sha256_file,
    write_dict_csv,
    write_json,
)


RISK_CLASSES = ("log", "person", "rubble")
PAPER_COMMON_CLASSES = ("log", "rubble")


def _load_split(
    split: str,
    manifest_path: Path,
    confusion_path: Path,
    summary_path: Path,
) -> list[dict[str, Any]]:
    records = read_manifest(manifest_path)
    payload = np.load(confusion_path, allow_pickle=False)
    sample_ids = payload["sample_ids"].astype(str).tolist()
    if sample_ids != [record.sample_id for record in records]:
        raise RuntimeError(f"Per-image confusion order differs from {split} manifest")
    confusions = payload["confusions"]
    ignored = payload["ignored_pixels"]
    if confusions.shape[0] != len(records) or ignored.shape[0] != len(records):
        raise RuntimeError(f"Per-image array length differs from {split} manifest")
    summary = read_json(summary_path)
    if summary["provenance"]["ordered_manifest_sha256"] != _manifest_digest(records):
        raise RuntimeError(f"Summary provenance differs from {split} manifest")
    return [
        {
            "split": split,
            "record": record,
            "confusion": confusions[index].astype(np.int64, copy=False),
            "ignored_pixels": int(ignored[index]),
            "checkpoint_sha256": summary["provenance"]["checkpoint_sha256"],
            "evaluation_contract_sha256": summary["provenance"][
                "evaluation_contract_sha256"
            ],
        }
        for index, record in enumerate(records)
    ]


def _manifest_digest(records: list[Any]) -> str:
    from _common import manifest_sha256

    return manifest_sha256(records)


def _select_unique(
    groups: dict[str, list[dict[str, Any]]], preference: tuple[str, ...]
) -> list[dict[str, Any]]:
    rank = {split: index for index, split in enumerate(preference)}
    return [
        min(group, key=lambda entry: rank.get(entry["split"], len(rank)))
        for _, group in sorted(groups.items())
    ]


def _cohort_metrics(
    name: str, entries: list[dict[str, Any]]
) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray]:
    confusion = np.sum(
        np.stack([entry["confusion"] for entry in entries]), axis=0, dtype=np.int64
    )
    ignored_pixels = sum(entry["ignored_pixels"] for entry in entries)
    common_classes = [
        name
        for name in PAPER_COMMON_CLASSES
        if confusion[SEMANTIC20_CLASSES.index(name)].sum() > 0
    ]
    metrics, per_class = metrics_from_confusion(
        confusion,
        ignored_pixels=ignored_pixels,
        common_classes=common_classes,
    )
    row = {
        "cohort": name,
        "manifest_rows": len(entries),
        "unique_images": len({entry["record"].image_sha256 for entry in entries}),
        "sequence_count": len({entry["record"].sequence for entry in entries}),
        "supported_classes": ";".join(metrics["supported_classes"]),
        "aAcc": metrics["aAcc"],
        "dataset_native_supported_mIoU": metrics["dataset_native_supported_mIoU"],
        "common_classes": ";".join(metrics["common_classes"]),
        "common_supported_mIoU": metrics["common_supported_mIoU"],
        "total_evaluated_pixels": metrics["total_evaluated_pixels"],
        "ignored_pixels": metrics["ignored_pixels"],
    }
    for class_name in RISK_CLASSES:
        class_row = per_class[SEMANTIC20_CLASSES.index(class_name)]
        row[f"{class_name}_gt_pixels"] = class_row["gt_pixel_count"]
        row[f"{class_name}_iou"] = class_row["iou"]
        row[f"{class_name}_precision"] = class_row["precision"]
        row[f"{class_name}_recall"] = class_row["recall"]
    return row, per_class, confusion


def _duplicate_conflicts(
    groups: dict[str, list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for image_sha256, group in sorted(groups.items()):
        if len(group) < 2:
            continue
        if len(group) != 2:
            raise RuntimeError(f"Unexpected duplicate multiplicity for {image_sha256}")
        first, second = group
        first_mask = np.asarray(Image.open(first["record"].annotation_path))
        second_mask = np.asarray(Image.open(second["record"].annotation_path))
        overlap = (first_mask != 255) & (second_mask != 255)
        conflicts = overlap & (first_mask != second_mask)
        pairs = Counter(
            zip(first_mask[conflicts].astype(int), second_mask[conflicts].astype(int))
        )
        output.append(
            {
                "image_sha256": image_sha256,
                "first_split": first["split"],
                "first_sample_id": first["record"].sample_id,
                "first_annotation_sha256": first["record"].annotation_sha256,
                "second_split": second["split"],
                "second_sample_id": second["record"].sample_id,
                "second_annotation_sha256": second["record"].annotation_sha256,
                "overlap_non_ignore_pixels": int(overlap.sum()),
                "conflicting_pixels": int(conflicts.sum()),
                "label_pairs": json.dumps(
                    {
                        f"{SEMANTIC20_CLASSES[a]}->{SEMANTIC20_CLASSES[b]}": count
                        for (a, b), count in sorted(pairs.items())
                    },
                    sort_keys=True,
                ),
            }
        )
    return output


def _prediction_distribution(
    cohort: str, confusion: np.ndarray
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for gt_name in RISK_CLASSES:
        gt_id = SEMANTIC20_CLASSES.index(gt_name)
        total = int(confusion[gt_id].sum())
        if total == 0:
            continue
        for predicted_id, pixels in enumerate(confusion[gt_id]):
            if not pixels:
                continue
            rows.append(
                {
                    "cohort": cohort,
                    "gt_class": gt_name,
                    "predicted_class": SEMANTIC20_CLASSES[predicted_id],
                    "pixels": int(pixels),
                    "percent_of_gt": 100.0 * int(pixels) / total,
                }
            )
    return rows


def _write_report(
    path: Path,
    cohort_rows: list[dict[str, Any]],
    conflicts: list[dict[str, Any]],
    checkpoint_sha256: str,
) -> None:
    by_name = {row["cohort"]: row for row in cohort_rows}
    primary = by_name["conflict_free_unique"]
    lines = [
        "# B0-E0 on all self-collected annotations",
        "",
        "## Scope",
        "",
        "B0-E0 was trained without the Korean collection. Train, validation, and held-out",
        "test annotations are therefore all out-of-domain observations for this checkpoint.",
        f"Checkpoint SHA-256: `{checkpoint_sha256}`.",
        "",
        "The source package has 215 manifest rows but only 203 unique RGB images. Twelve",
        "train/validation duplicate RGBs have conflicting non-ignore labels, so the primary",
        "whole-collection estimate excludes both annotations for those ambiguous images.",
        f"It contains {primary['unique_images']} unique, conflict-free images.",
        "",
        "## Primary result",
        "",
        f"- aAcc: {primary['aAcc']:.4f}%",
        f"- native supported mIoU: {primary['dataset_native_supported_mIoU']:.4f}%",
        f"- log+rubble common mIoU: {primary['common_supported_mIoU']:.4f}%",
        f"- log IoU/recall: {primary['log_iou']:.4f}% / {primary['log_recall']:.4f}%",
        f"- person IoU/recall: {primary['person_iou']:.4f}% / {primary['person_recall']:.4f}%",
        f"- rubble IoU/recall: {primary['rubble_iou']:.4f}% / {primary['rubble_recall']:.4f}%",
        "",
        "## Interpretation",
        "",
        "These observations are a concrete counterexample to assuming stable rare-hazard",
        "recognition after public-dataset training. They support wording that such stability",
        "is not guaranteed. They do not prove failure in every possible real-world domain,",
        "and the collection's partial-label policy plus the 12 contradictory duplicate masks",
        "must be disclosed.",
        "",
        "## Annotation conflict",
        "",
        f"- conflicting duplicate RGB groups: {len(conflicts)}",
        f"- conflicting labeled pixels: {sum(row['conflicting_pixels'] for row in conflicts)}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    paper_root = args.paper_eval_root.resolve()
    supplemental = args.supplemental_dir.resolve()
    split_inputs = {
        "train": (
            paper_root / "manifests/korean_train_manifest.csv",
            supplemental / "metrics/korean_train__b0_e0__per_image_confusions.npz",
            supplemental / "metrics/korean_train__b0_e0__summary.json",
        ),
        "val": (
            paper_root / "manifests/korean_val_manifest.csv",
            supplemental / "metrics/korean_val__b0_e0__per_image_confusions.npz",
            supplemental / "metrics/korean_val__b0_e0__summary.json",
        ),
        "test": (
            paper_root / "manifests/korean_test_manifest.csv",
            paper_root / "metrics/korean__b0_e0__per_image_confusions.npz",
            paper_root / "metrics/korean__b0_e0__summary.json",
        ),
    }
    entries_by_split = {
        split: _load_split(split, *paths) for split, paths in split_inputs.items()
    }
    all_entries = sum(entries_by_split.values(), [])
    checkpoint_hashes = {entry["checkpoint_sha256"] for entry in all_entries}
    contract_hashes = {entry["evaluation_contract_sha256"] for entry in all_entries}
    if len(checkpoint_hashes) != 1 or len(contract_hashes) != 1:
        raise RuntimeError("Split results do not share checkpoint/evaluation contract")

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in all_entries:
        groups[entry["record"].image_sha256].append(entry)
    duplicate_hashes = {key for key, group in groups.items() if len(group) > 1}
    conflict_free = [
        entry
        for entry in all_entries
        if entry["record"].image_sha256 not in duplicate_hashes
    ]
    cohorts = {
        "train": entries_by_split["train"],
        "val": entries_by_split["val"],
        "held_out_test": entries_by_split["test"],
        "all_rows_duplicate_weighted": all_entries,
        "unique_train_preferred": _select_unique(groups, ("train", "test", "val")),
        "unique_val_preferred": _select_unique(groups, ("val", "test", "train")),
        "conflict_free_unique": conflict_free,
    }

    cohort_rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    distributions: list[dict[str, Any]] = []
    confusion_dir = supplemental / "aggregate_confusions"
    confusion_dir.mkdir(parents=True, exist_ok=True)
    for cohort_name, entries in cohorts.items():
        row, per_class, confusion = _cohort_metrics(cohort_name, entries)
        cohort_rows.append(row)
        for class_row in per_class:
            per_class_rows.append({"cohort": cohort_name, **class_row})
        distributions.extend(_prediction_distribution(cohort_name, confusion))
        np.save(confusion_dir / f"{cohort_name}.npy", confusion, allow_pickle=False)

    conflicts = _duplicate_conflicts(groups)
    sequence_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in conflict_free:
        sequence_groups[entry["record"].sequence].append(entry)
    sequence_rows: list[dict[str, Any]] = []
    for sequence, entries in sorted(sequence_groups.items()):
        row, _, _ = _cohort_metrics(sequence, entries)
        sequence_rows.append({"sequence": sequence, **row})

    cohort_fields = list(cohort_rows[0])
    write_dict_csv(supplemental / "cohort_metrics.csv", cohort_rows, cohort_fields)
    write_dict_csv(
        supplemental / "per_sequence_metrics.csv",
        sequence_rows,
        ["sequence", *[key for key in sequence_rows[0] if key != "sequence"]],
    )
    write_dict_csv(
        supplemental / "per_class_metrics.csv",
        per_class_rows,
        ["cohort", *[key for key in per_class_rows[0] if key != "cohort"]],
    )
    write_dict_csv(
        supplemental / "prediction_distribution.csv",
        distributions,
        ("cohort", "gt_class", "predicted_class", "pixels", "percent_of_gt"),
    )
    conflict_fields = (
        "image_sha256",
        "first_split",
        "first_sample_id",
        "first_annotation_sha256",
        "second_split",
        "second_sample_id",
        "second_annotation_sha256",
        "overlap_non_ignore_pixels",
        "conflicting_pixels",
        "label_pairs",
    )
    write_dict_csv(supplemental / "duplicate_label_conflicts.csv", conflicts, conflict_fields)
    report_path = supplemental / "report.md"
    checkpoint_sha256 = next(iter(checkpoint_hashes))
    _write_report(report_path, cohort_rows, conflicts, checkpoint_sha256)
    summary = {
        "schema_version": "adom-self-collected-b0-e0-v1",
        "checkpoint_sha256": checkpoint_sha256,
        "evaluation_contract_sha256": next(iter(contract_hashes)),
        "source_rows": len(all_entries),
        "unique_images": len(groups),
        "conflicting_duplicate_groups": len(conflicts),
        "conflicting_labeled_pixels": sum(
            row["conflicting_pixels"] for row in conflicts
        ),
        "primary_cohort": "conflict_free_unique",
        "cohorts": {row["cohort"]: row for row in cohort_rows},
        "artifacts": {
            "cohort_metrics": str((supplemental / "cohort_metrics.csv").resolve()),
            "per_sequence_metrics": str(
                (supplemental / "per_sequence_metrics.csv").resolve()
            ),
            "per_class_metrics": str(
                (supplemental / "per_class_metrics.csv").resolve()
            ),
            "prediction_distribution": str(
                (supplemental / "prediction_distribution.csv").resolve()
            ),
            "duplicate_label_conflicts": str(
                (supplemental / "duplicate_label_conflicts.csv").resolve()
            ),
            "report": str(report_path.resolve()),
        },
        "source_sha256": {
            str(path.resolve()): sha256_file(path)
            for paths in split_inputs.values()
            for path in paths
        },
    }
    write_json(supplemental / "summary.json", summary)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate B0-E0 results across all self-collected annotations"
    )
    parser.add_argument("--paper-eval-root", required=True, type=Path)
    parser.add_argument("--supplemental-dir", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    result = summarize(parse_args(argv))
    print(json.dumps(result["cohorts"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
