from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


SOURCE_ADOM = "adom_zed2i"
SPLIT_NAMES = (
    "ta0_train",
    "ta1_train",
    "ta2_train",
    "val",
    "test",
    "adom_val_diagnostic",
    "adom_test_diagnostic",
)
MANIFEST_FIELDS = (
    "sample_key",
    "source",
    "source_split",
    "image_path",
    "mask_path",
)
ALLOWED_IDS = set(range(19)) | {255}
EXPECTED_SPLIT_COUNTS = {
    "ta0_train": 4435,
    "ta1_train": 4568,
    "ta2_train": 10001,
    "val": 900,
    "test": 899,
    "adom_val_diagnostic": 21,
    "adom_test_diagnostic": 61,
}
EXPECTED_SPLIT_SOURCES = {
    "ta0_train": {"rellis3d"},
    "ta1_train": {"rellis3d", SOURCE_ADOM},
    "ta2_train": {"rellis3d", "rugd", "ycor", SOURCE_ADOM},
    "val": {"rellis3d"},
    "test": {"rellis3d"},
    "adom_val_diagnostic": {SOURCE_ADOM},
    "adom_test_diagnostic": {SOURCE_ADOM},
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_lines(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    values = [
        line.strip()
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    if not values or len(values) != len(set(values)):
        raise ValueError(f"Split is empty or contains duplicates: {path}")
    return values


def _write_lines(path: Path, values: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{value}\n" for value in values), encoding="utf-8")


def _safe_relative(root: Path, stored: str, field: str) -> Path:
    value = Path(stored)
    if value.is_absolute() or ".." in value.parts:
        raise ValueError(f"Non-portable {field}: {stored}")
    resolved = (root / value).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"{field} escapes package root: {stored}") from exc
    return resolved


def _read_manifest(root: Path) -> list[dict[str, str]]:
    path = root / "manifest.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"sample_key", "image_path", "mask_path"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Manifest fields missing in {path}: {sorted(missing)}")
        rows = list(reader)
    keys = [row["sample_key"] for row in rows]
    if not rows or len(keys) != len(set(keys)):
        raise ValueError(f"Manifest is empty or contains duplicate keys: {path}")
    return rows


def _link_or_copy(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def _materialize_row(
    *,
    source_root: Path,
    output_root: Path,
    source_key: str,
    output_key: str,
    source_name: str,
    source_split: str,
    image_stored: str,
    mask_stored: str,
    destination_prefix: Path | None,
) -> tuple[dict[str, str], Counter[str]]:
    source_image = _safe_relative(source_root, image_stored, "image_path")
    source_mask = _safe_relative(source_root, mask_stored, "mask_path")
    if not source_image.is_file() or not source_mask.is_file():
        raise FileNotFoundError(f"Missing source pair: {source_key}")

    if destination_prefix is None:
        image_relative = Path(image_stored)
        mask_relative = Path(mask_stored)
    else:
        image_tail = Path(image_stored).relative_to("images")
        mask_tail = Path(mask_stored).relative_to("masks")
        image_relative = Path("images") / destination_prefix / image_tail
        mask_relative = Path("masks") / destination_prefix / mask_tail

    modes: Counter[str] = Counter()
    modes[f"image_{_link_or_copy(source_image, output_root / image_relative)}"] += 1
    modes[f"mask_{_link_or_copy(source_mask, output_root / mask_relative)}"] += 1
    return (
        {
            "sample_key": output_key,
            "source": source_name,
            "source_split": source_split,
            "image_path": image_relative.as_posix(),
            "mask_path": mask_relative.as_posix(),
        },
        modes,
    )


def build_package(e1_root: Path, standalone_root: Path, output_root: Path) -> dict[str, Any]:
    e1_root = e1_root.resolve()
    standalone_root = standalone_root.resolve()
    output_root = output_root.resolve()
    for root, name in ((e1_root, "E1"), (standalone_root, "standalone")):
        if not (root / "_SUCCESS").is_file():
            raise FileNotFoundError(f"{name} _SUCCESS is missing: {root / '_SUCCESS'}")
    if output_root in {e1_root, standalone_root}:
        raise ValueError("Output root must differ from both input roots")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Output root must be new or empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    e1_rows = _read_manifest(e1_root)
    standalone_rows = _read_manifest(standalone_root)
    output_rows: list[dict[str, str]] = []
    storage_modes: Counter[str] = Counter()

    for row in e1_rows:
        key = row["sample_key"]
        source = row.get("source") or key.split("/", 1)[0]
        output_row, modes = _materialize_row(
            source_root=e1_root,
            output_root=output_root,
            source_key=key,
            output_key=key,
            source_name=source,
            source_split=row.get("source_split", "unknown"),
            image_stored=row["image_path"],
            mask_stored=row["mask_path"],
            destination_prefix=None,
        )
        output_rows.append(output_row)
        storage_modes.update(modes)

    standalone_key_map: dict[str, str] = {}
    for row in standalone_rows:
        source_key = row["sample_key"]
        output_key = f"{SOURCE_ADOM}/{source_key}"
        standalone_key_map[source_key] = output_key
        output_row, modes = _materialize_row(
            source_root=standalone_root,
            output_root=output_root,
            source_key=source_key,
            output_key=output_key,
            source_name=SOURCE_ADOM,
            source_split=row.get("split", "unknown"),
            image_stored=row["image_path"],
            mask_stored=row["mask_path"],
            destination_prefix=Path(SOURCE_ADOM),
        )
        output_rows.append(output_row)
        storage_modes.update(modes)

    output_keys = [row["sample_key"] for row in output_rows]
    if len(output_keys) != len(set(output_keys)):
        raise ValueError("Combined target-adaptation manifest contains duplicate keys")

    e1_train = _read_lines(e1_root / "splits" / "train.txt")
    canonical_val = _read_lines(e1_root / "splits" / "val.txt")
    canonical_test = _read_lines(e1_root / "splits" / "test.txt")
    rellis_train = [key for key in e1_train if key.startswith("rellis3d/")]
    adom_train = [
        standalone_key_map[key]
        for key in _read_lines(standalone_root / "splits" / "train.txt")
    ]
    splits = {
        "ta0_train": rellis_train,
        "ta1_train": rellis_train + adom_train,
        "ta2_train": e1_train + adom_train,
        "val": canonical_val,
        "test": canonical_test,
        "adom_val_diagnostic": [
            standalone_key_map[key]
            for key in _read_lines(standalone_root / "splits" / "val.txt")
        ],
        "adom_test_diagnostic": [
            standalone_key_map[key]
            for key in _read_lines(standalone_root / "splits" / "test.txt")
        ],
    }
    for name, values in splits.items():
        _write_lines(output_root / "splits" / f"{name}.txt", values)

    with (output_root / "manifest.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)

    metadata_root = output_root / "metadata"
    metadata_root.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": "adom-semantic20-ta-package-v1",
        "dataset": "adom_semantic20_target_adaptation_v1",
        "num_classes": 19,
        "ignore_index": 255,
        "reduce_zero_label": False,
        "manifest_count": len(output_rows),
        "manifest_source_counts": dict(Counter(row["source"] for row in output_rows)),
        "split_counts": {name: len(values) for name, values in splits.items()},
        "split_source_counts": {
            name: dict(Counter(key.split("/", 1)[0] for key in values))
            for name, values in splits.items()
        },
        "storage_modes": dict(storage_modes),
        "input_contract": {
            "e1_manifest_sha256": sha256_file(e1_root / "manifest.csv"),
            "e1_success_sha256": sha256_file(e1_root / "_SUCCESS"),
            "standalone_manifest_sha256": sha256_file(standalone_root / "manifest.csv"),
            "standalone_success_sha256": sha256_file(standalone_root / "_SUCCESS"),
        },
        "validation_policy": {
            "checkpoint_selection": "canonical RELLIS val only",
            "canonical_test": "locked RELLIS test only",
            "standalone_val_test": "diagnostic only",
        },
    }
    (metadata_root / "package_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary


def validate_package(root: Path, *, write_success: bool) -> dict[str, Any]:
    root = root.resolve()
    summary_path = root / "metadata" / "package_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8-sig"))
    rows = _read_manifest(root)
    row_by_key = {row["sample_key"]: row for row in rows}
    splits = {name: _read_lines(root / "splits" / f"{name}.txt") for name in SPLIT_NAMES}
    split_counts = {name: len(values) for name, values in splits.items()}
    if split_counts != EXPECTED_SPLIT_COUNTS:
        raise ValueError(
            f"Target-adaptation split counts differ: actual={split_counts}, "
            f"expected={EXPECTED_SPLIT_COUNTS}"
        )
    for name, values in splits.items():
        missing = set(values) - set(row_by_key)
        if missing:
            raise ValueError(f"{name} contains keys absent from manifest: {sorted(missing)[:10]}")
        sources = {key.split("/", 1)[0] for key in values}
        if sources != EXPECTED_SPLIT_SOURCES[name]:
            raise ValueError(f"Unexpected sources in {name}: {sorted(sources)}")
    if not set(splits["ta0_train"]) < set(splits["ta1_train"]):
        raise ValueError("TA0 train must be a strict subset of TA1 train")
    if not set(splits["ta1_train"]) < set(splits["ta2_train"]):
        raise ValueError("TA1 train must be a strict subset of TA2 train")
    if set(splits["val"]) & set(splits["test"]):
        raise ValueError("Canonical validation and test splits overlap")
    if any(not key.startswith("rellis3d/") for key in splits["val"] + splits["test"]):
        raise ValueError("Canonical validation/test must remain RELLIS-only")
    adom_partitions = (
        set(key for key in splits["ta2_train"] if key.startswith(f"{SOURCE_ADOM}/")),
        set(splits["adom_val_diagnostic"]),
        set(splits["adom_test_diagnostic"]),
    )
    if any(left & right for index, left in enumerate(adom_partitions) for right in adom_partitions[index + 1 :]):
        raise ValueError("ADOM train/diagnostic partitions overlap")
    for key, row in row_by_key.items():
        if row["source"] != key.split("/", 1)[0]:
            raise ValueError(f"Manifest source does not match sample key: {key}")

    train_keys = set(splits["ta2_train"])
    image_digest = hashlib.sha256()
    mask_digest = hashlib.sha256()
    all_ignore_train: list[str] = []
    observed_ids: set[int] = set()
    for key in sorted(row_by_key):
        row = row_by_key[key]
        image_path = _safe_relative(root, row["image_path"], "image_path")
        mask_path = _safe_relative(root, row["mask_path"], "mask_path")
        if not image_path.is_file() or not mask_path.is_file():
            raise FileNotFoundError(f"Missing target-adaptation pair: {key}")
        with Image.open(image_path) as image:
            image.load()
            image_size = image.size
        with Image.open(mask_path) as mask_image:
            mask_image.load()
            if mask_image.mode not in {"L", "P"}:
                raise ValueError(f"Mask must be single-channel: {key}, mode={mask_image.mode}")
            mask = np.asarray(mask_image)
        if mask.dtype != np.uint8 or mask.ndim != 2:
            raise ValueError(f"Mask must be uint8 HxW: {key}, {mask.dtype}, {mask.shape}")
        if image_size != (mask.shape[1], mask.shape[0]):
            raise ValueError(f"Image/mask size mismatch: {key}")
        ids = {int(value) for value in np.unique(mask)}
        invalid = ids - ALLOWED_IDS
        if invalid:
            raise ValueError(f"Invalid Semantic20 IDs for {key}: {sorted(invalid)}")
        observed_ids.update(ids)
        if key in train_keys and not np.any(mask != 255):
            all_ignore_train.append(key)
        key_bytes = key.encode("utf-8")
        image_digest.update(key_bytes)
        image_digest.update(sha256_file(image_path).encode("ascii"))
        mask_digest.update(key_bytes)
        mask_digest.update(sha256_file(mask_path).encode("ascii"))
    if all_ignore_train:
        raise ValueError(f"TA2 train contains all-ignore masks: {all_ignore_train[:10]}")

    actual_summary_fields = {
        "manifest_count": len(rows),
        "manifest_source_counts": dict(Counter(row["source"] for row in rows)),
        "split_counts": split_counts,
        "split_source_counts": {
            name: dict(Counter(key.split("/", 1)[0] for key in values))
            for name, values in splits.items()
        },
    }
    for field, actual in actual_summary_fields.items():
        if summary.get(field) != actual:
            raise ValueError(f"Package summary mismatch for {field}")

    report = {
        "schema_version": "adom-semantic20-ta-validation-v1",
        "status": "PASS",
        **actual_summary_fields,
        "observed_target_ids": sorted(observed_ids),
        "all_ignore_train_masks": 0,
        "manifest_sha256": sha256_file(root / "manifest.csv"),
        "package_summary_sha256": sha256_file(summary_path),
        "dataset_images_sha256": image_digest.hexdigest(),
        "dataset_masks_sha256": mask_digest.hexdigest(),
    }
    results_root = root / "results"
    results_root.mkdir(parents=True, exist_ok=True)
    report_path = results_root / "final_check.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if write_success:
        (root / "_SUCCESS").write_text(
            json.dumps(
                {"status": "PASS", "final_check_sha256": sha256_file(report_path)},
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build or validate the shared Semantic20 TA0/TA1/TA2 package"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--e1-root", required=True, type=Path)
    build.add_argument("--standalone-root", required=True, type=Path)
    build.add_argument("--output-root", required=True, type=Path)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--input-root", required=True, type=Path)
    validate.add_argument("--write-success-marker", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            result = build_package(args.e1_root, args.standalone_root, args.output_root)
        else:
            result = validate_package(
                args.input_root, write_success=args.write_success_marker
            )
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.exit(2, f"ERROR: {error}\n")


if __name__ == "__main__":
    main()
