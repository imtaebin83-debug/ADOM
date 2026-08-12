from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any


SOURCES = ("rellis", "rugd", "ycor", "goose")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine four validated Cost4 bridge outputs into one package."
    )
    for source in SOURCES:
        parser.add_argument(f"--{source}-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def require_empty_or_absent(path: Path) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f"Output root must be empty or absent: {path}")


def resolve_relative(root: Path, value: str, field: str) -> Path:
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Non-portable {field}: {value}")
    resolved = root.joinpath(*relative.parts).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{field} escapes bridge root: {value}") from exc
    return resolved


def safe_sample_id(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"Unsafe sample_id: {value}")
    return path


def append_suffix(path: PurePosixPath, suffix: str) -> Path:
    return Path(*path.parent.parts, f"{path.name}{suffix}")


def link_or_copy(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Destination already exists: {destination}")
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def read_manifest(source: str, root: Path) -> list[dict[str, str]]:
    path = root / "metadata" / "manifest.csv"
    if not path.is_file():
        raise FileNotFoundError(f"{source} bridge manifest not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"{source} bridge manifest is empty")
    required = {
        "sample_key",
        "source",
        "source_split",
        "sample_id",
        "image_path",
        "mask_path",
        "non_ignore_ratio",
    }
    if not required <= set(rows[0]):
        raise ValueError(f"{source} manifest requires {sorted(required)}")
    if any(row["source"] != source for row in rows):
        raise ValueError(f"{source} manifest contains another source")
    keys = [row["sample_key"] for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError(f"Duplicate sample_key in {source} manifest")
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Combined manifest must not be empty")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    roots = {
        source: getattr(args, f"{source}_root").expanduser().resolve()
        for source in SOURCES
    }
    for source, root in roots.items():
        if not root.is_dir():
            raise FileNotFoundError(f"{source} bridge root not found: {root}")
    output_root = args.output_root.expanduser().resolve()
    require_empty_or_absent(output_root)
    (output_root / "metadata").mkdir(parents=True, exist_ok=True)
    (output_root / "splits").mkdir(parents=True, exist_ok=True)

    combined_rows: list[dict[str, Any]] = []
    main_splits: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    diagnostics: dict[str, list[str]] = {}
    source_counts: Counter[str] = Counter()
    storage_counts: Counter[str] = Counter()
    seen_keys: set[str] = set()

    for source in SOURCES:
        root = roots[source]
        for row in read_manifest(source, root):
            source_split = row["source_split"]
            if source_split not in {"train", "val", "test"}:
                raise ValueError(f"Invalid {source} source split: {source_split}")
            sample_id = safe_sample_id(row["sample_id"])
            source_image = resolve_relative(root, row["image_path"], "image_path")
            source_mask = resolve_relative(root, row["mask_path"], "mask_path")
            if not source_image.is_file() or not source_mask.is_file():
                raise FileNotFoundError(f"Missing bridge pair: {row['sample_key']}")

            destination_image = (
                output_root
                / "images"
                / source
                / source_split
                / append_suffix(sample_id, source_image.suffix.lower())
            )
            destination_mask = (
                output_root
                / "masks"
                / source
                / source_split
                / append_suffix(sample_id, ".png")
            )
            storage_counts[link_or_copy(source_image, destination_image)] += 1
            storage_counts[link_or_copy(source_mask, destination_mask)] += 1

            sample_key = f"{source}/{source_split}/{sample_id.as_posix()}"
            if sample_key in seen_keys:
                raise ValueError(f"Duplicate combined sample key: {sample_key}")
            seen_keys.add(sample_key)
            if source == "rellis":
                main_splits[source_split].append(sample_key)
                role = source_split
            elif source_split == "train":
                main_splits["train"].append(sample_key)
                role = "train"
            else:
                diagnostic_name = f"{source}_{source_split}_diagnostic"
                diagnostics.setdefault(diagnostic_name, []).append(sample_key)
                role = diagnostic_name

            combined_rows.append(
                {
                    "sample_key": sample_key,
                    "source": source,
                    "source_split": source_split,
                    "package_role": role,
                    "sample_id": sample_id.as_posix(),
                    "image_path": destination_image.relative_to(output_root).as_posix(),
                    "mask_path": destination_mask.relative_to(output_root).as_posix(),
                    "non_ignore_ratio": row["non_ignore_ratio"],
                }
            )
            source_counts[source] += 1

    write_csv(output_root / "metadata" / "manifest.csv", combined_rows)
    for split, keys in main_splits.items():
        (output_root / "splits" / f"{split}.txt").write_text(
            "".join(f"{key}\n" for key in keys), encoding="utf-8"
        )
    for name, keys in diagnostics.items():
        (output_root / "splits" / f"{name}.txt").write_text(
            "".join(f"{key}\n" for key in keys), encoding="utf-8"
        )

    summary = {
        "status": "PASS",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_name": "adom_cost4_rellis_rugd_ycor_goose_v1",
        "target_ontology": "ADOM Cost4 traversability",
        "num_classes": 4,
        "ignore_index": 255,
        "sample_count": len(combined_rows),
        "source_counts": dict(sorted(source_counts.items())),
        "main_split_counts": {key: len(value) for key, value in main_splits.items()},
        "diagnostic_split_counts": {key: len(value) for key, value in diagnostics.items()},
        "storage_counts": dict(sorted(storage_counts.items())),
        "evaluation_policy": "main val/test contain RELLIS only",
    }
    (output_root / "metadata" / "package_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Cost4 combined package completed: {output_root}")


if __name__ == "__main__":
    main()
