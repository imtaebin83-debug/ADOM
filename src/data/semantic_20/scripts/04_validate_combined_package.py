from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image


EXPECTED_SOURCE_COUNTS = {
    "rellis3d": 6234,
    "rugd": 7436,
    "ycor": 751,
}

EXPECTED_MAIN_SPLIT_COUNTS = {
    "train": 9868,
    "val": 900,
    "test": 899,
}

EXPECTED_TRAIN_SOURCE_COUNTS = {
    "rellis3d": 4435,
    "rugd": 4779,
    "ycor": 654,
}

EXPECTED_DIAGNOSTIC_COUNTS = {
    "rugd_val_diagnostic.txt": 733,
    "rugd_test_diagnostic.txt": 1924,
    "ycor_val_diagnostic.txt": 97,
    "rugd_train_source.txt": 4779,
    "ycor_train_source.txt": 654,
}

ALLOWED_IDS = {
    "rellis3d": set(range(19)) | {255},
    "rugd": {1, 2, 4, 5, 8, 9, 11, 13, 18, 255},
    "ycor": {1, 16, 255},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the combined ADOM Semantic20 package."
        )
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
    )
    return parser.parse_args()


def read_nonempty_lines(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(path)

    return [
        line.strip()
        for line in path.read_text(
            encoding="utf-8-sig"
        ).splitlines()
        if line.strip()
    ]


def main() -> None:
    args = parse_args()
    root = args.output_root.resolve()

    manifest_path = root / "manifest.csv"

    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)

    with manifest_path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        rows = list(csv.DictReader(file))

    expected_manifest_count = sum(
        EXPECTED_SOURCE_COUNTS.values()
    )

    if len(rows) != expected_manifest_count:
        raise RuntimeError(
            "Manifest count mismatch: "
            f"expected={expected_manifest_count}, "
            f"actual={len(rows)}"
        )

    sample_keys = [
        row["sample_key"]
        for row in rows
    ]

    if len(sample_keys) != len(set(sample_keys)):
        raise RuntimeError(
            "Duplicate sample_key in manifest."
        )

    row_by_key = {
        row["sample_key"]: row
        for row in rows
    }

    source_counts = Counter(
        row["source"]
        for row in rows
    )

    if dict(source_counts) != EXPECTED_SOURCE_COUNTS:
        raise RuntimeError(
            f"Source count mismatch: {dict(source_counts)}"
        )

    observed_ids = {
        source: set()
        for source in ALLOWED_IDS
    }

    for index, row in enumerate(rows, start=1):
        source = row["source"]

        if source not in ALLOWED_IDS:
            raise RuntimeError(
                f"Unknown source: {source}"
            )

        image_path = root / row["image_path"]
        mask_path = root / row["mask_path"]

        if not image_path.is_file():
            raise FileNotFoundError(
                f"Missing image: {image_path}"
            )

        if not mask_path.is_file():
            raise FileNotFoundError(
                f"Missing mask: {mask_path}"
            )

        with Image.open(image_path) as image:
            image_size = image.size

        with Image.open(mask_path) as image:
            mask_size = image.size
            mask = np.asarray(image)

        if image_size != mask_size:
            raise RuntimeError(
                "Image/mask size mismatch: "
                f"{image_path}={image_size}, "
                f"{mask_path}={mask_size}"
            )

        if mask.ndim != 2:
            raise RuntimeError(
                "Mask is not single-channel: "
                f"{mask_path}, shape={mask.shape}"
            )

        current_ids = {
            int(value)
            for value in np.unique(mask)
        }

        unexpected_ids = (
            current_ids - ALLOWED_IDS[source]
        )

        if unexpected_ids:
            raise RuntimeError(
                f"Unexpected IDs in {mask_path}: "
                f"{sorted(unexpected_ids)}"
            )

        observed_ids[source].update(
            current_ids
        )

        actual_ratio = float(
            np.count_nonzero(mask != 255)
            / mask.size
        )

        recorded_ratio = float(
            row["non_ignore_ratio"]
        )

        if abs(
            actual_ratio - recorded_ratio
        ) > 1e-6:
            raise RuntimeError(
                "non_ignore_ratio mismatch: "
                f"{mask_path}"
            )

        if index % 1000 == 0:
            print(
                f"Validated pairs: "
                f"{index}/{len(rows)}"
            )

    main_split_sets: dict[str, set[str]] = {}

    for split_name, expected_count in (
        EXPECTED_MAIN_SPLIT_COUNTS.items()
    ):
        split_path = (
            root
            / "splits"
            / f"{split_name}.txt"
        )

        entries = read_nonempty_lines(
            split_path
        )

        if len(entries) != expected_count:
            raise RuntimeError(
                f"{split_name} count mismatch: "
                f"expected={expected_count}, "
                f"actual={len(entries)}"
            )

        if len(entries) != len(set(entries)):
            raise RuntimeError(
                f"Duplicate entries in {split_name}"
            )

        unknown_entries = [
            entry
            for entry in entries
            if entry not in row_by_key
        ]

        if unknown_entries:
            raise RuntimeError(
                f"Unknown entries in {split_name}: "
                f"{unknown_entries[:10]}"
            )

        main_split_sets[split_name] = set(
            entries
        )

    if (
        main_split_sets["train"]
        & main_split_sets["val"]
        or main_split_sets["train"]
        & main_split_sets["test"]
        or main_split_sets["val"]
        & main_split_sets["test"]
    ):
        raise RuntimeError(
            "Overlap detected in main splits."
        )

    train_source_counts = Counter(
        entry.split("/", 1)[0]
        for entry in main_split_sets["train"]
    )

    if (
        dict(train_source_counts)
        != EXPECTED_TRAIN_SOURCE_COUNTS
    ):
        raise RuntimeError(
            "Main train source count mismatch: "
            f"{dict(train_source_counts)}"
        )

    for split_name in ("val", "test"):
        non_rellis = [
            entry
            for entry
            in main_split_sets[split_name]
            if not entry.startswith("rellis3d/")
        ]

        if non_rellis:
            raise RuntimeError(
                f"{split_name} contains non-RELLIS "
                f"samples: {non_rellis[:10]}"
            )

    diagnostic_counts = {}

    for file_name, expected_count in (
        EXPECTED_DIAGNOSTIC_COUNTS.items()
    ):
        entries = read_nonempty_lines(
            root / "splits" / file_name
        )

        if len(entries) != expected_count:
            raise RuntimeError(
                f"{file_name} count mismatch: "
                f"expected={expected_count}, "
                f"actual={len(entries)}"
            )

        if len(entries) != len(set(entries)):
            raise RuntimeError(
                f"Duplicate entries in {file_name}"
            )

        unknown_entries = [
            entry
            for entry in entries
            if entry not in row_by_key
        ]

        if unknown_entries:
            raise RuntimeError(
                f"Unknown entries in {file_name}: "
                f"{unknown_entries[:10]}"
            )

        diagnostic_counts[file_name] = len(
            entries
        )

    result = {
        "status": "PASS",
        "manifest_count": len(rows),
        "source_counts": dict(source_counts),
        "main_split_counts": {
            split: len(entries)
            for split, entries
            in main_split_sets.items()
        },
        "main_train_source_counts": dict(
            train_source_counts
        ),
        "diagnostic_split_counts": (
            diagnostic_counts
        ),
        "observed_ids": {
            source: sorted(ids)
            for source, ids
            in observed_ids.items()
        },
        "pair_validation": {
            "missing_images": 0,
            "missing_masks": 0,
            "size_mismatches": 0,
            "non_single_channel_masks": 0,
            "non_ignore_ratio_mismatches": 0,
        },
        "val_test_policy": "RELLIS-only",
    }

    output_path = (
        root
        / "results"
        / "final_check.json"
    )

    output_path.write_text(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print(
        "Final combined package validation passed."
    )
    print(f"Final check: {output_path}")


if __name__ == "__main__":
    main()
