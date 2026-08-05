from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


ALLOWED_TARGET_IDS = set(range(19)) | {255}

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Add RELLIS-3D Semantic20 to the existing "
            "RUGD/YCOR bridge package and create final "
            "training, validation, and test splits."
        )
    )

    parser.add_argument(
        "--rellis-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--overwrite-rellis",
        action="store_true",
    )

    return parser.parse_args()


def require_file(
    path: Path,
    description: str,
) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"{description} not found: {path}"
        )


def require_dir(
    path: Path,
    description: str,
) -> None:
    if not path.is_dir():
        raise FileNotFoundError(
            f"{description} not found: {path}"
        )


def read_lines(path: Path) -> list[str]:
    require_file(path, "Split file")

    entries = [
        line.strip()
        for line in path.read_text(
            encoding="utf-8-sig"
        ).splitlines()
        if line.strip()
    ]

    if len(entries) != len(set(entries)):
        raise ValueError(
            f"Duplicate split entries: {path}"
        )

    return entries


def write_lines(
    path: Path,
    entries: list[str],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    text = (
        "\n".join(entries) + "\n"
        if entries
        else ""
    )

    path.write_text(
        text,
        encoding="utf-8",
    )


def read_manifest(
    path: Path,
) -> list[dict[str, str]]:
    require_file(path, "Bridge manifest")

    with path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        reader = csv.DictReader(file)

        if reader.fieldnames != MANIFEST_FIELDS:
            raise ValueError(
                f"Unexpected manifest fields: "
                f"{reader.fieldnames}"
            )

        rows = list(reader)

    keys = [
        row["sample_key"]
        for row in rows
    ]

    if len(keys) != len(set(keys)):
        raise ValueError(
            "Duplicate sample_key in bridge manifest."
        )

    return rows


def write_manifest(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=MANIFEST_FIELDS,
        )
        writer.writeheader()
        writer.writerows(rows)


def resolve_rellis_image(
    rellis_root: Path,
    sample_id: str,
) -> Path:
    image_base = (
        rellis_root
        / "images"
        / Path(sample_id)
    )

    candidates = [
        Path(f"{image_base}.jpg"),
        Path(f"{image_base}.jpeg"),
        Path(f"{image_base}.png"),
    ]

    existing = [
        path
        for path in candidates
        if path.is_file()
    ]

    if len(existing) != 1:
        raise FileNotFoundError(
            f"Expected exactly one RELLIS image "
            f"for {sample_id}, found: {existing}"
        )

    return existing[0]


def resolve_rellis_mask(
    rellis_root: Path,
    sample_id: str,
) -> Path:
    mask_path = (
        rellis_root
        / "masks"
        / Path(f"{sample_id}.png")
    )

    require_file(
        mask_path,
        "RELLIS mask",
    )

    return mask_path


def link_or_copy(
    source: Path,
    destination: Path,
) -> str:
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if destination.exists():
        destination.unlink()

    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def inspect_mask(
    path: Path,
) -> tuple[float, set[int]]:
    with Image.open(path) as image:
        image.load()
        mask = np.asarray(image)

    if mask.ndim != 2:
        raise ValueError(
            f"RELLIS mask must be single-channel: "
            f"{path}, shape={mask.shape}"
        )

    observed_ids = {
        int(value)
        for value in np.unique(mask)
    }

    unexpected = (
        observed_ids
        - ALLOWED_TARGET_IDS
    )

    if unexpected:
        raise ValueError(
            f"Unexpected RELLIS target IDs "
            f"in {path}: {sorted(unexpected)}"
        )

    non_ignore_ratio = float(
        np.count_nonzero(mask != 255)
        / mask.size
    )

    return non_ignore_ratio, observed_ids


def relative_posix(
    path: Path,
    root: Path,
) -> str:
    return path.relative_to(
        root
    ).as_posix()


def backup_bridge_files(
    output_root: Path,
) -> Path:
    results_root = (
        output_root
        / "results"
    )
    results_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    current_manifest = (
        output_root
        / "manifest.csv"
    )
    backup_manifest = (
        results_root
        / "bridge_manifest.csv"
    )

    if not backup_manifest.exists():
        require_file(
            current_manifest,
            "Current bridge manifest",
        )
        shutil.copy2(
            current_manifest,
            backup_manifest,
        )

    for split_name in (
        "train",
        "val",
        "test",
    ):
        current_split = (
            output_root
            / "splits"
            / f"{split_name}.txt"
        )
        backup_split = (
            results_root
            / f"bridge_{split_name}.txt"
        )

        if not backup_split.exists():
            require_file(
                current_split,
                f"Bridge {split_name} split",
            )
            shutil.copy2(
                current_split,
                backup_split,
            )

    return backup_manifest


def main() -> None:
    args = parse_args()

    rellis_root = args.rellis_root.resolve()
    output_root = args.output_root.resolve()

    require_dir(
        rellis_root / "images",
        "RELLIS images root",
    )
    require_dir(
        rellis_root / "masks",
        "RELLIS masks root",
    )
    require_dir(
        rellis_root / "splits",
        "RELLIS splits root",
    )
    require_dir(
        output_root,
        "Existing bridge output root",
    )

    bridge_manifest_path = backup_bridge_files(
        output_root
    )

    bridge_rows = read_manifest(
        bridge_manifest_path
    )

    invalid_bridge_sources = sorted(
        {
            row["source"]
            for row in bridge_rows
        }
        - {"rugd", "ycor"}
    )

    if invalid_bridge_sources:
        raise ValueError(
            f"Unexpected bridge sources: "
            f"{invalid_bridge_sources}"
        )

    rellis_image_output = (
        output_root
        / "images"
        / "rellis3d"
    )
    rellis_mask_output = (
        output_root
        / "masks"
        / "rellis3d"
    )

    if (
        rellis_image_output.exists()
        or rellis_mask_output.exists()
    ):
        if not args.overwrite_rellis:
            raise FileExistsError(
                "RELLIS output already exists. "
                "Use --overwrite-rellis."
            )

        shutil.rmtree(
            rellis_image_output,
            ignore_errors=True,
        )
        shutil.rmtree(
            rellis_mask_output,
            ignore_errors=True,
        )

    rellis_rows: list[
        dict[str, Any]
    ] = []

    rellis_main_splits: dict[
        str,
        list[str],
    ] = {
        "train": [],
        "val": [],
        "test": [],
    }

    rellis_storage_modes: Counter[str] = Counter()
    rellis_observed_ids: set[int] = set()
    seen_rellis_ids: set[str] = set()

    for split_name in (
        "train",
        "val",
        "test",
    ):
        split_entries = read_lines(
            rellis_root
            / "splits"
            / f"{split_name}.txt"
        )

        for sample_id in split_entries:
            if sample_id in seen_rellis_ids:
                raise ValueError(
                    f"RELLIS sample occurs in "
                    f"multiple splits: {sample_id}"
                )

            seen_rellis_ids.add(
                sample_id
            )

            source_image = resolve_rellis_image(
                rellis_root,
                sample_id,
            )
            source_mask = resolve_rellis_mask(
                rellis_root,
                sample_id,
            )

            (
                non_ignore_ratio,
                observed_ids,
            ) = inspect_mask(
                source_mask
            )

            rellis_observed_ids.update(
                observed_ids
            )

            image_relative = (
                Path(sample_id)
                .with_suffix(
                    source_image.suffix
                )
            )
            mask_relative = (
                Path(sample_id)
                .with_suffix(".png")
            )

            destination_image = (
                rellis_image_output
                / image_relative
            )
            destination_mask = (
                rellis_mask_output
                / mask_relative
            )

            image_mode = link_or_copy(
                source_image,
                destination_image,
            )
            mask_mode = link_or_copy(
                source_mask,
                destination_mask,
            )

            rellis_storage_modes[
                f"image_{image_mode}"
            ] += 1
            rellis_storage_modes[
                f"mask_{mask_mode}"
            ] += 1

            sample_key = (
                f"rellis3d/{sample_id}"
            )

            rellis_main_splits[
                split_name
            ].append(
                sample_key
            )

            rellis_rows.append(
                {
                    "sample_key": sample_key,
                    "source": "rellis3d",
                    "source_split": split_name,
                    "output_split": split_name,
                    "sample_id": sample_id,
                    "image_path": relative_posix(
                        destination_image,
                        output_root,
                    ),
                    "mask_path": relative_posix(
                        destination_mask,
                        output_root,
                    ),
                    "non_ignore_ratio": (
                        f"{non_ignore_ratio:.8f}"
                    ),
                }
            )

    bridge_by_source_split: dict[
        tuple[str, str],
        list[str],
    ] = {}

    for row in bridge_rows:
        key = (
            row["source"],
            row["output_split"],
        )

        bridge_by_source_split.setdefault(
            key,
            [],
        ).append(
            row["sample_key"]
        )

    bridge_train_rows = [
        row
        for row in bridge_rows
        if row["output_split"] == "train"
    ]

    main_train = (
        rellis_main_splits["train"]
        + [
            row["sample_key"]
            for row in bridge_train_rows
        ]
    )

    main_val = list(
        rellis_main_splits["val"]
    )
    main_test = list(
        rellis_main_splits["test"]
    )

    main_split_sets = {
        "train": set(main_train),
        "val": set(main_val),
        "test": set(main_test),
    }

    if (
        main_split_sets["train"]
        & main_split_sets["val"]
        or main_split_sets["train"]
        & main_split_sets["test"]
        or main_split_sets["val"]
        & main_split_sets["test"]
    ):
        raise ValueError(
            "Overlap detected in final main splits."
        )

    combined_rows = (
        rellis_rows
        + bridge_rows
    )

    combined_keys = [
        row["sample_key"]
        for row in combined_rows
    ]

    if len(combined_keys) != len(
        set(combined_keys)
    ):
        raise ValueError(
            "Duplicate sample_key in "
            "combined manifest."
        )

    split_root = (
        output_root
        / "splits"
    )

    write_lines(
        split_root / "train.txt",
        main_train,
    )
    write_lines(
        split_root / "val.txt",
        main_val,
    )
    write_lines(
        split_root / "test.txt",
        main_test,
    )

    diagnostic_splits = {
        "rugd_val_diagnostic.txt": (
            bridge_by_source_split.get(
                ("rugd", "val"),
                [],
            )
        ),
        "rugd_test_diagnostic.txt": (
            bridge_by_source_split.get(
                ("rugd", "test"),
                [],
            )
        ),
        "ycor_val_diagnostic.txt": (
            bridge_by_source_split.get(
                ("ycor", "val"),
                [],
            )
        ),
        "rugd_train_source.txt": (
            bridge_by_source_split.get(
                ("rugd", "train"),
                [],
            )
        ),
        "ycor_train_source.txt": (
            bridge_by_source_split.get(
                ("ycor", "train"),
                [],
            )
        ),
    }

    for file_name, entries in (
        diagnostic_splits.items()
    ):
        write_lines(
            split_root / file_name,
            entries,
        )

    manifest_path = (
        output_root
        / "manifest.csv"
    )

    write_manifest(
        manifest_path,
        combined_rows,
    )

    source_counts = Counter(
        row["source"]
        for row in combined_rows
    )

    main_train_source_counts = Counter(
        key.split("/", 1)[0]
        for key in main_train
    )

    summary = {
        "dataset_name": (
            "adom_semantic20_"
            "rellis_rugd_ycor_v1"
        ),
        "target_dataset": (
            "rellis3d_semantic20_v1"
        ),
        "num_classes": 19,
        "ignore_index": 255,
        "total_manifest_count": len(
            combined_rows
        ),
        "source_counts": dict(
            source_counts
        ),
        "main_split_counts": {
            "train": len(main_train),
            "val": len(main_val),
            "test": len(main_test),
        },
        "main_train_source_counts": dict(
            main_train_source_counts
        ),
        "diagnostic_split_counts": {
            file_name: len(entries)
            for file_name, entries
            in diagnostic_splits.items()
        },
        "rellis_observed_ids": sorted(
            rellis_observed_ids
        ),
        "rellis_storage_modes": dict(
            rellis_storage_modes
        ),
        "validation_policy": {
            "train": (
                "RELLIS train + RUGD train "
                "+ YCOR train"
            ),
            "val": "RELLIS val only",
            "test": "RELLIS test only",
        },
    }

    summary_path = (
        output_root
        / "results"
        / "combined_package_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("Combined package completed.")
    print(f"Manifest: {manifest_path}")
    print(f"Summary: {summary_path}")
    print()
    print(
        f"Total manifest: "
        f"{len(combined_rows)}"
    )
    print(
        f"Source counts: "
        f"{dict(source_counts)}"
    )
    print(
        "Main splits: "
        f"train={len(main_train)}, "
        f"val={len(main_val)}, "
        f"test={len(main_test)}"
    )
    print(
        "Main train sources: "
        f"{dict(main_train_source_counts)}"
    )
    print(
        "RELLIS observed IDs: "
        f"{sorted(rellis_observed_ids)}"
    )


if __name__ == "__main__":
    main()
