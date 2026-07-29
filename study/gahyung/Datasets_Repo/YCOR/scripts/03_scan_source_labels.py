from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import common
from common import (
    OUTPUT_SPLITS,
    discover_dataset_root,
    is_dataset_root,
    load_source_mask,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan source label values in YCOR masks."
    )

    parser.add_argument(
        "--input-root",
        type=Path,
        default=None,
        help="YCOR dataset root containing train/ and valid/.",
    )

    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=common.MANIFEST_DIR,
        help="Directory containing train.csv and val.csv manifests.",
    )

    parser.add_argument(
        "--report-dir",
        type=Path,
        default=common.REPORT_DIR,
        help="Directory for label-scan CSV reports.",
    )

    parser.add_argument(
        "--mapping",
        type=Path,
        default=common.MAPPING_FILE,
        help="YCOR label_mapping.json path.",
    )

    return parser.parse_args()


def resolve_dataset_root(
    input_root: Path | None,
) -> Path:
    if input_root is None:
        return discover_dataset_root()

    dataset_root = input_root.expanduser().resolve()

    if not is_dataset_root(dataset_root):
        raise FileNotFoundError(
            "The selected input root must contain "
            f"non-empty train/ and valid/ folders: {dataset_root}"
        )

    return dataset_root


def resolve_manifest_source(
    dataset_root: Path,
    stored_value: str,
    field_name: str,
) -> Path:
    stored_path = Path(stored_value)

    if stored_path.is_absolute():
        raise ValueError(
            f"Absolute path found in manifest column "
            f"{field_name}: {stored_value}"
        )

    resolved_path = (
        dataset_root
        / stored_path
    ).resolve()

    try:
        resolved_path.relative_to(dataset_root)
    except ValueError as exc:
        raise ValueError(
            f"Manifest path escapes the dataset root: {stored_value}"
        ) from exc

    return resolved_path


def rgb_key_to_text(key: int) -> str:
    return (
        f"{(key >> 16) & 255},"
        f"{(key >> 8) & 255},"
        f"{key & 255}"
    )


def main() -> None:
    args = parse_args()

    dataset_root = resolve_dataset_root(
        args.input_root
    )

    manifest_dir = (
        args.manifest_dir
        .expanduser()
        .resolve()
    )

    report_dir = (
        args.report_dir
        .expanduser()
        .resolve()
    )

    mapping_file = (
        args.mapping
        .expanduser()
        .resolve()
    )

    if not mapping_file.is_file():
        raise FileNotFoundError(
            f"Mapping file not found: {mapping_file}"
        )

    # The original common.py is preserved.
    # The selected mapping is applied only to this process.
    common.MAPPING_FILE = mapping_file

    rgb_mapping, rgb_names = (
        common.load_rgb_mapping()
    )

    index_mapping = (
        common.load_index_mapping()
    )

    all_rows = []
    unknown_rows = []
    encoding_counts = Counter()

    for split in OUTPUT_SPLITS:
        manifest_path = (
            manifest_dir
            / f"{split}.csv"
        )

        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest not found: {manifest_path}\n"
                "Run 02_build_manifest.py first."
            )

        manifest = pd.read_csv(
            manifest_path,
            dtype=str,
        )

        split_rgb_counts = Counter()
        split_index_counts = Counter()

        for _, row in tqdm(
            manifest.iterrows(),
            total=len(manifest),
            desc=f"scan labels {split}",
        ):
            mask_path = resolve_manifest_source(
                dataset_root,
                row["source_mask"],
                "source_mask",
            )

            source, encoding = load_source_mask(
                mask_path
            )

            encoding_counts[encoding] += 1

            if encoding.startswith("rgb"):
                keys = (
                    (
                        source[:, :, 0].astype(
                            np.uint32
                        )
                        << 16
                    )
                    | (
                        source[:, :, 1].astype(
                            np.uint32
                        )
                        << 8
                    )
                    | source[:, :, 2].astype(
                        np.uint32
                    )
                )

                values, counts = np.unique(
                    keys,
                    return_counts=True,
                )

                for value, count in zip(
                    values,
                    counts,
                ):
                    key = int(value)
                    split_rgb_counts[key] += int(
                        count
                    )

                    if key not in rgb_mapping:
                        unknown_rows.append(
                            {
                                "split": split,
                                "sample_id": (
                                    row["sample_id"]
                                ),
                                "mask": (
                                    Path(
                                        row["source_mask"]
                                    ).as_posix()
                                ),
                                "encoding": encoding,
                                "unknown_value": (
                                    rgb_key_to_text(
                                        key
                                    )
                                ),
                            }
                        )
            else:
                values, counts = np.unique(
                    source,
                    return_counts=True,
                )

                for value, count in zip(
                    values,
                    counts,
                ):
                    source_id = int(value)

                    split_index_counts[
                        source_id
                    ] += int(count)

                    if source_id not in index_mapping:
                        unknown_rows.append(
                            {
                                "split": split,
                                "sample_id": (
                                    row["sample_id"]
                                ),
                                "mask": (
                                    Path(
                                        row["source_mask"]
                                    ).as_posix()
                                ),
                                "encoding": encoding,
                                "unknown_value": (
                                    str(source_id)
                                ),
                            }
                        )

        for key, count in sorted(
            split_rgb_counts.items()
        ):
            all_rows.append(
                {
                    "split": split,
                    "encoding": "rgb",
                    "source_value": (
                        rgb_key_to_text(key)
                    ),
                    "source_name": (
                        rgb_names.get(
                            key,
                            "UNKNOWN",
                        )
                    ),
                    "target_id": (
                        rgb_mapping.get(
                            key,
                            "",
                        )
                    ),
                    "pixel_count": count,
                }
            )

        for source_id, count in sorted(
            split_index_counts.items()
        ):
            all_rows.append(
                {
                    "split": split,
                    "encoding": "indexed",
                    "source_value": source_id,
                    "source_name": (
                        f"source_index_{source_id}"
                    ),
                    "target_id": (
                        index_mapping.get(
                            source_id,
                            "",
                        )
                    ),
                    "pixel_count": count,
                }
            )

    report_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    report = pd.DataFrame(all_rows)

    report_path = (
        report_dir
        / "source_label_values.csv"
    )

    report.to_csv(
        report_path,
        index=False,
        encoding="utf-8-sig",
    )

    encoding_path = (
        report_dir
        / "source_encoding_counts.csv"
    )

    pd.DataFrame(
        [
            {
                "encoding": encoding,
                "mask_count": count,
            }
            for encoding, count
            in sorted(
                encoding_counts.items()
            )
        ]
    ).to_csv(
        encoding_path,
        index=False,
        encoding="utf-8-sig",
    )

    unknown_path = (
        report_dir
        / "unknown_source_labels.csv"
    )

    pd.DataFrame(
        unknown_rows
    ).to_csv(
        unknown_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"[encodings] {dict(encoding_counts)}"
    )
    print(f"[saved] {report_path}")
    print(f"[saved] {encoding_path}")
    print(f"[saved] {unknown_path}")

    if unknown_rows:
        raise RuntimeError(
            f"Found {len(unknown_rows)} masks "
            "containing unknown label values. "
            f"Check {unknown_path}."
        )

    print(
        "03_scan_source_labels.py: PASS"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(
            f"\nERROR: {exc}",
            file=sys.stderr,
        )
        raise