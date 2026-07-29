from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from common import (
    OUTPUT_SPLITS,
    PROCESSED_ROOT,
    TARGET_CLASSES,
)


COUNT_IDS = (0, 1, 2, 3, 255)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate class-distribution and "
            "per-image statistics for processed YCOR."
        )
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROCESSED_ROOT,
        help=(
            "Processed YCOR_ADOM directory containing "
            "masks/ and metadata/."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    processed_root = (
        args.output_root
        .expanduser()
        .resolve()
    )

    masks_root = (
        processed_root
        / "masks"
    )

    metadata_root = (
        processed_root
        / "metadata"
    )

    qc_root = (
        processed_root
        / "qc"
    )

    qc_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    split_rows = []
    image_rows = []

    for split in OUTPUT_SPLITS:
        metadata_path = (
            metadata_root
            / f"{split}.csv"
        )

        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata not found: {metadata_path}\n"
                "Run 05_convert_dataset.py first."
            )

        metadata = pd.read_csv(
            metadata_path,
            dtype=str,
        )

        pixel_counts = {
            class_id: 0
            for class_id in COUNT_IDS
        }

        for _, row in tqdm(
            metadata.iterrows(),
            total=len(metadata),
            desc=f"QC {split}",
        ):
            mask_path = (
                masks_root
                / split
                / row["mask_filename"]
            )

            if not mask_path.is_file():
                raise FileNotFoundError(
                    f"Processed mask not found: {mask_path}"
                )

            with Image.open(mask_path) as image:
                mask = np.asarray(
                    image,
                    dtype=np.uint8,
                )

            counts = np.bincount(
                mask.ravel(),
                minlength=256,
            )

            used_ids = [
                class_id
                for class_id in COUNT_IDS
                if counts[class_id] > 0
            ]

            for class_id in COUNT_IDS:
                pixel_counts[class_id] += int(
                    counts[class_id]
                )

            image_rows.append(
                {
                    "split": split,
                    "sample_id": row["sample_id"],
                    "source_sample_name": (
                        row["source_sample_name"]
                    ),
                    "used_ids": " ".join(
                        map(
                            str,
                            used_ids,
                        )
                    ),
                    "paved_pixels": int(
                        counts[0]
                    ),
                    "natural_pixels": int(
                        counts[1]
                    ),
                    "medium_pixels": int(
                        counts[2]
                    ),
                    "high_pixels": int(
                        counts[3]
                    ),
                    "ignore_pixels": int(
                        counts[255]
                    ),
                }
            )

        total_pixels = sum(
            pixel_counts.values()
        )

        if total_pixels == 0:
            raise RuntimeError(
                f"No mask pixels were counted for {split}."
            )

        for class_id in COUNT_IDS:
            split_rows.append(
                {
                    "split": split,
                    "class_id": class_id,
                    "class_name": (
                        TARGET_CLASSES[
                            class_id
                        ]
                    ),
                    "pixel_count": (
                        pixel_counts[
                            class_id
                        ]
                    ),
                    "pixel_ratio": (
                        pixel_counts[class_id]
                        / total_pixels
                    ),
                }
            )

        print(f"[{split}]")

        for class_id in COUNT_IDS:
            ratio = (
                100.0
                * pixel_counts[class_id]
                / total_pixels
            )

            print(
                f"  {class_id:>3} "
                f"{TARGET_CLASSES[class_id]:<24} "
                f"{pixel_counts[class_id]:>15,} "
                f"({ratio:6.2f}%)"
            )

    distribution = pd.DataFrame(
        split_rows
    )

    distribution_path = (
        qc_root
        / "class_distribution.csv"
    )

    distribution.to_csv(
        distribution_path,
        index=False,
        encoding="utf-8-sig",
    )

    image_stats = pd.DataFrame(
        image_rows
    )

    image_stats_path = (
        qc_root
        / "per_image_class_counts.csv"
    )

    image_stats.to_csv(
        image_stats_path,
        index=False,
        encoding="utf-8-sig",
    )

    summary = {
        "dataset": "YCOR_ADOM",
        "classes": {
            str(class_id): class_name
            for class_id, class_name
            in TARGET_CLASSES.items()
        },
        "splits": {
            split: int(
                (
                    image_stats["split"]
                    == split
                ).sum()
            )
            for split in OUTPUT_SPLITS
        },
        "warning": (
            "YCOR has no paved/asphalt source class; "
            "target ID 0 may have zero pixels."
        ),
    }

    summary_path = (
        qc_root
        / "dataset_summary.json"
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\n[saved] {distribution_path}")
    print(f"[saved] {image_stats_path}")
    print(f"[saved] {summary_path}")
    print("06_qc_statistics.py: PASS")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(
            f"\nERROR: {exc}",
            file=sys.stderr,
        )
        raise