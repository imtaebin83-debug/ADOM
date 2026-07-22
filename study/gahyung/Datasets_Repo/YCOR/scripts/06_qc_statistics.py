from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from common import (
    MASKS_ROOT,
    METADATA_ROOT,
    OUTPUT_SPLITS,
    QC_ROOT,
    TARGET_CLASSES,
)


COUNT_IDS = (0, 1, 2, 3, 255)


def main() -> None:
    split_rows = []
    image_rows = []

    for split in OUTPUT_SPLITS:
        metadata_path = METADATA_ROOT / f"{split}.csv"
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata not found: {metadata_path}\n"
                "Run 05_convert_dataset.py first."
            )

        metadata = pd.read_csv(metadata_path, dtype=str)
        pixel_counts = {class_id: 0 for class_id in COUNT_IDS}

        for _, row in tqdm(
            metadata.iterrows(),
            total=len(metadata),
            desc=f"QC {split}",
        ):
            mask_path = MASKS_ROOT / split / row["mask_filename"]

            with Image.open(mask_path) as image:
                mask = np.asarray(image, dtype=np.uint8)

            counts = np.bincount(mask.ravel(), minlength=256)
            used_ids = [
                class_id for class_id in COUNT_IDS
                if counts[class_id] > 0
            ]

            for class_id in COUNT_IDS:
                pixel_counts[class_id] += int(counts[class_id])

            image_rows.append(
                {
                    "split": split,
                    "sample_id": row["sample_id"],
                    "source_sample_name": row["source_sample_name"],
                    "used_ids": " ".join(map(str, used_ids)),
                    "paved_pixels": int(counts[0]),
                    "natural_pixels": int(counts[1]),
                    "medium_pixels": int(counts[2]),
                    "high_pixels": int(counts[3]),
                    "ignore_pixels": int(counts[255]),
                }
            )

        total_pixels = sum(pixel_counts.values())

        for class_id in COUNT_IDS:
            split_rows.append(
                {
                    "split": split,
                    "class_id": class_id,
                    "class_name": TARGET_CLASSES[class_id],
                    "pixel_count": pixel_counts[class_id],
                    "pixel_ratio": (
                        pixel_counts[class_id] / total_pixels
                        if total_pixels
                        else 0.0
                    ),
                }
            )

        print(f"[{split}]")
        for class_id in COUNT_IDS:
            ratio = (
                100.0 * pixel_counts[class_id] / total_pixels
                if total_pixels
                else 0.0
            )
            print(
                f"  {class_id:>3} {TARGET_CLASSES[class_id]:<24} "
                f"{pixel_counts[class_id]:>15,} ({ratio:6.2f}%)"
            )

    distribution = pd.DataFrame(split_rows)
    distribution_path = QC_ROOT / "class_distribution.csv"
    distribution.to_csv(
        distribution_path,
        index=False,
        encoding="utf-8-sig",
    )

    image_stats = pd.DataFrame(image_rows)
    image_stats_path = QC_ROOT / "per_image_class_counts.csv"
    image_stats.to_csv(
        image_stats_path,
        index=False,
        encoding="utf-8-sig",
    )

    summary = {
        "dataset": "YCOR_ADOM",
        "classes": {str(k): v for k, v in TARGET_CLASSES.items()},
        "splits": {
            split: int((image_stats["split"] == split).sum())
            for split in OUTPUT_SPLITS
        },
        "warning": (
            "YCOR has no paved/asphalt source class; target ID 0 may have "
            "zero pixels."
        ),
    }
    summary_path = QC_ROOT / "dataset_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n[saved] {distribution_path}")
    print(f"[saved] {image_stats_path}")
    print(f"[saved] {summary_path}")
    print("06_qc_statistics.py: PASS")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise
