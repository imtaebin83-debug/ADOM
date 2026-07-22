from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import yaml
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "rellis_cost4_standard"
)

METADATA_PATH = OUTPUT_ROOT / "metadata.csv"
MAPPING_PATH = OUTPUT_ROOT / "class_mapping.yaml"

QC_DIR = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "qc_overlays"
)

VALID_TARGET_IDS = {0, 1, 2, 3, 255}

COLORS = {
    0: (128, 128, 128),   # paved_low_cost
    1: (60, 180, 75),     # natural_low_cost
    2: (255, 165, 0),     # medium_cost
    3: (220, 20, 60),     # high_cost_or_obstacle
    255: (0, 0, 0),       # ignore
}


def create_color_mask(mask: np.ndarray) -> Image.Image:
    color = np.zeros(
        (mask.shape[0], mask.shape[1], 3),
        dtype=np.uint8,
    )

    for class_id, rgb_color in COLORS.items():
        color[mask == class_id] = rgb_color

    return Image.fromarray(color, mode="RGB")


def create_overlay(
    rgb: Image.Image,
    mask: np.ndarray,
) -> Image.Image:
    rgb_rgba = rgb.convert("RGBA")

    overlay_array = np.zeros(
        (mask.shape[0], mask.shape[1], 4),
        dtype=np.uint8,
    )

    for class_id, rgb_color in COLORS.items():
        if class_id == 255:
            continue

        overlay_array[mask == class_id] = (
            rgb_color[0],
            rgb_color[1],
            rgb_color[2],
            115,
        )

    overlay = Image.fromarray(overlay_array, mode="RGBA")

    return Image.alpha_composite(
        rgb_rgba,
        overlay,
    ).convert("RGB")


def resize_panel(
    image: Image.Image,
    width: int = 720,
    nearest: bool = False,
) -> Image.Image:
    ratio = width / image.width
    height = round(image.height * ratio)

    resampling = (
        Image.Resampling.NEAREST
        if nearest
        else Image.Resampling.BILINEAR
    )

    return image.resize((width, height), resampling)


def main(sample_count: int) -> None:
    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            "metadata.csv가 없습니다. "
            "02_convert_masks.py를 먼저 실행하세요."
        )

    with MAPPING_PATH.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    class_names = {
        int(class_id): str(class_name)
        for class_id, class_name
        in config["target_classes"].items()
    }

    with METADATA_PATH.open(
        "r",
        encoding="utf-8-sig",
    ) as file:
        rows = list(csv.DictReader(file))

    pixel_counts = {
        class_id: 0
        for class_id in VALID_TARGET_IDS
    }

    image_counts = {
        class_id: 0
        for class_id in VALID_TARGET_IDS
    }

    qc_rows: list[dict[str, str]] = []
    valid_rows: list[dict[str, str]] = []

    for row in rows:
        if row["status"] != "ok":
            qc_rows.append(
                {
                    "sample_id": row["sample_id"],
                    "status": "conversion_error",
                    "details": row["status"],
                }
            )
            continue

        mask_path = PROJECT_ROOT / row["converted_mask_path"]

        try:
            with Image.open(mask_path) as image:
                mask = np.asarray(image)

            if mask.ndim != 2:
                qc_rows.append(
                    {
                        "sample_id": row["sample_id"],
                        "status": "invalid_channel",
                        "details": f"shape={mask.shape}",
                    }
                )
                continue

            unique_ids = {
                int(value) for value in np.unique(mask)
            }

            invalid_ids = unique_ids - VALID_TARGET_IDS

            if invalid_ids:
                qc_rows.append(
                    {
                        "sample_id": row["sample_id"],
                        "status": "invalid_id",
                        "details": ";".join(
                            map(str, sorted(invalid_ids))
                        ),
                    }
                )
                continue

            if np.all(mask == 255):
                qc_rows.append(
                    {
                        "sample_id": row["sample_id"],
                        "status": "all_ignore",
                        "details": "모든 픽셀이 255",
                    }
                )

            for class_id in VALID_TARGET_IDS:
                count = int(
                    np.count_nonzero(mask == class_id)
                )

                pixel_counts[class_id] += count

                if count > 0:
                    image_counts[class_id] += 1

            valid_rows.append(row)

        except Exception as error:
            qc_rows.append(
                {
                    "sample_id": row["sample_id"],
                    "status": "read_error",
                    "details": str(error),
                }
            )

    total_pixels = sum(pixel_counts.values())

    statistics_path = OUTPUT_ROOT / "class_statistics.csv"

    with statistics_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "class_id",
                "class_name",
                "pixel_count",
                "pixel_ratio",
                "image_count",
            ],
        )

        writer.writeheader()

        for class_id in [0, 1, 2, 3, 255]:
            pixel_ratio = (
                pixel_counts[class_id] / total_pixels
                if total_pixels > 0
                else 0.0
            )

            writer.writerow(
                {
                    "class_id": class_id,
                    "class_name": class_names[class_id],
                    "pixel_count": pixel_counts[class_id],
                    "pixel_ratio": pixel_ratio,
                    "image_count": image_counts[class_id],
                }
            )

    qc_report_path = OUTPUT_ROOT / "qc_report.csv"

    with qc_report_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "sample_id",
                "status",
                "details",
            ],
        )

        writer.writeheader()
        writer.writerows(qc_rows)

    QC_DIR.mkdir(parents=True, exist_ok=True)

    random.seed(42)

    selected_rows = random.sample(
        valid_rows,
        k=min(sample_count, len(valid_rows)),
    )

    for row in selected_rows:
        rgb_path = PROJECT_ROOT / row["rgb_path"]
        mask_path = PROJECT_ROOT / row["converted_mask_path"]

        with Image.open(rgb_path) as rgb_image:
            rgb = rgb_image.convert("RGB")

        with Image.open(mask_path) as mask_image:
            mask = np.asarray(mask_image)

        color_mask = create_color_mask(mask)
        overlay = create_overlay(rgb, mask)

        rgb_small = resize_panel(rgb)
        mask_small = resize_panel(
            color_mask,
            nearest=True,
        )
        overlay_small = resize_panel(overlay)

        panel = Image.new(
            "RGB",
            (
                rgb_small.width * 3,
                rgb_small.height,
            ),
        )

        panel.paste(rgb_small, (0, 0))
        panel.paste(mask_small, (rgb_small.width, 0))
        panel.paste(
            overlay_small,
            (rgb_small.width * 2, 0),
        )

        panel.save(
            QC_DIR / f"{row['sample_id']}.jpg",
            quality=90,
        )

    print(f"통계 파일: {statistics_path}")
    print(f"QC 보고서: {qc_report_path}")
    print(f"QC overlay: {QC_DIR}")
    print(f"QC 문제 수: {len(qc_rows)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--samples",
        type=int,
        default=20,
        help="생성할 QC overlay 수",
    )

    args = parser.parse_args()

    main(args.samples)