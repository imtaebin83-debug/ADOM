from pathlib import Path
from PIL import Image
from collections import Counter
import numpy as np
import json
import csv

ROOT = Path(r"C:\Users\gahyu\RUGD")

MASK_ROOT = ROOT / "processed" / "annotations"
METADATA_DIR = ROOT / "processed" / "metadata"

METADATA_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CLASS_NAMES = {
    0: "paved_low_cost",
    1: "natural_low_cost",
    2: "medium_cost",
    3: "high_cost_or_obstacle",
    255: "ignore",
}

VALID_IDS = set(CLASS_NAMES)

statistics = {}
csv_rows = []

for split in ("train", "val", "test"):
    mask_dir = MASK_ROOT / split
    mask_paths = sorted(mask_dir.glob("*.png"))

    pixel_counts = Counter()
    images_per_class = Counter()

    for mask_path in mask_paths:
        mask = np.array(Image.open(mask_path))

        if mask.ndim != 2:
            raise RuntimeError(
                f"단일 채널 마스크가 아닙니다: "
                f"{mask_path.name}"
            )

        values, counts = np.unique(
            mask,
            return_counts=True,
        )

        current_ids = {
            int(value)
            for value in values
        }

        invalid_ids = current_ids - VALID_IDS

        if invalid_ids:
            raise RuntimeError(
                f"잘못된 ID 발견: "
                f"{mask_path.name}, {sorted(invalid_ids)}"
            )

        for value, count in zip(values, counts):
            class_id = int(value)
            pixel_counts[class_id] += int(count)
            images_per_class[class_id] += 1

    total_pixels = sum(pixel_counts.values())

    valid_pixels = sum(
        pixel_counts[class_id]
        for class_id in (0, 1, 2, 3)
    )

    split_result = {
        "image_count": len(mask_paths),
        "total_pixels": total_pixels,
        "valid_pixels_excluding_ignore": valid_pixels,
        "classes": {},
    }

    for class_id, class_name in CLASS_NAMES.items():
        count = pixel_counts[class_id]

        total_percentage = (
            count / total_pixels * 100
            if total_pixels > 0
            else 0
        )

        if class_id == 255:
            valid_percentage = None
        else:
            valid_percentage = (
                count / valid_pixels * 100
                if valid_pixels > 0
                else 0
            )

        class_result = {
            "class_id": class_id,
            "class_name": class_name,
            "pixel_count": count,
            "image_count_containing_class":
                images_per_class[class_id],
            "percentage_of_all_pixels":
                round(total_percentage, 4),
            "percentage_excluding_ignore":
                (
                    round(valid_percentage, 4)
                    if valid_percentage is not None
                    else None
                ),
        }

        split_result["classes"][str(class_id)] = (
            class_result
        )

        csv_rows.append({
            "split": split,
            **class_result,
        })

    statistics[split] = split_result

json_path = (
    METADATA_DIR / "class_statistics.json"
)

json_path.write_text(
    json.dumps(
        statistics,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)

csv_path = (
    METADATA_DIR / "class_statistics.csv"
)

with csv_path.open(
    "w",
    newline="",
    encoding="utf-8-sig",
) as file:
    fieldnames = [
        "split",
        "class_id",
        "class_name",
        "pixel_count",
        "image_count_containing_class",
        "percentage_of_all_pixels",
        "percentage_excluding_ignore",
    ]

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames,
    )

    writer.writeheader()
    writer.writerows(csv_rows)

print("클래스 통계 생성 완료")

for split, result in statistics.items():
    print(f"\n[{split}]")
    print("이미지 수:", result["image_count"])

    for class_id in ("0", "1", "2", "3", "255"):
        item = result["classes"][class_id]

        print(
            f"{class_id}: "
            f"{item['class_name']} | "
            f"{item['pixel_count']} pixels | "
            f"{item['percentage_of_all_pixels']}%"
        )

print("\nJSON:", json_path)
print("CSV :", csv_path)