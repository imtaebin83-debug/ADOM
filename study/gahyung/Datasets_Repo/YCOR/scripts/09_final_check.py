from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from common import (
    ALLOWED_TARGET_IDS,
    EXPECTED_COUNTS,
    IMAGES_ROOT,
    MASKS_ROOT,
    METADATA_ROOT,
    OUTPUT_SPLITS,
    PROCESSED_ROOT,
)


def main() -> None:
    total_errors = 0
    split_ids = {}

    for split in OUTPUT_SPLITS:
        image_dir = IMAGES_ROOT / split
        mask_dir = MASKS_ROOT / split
        metadata_path = METADATA_ROOT / f"{split}.csv"

        image_paths = sorted(image_dir.glob("*.jpg"))
        mask_paths = sorted(mask_dir.glob("*.png"))
        metadata = pd.read_csv(metadata_path, dtype=str)

        image_stems = {path.stem for path in image_paths}
        mask_stems = {path.stem for path in mask_paths}
        metadata_ids = set(metadata["sample_id"])

        missing_masks = image_stems - mask_stems
        missing_images = mask_stems - image_stems
        metadata_mismatch = (
            image_stems ^ metadata_ids
        ) | (
            mask_stems ^ metadata_ids
        )

        used_ids = set()
        size_errors = 0

        for stem in tqdm(
            sorted(image_stems & mask_stems),
            desc=f"final check {split}",
        ):
            image_path = image_dir / f"{stem}.jpg"
            mask_path = mask_dir / f"{stem}.png"

            with Image.open(image_path) as image:
                image.load()
                image_size = image.size

            with Image.open(mask_path) as mask_image:
                mask = np.asarray(mask_image, dtype=np.uint8)

            mask_size = (mask.shape[1], mask.shape[0])

            if image_size != mask_size:
                size_errors += 1

            used_ids.update(int(v) for v in np.unique(mask))

        invalid_ids = used_ids - ALLOWED_TARGET_IDS
        split_ids[split] = metadata_ids

        expected = EXPECTED_COUNTS[split]
        count_warning = len(image_paths) != expected

        split_errors = (
            len(missing_masks)
            + len(missing_images)
            + len(metadata_mismatch)
            + size_errors
            + len(invalid_ids)
        )
        total_errors += split_errors

        print(f"\n[{split}]")
        print(f"이미지 수: {len(image_paths)}")
        print(f"마스크 수: {len(mask_paths)}")
        print(f"메타데이터 수: {len(metadata)}")
        print(f"공식 예상 수: {expected}")
        print(f"사용된 ID: {sorted(used_ids)}")
        print(f"누락 마스크: {len(missing_masks)}")
        print(f"누락 이미지: {len(missing_images)}")
        print(f"메타데이터 불일치: {len(metadata_mismatch)}")
        print(f"크기 불일치: {size_errors}")
        print(f"잘못된 ID: {sorted(invalid_ids)}")

        if count_warning:
            print(
                "WARNING: 파일 수가 공식 931/145와 다릅니다. "
                "다운로드가 완전한지 확인하세요."
            )

    overlap = split_ids["train"] & split_ids["val"]
    if overlap:
        total_errors += len(overlap)
        print(f"[split overlap] train vs val: {len(overlap)}")

    required_files = [
        PROCESSED_ROOT / "dataset_info.json",
        PROCESSED_ROOT / "mmseg_dataset_snippet.py",
    ]
    missing_required = [path for path in required_files if not path.exists()]
    total_errors += len(missing_required)

    if total_errors:
        raise RuntimeError(f"Final check failed with {total_errors} errors.")

    print("\n참고: YCOR에는 포장도로 클래스가 없어 ID 0이 없어도 정상입니다.")
    print("공식 test split은 없으며 train/val만 생성했습니다.")
    print("09_final_check.py: PASS")
    print(f"학습용 데이터 루트: {PROCESSED_ROOT}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise
