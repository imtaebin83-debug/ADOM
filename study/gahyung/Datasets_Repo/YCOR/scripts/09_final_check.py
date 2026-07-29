from __future__ import annotations

import argparse
import sys
from pathlib import Path, PureWindowsPath

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from common import (
    ALLOWED_TARGET_IDS,
    EXPECTED_COUNTS,
    OUTPUT_SPLITS,
    PROCESSED_ROOT,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the final validation for processed YCOR."
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROCESSED_ROOT,
        help=(
            "Processed YCOR_ADOM directory containing "
            "images/, masks/, metadata/, and training information."
        ),
    )

    return parser.parse_args()


def is_absolute_path_text(
    value: object,
) -> bool:
    text = str(value)

    return (
        Path(text).is_absolute()
        or PureWindowsPath(text).is_absolute()
    )


def main() -> None:
    args = parse_args()

    processed_root = (
        args.output_root
        .expanduser()
        .resolve()
    )

    images_root = (
        processed_root
        / "images"
    )

    masks_root = (
        processed_root
        / "masks"
    )

    metadata_root = (
        processed_root
        / "metadata"
    )

    total_errors = 0
    split_ids = {}

    for split in OUTPUT_SPLITS:
        image_dir = (
            images_root
            / split
        )

        mask_dir = (
            masks_root
            / split
        )

        metadata_path = (
            metadata_root
            / f"{split}.csv"
        )

        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"Metadata not found: {metadata_path}"
            )

        image_paths = sorted(
            image_dir.glob("*.jpg")
        )

        mask_paths = sorted(
            mask_dir.glob("*.png")
        )

        metadata = pd.read_csv(
            metadata_path,
            dtype=str,
        )

        required_columns = {
            "sample_id",
            "image_relpath",
            "mask_relpath",
            "source_image",
            "source_mask",
        }

        missing_columns = (
            required_columns
            - set(metadata.columns)
        )

        if missing_columns:
            raise ValueError(
                f"{split} metadata columns are missing: "
                f"{sorted(missing_columns)}"
            )

        image_stems = {
            path.stem
            for path in image_paths
        }

        mask_stems = {
            path.stem
            for path in mask_paths
        }

        metadata_ids = set(
            metadata["sample_id"]
        )

        missing_masks = (
            image_stems
            - mask_stems
        )

        missing_images = (
            mask_stems
            - image_stems
        )

        metadata_mismatch = (
            image_stems
            ^ metadata_ids
        ) | (
            mask_stems
            ^ metadata_ids
        )

        duplicate_metadata_ids = int(
            metadata[
                "sample_id"
            ].duplicated().sum()
        )

        absolute_metadata_paths = 0

        for column in (
            "image_relpath",
            "mask_relpath",
            "source_image",
            "source_mask",
        ):
            absolute_metadata_paths += int(
                metadata[column]
                .map(
                    is_absolute_path_text
                )
                .sum()
            )

        used_ids = set()
        size_errors = 0

        for stem in tqdm(
            sorted(
                image_stems
                & mask_stems
            ),
            desc=f"final check {split}",
        ):
            image_path = (
                image_dir
                / f"{stem}.jpg"
            )

            mask_path = (
                mask_dir
                / f"{stem}.png"
            )

            with Image.open(
                image_path
            ) as image:
                image.load()
                image_size = image.size

            with Image.open(
                mask_path
            ) as mask_image:
                mask = np.asarray(
                    mask_image,
                    dtype=np.uint8,
                )

            mask_size = (
                mask.shape[1],
                mask.shape[0],
            )

            if image_size != mask_size:
                size_errors += 1

            used_ids.update(
                int(value)
                for value
                in np.unique(mask)
            )

        invalid_ids = (
            used_ids
            - ALLOWED_TARGET_IDS
        )

        split_ids[split] = metadata_ids

        expected = EXPECTED_COUNTS[
            split
        ]

        image_count_error = int(
            len(image_paths)
            != expected
        )

        mask_count_error = int(
            len(mask_paths)
            != expected
        )

        metadata_count_error = int(
            len(metadata)
            != expected
        )

        split_errors = (
            len(missing_masks)
            + len(missing_images)
            + len(metadata_mismatch)
            + duplicate_metadata_ids
            + absolute_metadata_paths
            + size_errors
            + len(invalid_ids)
            + image_count_error
            + mask_count_error
            + metadata_count_error
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
        print(
            f"메타데이터 불일치: "
            f"{len(metadata_mismatch)}"
        )
        print(
            f"메타데이터 중복 ID: "
            f"{duplicate_metadata_ids}"
        )
        print(
            f"메타데이터 절대경로: "
            f"{absolute_metadata_paths}"
        )
        print(f"크기 불일치: {size_errors}")
        print(f"잘못된 ID: {sorted(invalid_ids)}")

        if (
            image_count_error
            or mask_count_error
            or metadata_count_error
        ):
            print(
                "ERROR: image, mask, or metadata count "
                "does not match the official split count."
            )

    overlap = (
        split_ids["train"]
        & split_ids["val"]
    )

    if overlap:
        total_errors += len(overlap)

        print(
            f"[split overlap] "
            f"train vs val: {len(overlap)}"
        )

    required_files = [
        processed_root
        / "dataset_info.json",
        processed_root
        / "mmseg_dataset_snippet.py",
    ]

    missing_required = [
        path
        for path in required_files
        if not path.exists()
    ]

    total_errors += len(
        missing_required
    )

    if missing_required:
        print(
            "[missing required files] "
            + ", ".join(
                path.name
                for path in missing_required
            )
        )

    if total_errors:
        raise RuntimeError(
            "Final check failed with "
            f"{total_errors} errors."
        )

    print(
        "\n참고: YCOR에는 포장도로 클래스가 없어 "
        "ID 0이 없어도 정상입니다."
    )

    print(
        "공식 test split은 없으며 "
        "train/val만 생성했습니다."
    )

    print(
        "09_final_check.py: PASS"
    )

    print(
        "학습용 데이터 루트: "
        "<YCOR_PROCESSED_ROOT>"
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