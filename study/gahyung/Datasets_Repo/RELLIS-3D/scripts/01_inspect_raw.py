from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 현재 실제 폴더 이름에 맞춤
RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "Rellis-3D"
REPORT_DIR = PROJECT_ROOT / "reports"

RGB_SUFFIXES = {".jpg", ".jpeg", ".png"}
MASK_SUFFIXES = {".png"}

VALID_RELLIS_IDS = {
    0, 1, 3, 4, 5, 6, 7, 8, 9, 10,
    12, 15, 17, 18, 19, 23, 27, 31, 33, 34,
}


def collect_files(
    directory: Path,
    allowed_suffixes: set[str],
) -> dict[str, Path]:
    """허용된 확장자의 파일을 stem 기준 사전으로 반환한다."""
    if not directory.exists():
        return {}

    files: dict[str, Path] = {}

    for path in sorted(directory.iterdir()):
        if not path.is_file():
            continue

        if path.suffix.lower() not in allowed_suffixes:
            continue

        if path.stem in files:
            raise ValueError(
                f"동일한 stem의 파일이 중복되었습니다: "
                f"{files[path.stem]} / {path}"
            )

        files[path.stem] = path

    return files


def read_id_mask(mask_path: Path) -> tuple[np.ndarray, str]:
    """ID mask를 2차원 배열로 읽는다."""
    with Image.open(mask_path) as image:
        mask = np.asarray(image)

    if mask.ndim == 2:
        return mask, "1-channel"

    if mask.ndim == 3:
        channels_equal = all(
            np.array_equal(mask[:, :, 0], mask[:, :, index])
            for index in range(1, mask.shape[2])
        )

        if not channels_equal:
            raise ValueError(
                "3채널 mask이지만 각 채널의 값이 서로 다릅니다."
            )

        return mask[:, :, 0], f"{mask.shape[2]}-channel-identical"

    raise ValueError(f"지원하지 않는 mask shape: {mask.shape}")


def main(limit: int | None) -> None:
    if not RAW_ROOT.exists():
        raise FileNotFoundError(f"원본 데이터가 없습니다: {RAW_ROOT}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    report_path = REPORT_DIR / "raw_inspection.csv"
    summary_path = REPORT_DIR / "raw_summary.txt"

    rows: list[dict[str, str | int | bool]] = []

    sequence_dirs = sorted(
        path
        for path in RAW_ROOT.iterdir()
        if path.is_dir() and path.name.isdigit()
    )

    processed_count = 0

    for sequence_dir in sequence_dirs:
        sequence = sequence_dir.name

        rgb_dir = sequence_dir / "pylon_camera_node"
        mask_dir = sequence_dir / "pylon_camera_node_label_id"

        rgb_files = collect_files(
            rgb_dir,
            RGB_SUFFIXES,
        )

        mask_files = collect_files(
            mask_dir,
            MASK_SUFFIXES,
        )

        rgb_stems = set(rgb_files)
        mask_stems = set(mask_files)

        paired_stems = sorted(rgb_stems & mask_stems)
        rgb_only_stems = sorted(rgb_stems - mask_stems)
        mask_only_stems = sorted(mask_stems - rgb_stems)

        for stem in paired_stems:
            if limit is not None and processed_count >= limit:
                break

            rgb_path = rgb_files[stem]
            mask_path = mask_files[stem]

            row: dict[str, str | int | bool] = {
                "sequence": sequence,
                "stem": stem,
                "rgb_exists": rgb_path is not None,
                "mask_exists": mask_path is not None,
                "rgb_path": str(rgb_path or ""),
                "mask_path": str(mask_path or ""),
                "rgb_size": "",
                "mask_size": "",
                "size_match": False,
                "mask_dtype": "",
                "mask_channel_type": "",
                "unique_ids": "",
                "unknown_ids": "",
                "status": "",
                "error": "",
            }

            """ if rgb_path is None:
                row["status"] = "missing_rgb"
                rows.append(row)
                processed_count += 1
                continue

            if mask_path is None:
                row["status"] = "missing_mask"
                rows.append(row)
                processed_count += 1
                continue """

            try:
                with Image.open(rgb_path) as rgb_image:
                    rgb_size = rgb_image.size
                    rgb_image.verify()

                with Image.open(mask_path) as mask_image:
                    mask_size = mask_image.size
                    mask_image.verify()

                mask, channel_type = read_id_mask(mask_path)

                unique_ids = sorted(
                    int(value) for value in np.unique(mask)
                )

                unknown_ids = sorted(
                    set(unique_ids) - VALID_RELLIS_IDS
                )

                row["rgb_size"] = f"{rgb_size[0]}x{rgb_size[1]}"
                row["mask_size"] = f"{mask_size[0]}x{mask_size[1]}"
                row["size_match"] = rgb_size == mask_size
                row["mask_dtype"] = str(mask.dtype)
                row["mask_channel_type"] = channel_type
                row["unique_ids"] = ";".join(map(str, unique_ids))
                row["unknown_ids"] = ";".join(map(str, unknown_ids))

                if rgb_size != mask_size:
                    row["status"] = "size_mismatch"
                elif unknown_ids:
                    row["status"] = "unknown_id"
                else:
                    row["status"] = "ok"

            except (OSError, ValueError, UnidentifiedImageError) as error:
                row["status"] = "read_error"
                row["error"] = str(error)

            rows.append(row)
            processed_count += 1

        if limit is not None and processed_count >= limit:
            break

    fieldnames = [
        "sequence",
        "stem",
        "rgb_exists",
        "mask_exists",
        "rgb_path",
        "mask_path",
        "rgb_size",
        "mask_size",
        "size_match",
        "mask_dtype",
        "mask_channel_type",
        "unique_ids",
        "unknown_ids",
        "status",
        "error",
    ]

    with report_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    status_counts: dict[str, int] = {}

    for row in rows:
        status = str(row["status"])
        status_counts[status] = status_counts.get(status, 0) + 1

    with summary_path.open("w", encoding="utf-8") as file:
        file.write(f"RAW_ROOT: {RAW_ROOT}\n")
        file.write(f"검사한 sequence 수: {len(sequence_dirs)}\n")
        file.write(f"검사한 sample 수: {len(rows)}\n\n")

        for status, count in sorted(status_counts.items()):
            file.write(f"{status}: {count}\n")

    print(f"검사 완료: {len(rows)}개")
    print(f"상세 결과: {report_path}")
    print(f"요약 결과: {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="시험용으로 검사할 최대 sample 수",
    )
    args = parser.parse_args()

    main(args.limit)