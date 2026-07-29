from __future__ import annotations

## 0728 절대경로 제거 및 CLI·환경변수 기반 통계 생성
import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError
##


## 0728 기존 ADOM Cost4 클래스 정책과 출력 파일명 유지
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_PROCESSED_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "rugd_cost4_standard"
)

SPLIT_NAMES = (
    "train",
    "val",
    "test",
)

CLASS_ID_ORDER = (
    0,
    1,
    2,
    3,
    255,
)

CLASS_NAMES = {
    0: "paved_low_cost",
    1: "natural_low_cost",
    2: "medium_cost",
    3: "high_cost_or_obstacle",
    255: "ignore",
}

VALID_IDS = set(CLASS_NAMES)

JSON_FILE_NAME = "class_statistics.json"
CSV_FILE_NAME = "class_statistics.csv"

CSV_FIELD_NAMES = [
    "split",
    "class_id",
    "class_name",
    "pixel_count",
    "image_count_containing_class",
    "percentage_of_all_pixels",
    "percentage_excluding_ignore",
]
##


## 0728 CLI → 환경변수 → 저장소 상대 기본값 순서로 경로 결정
def resolve_path(
    cli_value: Path | None,
    env_name: str,
    default: Path,
) -> Path:
    value = cli_value or os.getenv(env_name)

    if value is None:
        return default.expanduser().resolve()

    return Path(value).expanduser().resolve()
##


## 0728 split별 PNG mask를 결정적인 이름순으로 수집
def collect_mask_paths(
    mask_directory: Path,
    split_name: str,
) -> list[Path]:
    if not mask_directory.is_dir():
        raise FileNotFoundError(
            "RUGD mask directory does not exist: "
            f"split={split_name}, "
            f"path={mask_directory}"
        )

    mask_paths = sorted(
        mask_directory.glob("*.png")
    )

    if not mask_paths:
        raise RuntimeError(
            "No PNG masks were found: "
            f"split={split_name}, "
            f"path={mask_directory}"
        )

    file_names = [
        mask_path.name
        for mask_path in mask_paths
    ]

    if len(file_names) != len(set(file_names)):
        raise RuntimeError(
            "Duplicate RUGD mask file names were found: "
            f"split={split_name}"
        )

    return mask_paths
##


## 0728 PNG·L mode·단일 채널·Cost4 ID를 검사한 뒤 split 통계 계산
def analyze_split(
    split_name: str,
    mask_paths: list[Path],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    pixel_counts: Counter[int] = Counter()
    images_per_class: Counter[int] = Counter()

    for mask_path in mask_paths:
        try:
            with Image.open(mask_path) as mask_file:
                mask_format = mask_file.format
                mask_mode = mask_file.mode

                mask = np.asarray(
                    mask_file,
                    dtype=np.uint8,
                )

        except (UnidentifiedImageError, OSError) as error:
            raise RuntimeError(
                "Failed to read RUGD mask: "
                f"{mask_path}"
            ) from error

        if mask_format != "PNG":
            raise RuntimeError(
                "Mask is not PNG: "
                f"split={split_name}, "
                f"file={mask_path.name}, "
                f"format={mask_format}"
            )

        if mask_mode != "L":
            raise RuntimeError(
                "Mask mode is not L: "
                f"split={split_name}, "
                f"file={mask_path.name}, "
                f"mode={mask_mode}"
            )

        if mask.ndim != 2:
            raise RuntimeError(
                "Mask is not single-channel: "
                f"split={split_name}, "
                f"file={mask_path.name}, "
                f"shape={mask.shape}"
            )

        values, counts = np.unique(
            mask,
            return_counts=True,
        )

        current_ids = {
            int(value)
            for value in values.tolist()
        }

        invalid_ids = sorted(
            current_ids - VALID_IDS
        )

        if invalid_ids:
            raise RuntimeError(
                "Mask contains invalid IDs: "
                f"split={split_name}, "
                f"file={mask_path.name}, "
                f"invalid={invalid_ids}"
            )

        for value, count in zip(
            values.tolist(),
            counts.tolist(),
        ):
            class_id = int(value)

            pixel_counts[class_id] += int(count)
            images_per_class[class_id] += 1

    total_pixels = sum(
        pixel_counts.values()
    )

    valid_pixels = sum(
        pixel_counts[class_id]
        for class_id in (0, 1, 2, 3)
    )

    if total_pixels <= 0:
        raise RuntimeError(
            "No pixels were counted: "
            f"split={split_name}"
        )

    if valid_pixels <= 0:
        raise RuntimeError(
            "No valid Cost4 pixels were found: "
            f"split={split_name}"
        )

    split_result: dict[str, object] = {
        "image_count": len(mask_paths),
        "total_pixels": total_pixels,
        "valid_pixels_excluding_ignore": valid_pixels,
        "classes": {},
    }

    class_results: dict[str, dict[str, object]] = {}
    csv_rows: list[dict[str, object]] = []

    for class_id in CLASS_ID_ORDER:
        class_name = CLASS_NAMES[class_id]
        pixel_count = pixel_counts[class_id]

        percentage_of_all_pixels = round(
            pixel_count / total_pixels * 100,
            4,
        )

        if class_id == 255:
            percentage_excluding_ignore = None
        else:
            percentage_excluding_ignore = round(
                pixel_count / valid_pixels * 100,
                4,
            )

        class_result: dict[str, object] = {
            "class_id": class_id,
            "class_name": class_name,
            "pixel_count": pixel_count,
            "image_count_containing_class": (
                images_per_class[class_id]
            ),
            "percentage_of_all_pixels": (
                percentage_of_all_pixels
            ),
            "percentage_excluding_ignore": (
                percentage_excluding_ignore
            ),
        }

        class_results[str(class_id)] = class_result

        csv_rows.append(
            {
                "split": split_name,
                **class_result,
            }
        )

    split_result["classes"] = class_results

    return split_result, csv_rows
##


## 0728 통계 합계와 백분율이 서로 일치하는지 재검증
def validate_statistics(
    statistics: dict[str, dict[str, object]],
) -> None:
    for split_name in SPLIT_NAMES:
        result = statistics[split_name]

        image_count = int(result["image_count"])
        total_pixels = int(result["total_pixels"])
        valid_pixels = int(
            result["valid_pixels_excluding_ignore"]
        )

        classes = result["classes"]

        if not isinstance(classes, dict):
            raise RuntimeError(
                "Invalid classes statistics structure: "
                f"split={split_name}"
            )

        class_pixel_sum = sum(
            int(classes[str(class_id)]["pixel_count"])
            for class_id in CLASS_ID_ORDER
        )

        valid_pixel_sum = sum(
            int(classes[str(class_id)]["pixel_count"])
            for class_id in (0, 1, 2, 3)
        )

        if class_pixel_sum != total_pixels:
            raise RuntimeError(
                "Total pixel count mismatch: "
                f"split={split_name}, "
                f"expected={total_pixels}, "
                f"actual={class_pixel_sum}"
            )

        if valid_pixel_sum != valid_pixels:
            raise RuntimeError(
                "Valid pixel count mismatch: "
                f"split={split_name}, "
                f"expected={valid_pixels}, "
                f"actual={valid_pixel_sum}"
            )

        for class_id in CLASS_ID_ORDER:
            containing_count = int(
                classes[str(class_id)][
                    "image_count_containing_class"
                ]
            )

            if containing_count > image_count:
                raise RuntimeError(
                    "Class-containing image count exceeds "
                    "split image count: "
                    f"split={split_name}, "
                    f"class_id={class_id}"
                )

        all_pixel_percentage_sum = sum(
            float(
                classes[str(class_id)][
                    "percentage_of_all_pixels"
                ]
            )
            for class_id in CLASS_ID_ORDER
        )

        if abs(all_pixel_percentage_sum - 100.0) > 0.01:
            raise RuntimeError(
                "All-pixel percentages do not sum to 100: "
                f"split={split_name}, "
                f"sum={all_pixel_percentage_sum}"
            )

        valid_percentage_sum = sum(
            float(
                classes[str(class_id)][
                    "percentage_excluding_ignore"
                ]
            )
            for class_id in (0, 1, 2, 3)
        )

        if abs(valid_percentage_sum - 100.0) > 0.01:
            raise RuntimeError(
                "Valid-pixel percentages do not sum to 100: "
                f"split={split_name}, "
                f"sum={valid_percentage_sum}"
            )
##


## 0728 통계 출력이 mask source 폴더를 침범하지 않는지 검사
def validate_output_location(
    processed_root: Path,
    output_directory: Path,
) -> None:
    processed_root = processed_root.resolve()
    mask_root = (processed_root / "annotations").resolve()
    output_directory = output_directory.resolve()

    if output_directory == processed_root:
        raise ValueError(
            "Statistics output directory must not equal "
            "the processed root."
        )

    if (
        output_directory == mask_root
        or output_directory.is_relative_to(mask_root)
    ):
        raise ValueError(
            "Statistics output directory must not be "
            "the mask root or a directory inside it."
        )
##


## 0728 기존 JSON·CSV를 --overwrite 없이 덮어쓰지 않도록 보호
def prepare_output_paths(
    output_directory: Path,
    overwrite: bool,
) -> tuple[Path, Path]:
    json_path = output_directory / JSON_FILE_NAME
    csv_path = output_directory / CSV_FILE_NAME

    existing_outputs = [
        output_path
        for output_path in (json_path, csv_path)
        if output_path.exists()
    ]

    if existing_outputs and not overwrite:
        raise FileExistsError(
            "RUGD statistics outputs already exist. "
            "Use a new --output-dir or add --overwrite. "
            f"Existing: {[str(path) for path in existing_outputs]}"
        )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    if overwrite:
        for output_path in (json_path, csv_path):
            if output_path.is_file():
                output_path.unlink()

    return json_path, csv_path
##


## 0728 임시 파일 작성 후 교체하여 부분 저장 방지
def write_json_atomic(
    json_path: Path,
    statistics: dict[str, dict[str, object]],
) -> None:
    temporary_path = json_path.with_name(
        json_path.name + ".tmp"
    )

    try:
        temporary_path.write_text(
            json.dumps(
                statistics,
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )

        temporary_path.replace(json_path)

    finally:
        if temporary_path.exists():
            temporary_path.unlink()
##


## 0728 CSV를 결정적인 split·class 순서로 저장
def write_csv_atomic(
    csv_path: Path,
    csv_rows: list[dict[str, object]],
) -> None:
    temporary_path = csv_path.with_name(
        csv_path.name + ".tmp"
    )

    try:
        with temporary_path.open(
            "w",
            newline="",
            encoding="utf-8-sig",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=CSV_FIELD_NAMES,
            )

            writer.writeheader()
            writer.writerows(csv_rows)

        temporary_path.replace(csv_path)

    finally:
        if temporary_path.exists():
            temporary_path.unlink()
##


## 0728 전체 split 통계 생성·검증·저장
def generate_statistics(
    processed_root: Path,
    output_directory: Path,
    overwrite: bool,
) -> None:
    mask_root = processed_root / "annotations"

    validate_output_location(
        processed_root,
        output_directory,
    )

    statistics: dict[str, dict[str, object]] = {}
    csv_rows: list[dict[str, object]] = []

    # 모든 mask를 먼저 검사하여 입력 오류 시
    # 기존 결과 또는 새 출력이 생성되지 않게 한다.
    for split_name in SPLIT_NAMES:
        mask_paths = collect_mask_paths(
            mask_root / split_name,
            split_name,
        )

        split_result, split_csv_rows = analyze_split(
            split_name,
            mask_paths,
        )

        statistics[split_name] = split_result
        csv_rows.extend(split_csv_rows)

    validate_statistics(statistics)

    json_path, csv_path = prepare_output_paths(
        output_directory,
        overwrite=overwrite,
    )

    write_json_atomic(
        json_path,
        statistics,
    )

    write_csv_atomic(
        csv_path,
        csv_rows,
    )

    expected_csv_rows = (
        len(SPLIT_NAMES)
        * len(CLASS_ID_ORDER)
    )

    if len(csv_rows) != expected_csv_rows:
        raise RuntimeError(
            "Statistics CSV row count mismatch: "
            f"expected={expected_csv_rows}, "
            f"actual={len(csv_rows)}"
        )

    if not json_path.is_file():
        raise RuntimeError(
            f"Statistics JSON was not generated: {json_path}"
        )

    if not csv_path.is_file():
        raise RuntimeError(
            f"Statistics CSV was not generated: {csv_path}"
        )

    print("")
    print("RUGD class statistics summary")

    for split_name in SPLIT_NAMES:
        result = statistics[split_name]

        print("")
        print(f"[{split_name}]")
        print(f"Images: {result['image_count']}")
        print(f"Total pixels: {result['total_pixels']}")
        print(
            "Valid pixels excluding ignore: "
            f"{result['valid_pixels_excluding_ignore']}"
        )

        classes = result["classes"]

        if not isinstance(classes, dict):
            raise RuntimeError(
                "Invalid classes statistics structure."
            )

        for class_id in CLASS_ID_ORDER:
            class_result = classes[str(class_id)]

            print(
                f"{class_id}: "
                f"{class_result['class_name']} | "
                f"{class_result['pixel_count']} pixels | "
                f"{class_result['percentage_of_all_pixels']}%"
            )

    print("")
    print(f"JSON: {json_path}")
    print(f"CSV: {csv_path}")
    print(
        "[PASS] RUGD class statistics generation "
        "completed."
    )
##


## 0728 CLI·환경변수 기반 실행 진입점
def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate validated RUGD Cost4 "
            "class statistics."
        )
    )

    parser.add_argument(
        "--processed-root",
        type=Path,
        default=None,
        help=(
            "Processed RUGD root containing "
            "annotations/train|val|test. "
            "Environment variable: RUGD_OUTPUT_ROOT"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for class_statistics.json "
            "and class_statistics.csv. "
            "Environment variable: "
            "RUGD_STATISTICS_OUTPUT_DIR. "
            "Default: <processed-root>/metadata"
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Replace existing generated statistics "
            "JSON and CSV files."
        ),
    )

    args = parser.parse_args()

    processed_root = resolve_path(
        args.processed_root,
        "RUGD_OUTPUT_ROOT",
        DEFAULT_PROCESSED_ROOT,
    )

    output_environment = os.getenv(
        "RUGD_STATISTICS_OUTPUT_DIR"
    )

    if args.output_dir is not None:
        output_directory = (
            args.output_dir
            .expanduser()
            .resolve()
        )

    elif output_environment:
        output_directory = (
            Path(output_environment)
            .expanduser()
            .resolve()
        )

    else:
        output_directory = (
            processed_root
            / "metadata"
        ).resolve()

    if not processed_root.is_dir():
        raise FileNotFoundError(
            "Processed root does not exist: "
            f"{processed_root}"
        )

    generate_statistics(
        processed_root=processed_root,
        output_directory=output_directory,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
##
