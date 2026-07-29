from __future__ import annotations

## 0728 절대경로 제거와 CLI·환경변수 지원을 위한 모듈 추가
import argparse
import csv
import os
from collections import Counter
from pathlib import Path, PureWindowsPath

import numpy as np
from PIL import Image
from tqdm import tqdm
##


## 0728 저장소 상대 기본 경로와 검증 기준 정의
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_PROCESSED_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "rugd_cost4_standard"
)

IMAGE_RELATIVE_DIR = (
    Path("images")
    / "all"
)

MASK_RELATIVE_DIR = (
    Path("annotations")
    / "all"
)

DEFAULT_METADATA_FILE_NAME = "metadata.csv"
QC_REPORT_FILE_NAME = "qc_report.csv"

VALID_TARGET_IDS = {
    0,
    1,
    2,
    3,
    255,
}

REQUIRED_METADATA_COLUMNS = {
    "sample_id",
    "file_name",
    "rgb_path",
    "source_color_mask_path",
    "converted_image_path",
    "converted_mask_path",
    "width",
    "height",
    "status",
    "details",
}

QC_FIELDS = [
    "sample_id",
    "status",
    "details",
]
##


## 0728 CLI → 환경변수 → 기본값 순서로 경로 결정
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


## 0728 metadata 또는 결과 파일에 절대경로가 포함됐는지 검사
def is_absolute_path_text(
    value: str,
) -> bool:
    stripped_value = value.strip()

    if not stripped_value:
        return False

    return (
        Path(stripped_value).is_absolute()
        or PureWindowsPath(
            stripped_value
        ).is_absolute()
    )
##


## 0728 PNG 파일을 파일 이름 기준으로 수집
def collect_png_map(
    directory: Path,
    description: str,
) -> dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(
            f"{description} directory does not exist: "
            f"{directory}"
        )

    file_map: dict[str, Path] = {}

    for file_path in sorted(
        directory.glob("*.png")
    ):
        if file_path.name in file_map:
            raise RuntimeError(
                f"Duplicate {description} file name: "
                f"{file_path.name}"
            )

        file_map[file_path.name] = file_path

    return file_map
##


## 0728 metadata.csv 필수 열·중복·절대경로 검사
def load_metadata(
    metadata_path: Path,
) -> tuple[
    dict[str, dict[str, str]],
    dict[str, str],
]:
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Metadata file does not exist: "
            f"{metadata_path}"
        )

    with metadata_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        actual_columns = set(
            reader.fieldnames or []
        )

        missing_columns = sorted(
            REQUIRED_METADATA_COLUMNS
            - actual_columns
        )

        if missing_columns:
            raise RuntimeError(
                "Metadata is missing required columns: "
                f"{missing_columns}"
            )

        metadata_map: dict[
            str,
            dict[str, str]
        ] = {}

        metadata_issues: dict[
            str,
            str
        ] = {}

        for row_number, row in enumerate(
            reader,
            start=2,
        ):
            file_name = (
                row.get("file_name", "")
                .strip()
            )

            if not file_name:
                metadata_issues[
                    f"__row_{row_number}__"
                ] = (
                    "metadata_missing_file_name: "
                    f"row={row_number}"
                )

                continue

            if file_name in metadata_map:
                metadata_issues[
                    file_name
                ] = (
                    "metadata_duplicate_file_name: "
                    f"row={row_number}"
                )

                continue

            absolute_path_fields = []

            for field_name in (
                "rgb_path",
                "source_color_mask_path",
                "converted_image_path",
                "converted_mask_path",
            ):
                field_value = (
                    row.get(field_name, "")
                    .strip()
                )

                if is_absolute_path_text(
                    field_value
                ):
                    absolute_path_fields.append(
                        field_name
                    )

            if absolute_path_fields:
                metadata_issues[
                    file_name
                ] = (
                    "metadata_absolute_path: "
                    + ", ".join(
                        absolute_path_fields
                    )
                )

            metadata_map[file_name] = row

    if not metadata_map:
        raise RuntimeError(
            "Metadata contains no usable rows."
        )

    return (
        metadata_map,
        metadata_issues,
    )
##


## 0728 한 sample의 파일·크기·채널·ID·ignore 여부 검증
def validate_sample(
    file_name: str,
    image_path: Path | None,
    mask_path: Path | None,
    metadata_row: dict[str, str] | None,
    metadata_issue: str | None,
) -> tuple[str, str]:
    if image_path is None:
        return (
            "missing_image",
            "Processed RGB image is missing.",
        )

    if mask_path is None:
        return (
            "missing_mask",
            "Processed mask is missing.",
        )

    if metadata_row is None:
        return (
            "missing_metadata",
            "Metadata row is missing.",
        )

    if metadata_issue is not None:
        return (
            "metadata_error",
            metadata_issue,
        )

    metadata_status = (
        metadata_row.get(
            "status",
            "",
        ).strip()
    )

    if metadata_status != "ok":
        return (
            "metadata_conversion_error",
            (
                "metadata status is not ok: "
                f"status={metadata_status}, "
                "details="
                f"{metadata_row.get('details', '')}"
            ),
        )

    expected_image_path = (
        IMAGE_RELATIVE_DIR
        / file_name
    ).as_posix()

    expected_mask_path = (
        MASK_RELATIVE_DIR
        / file_name
    ).as_posix()

    actual_image_path = (
        metadata_row.get(
            "converted_image_path",
            "",
        )
        .strip()
        .replace("\\", "/")
    )

    actual_mask_path = (
        metadata_row.get(
            "converted_mask_path",
            "",
        )
        .strip()
        .replace("\\", "/")
    )

    if (
        actual_image_path
        != expected_image_path
    ):
        return (
            "metadata_path_mismatch",
            (
                "converted_image_path mismatch: "
                f"expected={expected_image_path}, "
                f"actual={actual_image_path}"
            ),
        )

    if (
        actual_mask_path
        != expected_mask_path
    ):
        return (
            "metadata_path_mismatch",
            (
                "converted_mask_path mismatch: "
                f"expected={expected_mask_path}, "
                f"actual={actual_mask_path}"
            ),
        )

    try:
        with Image.open(
            image_path
        ) as image:
            image.load()
            image_size = image.size

        with Image.open(
            mask_path
        ) as mask_image:
            mask_image.load()

            mask_format = mask_image.format
            mask_mode = mask_image.mode
            mask_size = mask_image.size

            mask = np.asarray(
                mask_image
            )

    except OSError as exc:
        return (
            "read_error",
            (
                f"{type(exc).__name__}: "
                f"{exc}"
            ),
        )

    if mask_format != "PNG":
        return (
            "invalid_format",
            (
                "Mask format is not PNG: "
                f"{mask_format}"
            ),
        )

    if (
        mask_mode != "L"
        or mask.ndim != 2
    ):
        return (
            "invalid_channel",
            (
                "Mask must be a single-channel "
                "mode-L image: "
                f"mode={mask_mode}, "
                f"shape={mask.shape}"
            ),
        )

    if image_size != mask_size:
        return (
            "size_mismatch",
            (
                "RGB and mask dimensions differ: "
                f"image={image_size}, "
                f"mask={mask_size}"
            ),
        )

    unique_ids = {
        int(value)
        for value in np.unique(mask)
    }

    invalid_ids = (
        unique_ids
        - VALID_TARGET_IDS
    )

    if invalid_ids:
        return (
            "invalid_id",
            (
                "Mask contains invalid target IDs: "
                f"{sorted(invalid_ids)}"
            ),
        )

    if unique_ids == {255}:
        return (
            "all_ignore",
            "Mask contains only ignore ID 255.",
        )

    return (
        "ok",
        "",
    )
##


## 0728 QC 결과를 CSV로 저장
def write_qc_report(
    qc_report_path: Path,
    rows: list[dict[str, str]],
) -> None:
    qc_report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with qc_report_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=QC_FIELDS,
        )

        writer.writeheader()
        writer.writerows(rows)
##


## 0728 기존 QC 파일 덮어쓰기 방지
def prepare_qc_report_path(
    results_directory: Path,
    overwrite: bool,
) -> Path:
    qc_report_path = (
        results_directory
        / QC_REPORT_FILE_NAME
    )

    if (
        qc_report_path.exists()
        and not overwrite
    ):
        raise FileExistsError(
            "QC report already exists. "
            "Use a new --results-dir or add "
            f"--overwrite: {qc_report_path}"
        )

    results_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    return qc_report_path
##


## 0728 전체 processed dataset을 검사하고 실패 시 nonzero exit
def validate_processed_dataset(
    processed_root: Path,
    metadata_path: Path,
    results_directory: Path,
    overwrite: bool,
) -> None:
    image_directory = (
        processed_root
        / IMAGE_RELATIVE_DIR
    )

    mask_directory = (
        processed_root
        / MASK_RELATIVE_DIR
    )

    image_map = collect_png_map(
        image_directory,
        "processed RGB image",
    )

    mask_map = collect_png_map(
        mask_directory,
        "processed mask",
    )

    (
        metadata_map,
        metadata_issues,
    ) = load_metadata(
        metadata_path
    )

    all_file_names = sorted(
        set(image_map)
        | set(mask_map)
        | set(metadata_map)
        | {
            key
            for key in metadata_issues
            if not key.startswith("__row_")
        }
    )

    if not all_file_names:
        raise RuntimeError(
            "No processed RUGD samples were found."
        )

    qc_report_path = (
        prepare_qc_report_path(
            results_directory,
            overwrite=overwrite,
        )
    )

    qc_rows: list[
        dict[str, str]
    ] = []

    for file_name in tqdm(
        all_file_names,
        desc="Validating RUGD processed data",
    ):
        status, details = (
            validate_sample(
                file_name=file_name,
                image_path=image_map.get(
                    file_name
                ),
                mask_path=mask_map.get(
                    file_name
                ),
                metadata_row=metadata_map.get(
                    file_name
                ),
                metadata_issue=metadata_issues.get(
                    file_name
                ),
            )
        )

        qc_rows.append(
            {
                "sample_id": Path(
                    file_name
                ).stem,
                "status": status,
                "details": details,
            }
        )

    for key, details in sorted(
        metadata_issues.items()
    ):
        if not key.startswith("__row_"):
            continue

        qc_rows.append(
            {
                "sample_id": key,
                "status": "metadata_error",
                "details": details,
            }
        )

    write_qc_report(
        qc_report_path,
        qc_rows,
    )

    status_counts = Counter(
        row["status"]
        for row in qc_rows
    )

    failure_rows = [
        row
        for row in qc_rows
        if row["status"] != "ok"
    ]

    print("")
    print("RUGD processed validation summary")
    print(
        f"Processed RGB images: "
        f"{len(image_map)}"
    )
    print(
        f"Processed masks: "
        f"{len(mask_map)}"
    )
    print(
        f"Metadata rows: "
        f"{len(metadata_map)}"
    )
    print(
        f"QC rows: "
        f"{len(qc_rows)}"
    )

    print("")
    print("QC status counts")

    for status in sorted(
        status_counts
    ):
        print(
            f"{status}: "
            f"{status_counts[status]}"
        )

    print("")
    print(
        f"QC failures: "
        f"{len(failure_rows)}"
    )

    print(
        f"QC report: "
        f"{qc_report_path}"
    )

    if not qc_rows:
        raise RuntimeError(
            "QC report contains no rows."
        )

    if failure_rows:
        examples = "\n".join(
            (
                f"- {row['sample_id']}: "
                f"{row['status']} / "
                f"{row['details']}"
            )
            for row in failure_rows[:10]
        )

        raise RuntimeError(
            "RUGD processed validation failed "
            f"for {len(failure_rows)} sample(s). "
            "See qc_report.csv.\n"
            f"{examples}"
        )

    if len(image_map) != len(mask_map):
        raise RuntimeError(
            "Processed image and mask counts differ "
            "despite zero QC failures."
        )

    if len(qc_rows) != len(image_map):
        raise RuntimeError(
            "QC row count does not match "
            "processed image count."
        )

    print(
        "[PASS] RUGD processed validation completed."
    )
##


## 0728 CLI·환경변수 기반 실행 진입점 추가
def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate processed RUGD images, "
            "Cost4 masks, and metadata."
        )
    )

    parser.add_argument(
        "--processed-root",
        type=Path,
        default=None,
        help=(
            "Processed RUGD dataset root. "
            "Environment variable: "
            "RUGD_OUTPUT_ROOT"
        ),
    )

    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help=(
            "Path to metadata.csv. "
            "Environment variable: "
            "RUGD_METADATA_PATH"
        ),
    )

    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help=(
            "Directory for qc_report.csv. "
            "Environment variable: "
            "RUGD_RESULTS_DIR"
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Replace an existing qc_report.csv."
        ),
    )

    args = parser.parse_args()

    processed_root = resolve_path(
        args.processed_root,
        "RUGD_OUTPUT_ROOT",
        DEFAULT_PROCESSED_ROOT,
    )

    metadata_path = resolve_path(
        args.metadata,
        "RUGD_METADATA_PATH",
        (
            processed_root
            / DEFAULT_METADATA_FILE_NAME
        ),
    )

    results_directory = resolve_path(
        args.results_dir,
        "RUGD_RESULTS_DIR",
        (
            processed_root
            / "results"
        ),
    )

    if not processed_root.is_dir():
        raise FileNotFoundError(
            "Processed root does not exist: "
            f"{processed_root}"
        )

    if (
        processed_root.resolve()
        == results_directory.resolve()
    ):
        raise ValueError(
            "Results directory must not be the "
            "same directory as processed root."
        )

    validate_processed_dataset(
        processed_root=processed_root,
        metadata_path=metadata_path,
        results_directory=results_directory,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()