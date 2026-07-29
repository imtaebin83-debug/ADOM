from __future__ import annotations

## 0728 절대경로 제거와 CLI·환경변수 입력을 위한 모듈 추가
import argparse
import ast
import csv
import json
import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm
##


## 0728 저장소 상대 기본 경로와 출력 구조 정의
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "RUGD"
)

DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "rugd_cost4_standard"
)

DEFAULT_MAPPING_PATH = (
    PROJECT_ROOT
    / "config"
    / "label_mapping.json"
)

IMAGE_RELATIVE_DIR = (
    Path("3.after join creek")
    / "image"
)

OUTPUT_IMAGE_RELATIVE_DIR = (
    Path("images")
    / "all"
)

OUTPUT_MASK_RELATIVE_DIR = (
    Path("annotations")
    / "all"
)

METADATA_FILE_NAME = "metadata.csv"

ALLOWED_TARGET_IDS = {
    0,
    1,
    2,
    3,
    255,
}

METADATA_FIELDS = [
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
        return default.resolve()

    return Path(value).expanduser().resolve()
##


## 0728 JSON의 "(R, G, B)" 문자열을 RGB tuple로 안전하게 변환
def parse_rgb_key(
    key: str,
) -> tuple[int, int, int]:
    try:
        value = ast.literal_eval(key)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(
            f"Invalid RGB mapping key: {key}"
        ) from exc

    if (
        not isinstance(value, tuple)
        or len(value) != 3
        or not all(
            isinstance(channel, int)
            for channel in value
        )
    ):
        raise ValueError(
            f"RGB mapping key must be an integer tuple: {key}"
        )

    if not all(
        0 <= channel <= 255
        for channel in value
    ):
        raise ValueError(
            f"RGB channel is outside 0~255: {key}"
        )

    return value
##


## 0728 label_mapping.json을 읽고 3개 mapping 구간의 일관성 검증
def load_mapping(
    mapping_path: Path,
) -> dict[tuple[int, int, int], int]:
    if not mapping_path.is_file():
        raise FileNotFoundError(
            f"Mapping file does not exist: {mapping_path}"
        )

    with mapping_path.open(
        "r",
        encoding="utf-8-sig",
    ) as file:
        data = json.load(file)

    required_sections = {
        "RGB_TO_NAME",
        "RUGD_TO_ADOM",
        "RGB_TO_ADOM",
    }

    missing_sections = sorted(
        required_sections - set(data)
    )

    if missing_sections:
        raise RuntimeError(
            "Mapping file is missing sections: "
            f"{missing_sections}"
        )

    rgb_to_name = {
        parse_rgb_key(key): value
        for key, value
        in data["RGB_TO_NAME"].items()
        if key != "_comment"
    }

    name_to_adom = {
        key: int(value)
        for key, value
        in data["RUGD_TO_ADOM"].items()
        if key != "_comment"
    }

    direct_rgb_to_adom = {
        parse_rgb_key(key): int(value)
        for key, value
        in data["RGB_TO_ADOM"].items()
        if key != "_comment"
    }

    missing_class_names = sorted(
        {
            class_name
            for class_name in rgb_to_name.values()
            if class_name not in name_to_adom
        }
    )

    if missing_class_names:
        raise RuntimeError(
            "Classes are missing from RUGD_TO_ADOM: "
            f"{missing_class_names}"
        )

    composed_rgb_to_adom = {
        rgb: name_to_adom[class_name]
        for rgb, class_name
        in rgb_to_name.items()
    }

    if composed_rgb_to_adom != direct_rgb_to_adom:
        all_colors = sorted(
            set(composed_rgb_to_adom)
            | set(direct_rgb_to_adom)
        )

        mismatches = {
            rgb: {
                "composed": composed_rgb_to_adom.get(rgb),
                "direct": direct_rgb_to_adom.get(rgb),
            }
            for rgb in all_colors
            if (
                composed_rgb_to_adom.get(rgb)
                != direct_rgb_to_adom.get(rgb)
            )
        }

        raise RuntimeError(
            "RGB_TO_ADOM does not match "
            "RGB_TO_NAME + RUGD_TO_ADOM: "
            f"{mismatches}"
        )

    target_ids = set(
        composed_rgb_to_adom.values()
    )

    if target_ids != ALLOWED_TARGET_IDS:
        raise RuntimeError(
            "Unexpected target IDs in mapping: "
            f"{sorted(target_ids)}"
        )

    return composed_rgb_to_adom
##


## 0728 입력 root에서 RUGD color annotation 폴더 탐색
def find_color_directories(
    input_root: Path,
) -> list[Path]:
    color_directories = sorted(
        path
        for path in input_root.iterdir()
        if (
            path.is_dir()
            and "indexlabel" in path.name.lower()
            and "color" in path.name.lower()
        )
    )

    if not color_directories:
        raise RuntimeError(
            "No indexLabel-color directory was found "
            f"under: {input_root}"
        )

    return color_directories
##


## 0728 파일 이름을 기준으로 RGB 이미지 목록 생성
def collect_image_map(
    image_directory: Path,
) -> dict[str, Path]:
    if not image_directory.is_dir():
        raise FileNotFoundError(
            f"RGB image directory does not exist: "
            f"{image_directory}"
        )

    image_map: dict[str, Path] = {}

    for image_path in sorted(
        image_directory.glob("*.png")
    ):
        if image_path.name in image_map:
            raise RuntimeError(
                f"Duplicate RGB image name: "
                f"{image_path.name}"
            )

        image_map[image_path.name] = image_path

    return image_map
##


## 0728 여러 color annotation 폴더에서 mask 목록 생성 및 중복 검사
def collect_color_mask_map(
    color_directories: list[Path],
) -> dict[str, Path]:
    color_mask_map: dict[str, Path] = {}

    for color_directory in color_directories:
        for mask_path in sorted(
            color_directory.rglob("*.png")
        ):
            if mask_path.name in color_mask_map:
                previous_path = color_mask_map[
                    mask_path.name
                ]

                raise RuntimeError(
                    "Duplicate color mask name: "
                    f"{mask_path.name}\n"
                    f"first={previous_path}\n"
                    f"second={mask_path}"
                )

            color_mask_map[
                mask_path.name
            ] = mask_path

    return color_mask_map
##


## 0728 RGB만 있거나 mask만 있는 누락 pair를 모두 오류로 탐지
def validate_pairs(
    image_map: dict[str, Path],
    color_mask_map: dict[str, Path],
) -> list[str]:
    image_names = set(image_map)
    mask_names = set(color_mask_map)

    images_without_masks = sorted(
        image_names - mask_names
    )

    masks_without_images = sorted(
        mask_names - image_names
    )

    if (
        images_without_masks
        or masks_without_images
    ):
        messages = [
            "RGB-color-mask pair mismatch.",
            (
                "RGB images without masks: "
                f"{len(images_without_masks)}"
            ),
            (
                "Color masks without RGB images: "
                f"{len(masks_without_images)}"
            ),
        ]

        if images_without_masks:
            messages.append(
                "RGB-only examples: "
                + ", ".join(
                    images_without_masks[:10]
                )
            )

        if masks_without_images:
            messages.append(
                "Mask-only examples: "
                + ", ".join(
                    masks_without_images[:10]
                )
            )

        raise RuntimeError(
            "\n".join(messages)
        )

    paired_names = sorted(
        image_names & mask_names
    )

    if not paired_names:
        raise RuntimeError(
            "No paired RGB and color-mask samples "
            "were found."
        )

    return paired_names
##


## 0728 metadata에 개인 절대경로 대신 root 기준 상대경로 기록
def relative_path_text(
    path: Path,
    root: Path,
) -> str:
    try:
        relative_path = (
            path.resolve()
            .relative_to(root.resolve())
        )
    except ValueError as exc:
        raise RuntimeError(
            f"Path is outside its configured root: "
            f"path={path}, root={root}"
        ) from exc

    return relative_path.as_posix()
##


## 0728 기존 출력 보호 및 --overwrite 사용 시 생성 파일만 정리
def prepare_output_directories(
    output_root: Path,
    overwrite: bool,
) -> tuple[Path, Path, Path]:
    output_image_directory = (
        output_root
        / OUTPUT_IMAGE_RELATIVE_DIR
    )

    output_mask_directory = (
        output_root
        / OUTPUT_MASK_RELATIVE_DIR
    )

    metadata_path = (
        output_root
        / METADATA_FILE_NAME
    )

    existing_output_files = []

    if output_image_directory.is_dir():
        existing_output_files.extend(
            output_image_directory.glob("*.png")
        )

    if output_mask_directory.is_dir():
        existing_output_files.extend(
            output_mask_directory.glob("*.png")
        )

    if metadata_path.is_file():
        existing_output_files.append(
            metadata_path
        )

    if (
        existing_output_files
        and not overwrite
    ):
        raise FileExistsError(
            "RUGD processed output already exists. "
            "Use a new --output-root or add "
            f"--overwrite: {output_root}"
        )

    if overwrite:
        if output_image_directory.is_dir():
            for image_path in (
                output_image_directory.glob("*.png")
            ):
                image_path.unlink()

        if output_mask_directory.is_dir():
            for mask_path in (
                output_mask_directory.glob("*.png")
            ):
                mask_path.unlink()

        if metadata_path.is_file():
            metadata_path.unlink()

    output_image_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_mask_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    return (
        output_image_directory,
        output_mask_directory,
        metadata_path,
    )
##


## 0728 변환 결과와 오류 상태를 metadata.csv로 저장
def write_metadata(
    metadata_path: Path,
    rows: list[dict[str, object]],
) -> None:
    metadata_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with metadata_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=METADATA_FIELDS,
        )

        writer.writeheader()
        writer.writerows(rows)
##


## 0728 RGB color mask를 ADOM Cost4 단일 채널 mask로 변환
def convert_dataset(
    input_root: Path,
    output_root: Path,
    mapping_path: Path,
    limit: int | None,
    overwrite: bool,
) -> None:
    rgb_to_adom = load_mapping(
        mapping_path
    )

    image_directory = (
        input_root
        / IMAGE_RELATIVE_DIR
    )

    color_directories = (
        find_color_directories(
            input_root
        )
    )

    image_map = collect_image_map(
        image_directory
    )

    color_mask_map = (
        collect_color_mask_map(
            color_directories
        )
    )

    paired_names = validate_pairs(
        image_map,
        color_mask_map,
    )

    selected_names = paired_names

    if limit is not None:
        selected_names = paired_names[:limit]

    (
        output_image_directory,
        output_mask_directory,
        metadata_path,
    ) = prepare_output_directories(
        output_root,
        overwrite=overwrite,
    )

    print(
        f"RGB images: {len(image_map)}"
    )

    print(
        f"Color masks: {len(color_mask_map)}"
    )

    print(
        f"Paired samples: {len(paired_names)}"
    )

    print(
        f"Selected samples: {len(selected_names)}"
    )

    known_colors = set(
        rgb_to_adom
    )

    metadata_rows: list[
        dict[str, object]
    ] = []

    conversion_errors: list[
        tuple[str, str]
    ] = []

    for file_name in tqdm(
        selected_names,
        desc="Converting RUGD masks",
    ):
        image_path = image_map[
            file_name
        ]

        color_mask_path = (
            color_mask_map[
                file_name
            ]
        )

        output_image_path = (
            output_image_directory
            / file_name
        )

        output_mask_path = (
            output_mask_directory
            / file_name
        )

        sample_id = Path(
            file_name
        ).stem

        width = ""
        height = ""
        status = "ok"
        details = ""

        try:
            with Image.open(
                image_path
            ) as image_file:
                rgb_image = (
                    image_file.convert("RGB")
                )

                image_size = rgb_image.size

            with Image.open(
                color_mask_path
            ) as mask_file:
                color_mask_image = (
                    mask_file.convert("RGB")
                )

                color_mask = np.asarray(
                    color_mask_image,
                    dtype=np.uint8,
                )

                mask_size = (
                    color_mask_image.size
                )

            if image_size != mask_size:
                raise ValueError(
                    "RGB and color-mask dimensions "
                    "do not match: "
                    f"rgb_size={image_size}, "
                    f"mask_size={mask_size}"
                )

            width, height = image_size

            unique_colors = {
                tuple(
                    int(channel)
                    for channel in color
                )
                for color in np.unique(
                    color_mask.reshape(-1, 3),
                    axis=0,
                )
            }

            unknown_colors = sorted(
                unique_colors
                - known_colors
            )

            if unknown_colors:
                raise RuntimeError(
                    "Unknown RGB colors were found: "
                    f"count={len(unknown_colors)}, "
                    f"examples={unknown_colors[:20]}"
                )

            target_mask = np.full(
                color_mask.shape[:2],
                fill_value=255,
                dtype=np.uint8,
            )

            for (
                rgb,
                target_id,
            ) in rgb_to_adom.items():
                pixels = np.all(
                    color_mask
                    == np.asarray(
                        rgb,
                        dtype=np.uint8,
                    ),
                    axis=2,
                )

                target_mask[
                    pixels
                ] = target_id

            converted_ids = {
                int(value)
                for value in np.unique(
                    target_mask
                )
            }

            invalid_ids = (
                converted_ids
                - ALLOWED_TARGET_IDS
            )

            if invalid_ids:
                raise RuntimeError(
                    "Converted mask contains "
                    "invalid target IDs: "
                    f"{sorted(invalid_ids)}"
                )

            Image.fromarray(
                target_mask,
                mode="L",
            ).save(
                output_mask_path
            )

            shutil.copy2(
                image_path,
                output_image_path,
            )

        except (
            OSError,
            ValueError,
            RuntimeError,
        ) as exc:
            status = "conversion_error"
            details = (
                f"{type(exc).__name__}: "
                f"{exc}"
            )

            conversion_errors.append(
                (
                    file_name,
                    details,
                )
            )

        metadata_rows.append(
            {
                "sample_id": sample_id,
                "file_name": file_name,
                "rgb_path": relative_path_text(
                    image_path,
                    input_root,
                ),
                "source_color_mask_path":
                    relative_path_text(
                        color_mask_path,
                        input_root,
                    ),
                "converted_image_path":
                    relative_path_text(
                        output_image_path,
                        output_root,
                    ),
                "converted_mask_path":
                    relative_path_text(
                        output_mask_path,
                        output_root,
                    ),
                "width": width,
                "height": height,
                "status": status,
                "details": details,
            }
        )

    write_metadata(
        metadata_path,
        metadata_rows,
    )

    converted_image_count = len(
        list(
            output_image_directory.glob(
                "*.png"
            )
        )
    )

    converted_mask_count = len(
        list(
            output_mask_directory.glob(
                "*.png"
            )
        )
    )

    print("")
    print("RUGD conversion summary")
    print(
        f"Metadata rows: "
        f"{len(metadata_rows)}"
    )
    print(
        f"Converted images: "
        f"{converted_image_count}"
    )
    print(
        f"Converted masks: "
        f"{converted_mask_count}"
    )
    print(
        f"Conversion failures: "
        f"{len(conversion_errors)}"
    )
    print(
        f"Metadata: {metadata_path}"
    )

    if conversion_errors:
        examples = "\n".join(
            f"- {file_name}: {details}"
            for (
                file_name,
                details,
            ) in conversion_errors[:10]
        )

        raise RuntimeError(
            "RUGD conversion failed for "
            f"{len(conversion_errors)} sample(s). "
            f"See metadata.csv.\n{examples}"
        )

    if (
        converted_image_count
        != len(selected_names)
    ):
        raise RuntimeError(
            "Converted image count mismatch: "
            f"expected={len(selected_names)}, "
            f"actual={converted_image_count}"
        )

    if (
        converted_mask_count
        != len(selected_names)
    ):
        raise RuntimeError(
            "Converted mask count mismatch: "
            f"expected={len(selected_names)}, "
            f"actual={converted_mask_count}"
        )

    print("[PASS] RUGD conversion completed.")
##


## 0728 CLI·환경변수 기반 실행 진입점 추가
def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert RUGD color masks into "
            "ADOM Cost4 single-channel masks."
        )
    )

    parser.add_argument(
        "--input-root",
        type=Path,
        default=None,
        help=(
            "RUGD raw dataset root containing "
            "'3.after join creek'. "
            "Environment variable: "
            "RUGD_INPUT_ROOT"
        ),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Processed dataset output root. "
            "Environment variable: "
            "RUGD_OUTPUT_ROOT"
        ),
    )

    parser.add_argument(
        "--mapping",
        type=Path,
        default=None,
        help=(
            "Path to label_mapping.json. "
            "Environment variable: "
            "RUGD_MAPPING_PATH"
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Convert only the first N paired "
            "samples for a dry run."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Replace existing generated PNG "
            "files and metadata.csv."
        ),
    )

    args = parser.parse_args()

    input_root = resolve_path(
        args.input_root,
        "RUGD_INPUT_ROOT",
        DEFAULT_INPUT_ROOT,
    )

    output_root = resolve_path(
        args.output_root,
        "RUGD_OUTPUT_ROOT",
        DEFAULT_OUTPUT_ROOT,
    )

    mapping_path = resolve_path(
        args.mapping,
        "RUGD_MAPPING_PATH",
        DEFAULT_MAPPING_PATH,
    )

    if not input_root.is_dir():
        raise FileNotFoundError(
            f"Input root does not exist: "
            f"{input_root}"
        )

    if not mapping_path.is_file():
        raise FileNotFoundError(
            f"Mapping file does not exist: "
            f"{mapping_path}"
        )

    if (
        args.limit is not None
        and args.limit < 1
    ):
        raise ValueError(
            "--limit must be at least 1."
        )

    if (
        input_root.resolve()
        == output_root.resolve()
    ):
        raise ValueError(
            "Input root and output root "
            "must be different directories."
        )

    convert_dataset(
        input_root=input_root,
        output_root=output_root,
        mapping_path=mapping_path,
        limit=args.limit,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
##
