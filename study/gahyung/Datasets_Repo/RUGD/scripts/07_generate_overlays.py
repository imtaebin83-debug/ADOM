from __future__ import annotations

## 0728 overlay 경로·옵션을 CLI와 환경변수로 입력하도록 변경
import argparse
import csv
import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError
##


## 0728 저장소 상대 기본 경로와 기존 시각화 정책 정의
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

ALLOWED_MASK_IDS = {
    0,
    1,
    2,
    3,
    255,
}

# 기존 ADOM Cost4 시각화 색상 유지
CLASS_COLORS = {
    0: np.array(
        [128, 64, 128],
        dtype=np.uint8,
    ),
    1: np.array(
        [107, 142, 35],
        dtype=np.uint8,
    ),
    2: np.array(
        [255, 165, 0],
        dtype=np.uint8,
    ),
    3: np.array(
        [220, 20, 60],
        dtype=np.uint8,
    ),
}

DEFAULT_SAMPLES_PER_SPLIT = 100
DEFAULT_ALPHA = 0.45

MANIFEST_FILE_NAME = "overlay_manifest.csv"
##


## 0728 CLI → 환경변수 → 저장소 상대 기본값 순서로 경로 결정
def resolve_path(
    cli_value: Path | None,
    env_name: str,
    default: Path,
) -> Path:
    value = cli_value or os.getenv(
        env_name
    )

    if value is None:
        return default.expanduser().resolve()

    return Path(
        value
    ).expanduser().resolve()
##


## 0728 split 폴더의 PNG 파일을 이름 기준으로 수집
def collect_png_map(
    directory: Path,
    description: str,
) -> dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(
            f"{description} directory does not exist: "
            f"{directory}"
        )

    file_map: dict[
        str,
        Path,
    ] = {}

    for file_path in sorted(
        directory.glob("*.png")
    ):
        if file_path.name in file_map:
            raise RuntimeError(
                f"Duplicate {description} file name: "
                f"{file_path.name}"
            )

        file_map[
            file_path.name
        ] = file_path

    return file_map
##


## 0728 RGB만 있거나 mask만 있는 누락 pair를 양방향 검사
def validate_pairs(
    image_map: dict[str, Path],
    mask_map: dict[str, Path],
    split_name: str,
) -> list[str]:
    image_names = set(
        image_map
    )

    mask_names = set(
        mask_map
    )

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
            (
                "RUGD overlay pair mismatch "
                f"in split={split_name}."
            ),
            (
                "RGB images without masks: "
                f"{len(images_without_masks)}"
            ),
            (
                "Masks without RGB images: "
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
            "No paired RGB-mask samples were found "
            f"for split={split_name}."
        )

    return paired_names
##


## 0728 정렬된 split 전체에서 결정적으로 샘플 선택
def select_evenly(
    file_names: list[str],
    sample_count: int,
) -> list[str]:
    if sample_count <= 0:
        raise ValueError(
            "samples-per-split must be positive: "
            f"{sample_count}"
        )

    if len(file_names) <= sample_count:
        return list(
            file_names
        )

    indices = np.linspace(
        0,
        len(file_names) - 1,
        sample_count,
        dtype=int,
    )

    selected = [
        file_names[index]
        for index in indices
    ]

    if len(selected) != len(set(selected)):
        raise RuntimeError(
            "Even sampling produced duplicate "
            "file names."
        )

    return selected
##


## 0728 overlay 출력 경로가 RGB·mask source를 침범하지 않는지 검사
def validate_output_location(
    processed_root: Path,
    image_root: Path,
    mask_root: Path,
    output_root: Path,
) -> None:
    processed_root = (
        processed_root.resolve()
    )

    image_root = (
        image_root.resolve()
    )

    mask_root = (
        mask_root.resolve()
    )

    output_root = (
        output_root.resolve()
    )

    if output_root == processed_root:
        raise ValueError(
            "Overlay output root must not equal "
            "the processed root."
        )

    if processed_root.is_relative_to(
        output_root
    ):
        raise ValueError(
            "Overlay output root must not contain "
            "the processed root."
        )

    if (
        output_root == image_root
        or output_root.is_relative_to(
            image_root
        )
    ):
        raise ValueError(
            "Overlay output root must not be "
            "the image root or a directory "
            "inside it."
        )

    if (
        output_root == mask_root
        or output_root.is_relative_to(
            mask_root
        )
    ):
        raise ValueError(
            "Overlay output root must not be "
            "the mask root or a directory "
            "inside it."
        )
##


## 0728 기존 overlay를 --overwrite 없이 덮어쓰지 않도록 보호
def prepare_output_root(
    output_root: Path,
    overwrite: bool,
) -> dict[str, Path]:
    output_directories = {
        split_name: (
            output_root
            / split_name
        )
        for split_name in SPLIT_NAMES
    }

    manifest_path = (
        output_root
        / MANIFEST_FILE_NAME
    )

    existing_outputs: list[
        Path
    ] = []

    for directory in (
        output_directories.values()
    ):
        if directory.exists():
            existing_outputs.extend(
                sorted(
                    directory.rglob("*")
                )
            )

    if manifest_path.exists():
        existing_outputs.append(
            manifest_path
        )

    if (
        existing_outputs
        and not overwrite
    ):
        examples = [
            str(path)
            for path in existing_outputs[:10]
        ]

        raise FileExistsError(
            "RUGD overlay outputs already exist. "
            "Use a new --output-root or add "
            "--overwrite. "
            f"Examples: {examples}"
        )

    if overwrite:
        for directory in (
            output_directories.values()
        ):
            if directory.exists():
                shutil.rmtree(
                    directory
                )

        if manifest_path.is_file():
            manifest_path.unlink()

    for directory in (
        output_directories.values()
    ):
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    return output_directories
##


## 0728 RGB·mask 형식과 Cost4 ID를 검사한 뒤 overlay 생성
def make_overlay(
    image_path: Path,
    mask_path: Path,
    output_path: Path,
    alpha: float,
) -> tuple[
    int,
    int,
    list[int],
]:
    try:
        with Image.open(
            image_path
        ) as image_file:
            image = np.asarray(
                image_file.convert("RGB"),
                dtype=np.uint8,
            )

        with Image.open(
            mask_path
        ) as mask_file:
            mask_format = (
                mask_file.format
            )

            mask_mode = (
                mask_file.mode
            )

            mask = np.asarray(
                mask_file
            )

    except (
        UnidentifiedImageError,
        OSError,
    ) as error:
        raise RuntimeError(
            "Failed to read image or mask: "
            f"image={image_path}, "
            f"mask={mask_path}"
        ) from error

    if mask_format != "PNG":
        raise RuntimeError(
            f"Mask is not PNG: "
            f"{mask_path.name}, "
            f"format={mask_format}"
        )

    if mask_mode != "L":
        raise RuntimeError(
            f"Mask mode is not L: "
            f"{mask_path.name}, "
            f"mode={mask_mode}"
        )

    if mask.ndim != 2:
        raise RuntimeError(
            "Mask is not single-channel: "
            f"{mask_path.name}, "
            f"shape={mask.shape}"
        )

    if image.shape[:2] != mask.shape:
        raise RuntimeError(
            "RGB-mask size mismatch: "
            f"{image_path.name}, "
            f"image={image.shape[:2]}, "
            f"mask={mask.shape}"
        )

    unique_ids = sorted(
        int(value)
        for value in np.unique(
            mask
        ).tolist()
    )

    invalid_ids = sorted(
        set(unique_ids)
        - ALLOWED_MASK_IDS
    )

    if invalid_ids:
        raise RuntimeError(
            "Mask contains invalid IDs: "
            f"{mask_path.name}, "
            f"invalid={invalid_ids}"
        )

    if set(unique_ids) == {255}:
        raise RuntimeError(
            "Mask contains only ignore pixels: "
            f"{mask_path.name}"
        )

    color_mask = np.zeros_like(
        image
    )

    valid_pixels = np.zeros(
        mask.shape,
        dtype=bool,
    )

    for class_id, color in (
        CLASS_COLORS.items()
    ):
        class_pixels = (
            mask == class_id
        )

        color_mask[
            class_pixels
        ] = color

        valid_pixels |= class_pixels

    blended = (
        image.astype(
            np.float32
        )
        * (1.0 - alpha)
        + color_mask.astype(
            np.float32
        )
        * alpha
    ).clip(
        0,
        255,
    ).astype(
        np.uint8
    )

    # 0~3 영역만 blending하고,
    # 255 ignore는 원본 RGB를 유지한다.
    overlay = image.copy()

    overlay[
        valid_pixels
    ] = blended[
        valid_pixels
    ]

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    Image.fromarray(
        overlay
    ).save(
        output_path
    )

    height, width = mask.shape

    return (
        width,
        height,
        unique_ids,
    )
##


## 0728 선택 샘플과 상대경로를 manifest CSV에 기록
def write_manifest(
    manifest_path: Path,
    records: list[
        dict[
            str,
            str | int,
        ]
    ],
) -> None:
    field_names = [
        "split",
        "sample_id",
        "source_image",
        "source_mask",
        "overlay_path",
        "width",
        "height",
        "mask_ids",
    ]

    with manifest_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=field_names,
        )

        writer.writeheader()
        writer.writerows(
            records
        )
##


## 0728 생성된 overlay 수와 manifest 행 수를 재검증
def validate_generated_outputs(
    output_directories: dict[
        str,
        Path,
    ],
    manifest_path: Path,
    expected_counts: dict[
        str,
        int,
    ],
    records: list[
        dict[
            str,
            str | int,
        ]
    ],
) -> None:
    if not manifest_path.is_file():
        raise RuntimeError(
            "Overlay manifest was not generated: "
            f"{manifest_path}"
        )

    expected_total = sum(
        expected_counts.values()
    )

    if len(records) != expected_total:
        raise RuntimeError(
            "Overlay manifest record count mismatch: "
            f"expected={expected_total}, "
            f"actual={len(records)}"
        )

    for split_name in SPLIT_NAMES:
        generated_files = sorted(
            output_directories[
                split_name
            ].glob(
                "*_overlay.png"
            )
        )

        if (
            len(generated_files)
            != expected_counts[
                split_name
            ]
        ):
            raise RuntimeError(
                f"{split_name} overlay count mismatch: "
                f"expected="
                f"{expected_counts[split_name]}, "
                f"actual={len(generated_files)}"
            )
##


## 0728 전체 overlay 생성 파이프라인
def generate_overlays(
    processed_root: Path,
    output_root: Path,
    samples_per_split: int,
    alpha: float,
    overwrite: bool,
) -> None:
    if samples_per_split <= 0:
        raise ValueError(
            "samples-per-split must be positive: "
            f"{samples_per_split}"
        )

    if not 0.0 <= alpha <= 1.0:
        raise ValueError(
            "alpha must be between 0 and 1: "
            f"{alpha}"
        )

    image_root = (
        processed_root
        / "images"
    )

    mask_root = (
        processed_root
        / "annotations"
    )

    validate_output_location(
        processed_root,
        image_root,
        mask_root,
        output_root,
    )

    split_inputs: dict[
        str,
        tuple[
            dict[str, Path],
            dict[str, Path],
            list[str],
        ],
    ] = {}

    # 모든 split과 모든 pair를 먼저 검사한 뒤
    # 출력 폴더를 생성한다.
    for split_name in SPLIT_NAMES:
        image_map = collect_png_map(
            image_root
            / split_name,
            f"{split_name} RGB image",
        )

        mask_map = collect_png_map(
            mask_root
            / split_name,
            f"{split_name} mask",
        )

        paired_names = validate_pairs(
            image_map,
            mask_map,
            split_name,
        )

        selected_names = select_evenly(
            paired_names,
            samples_per_split,
        )

        split_inputs[
            split_name
        ] = (
            image_map,
            mask_map,
            selected_names,
        )

    output_directories = (
        prepare_output_root(
            output_root,
            overwrite=overwrite,
        )
    )

    records: list[
        dict[
            str,
            str | int,
        ]
    ] = []

    expected_counts: dict[
        str,
        int,
    ] = {}

    for split_name in SPLIT_NAMES:
        (
            image_map,
            mask_map,
            selected_names,
        ) = split_inputs[
            split_name
        ]

        expected_counts[
            split_name
        ] = len(
            selected_names
        )

        print(
            f"{split_name}: "
            f"selected={len(selected_names)} "
            f"from paired={len(image_map)}"
        )

        for file_name in selected_names:
            image_path = image_map[
                file_name
            ]

            mask_path = mask_map[
                file_name
            ]

            output_path = (
                output_directories[
                    split_name
                ]
                / (
                    f"{Path(file_name).stem}"
                    "_overlay.png"
                )
            )

            (
                width,
                height,
                unique_ids,
            ) = make_overlay(
                image_path=image_path,
                mask_path=mask_path,
                output_path=output_path,
                alpha=alpha,
            )

            records.append(
                {
                    "split": split_name,
                    "sample_id": (
                        Path(
                            file_name
                        ).stem
                    ),
                    "source_image": (
                        image_path.relative_to(
                            processed_root
                        ).as_posix()
                    ),
                    "source_mask": (
                        mask_path.relative_to(
                            processed_root
                        ).as_posix()
                    ),
                    "overlay_path": (
                        output_path.relative_to(
                            output_root
                        ).as_posix()
                    ),
                    "width": width,
                    "height": height,
                    "mask_ids": ";".join(
                        str(value)
                        for value in unique_ids
                    ),
                }
            )

    manifest_path = (
        output_root
        / MANIFEST_FILE_NAME
    )

    write_manifest(
        manifest_path,
        records,
    )

    validate_generated_outputs(
        output_directories,
        manifest_path,
        expected_counts,
        records,
    )

    print("")
    print("RUGD overlay summary")

    for split_name in SPLIT_NAMES:
        print(
            f"{split_name}: "
            f"{expected_counts[split_name]}"
        )

    print(
        f"Total overlays: "
        f"{len(records)}"
    )

    print(
        f"Alpha: {alpha}"
    )

    print(
        f"Manifest: "
        f"{manifest_path}"
    )

    print(
        f"Output root: "
        f"{output_root}"
    )

    print(
        "[PASS] RUGD overlay generation completed."
    )
##


## 0728 CLI·환경변수 기반 실행 진입점 추가
def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate deterministic RUGD "
            "Cost4 overlay previews."
        )
    )

    parser.add_argument(
        "--processed-root",
        type=Path,
        default=None,
        help=(
            "Processed RUGD root containing "
            "images/train|val|test and "
            "annotations/train|val|test. "
            "Environment variable: "
            "RUGD_OUTPUT_ROOT"
        ),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Overlay output root. "
            "Environment variable: "
            "RUGD_OVERLAY_OUTPUT_ROOT. "
            "Default: "
            "<processed-root>/qc/overlays"
        ),
    )

    parser.add_argument(
        "--samples-per-split",
        type=int,
        default=(
            DEFAULT_SAMPLES_PER_SPLIT
        ),
        help=(
            "Maximum number of evenly selected "
            "overlays per split."
        ),
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help=(
            "Mask blending opacity "
            "between 0 and 1."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Replace existing generated "
            "overlay outputs."
        ),
    )

    args = parser.parse_args()

    processed_root = resolve_path(
        args.processed_root,
        "RUGD_OUTPUT_ROOT",
        DEFAULT_PROCESSED_ROOT,
    )

    output_env = os.getenv(
        "RUGD_OVERLAY_OUTPUT_ROOT"
    )

    if args.output_root is not None:
        output_root = (
            args.output_root
            .expanduser()
            .resolve()
        )

    elif output_env:
        output_root = (
            Path(
                output_env
            )
            .expanduser()
            .resolve()
        )

    else:
        output_root = (
            processed_root
            / "qc"
            / "overlays"
        ).resolve()

    if not processed_root.is_dir():
        raise FileNotFoundError(
            "Processed root does not exist: "
            f"{processed_root}"
        )

    generate_overlays(
        processed_root=processed_root,
        output_root=output_root,
        samples_per_split=(
            args.samples_per_split
        ),
        alpha=args.alpha,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
##
