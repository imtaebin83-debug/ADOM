from __future__ import annotations

## 0728 절대경로 제거와 CLI·환경변수 지원
import argparse
import os
import shutil
from collections import Counter
from pathlib import Path

from tqdm import tqdm
##


## 0728 저장소 상대 기본 경로와 RUGD split 정책 정의
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_PROCESSED_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "rugd_cost4_standard"
)

DEFAULT_SPLIT_OUTPUT_ROOT = (
    DEFAULT_PROCESSED_ROOT
)

SOURCE_IMAGE_RELATIVE_DIR = (
    Path("images")
    / "all"
)

SOURCE_MASK_RELATIVE_DIR = (
    Path("annotations")
    / "all"
)

SPLIT_DIRECTORY_NAME = "splits"

SPLIT_NAMES = (
    "train",
    "val",
    "test",
)

TRAIN_SEQUENCES = {
    "park-2",
    "trail",
    "trail-3",
    "trail-4",
    "trail-6",
    "trail-9",
    "trail-10",
    "trail-11",
    "trail-12",
    "trail-14",
    "trail-15",
    "village",
}

VAL_SEQUENCES = {
    "park-8",
    "trail-5",
}

TEST_SEQUENCES = {
    "creek",
    "park-1",
    "trail-7",
    "trail-13",
}

SEQUENCES_BY_SPLIT = {
    "train": TRAIN_SEQUENCES,
    "val": VAL_SEQUENCES,
    "test": TEST_SEQUENCES,
}
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


## 0728 split별 sequence 목록이 서로 겹치지 않는지 검사
def build_sequence_to_split() -> dict[str, str]:
    sequence_to_split: dict[str, str] = {}

    for split_name in SPLIT_NAMES:
        sequences = SEQUENCES_BY_SPLIT[
            split_name
        ]

        for sequence in sorted(sequences):
            if sequence in sequence_to_split:
                previous_split = sequence_to_split[
                    sequence
                ]

                raise RuntimeError(
                    "Sequence is assigned to multiple "
                    "splits: "
                    f"sequence={sequence}, "
                    f"first={previous_split}, "
                    f"second={split_name}"
                )

            sequence_to_split[
                sequence
            ] = split_name

    return sequence_to_split
##


## 0728 파일 이름에서 RUGD sequence 이름 추출
def extract_sequence(
    stem: str,
) -> str:
    parts = stem.rsplit(
        "_",
        1,
    )

    if len(parts) != 2:
        raise ValueError(
            "RUGD sample stem does not contain "
            f"a frame separator: {stem}"
        )

    sequence, frame_number = parts

    if not sequence:
        raise ValueError(
            f"RUGD sequence name is empty: {stem}"
        )

    if not frame_number.isdigit():
        raise ValueError(
            "RUGD frame suffix is not numeric: "
            f"{stem}"
        )

    return sequence
##


## 0728 PNG 파일을 이름 기준으로 수집
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

        file_map[
            file_path.name
        ] = file_path

    return file_map
##


## 0728 RGB만 있거나 mask만 있는 pair 누락을 양방향 검사
def validate_pairs(
    image_map: dict[str, Path],
    mask_map: dict[str, Path],
) -> list[str]:
    image_names = set(image_map)
    mask_names = set(mask_map)

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
            "Processed RGB-mask pair mismatch.",
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
            "No paired processed RGB-mask samples "
            "were found."
        )

    return paired_names
##


## 0728 sample을 기존 RUGD sequence 정책에 따라 분류
def assign_samples_to_splits(
    paired_names: list[str],
    sequence_to_split: dict[str, str],
) -> tuple[
    list[dict[str, str]],
    dict[str, list[str]],
    Counter[str],
]:
    records: list[
        dict[str, str]
    ] = []

    split_stems = {
        split_name: []
        for split_name in SPLIT_NAMES
    }

    sequence_counts: Counter[
        str
    ] = Counter()

    unknown_sequences: dict[
        str,
        list[str]
    ] = {}

    seen_stems: dict[
        str,
        str
    ] = {}

    for file_name in paired_names:
        stem = Path(
            file_name
        ).stem

        if stem in seen_stems:
            raise RuntimeError(
                "Duplicate sample stem: "
                f"stem={stem}, "
                f"first={seen_stems[stem]}, "
                f"second={file_name}"
            )

        seen_stems[
            stem
        ] = file_name

        sequence = extract_sequence(
            stem
        )

        split_name = sequence_to_split.get(
            sequence
        )

        if split_name is None:
            unknown_sequences.setdefault(
                sequence,
                [],
            ).append(
                file_name
            )

            continue

        records.append(
            {
                "file_name": file_name,
                "stem": stem,
                "sequence": sequence,
                "split": split_name,
            }
        )

        split_stems[
            split_name
        ].append(
            stem
        )

        sequence_counts[
            sequence
        ] += 1

    if unknown_sequences:
        messages = [
            "Unclassified RUGD sequences were found."
        ]

        for sequence in sorted(
            unknown_sequences
        ):
            examples = unknown_sequences[
                sequence
            ][:5]

            messages.append(
                f"- {sequence}: "
                f"count="
                f"{len(unknown_sequences[sequence])}, "
                f"examples={examples}"
            )

        raise RuntimeError(
            "\n".join(messages)
        )

    assigned_count = sum(
        len(stems)
        for stems in split_stems.values()
    )

    if assigned_count != len(paired_names):
        raise RuntimeError(
            "Not all paired samples were assigned: "
            f"paired={len(paired_names)}, "
            f"assigned={assigned_count}"
        )

    validate_split_membership(
        split_stems
    )

    return (
        records,
        split_stems,
        sequence_counts,
    )
##


## 0728 split 내부 중복과 split 간 교차 중복 검사
def validate_split_membership(
    split_stems: dict[str, list[str]],
) -> None:
    split_sets: dict[
        str,
        set[str]
    ] = {}

    for split_name in SPLIT_NAMES:
        stems = split_stems[
            split_name
        ]

        stem_set = set(
            stems
        )

        if len(stems) != len(stem_set):
            duplicate_counts = Counter(
                stems
            )

            duplicates = sorted(
                stem
                for stem, count
                in duplicate_counts.items()
                if count > 1
            )

            raise RuntimeError(
                f"Duplicate samples inside {split_name}: "
                f"{duplicates[:20]}"
            )

        split_sets[
            split_name
        ] = stem_set

    for index, first_split in enumerate(
        SPLIT_NAMES
    ):
        for second_split in SPLIT_NAMES[
            index + 1:
        ]:
            overlap = sorted(
                split_sets[first_split]
                & split_sets[second_split]
            )

            if overlap:
                raise RuntimeError(
                    "Samples overlap between splits: "
                    f"{first_split} vs "
                    f"{second_split}, "
                    f"examples={overlap[:20]}"
                )
##


## 0728 출력 경로가 source all 폴더 내부를 침범하지 않는지 검사
def validate_output_location(
    source_image_directory: Path,
    source_mask_directory: Path,
    split_output_root: Path,
) -> None:
    source_image_directory = (
        source_image_directory.resolve()
    )

    source_mask_directory = (
        source_mask_directory.resolve()
    )

    split_output_root = (
        split_output_root.resolve()
    )

    if (
        split_output_root
        == source_image_directory
        or split_output_root
        == source_mask_directory
    ):
        raise ValueError(
            "Split output root must not be the "
            "source images/all or annotations/all "
            "directory."
        )

    if split_output_root.is_relative_to(
        source_image_directory
    ):
        raise ValueError(
            "Split output root must not be inside "
            f"the source image directory: "
            f"{source_image_directory}"
        )

    if split_output_root.is_relative_to(
        source_mask_directory
    ):
        raise ValueError(
            "Split output root must not be inside "
            f"the source mask directory: "
            f"{source_mask_directory}"
        )
##


## 0728 기존 split 이미지·mask·txt를 --overwrite 없이 덮어쓰지 않도록 보호
def prepare_output_directories(
    split_output_root: Path,
    overwrite: bool,
) -> tuple[
    dict[str, Path],
    dict[str, Path],
    Path,
]:
    image_directories = {
        split_name: (
            split_output_root
            / "images"
            / split_name
        )
        for split_name in SPLIT_NAMES
    }

    mask_directories = {
        split_name: (
            split_output_root
            / "annotations"
            / split_name
        )
        for split_name in SPLIT_NAMES
    }

    split_directory = (
        split_output_root
        / SPLIT_DIRECTORY_NAME
    )

    generated_directories = [
        *image_directories.values(),
        *mask_directories.values(),
    ]

    split_files = [
        split_directory
        / f"{split_name}.txt"
        for split_name in SPLIT_NAMES
    ]

    existing_outputs: list[
        Path
    ] = []

    for directory in generated_directories:
        if not directory.exists():
            continue

        existing_outputs.extend(
            sorted(
                directory.rglob("*")
            )
        )

    existing_outputs.extend(
        split_file
        for split_file in split_files
        if split_file.exists()
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
            "RUGD split outputs already exist. "
            "Use a new --split-output-root or "
            "add --overwrite. "
            f"Examples: {examples}"
        )

    if overwrite:
        for directory in generated_directories:
            if directory.exists():
                shutil.rmtree(
                    directory
                )

        for split_file in split_files:
            if split_file.is_file():
                split_file.unlink()

    for directory in generated_directories:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    split_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    return (
        image_directories,
        mask_directories,
        split_directory,
    )
##


## 0728 split 파일을 LF 줄바꿈으로 결정적으로 저장
def write_split_files(
    split_directory: Path,
    split_stems: dict[str, list[str]],
) -> None:
    for split_name in SPLIT_NAMES:
        split_file = (
            split_directory
            / f"{split_name}.txt"
        )

        stems = split_stems[
            split_name
        ]

        text = "\n".join(
            stems
        )

        if stems:
            text += "\n"

        split_file.write_text(
            text,
            encoding="utf-8",
            newline="\n",
        )
##


## 0728 split 복사 결과와 txt 행 수 재검증
def validate_generated_outputs(
    image_directories: dict[str, Path],
    mask_directories: dict[str, Path],
    split_directory: Path,
    split_stems: dict[str, list[str]],
) -> None:
    for split_name in SPLIT_NAMES:
        expected_stems = split_stems[
            split_name
        ]

        generated_images = sorted(
            image_directories[
                split_name
            ].glob("*.png")
        )

        generated_masks = sorted(
            mask_directories[
                split_name
            ].glob("*.png")
        )

        split_file = (
            split_directory
            / f"{split_name}.txt"
        )

        if not split_file.is_file():
            raise RuntimeError(
                f"Split file was not generated: "
                f"{split_file}"
            )

        split_file_stems = [
            line.strip()
            for line in split_file.read_text(
                encoding="utf-8-sig",
            ).splitlines()
            if line.strip()
        ]

        if (
            len(generated_images)
            != len(expected_stems)
        ):
            raise RuntimeError(
                f"{split_name} image count mismatch: "
                f"expected={len(expected_stems)}, "
                f"actual={len(generated_images)}"
            )

        if (
            len(generated_masks)
            != len(expected_stems)
        ):
            raise RuntimeError(
                f"{split_name} mask count mismatch: "
                f"expected={len(expected_stems)}, "
                f"actual={len(generated_masks)}"
            )

        if split_file_stems != expected_stems:
            raise RuntimeError(
                f"{split_name}.txt content or order "
                "does not match the assigned samples."
            )

        generated_image_names = {
            path.name
            for path in generated_images
        }

        generated_mask_names = {
            path.name
            for path in generated_masks
        }

        if (
            generated_image_names
            != generated_mask_names
        ):
            raise RuntimeError(
                f"{split_name} generated RGB-mask "
                "file names do not match."
            )
##


## 0728 전체 split 생성과 검증 실행
def make_splits(
    processed_root: Path,
    split_output_root: Path,
    overwrite: bool,
) -> None:
    source_image_directory = (
        processed_root
        / SOURCE_IMAGE_RELATIVE_DIR
    )

    source_mask_directory = (
        processed_root
        / SOURCE_MASK_RELATIVE_DIR
    )

    validate_output_location(
        source_image_directory,
        source_mask_directory,
        split_output_root,
    )

    image_map = collect_png_map(
        source_image_directory,
        "processed RGB image",
    )

    mask_map = collect_png_map(
        source_mask_directory,
        "processed mask",
    )

    paired_names = validate_pairs(
        image_map,
        mask_map,
    )

    sequence_to_split = (
        build_sequence_to_split()
    )

    (
        records,
        split_stems,
        sequence_counts,
    ) = assign_samples_to_splits(
        paired_names,
        sequence_to_split,
    )

    (
        image_directories,
        mask_directories,
        split_directory,
    ) = prepare_output_directories(
        split_output_root,
        overwrite=overwrite,
    )

    for record in tqdm(
        records,
        desc="Creating RUGD split files",
    ):
        file_name = record[
            "file_name"
        ]

        split_name = record[
            "split"
        ]

        shutil.copy2(
            image_map[
                file_name
            ],
            image_directories[
                split_name
            ]
            / file_name,
        )

        shutil.copy2(
            mask_map[
                file_name
            ],
            mask_directories[
                split_name
            ]
            / file_name,
        )

    write_split_files(
        split_directory,
        split_stems,
    )

    validate_generated_outputs(
        image_directories,
        mask_directories,
        split_directory,
        split_stems,
    )

    configured_sequences = set(
        sequence_to_split
    )

    present_sequences = set(
        sequence_counts
    )

    absent_sequences = sorted(
        configured_sequences
        - present_sequences
    )

    total_assigned = sum(
        len(stems)
        for stems in split_stems.values()
    )

    print("")
    print("RUGD split summary")
    print(
        f"Source RGB images: "
        f"{len(image_map)}"
    )
    print(
        f"Source masks: "
        f"{len(mask_map)}"
    )
    print(
        f"Paired samples: "
        f"{len(paired_names)}"
    )

    for split_name in SPLIT_NAMES:
        print(
            f"{split_name}: "
            f"{len(split_stems[split_name])}"
        )

    print(
        f"Total assigned samples: "
        f"{total_assigned}"
    )

    print(
        "Unassigned samples: "
        f"{len(paired_names) - total_assigned}"
    )

    print(
        "Configured sequences absent "
        "from this input: "
        f"{absent_sequences}"
    )

    print(
        f"Split directory: "
        f"{split_directory}"
    )

    print(
        "[PASS] RUGD split generation completed."
    )
##


## 0728 CLI·환경변수 기반 실행 진입점 추가
def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create deterministic RUGD "
            "train/val/test splits."
        )
    )

    parser.add_argument(
        "--processed-root",
        type=Path,
        default=None,
        help=(
            "Processed RUGD root containing "
            "images/all and annotations/all. "
            "Environment variable: "
            "RUGD_OUTPUT_ROOT"
        ),
    )

    parser.add_argument(
        "--split-output-root",
        type=Path,
        default=None,
        help=(
            "Root where split image, mask, "
            "and text outputs are created. "
            "Environment variable: "
            "RUGD_SPLIT_OUTPUT_ROOT"
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Replace existing generated "
            "train/val/test outputs."
        ),
    )

    args = parser.parse_args()

    processed_root = resolve_path(
        args.processed_root,
        "RUGD_OUTPUT_ROOT",
        DEFAULT_PROCESSED_ROOT,
    )

    split_output_root = resolve_path(
        args.split_output_root,
        "RUGD_SPLIT_OUTPUT_ROOT",
        DEFAULT_SPLIT_OUTPUT_ROOT,
    )

    if not processed_root.is_dir():
        raise FileNotFoundError(
            "Processed root does not exist: "
            f"{processed_root}"
        )

    make_splits(
        processed_root=processed_root,
        split_output_root=split_output_root,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
##
