from __future__ import annotations

import argparse
import csv
import os
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

## 0727 외부 데이터 경로를 지정할 수 있도록 기본 경로와 split 이름 정의
DEFAULT_SOURCE_SPLIT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "splits_original"
)

DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "rellis_cost4_standard"
)

SPLIT_NAMES = ("train", "val", "test")
##

## 0727 CLI - 환경변수 - 기본값 순서로 경로를 결정하는 함수 추가
def resolve_path(
    cli_value: Path | None,
    env_name: str,
    default: Path,
) -> Path:
    """
    Resolve a path in this order:

    CLI argument -> environment variable -> repository-relative default.
    """
    value = cli_value or os.getenv(env_name)

    if value is None:
        return default.resolve()

    return Path(value).expanduser().resolve()
##

def determine_split_name(filename: str) -> str | None:

    ## 0727 파일명으로 train, val, test split을 판별한다는 설명 추가
    """Infer train, val, or test from a split filename."""
    ##

    name = filename.lower()

    if "train" in name:
        return "train"

    if "val" in name or "valid" in name:
        return "val"

    if "test" in name:
        return "test"

    return None


## 0727 split 원본의 RGB 경로에서 sequence와 stem을 추출하도록 정리
def convert_line(line: str) -> str | None:
    """
    Convert a source split line into a converted-mask sample ID.

    Example:
        00000/pylon_camera_node/frame000000.jpg ...

    Returns:
        00000_frame000000
    """
    line = line.strip()

    if not line or line.startswith("#"):
        return None

    # RGB와 label 경로가 함께 있으면 첫 번째 RGB 경로를 사용한다.
    first_token = line.split()[0].replace("\\", "/")

    sequence_match = re.search(
        r"(?<!\d)(\d{5})(?!\d)",
        first_token,
    )

    if sequence_match is None:
        raise ValueError(
            f"Could not find a five-digit sequence ID: {line}"
        )

    sequence = sequence_match.group(1)
    stem = Path(first_token).stem

    return f"{sequence}_{stem}"
##

## 0727 train·val·test 원본 split 파일 탐색 기능 추가
def find_split_files(
    source_split_root: Path,
) -> list[Path]:
    """Find supported source split files."""
    split_files = sorted(
        path
        for path in source_split_root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".txt", ".lst"}
        and determine_split_name(path.name) is not None
    )

    if not split_files:
        raise RuntimeError(
            "No train, val, or test split files were found under: "
            f"{source_split_root}"
        )

    return split_files
##


## 0727 변환 mask가 없는 split 항목을 CSV로 기록하는 기능 추가
def write_missing_report(
    missing_rows: list[dict[str, str]],
    report_path: Path,
) -> None:
    """Write source split entries that have no converted mask."""
    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with report_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "split",
                "sample_id",
                "source_file",
                "line_number",
            ],
        )

        writer.writeheader()
        writer.writerows(missing_rows)
##


## 0727 동일 sample이 여러 split에 포함되는지 검사하는 기능 추가
def find_cross_split_overlaps(
    converted: dict[str, set[str]],
) -> dict[str, set[str]]:
    """Find sample IDs assigned to more than one split."""
    return {
        "train_val": converted["train"] & converted["val"],
        "train_test": converted["train"] & converted["test"],
        "val_test": converted["val"] & converted["test"],
    }
##

## 0727 외부 경로 지원과 split 무결성 검증을 반영하도록 main 함수 수정
def main(
    source_split_root: Path,
    output_root: Path,
    overwrite: bool,
) -> None:
    output_split_root = output_root / "splits"
    mask_root = output_root / "masks"

    if not source_split_root.is_dir():
        raise FileNotFoundError(
            "Source split directory does not exist: "
            f"{source_split_root}"
        )

    if not mask_root.is_dir():
        raise FileNotFoundError(
            "Converted mask directory does not exist: "
            f"{mask_root}"
        )

    split_files = find_split_files(
        source_split_root
    )

    converted: dict[str, set[str]] = {
        split_name: set()
        for split_name in SPLIT_NAMES
    }

    source_entry_counts: dict[str, int] = {
        split_name: 0
        for split_name in SPLIT_NAMES
    }

    duplicate_entries: dict[str, set[str]] = {
        split_name: set()
        for split_name in SPLIT_NAMES
    }

    missing_masks: list[dict[str, str]] = []

    for split_file in split_files:
        split_name = determine_split_name(
            split_file.name
        )

        if split_name is None:
            continue

        with split_file.open(
            "r",
            encoding="utf-8",
            errors="strict",
        ) as file:
            for line_number, line in enumerate(
                file,
                start=1,
            ):
                sample_id = convert_line(line)

                if sample_id is None:
                    continue

                source_entry_counts[split_name] += 1

                if sample_id in converted[split_name]:
                    duplicate_entries[split_name].add(
                        sample_id
                    )

                mask_path = mask_root / f"{sample_id}.png"

                if not mask_path.is_file():
                    missing_masks.append(
                        {
                            "split": split_name,
                            "sample_id": sample_id,
                            "source_file": (
                                split_file
                                .relative_to(source_split_root)
                                .as_posix()
                            ),
                            "line_number": str(line_number),
                        }
                    )
                    continue

                converted[split_name].add(
                    sample_id
                )

    duplicate_count = sum(
        len(sample_ids)
        for sample_ids in duplicate_entries.values()
    )

    if duplicate_count:
        duplicate_examples = {
            split_name: sorted(sample_ids)[:10]
            for split_name, sample_ids
            in duplicate_entries.items()
            if sample_ids
        }

        raise RuntimeError(
            "Duplicate sample IDs were found within split files: "
            f"count={duplicate_count}, "
            f"examples={duplicate_examples}"
        )

    if missing_masks:
        missing_report_path = (
            output_split_root
            / "missing_split_masks.csv"
        )

        if (
            missing_report_path.exists()
            and not overwrite
        ):
            raise FileExistsError(
                "Missing-mask report already exists. "
                "Use --overwrite to replace it: "
                f"{missing_report_path}"
            )

        write_missing_report(
            missing_masks,
            missing_report_path,
        )

        raise RuntimeError(
            "Source split entries without converted masks were found: "
            f"count={len(missing_masks)}. "
            f"See: {missing_report_path}"
        )

    overlaps = find_cross_split_overlaps(
        converted
    )

    overlap_count = sum(
        len(sample_ids)
        for sample_ids in overlaps.values()
    )

    if overlap_count:
        overlap_examples = {
            name: sorted(sample_ids)[:10]
            for name, sample_ids in overlaps.items()
            if sample_ids
        }

        raise RuntimeError(
            "Samples are assigned to multiple splits: "
            f"count={overlap_count}, "
            f"examples={overlap_examples}"
        )

    output_paths = {
        split_name: (
            output_split_root
            / f"{split_name}.txt"
        )
        for split_name in SPLIT_NAMES
    }

    existing_outputs = [
        path
        for path in output_paths.values()
        if path.exists()
    ]

    if existing_outputs and not overwrite:
        raise FileExistsError(
            "Split output files already exist. "
            "Use --overwrite to replace them: "
            f"{existing_outputs}"
        )

    output_split_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    for split_name in SPLIT_NAMES:
        output_path = output_paths[split_name]

        with output_path.open(
            "w",
            encoding="utf-8",
            newline="\n",
        ) as file:
            for sample_id in sorted(
                converted[split_name]
            ):
                file.write(f"{sample_id}\n")

        print(
            f"{split_name}: "
            f"source_entries={source_entry_counts[split_name]}, "
            f"unique_samples={len(converted[split_name])}, "
            f"output={output_path}"
        )

    assigned_samples = set().union(
        *converted.values()
    )

    available_masks = {
        path.stem
        for path in mask_root.glob("*.png")
        if path.is_file()
    }

    unassigned_masks = (
        available_masks - assigned_samples
    )

    print(
        f"Assigned split samples: "
        f"{len(assigned_samples)}"
    )

    print(
        f"Available converted masks: "
        f"{len(available_masks)}"
    )

    print(
        f"Converted masks not assigned to a split: "
        f"{len(unassigned_masks)}"
    )

    print(
        f"Split output directory: "
        f"{output_split_root}"
    )
##

## 0727 split 원본과 처리 결과 경로를 CLI 또는 환경변수로 지정하도록 변경
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source-split-root",
        type=Path,
        default=None,
        help=(
            "Directory containing the original train/val/test "
            "split files. Environment variable: "
            "RELLIS_SPLIT_SOURCE_ROOT"
        ),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Processed dataset root containing masks/. "
            "Converted splits are written to output-root/splits. "
            "Environment variable: RELLIS_OUTPUT_ROOT"
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Replace existing train.txt, val.txt, "
            "and test.txt."
        ),
    )

    args = parser.parse_args()

    resolved_source_split_root = resolve_path(
        args.source_split_root,
        "RELLIS_SPLIT_SOURCE_ROOT",
        DEFAULT_SOURCE_SPLIT_ROOT,
    )

    resolved_output_root = resolve_path(
        args.output_root,
        "RELLIS_OUTPUT_ROOT",
        DEFAULT_OUTPUT_ROOT,
    )

    main(
        source_split_root=resolved_source_split_root,
        output_root=resolved_output_root,
        overwrite=args.overwrite,
    )
##