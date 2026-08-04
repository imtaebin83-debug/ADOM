from __future__ import annotations

import argparse
import csv
import shutil
import os
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "Rellis-3D"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "rellis_cost4_standard"
)

MASK_OUTPUT_DIR = OUTPUT_ROOT / "masks"

MAPPING_PATH = (
    PROJECT_ROOT
    / "config"
    / "class_mapping.yaml"
)

## 0727 CLI-환경변수 경로 처리 함수 추가
def resolve_path(
    cli_value: Path | None,
    env_name: str,
    default: Path,
) -> Path:
    """
    Resolve a path in the following order:
    CLI argument -> environment variable -> repository-relative default.
    """
    value = cli_value or os.getenv(env_name)

    if value is None:
        return default.resolve()

    return Path(value).expanduser().resolve()
##

RGB_SUFFIXES = {".jpg", ".jpeg", ".png"}
MASK_SUFFIXES = {".png"}


def collect_files(
    directory: Path,
    allowed_suffixes: set[str],
) -> dict[str, Path]:
    """
    지정한 폴더에서 허용된 확장자의 파일만 수집한다.

    반환 형식:
    {
        "파일 stem": Path 객체
    }
    """
    if not directory.exists():
        raise FileNotFoundError(
            f"폴더가 존재하지 않습니다: {directory}"
        )

    files: dict[str, Path] = {}

    for path in sorted(directory.iterdir()):
        if not path.is_file():
            continue

        if path.suffix.lower() not in allowed_suffixes:
            continue

        if path.stem in files:
            raise ValueError(
                "동일한 stem을 가진 파일이 중복되었습니다.\n"
                f"기존 파일: {files[path.stem]}\n"
                f"중복 파일: {path}"
            )

        files[path.stem] = path

    return files


def read_id_mask(mask_path: Path) -> np.ndarray:
    """
    RELLIS ID mask를 2차원 numpy 배열로 읽는다.
    """
    with Image.open(mask_path) as image:
        mask = np.asarray(image)

    if mask.ndim == 2:
        return mask

    if mask.ndim == 3:
        channels_equal = all(
            np.array_equal(
                mask[:, :, 0],
                mask[:, :, channel_index],
            )
            for channel_index in range(
                1,
                mask.shape[2],
            )
        )

        if not channels_equal:
            raise ValueError(
                "3채널 mask의 채널 값이 서로 다릅니다: "
                f"{mask_path}"
            )

        return mask[:, :, 0]

    raise ValueError(
        f"지원하지 않는 mask shape입니다: "
        f"{mask.shape}, 파일={mask_path}"
    )


def load_mapping() -> tuple[dict[int, int], dict]:
    """
    class_mapping.yaml에서 RELLIS → target ID 매핑을 읽는다.
    """
    if not MAPPING_PATH.exists():
        raise FileNotFoundError(
            f"매핑 파일이 없습니다: {MAPPING_PATH}"
        )

    with MAPPING_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    mapping = {
        int(source_id): int(target_id)
        for source_id, target_id
        in config["rellis_to_target"].items()
    }

    return mapping, config


def collect_rgb_mask_pairs() -> list[
    tuple[str, str, Path, Path]
]:
    """
    각 sequence에서 RGB와 PNG ID mask가 모두 존재하는
    교집합만 수집한다.
    """
    if not RAW_ROOT.exists():
        raise FileNotFoundError(
            f"원본 데이터 폴더가 없습니다: {RAW_ROOT}"
        )

    sequence_dirs = sorted(
        path
        for path in RAW_ROOT.iterdir()
        if path.is_dir()
        and path.name.isdigit()
    )

    if not sequence_dirs:
        raise RuntimeError(
            f"sequence 폴더를 찾지 못했습니다: {RAW_ROOT}"
        )

    pairs: list[
        tuple[str, str, Path, Path]
    ] = []

    for sequence_dir in sequence_dirs:
        sequence = sequence_dir.name

        rgb_dir = (
            sequence_dir
            / "pylon_camera_node"
        )

        mask_dir = (
            sequence_dir
            / "pylon_camera_node_label_id"
        )

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

        ##0727 누락 RGB-mask pair를 오류로 처리
        ## unlabeled RGB는 제외하고, 대응 RGB가 없는 mask만 오류 처리
        unlabeled_rgb_stems = sorted(
            rgb_stems - mask_stems
        )

        missing_rgb_stems = sorted(
            mask_stems - rgb_stems
        )

        if unlabeled_rgb_stems:
            print(
                f"[INFO] sequence {sequence}: "
                f"{len(unlabeled_rgb_stems)} unlabeled RGB frame(s) "
                "will be excluded from mask conversion."
            )

        if missing_rgb_stems:
            raise RuntimeError(
                f"Mask without matching RGB in sequence {sequence}: "
                f"count={len(missing_rgb_stems)} "
                f"examples={missing_rgb_stems[:10]}"
            )
        ##

        paired_stems = sorted(
            rgb_stems & mask_stems
        )

        for stem in paired_stems:
            rgb_path = rgb_files[stem]
            mask_path = mask_files[stem]

            pairs.append(
                (
                    sequence,
                    stem,
                    rgb_path,
                    mask_path,
                )
            )

    ##0727 pair가 0개인 경우 오류 추가
    if not pairs:
        raise RuntimeError(
            f"No valid RGB-mask pairs were found under: {RAW_ROOT}"
        )
    ##

    return pairs


def main(
    limit: int | None,
    overwrite: bool,
) -> None:
    mapping, _ = load_mapping()

    valid_source_ids = set(mapping)

    lookup = np.full(
        65536,
        255,
        dtype=np.uint8,
    )

    for source_id, target_id in mapping.items():
        if source_id < 0 or source_id >= len(lookup):
            raise ValueError(
                f"매핑 ID 범위가 잘못되었습니다: "
                f"{source_id}"
            )

        lookup[source_id] = target_id

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    MASK_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    pairs = collect_rgb_mask_pairs()

    if limit is not None:
        pairs = pairs[:limit]

    metadata_rows: list[
        dict[str, str | int | float]
    ] = []

    for (
        sequence,
        stem,
        rgb_path,
        mask_path,
    ) in tqdm(
        pairs,
        desc="4-class mask 변환",
    ):
        sample_id = f"{sequence}_{stem}"

        output_path = (
            MASK_OUTPUT_DIR
            / f"{sample_id}.png"
        )

        try:
            mask = read_id_mask(mask_path)

            if mask.size == 0:
                raise ValueError(
                    f"빈 mask입니다: {mask_path}"
                )

            mask_min = int(np.min(mask))
            mask_max = int(np.max(mask))

            if mask_min < 0:
                raise ValueError(
                    f"음수 클래스 ID가 있습니다: "
                    f"{mask_min}"
                )

            if mask_max >= len(lookup):
                raise ValueError(
                    f"mask ID 범위가 너무 큽니다: "
                    f"{mask_min}~{mask_max}"
                )

            with Image.open(rgb_path) as rgb_image:
                rgb_size = rgb_image.size

            mask_size = (
                mask.shape[1],
                mask.shape[0],
            )

            if rgb_size != mask_size:
                raise ValueError(
                    "RGB-mask 해상도가 다릅니다: "
                    f"RGB={rgb_size}, "
                    f"mask={mask_size}"
                )

            source_ids = sorted(
                int(value)
                for value in np.unique(mask)
            )

            unknown_ids = sorted(
                set(source_ids)
                - valid_source_ids
            )

            if unknown_ids:
                raise ValueError(
                    "매핑되지 않은 원본 ID가 있습니다: "
                    + ";".join(
                        map(str, unknown_ids)
                    )
                )

            converted = lookup[
                mask.astype(np.int64)
            ]

            output_ids = sorted(
                int(value)
                for value in np.unique(converted)
            )

            valid_target_ids = {
                0,
                1,
                2,
                3,
                255,
            }

            invalid_target_ids = sorted(
                set(output_ids)
                - valid_target_ids
            )

            if invalid_target_ids:
                raise ValueError(
                    "변환 후 허용되지 않은 ID가 있습니다: "
                    + ";".join(
                        map(str, invalid_target_ids)
                    )
                )

            if overwrite or not output_path.exists():
                Image.fromarray(
                    converted.astype(np.uint8),
                    mode="L",
                ).save(output_path)

            ignore_ratio = float(
                np.count_nonzero(
                    converted == 255
                )
                / converted.size
            )

            metadata_rows.append(
                {
                    "sample_id": sample_id,
                    "sequence": sequence,
                    "original_stem": stem,
                    "rgb_path": (
                        rgb_path
                        .relative_to(RAW_ROOT)
                        .as_posix()
                    ),
                    "source_mask_path": (
                        mask_path
                        .relative_to(RAW_ROOT)
                        .as_posix()
                    ),
                    "converted_mask_path": (
                        output_path
                        .relative_to(OUTPUT_ROOT)
                        .as_posix()
                    ),
                    "width": rgb_size[0],
                    "height": rgb_size[1],
                    "source_ids": ";".join(
                        map(str, source_ids)
                    ),
                    "unknown_source_ids": "",
                    "output_ids": ";".join(
                        map(str, output_ids)
                    ),
                    "ignore_ratio": ignore_ratio,
                    "status": "ok",
                }
            )

        except Exception as error:
            metadata_rows.append(
                {
                    "sample_id": sample_id,
                    "sequence": sequence,
                    "original_stem": stem,
                    "rgb_path": (
                        rgb_path
                        .relative_to(RAW_ROOT)
                        .as_posix()
                    ),
                    "source_mask_path": (
                        mask_path
                        .relative_to(RAW_ROOT)
                        .as_posix()
                    ),
                    "converted_mask_path": "",
                    "width": "",
                    "height": "",
                    "source_ids": "",
                    "unknown_source_ids": "",
                    "output_ids": "",
                    "ignore_ratio": "",
                    "status": f"error: {error}",
                }
            )

    metadata_path = (
        OUTPUT_ROOT
        / "metadata.csv"
    )

    fieldnames = [
        "sample_id",
        "sequence",
        "original_stem",
        "rgb_path",
        "source_mask_path",
        "converted_mask_path",
        "width",
        "height",
        "source_ids",
        "unknown_source_ids",
        "output_ids",
        "ignore_ratio",
        "status",
    ]

    with metadata_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(metadata_rows)

    shutil.copy2(
        MAPPING_PATH,
        OUTPUT_ROOT / "class_mapping.yaml",
    )

    success_count = sum(
        row["status"] == "ok"
        for row in metadata_rows
    )

    error_count = (
        len(metadata_rows)
        - success_count
    )

    print(
        f"전체 처리: {len(metadata_rows)}"
    )

    print(
        f"정상 변환: {success_count}"
    )

    print(
        f"오류: {error_count}"
    )

    print(
        f"출력 mask 폴더: "
        f"{MASK_OUTPUT_DIR}"
    )

    print(
        f"metadata: {metadata_path}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ##0727 CLI 인자 추가
    parser.add_argument(
        "--input-root",
        type=Path,
        default=None,
        help=(
            "RELLIS-3D raw dataset root. "
            "Environment variable: RELLIS_INPUT_ROOT"
        ),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Directory for converted masks and metadata. "
            "Environment variable: RELLIS_OUTPUT_ROOT"
        ),
    )

    parser.add_argument(
        "--mapping",
        type=Path,
        default=None,
        help=(
            "Path to class_mapping.yaml. "
            "Environment variable: RELLIS_MAPPING_PATH"
        ),
    )
    ##

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="시험용 최대 변환 개수",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="기존 변환 mask 덮어쓰기",
    )

    args = parser.parse_args()

    ##0727 CLI 인자 추가
    RAW_ROOT = resolve_path(
        args.input_root,
        "RELLIS_INPUT_ROOT",
        RAW_ROOT,
    )

    OUTPUT_ROOT = resolve_path(
        args.output_root,
        "RELLIS_OUTPUT_ROOT",
        OUTPUT_ROOT,
    )

    MAPPING_PATH = resolve_path(
        args.mapping,
        "RELLIS_MAPPING_PATH",
        MAPPING_PATH,
    )

    MASK_OUTPUT_DIR = OUTPUT_ROOT / "masks"
    ##

    main(
        limit=args.limit,
        overwrite=args.overwrite,
    )