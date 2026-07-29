from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, UnidentifiedImageError


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
RUGD_REPOSITORY_ROOT = SCRIPT_DIRECTORY.parent
DEFAULT_INPUT_ROOT = RUGD_REPOSITORY_ROOT / "raw"
DEFAULT_MAPPING_PATH = (
    RUGD_REPOSITORY_ROOT
    / "config"
    / "label_mapping.json"
)
DEFAULT_INDEX_RELATIVE_PATH = (
    Path("3.after join creek")
    / "indexLabel"
)


def optional_environment_path(
    variable_name: str,
) -> Path | None:
    value = os.environ.get(variable_name)

    if not value:
        return None

    return Path(value)


def parse_arguments() -> argparse.Namespace:
    environment_input_root = optional_environment_path(
        "RUGD_INPUT_ROOT"
    )

    environment_mapping_path = optional_environment_path(
        "RUGD_MAPPING_PATH"
    )

    parser = argparse.ArgumentParser(
        description=(
            "Analyze the joint distribution of RUGD index-label "
            "values and official RGB color masks, while verifying "
            "file pairing, dimensions, and mapping coverage."
        )
    )

    parser.add_argument(
        "--input-root",
        type=Path,
        default=(
            environment_input_root
            if environment_input_root is not None
            else DEFAULT_INPUT_ROOT
        ),
        help=(
            "Extracted RUGD dataset root. "
            "RUGD_INPUT_ROOT is used when omitted."
        ),
    )

    parser.add_argument(
        "--index-dir",
        type=Path,
        default=optional_environment_path(
            "RUGD_INDEX_MASK_DIR"
        ),
        help=(
            "Index-label directory. Defaults to "
            "'3.after join creek/indexLabel' under input root. "
            "RUGD_INDEX_MASK_DIR may also be used."
        ),
    )

    parser.add_argument(
        "--color-dir",
        type=Path,
        action="append",
        default=None,
        help=(
            "Color-mask directory. This option may be repeated. "
            "When omitted, directories whose names contain both "
            "'indexlabel' and 'color' are discovered under input root."
        ),
    )

    parser.add_argument(
        "--mapping",
        type=Path,
        default=(
            environment_mapping_path
            if environment_mapping_path is not None
            else DEFAULT_MAPPING_PATH
        ),
        help=(
            "Path to label_mapping.json. "
            "RUGD_MAPPING_PATH is used when omitted."
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Verify only the first N paired masks after sorting. "
            "Omit to verify all pairs."
        ),
    )

    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path.expanduser().resolve()


def require_directory(
    directory: Path,
    name: str,
) -> None:
    if not directory.exists():
        raise FileNotFoundError(
            f"{name} directory does not exist: {directory}"
        )

    if not directory.is_dir():
        raise NotADirectoryError(
            f"{name} path is not a directory: {directory}"
        )


def parse_rgb_key(key: str) -> tuple[int, int, int]:
    try:
        value = ast.literal_eval(key)
    except (SyntaxError, ValueError) as error:
        raise ValueError(
            f"Invalid RGB mapping key: {key}"
        ) from error

    if (
        not isinstance(value, tuple)
        or len(value) != 3
        or any(
            not isinstance(channel, int)
            for channel in value
        )
        or any(
            channel < 0 or channel > 255
            for channel in value
        )
    ):
        raise ValueError(
            f"Invalid RGB tuple in mapping: {key}"
        )

    return value


def load_mapping(
    mapping_path: Path,
) -> tuple[
    dict[tuple[int, int, int], str],
    dict[str, int],
    dict[tuple[int, int, int], int],
]:
    if not mapping_path.is_file():
        raise FileNotFoundError(
            f"Mapping file does not exist: {mapping_path}"
        )

    try:
        with mapping_path.open(
            "r",
            encoding="utf-8-sig",
        ) as file:
            mapping_data = json.load(file)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Mapping JSON is invalid: {mapping_path}"
        ) from error

    required_sections = (
        "RGB_TO_NAME",
        "RUGD_TO_ADOM",
        "RGB_TO_ADOM",
    )

    missing_sections = [
        section
        for section in required_sections
        if section not in mapping_data
    ]

    if missing_sections:
        raise ValueError(
            "Mapping JSON is missing sections: "
            f"{missing_sections}"
        )

    rgb_to_name = {
        parse_rgb_key(key): str(value)
        for key, value in mapping_data["RGB_TO_NAME"].items()
        if key != "_comment"
    }

    rugd_to_adom = {
        str(key): int(value)
        for key, value in mapping_data["RUGD_TO_ADOM"].items()
        if key != "_comment"
    }

    rgb_to_adom = {
        parse_rgb_key(key): int(value)
        for key, value in mapping_data["RGB_TO_ADOM"].items()
        if key != "_comment"
    }

    if not rgb_to_name:
        raise ValueError(
            "RGB_TO_NAME contains no mappings."
        )

    if not rugd_to_adom:
        raise ValueError(
            "RUGD_TO_ADOM contains no mappings."
        )

    if not rgb_to_adom:
        raise ValueError(
            "RGB_TO_ADOM contains no mappings."
        )

    if set(rgb_to_name) != set(rgb_to_adom):
        raise ValueError(
            "RGB_TO_NAME and RGB_TO_ADOM use different RGB keys."
        )

    composed_rgb_to_adom: dict[
        tuple[int, int, int],
        int,
    ] = {}

    missing_class_names: list[str] = []

    for rgb, class_name in rgb_to_name.items():
        if class_name not in rugd_to_adom:
            missing_class_names.append(class_name)
            continue

        composed_rgb_to_adom[rgb] = (
            rugd_to_adom[class_name]
        )

    if missing_class_names:
        raise ValueError(
            "RUGD_TO_ADOM is missing class names: "
            f"{sorted(set(missing_class_names))}"
        )

    if composed_rgb_to_adom != rgb_to_adom:
        mismatches = [
            (
                rgb,
                composed_rgb_to_adom.get(rgb),
                rgb_to_adom.get(rgb),
            )
            for rgb in sorted(
                set(composed_rgb_to_adom)
                | set(rgb_to_adom)
            )
            if composed_rgb_to_adom.get(rgb)
            != rgb_to_adom.get(rgb)
        ]

        raise ValueError(
            "RGB_TO_ADOM is inconsistent with "
            "RGB_TO_NAME + RUGD_TO_ADOM: "
            f"{mismatches[:10]}"
        )

    valid_adom_ids = {0, 1, 2, 3, 255}
    actual_adom_ids = set(rgb_to_adom.values())

    if not actual_adom_ids.issubset(valid_adom_ids):
        raise ValueError(
            "Mapping contains invalid ADOM IDs: "
            f"{sorted(actual_adom_ids - valid_adom_ids)}"
        )

    return rgb_to_name, rugd_to_adom, rgb_to_adom


def discover_color_directories(
    input_root: Path,
) -> list[Path]:
    color_directories = sorted(
        (
            path.resolve()
            for path in input_root.iterdir()
            if (
                path.is_dir()
                and "indexlabel" in path.name.casefold()
                and "color" in path.name.casefold()
            )
        ),
        key=lambda path: path.as_posix().casefold(),
    )

    if not color_directories:
        raise FileNotFoundError(
            "No color-mask directories containing both "
            "'indexLabel' and 'color' were found under: "
            f"{input_root}"
        )

    return color_directories


def collect_png_map(
    directories: Iterable[Path],
    name: str,
) -> dict[str, Path]:
    path_map: dict[str, Path] = {}
    duplicate_names: list[str] = []

    for directory in directories:
        require_directory(directory, name)

        for path in sorted(
            (
                candidate
                for candidate in directory.rglob("*")
                if (
                    candidate.is_file()
                    and candidate.suffix.casefold() == ".png"
                )
            ),
            key=lambda candidate: (
                candidate.as_posix().casefold()
            ),
        ):
            normalized_name = path.name.casefold()

            if normalized_name in path_map:
                duplicate_names.append(path.name)
                continue

            path_map[normalized_name] = path

    if duplicate_names:
        raise RuntimeError(
            f"Duplicate PNG file names were found in {name}: "
            f"{sorted(set(duplicate_names))[:10]}"
        )

    if not path_map:
        raise RuntimeError(
            f"No PNG files were found in {name}."
        )

    return path_map


def select_pair_names(
    index_map: dict[str, Path],
    color_map: dict[str, Path],
    limit: int | None,
) -> list[str]:
    index_names = set(index_map)
    color_names = set(color_map)

    only_index = sorted(index_names - color_names)
    only_color = sorted(color_names - index_names)

    if only_index or only_color:
        raise RuntimeError(
            "Index/color mask file-name mismatch. "
            f"Index masks without color masks: {only_index[:10]}; "
            f"Color masks without index masks: {only_color[:10]}"
        )

    pair_names = sorted(index_names)

    if limit is not None:
        if limit <= 0:
            raise ValueError(
                "--limit must be greater than zero."
            )

        pair_names = pair_names[:limit]

    if not pair_names:
        raise RuntimeError(
            "No paired masks were selected."
        )

    return pair_names


def load_index_mask(
    path: Path,
) -> np.ndarray:
    try:
        with Image.open(path) as image:
            if image.format != "PNG":
                raise RuntimeError(
                    f"Index mask is not PNG: {path}"
                )

            array = np.asarray(image)
    except (OSError, UnidentifiedImageError) as error:
        raise RuntimeError(
            f"Failed to read index mask: {path}"
        ) from error

    if array.ndim != 2:
        raise RuntimeError(
            "Index mask is not single-channel: "
            f"{path.name}, shape={array.shape}"
        )

    return array


def load_color_mask(
    path: Path,
) -> np.ndarray:
    try:
        with Image.open(path) as image:
            if image.format != "PNG":
                raise RuntimeError(
                    f"Color mask is not PNG: {path}"
                )

            array = np.asarray(
                image.convert("RGB"),
                dtype=np.uint8,
            )
    except (OSError, UnidentifiedImageError) as error:
        raise RuntimeError(
            f"Failed to read color mask: {path}"
        ) from error

    return array


def verify_pairs(
    pair_names: list[str],
    index_map: dict[str, Path],
    color_map: dict[str, Path],
    rgb_to_name: dict[tuple[int, int, int], str],
) -> dict[int, Counter[tuple[int, int, int]]]:
    id_color_counts: dict[
        int,
        Counter[tuple[int, int, int]],
    ] = defaultdict(Counter)

    known_rgbs = set(rgb_to_name)

    for number, pair_name in enumerate(
        pair_names,
        start=1,
    ):
        index_path = index_map[pair_name]
        color_path = color_map[pair_name]

        index_mask = load_index_mask(index_path)
        color_mask = load_color_mask(color_path)

        if index_mask.shape != color_mask.shape[:2]:
            raise RuntimeError(
                "Index/color mask size mismatch: "
                f"{index_path.name}, "
                f"index={index_mask.shape}, "
                f"color={color_mask.shape[:2]}"
            )

        unique_colors = {
            tuple(int(channel) for channel in color)
            for color in np.unique(
                color_mask.reshape(-1, 3),
                axis=0,
            )
        }

        unknown_colors = sorted(
            unique_colors - known_rgbs
        )

        if unknown_colors:
            raise RuntimeError(
                "Color mask contains RGB values not present in "
                f"label_mapping.json: {index_path.name}, "
                f"unknown={unknown_colors[:10]}"
            )

        for index_id in np.unique(index_mask):
            pixels = color_mask[index_mask == index_id]

            colors, counts = np.unique(
                pixels,
                axis=0,
                return_counts=True,
            )

            observed_colors = [
                tuple(int(channel) for channel in color)
                for color in colors
            ]

            for rgb, count in zip(observed_colors, counts):
                id_color_counts[int(index_id)][rgb] += int(
                    count
                )

        if number % 500 == 0:
            print(
                f"진행: {number}/{len(pair_names)}"
            )

    return id_color_counts

def print_summary(
    input_root: Path,
    index_directory: Path,
    color_directories: list[Path],
    mapping_path: Path,
    total_index_count: int,
    total_color_count: int,
    selected_pair_count: int,
    id_color_counts: dict[
        int,
        Counter[tuple[int, int, int]],
    ],
    rgb_to_name: dict[tuple[int, int, int], str],
    rgb_to_adom: dict[tuple[int, int, int], int],
) -> None:
    print("RUGD index/color mapping verification")
    print(f"입력 루트: {input_root}")
    print(f"indexLabel 폴더: {index_directory}")
    print(f"매핑 파일: {mapping_path}")
    print(f"indexLabel 수: {total_index_count}")
    print(f"색상 마스크 수: {total_color_count}")
    print(f"검증 pair 수: {selected_pair_count}")
    print("색상 마스크 폴더:")

    for directory in color_directories:
        print(f" - {directory}")

    print("\n=== index 값과 RGB 클래스 관측 결과 ===")
    print(
        "참고: 이 데이터 구성에서는 하나의 index 값이 같은 파일 안에서도 "
        "여러 RGB 클래스와 함께 관측될 수 있으므로, index 값을 전역 semantic "
        "class ID로 가정하지 않고 공동 분포만 집계합니다."
    )

    for index_id in sorted(id_color_counts):
        counter = id_color_counts[index_id]
        total = sum(counter.values())

        print(f"\n[index value {index_id}]")
        print(f"  관측 RGB 종류 수: {len(counter)}")

        for rgb, pixel_count in counter.most_common(10):
            ratio = pixel_count / total * 100
            class_name = rgb_to_name[rgb]
            adom_id = rgb_to_adom[rgb]

            print(
                f"  {rgb} -> {class_name} -> ADOM {adom_id}: "
                f"{pixel_count:,} pixels ({ratio:.4f}%)"
            )

    print("\nIndex/RGB joint-distribution collection: PASS")
    print("Pairing, dimensions, and RGB coverage: PASS")
    print("Mapping JSON consistency: PASS")
    print(
        "[PASS] RUGD index/color mapping "
        "verification completed."
    )

def main() -> int:
    arguments = parse_arguments()

    input_root = resolve_path(arguments.input_root)
    mapping_path = resolve_path(arguments.mapping)

    try:
        require_directory(input_root, "Input root")

        index_directory = resolve_path(
            arguments.index_dir
            if arguments.index_dir is not None
            else input_root / DEFAULT_INDEX_RELATIVE_PATH
        )

        require_directory(
            index_directory,
            "Index-label",
        )

        if arguments.color_dir:
            color_directories = [
                resolve_path(path)
                for path in arguments.color_dir
            ]
        else:
            color_directories = (
                discover_color_directories(input_root)
            )

        rgb_to_name, _, rgb_to_adom = load_mapping(
            mapping_path
        )

        index_map = collect_png_map(
            [index_directory],
            "index-label directory",
        )

        color_map = collect_png_map(
            color_directories,
            "color-mask directories",
        )

        pair_names = select_pair_names(
            index_map=index_map,
            color_map=color_map,
            limit=arguments.limit,
        )

        id_color_counts = verify_pairs(
            pair_names=pair_names,
            index_map=index_map,
            color_map=color_map,
            rgb_to_name=rgb_to_name,
        )

    except (
        FileNotFoundError,
        NotADirectoryError,
        RuntimeError,
        ValueError,
    ) as error:
        print(
            f"[ERROR] {error}",
            file=sys.stderr,
        )

        return 1

    print_summary(
        input_root=input_root,
        index_directory=index_directory,
        color_directories=color_directories,
        mapping_path=mapping_path,
        total_index_count=len(index_map),
        total_color_count=len(color_map),
        selected_pair_count=len(pair_names),
        id_color_counts=id_color_counts,
        rgb_to_name=rgb_to_name,
        rgb_to_adom=rgb_to_adom,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())