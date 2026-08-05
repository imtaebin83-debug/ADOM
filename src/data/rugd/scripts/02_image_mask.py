from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
RUGD_REPOSITORY_ROOT = SCRIPT_DIRECTORY.parent
DEFAULT_INPUT_ROOT = RUGD_REPOSITORY_ROOT / "raw"

DEFAULT_IMAGE_RELATIVE_PATH = (
    Path("3.after join creek")
    / "image"
)

DEFAULT_MASK_RELATIVE_PATH = (
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

    default_input_root = (
        environment_input_root
        if environment_input_root is not None
        else DEFAULT_INPUT_ROOT
    )

    parser = argparse.ArgumentParser(
        description=(
            "Inspect RUGD image and index-label PNG files "
            "and verify file-name pairing."
        )
    )

    parser.add_argument(
        "--input-root",
        type=Path,
        default=default_input_root,
        help=(
            "Extracted RUGD dataset root. "
            "RUGD_INPUT_ROOT is used when omitted."
        ),
    )

    parser.add_argument(
        "--image-dir",
        type=Path,
        default=optional_environment_path(
            "RUGD_IMAGE_DIR"
        ),
        help=(
            "Image directory. Defaults to "
            "'3.after join creek/image' under input root. "
            "RUGD_IMAGE_DIR may also be used."
        ),
    )

    parser.add_argument(
        "--mask-dir",
        type=Path,
        default=optional_environment_path(
            "RUGD_INDEX_MASK_DIR"
        ),
        help=(
            "Index-label mask directory. Defaults to "
            "'3.after join creek/indexLabel' under input root. "
            "RUGD_INDEX_MASK_DIR may also be used."
        ),
    )

    parser.add_argument(
        "--example-count",
        type=int,
        default=3,
        help=(
            "Number of image and mask example paths "
            "to print. Default: 3."
        ),
    )

    return parser.parse_args()


def resolve_directory(
    explicit_directory: Path | None,
    input_root: Path,
    default_relative_path: Path,
) -> Path:
    directory = (
        explicit_directory
        if explicit_directory is not None
        else input_root / default_relative_path
    )

    return directory.expanduser().resolve()


def collect_png_map(
    directory: Path,
    directory_name: str,
) -> dict[str, Path]:
    if not directory.exists():
        raise FileNotFoundError(
            f"{directory_name} directory does not exist: "
            f"{directory}"
        )

    if not directory.is_dir():
        raise NotADirectoryError(
            f"{directory_name} path is not a directory: "
            f"{directory}"
        )

    png_paths = sorted(
        (
            path
            for path in directory.rglob("*")
            if (
                path.is_file()
                and path.suffix.lower() == ".png"
            )
        ),
        key=lambda path: path.as_posix().casefold(),
    )

    if not png_paths:
        raise RuntimeError(
            f"No PNG files were found in "
            f"{directory_name} directory: {directory}"
        )

    path_map: dict[str, Path] = {}
    duplicate_names: list[str] = []

    for path in png_paths:
        normalized_name = path.name.casefold()

        if normalized_name in path_map:
            duplicate_names.append(path.name)
            continue

        path_map[normalized_name] = path

    if duplicate_names:
        raise RuntimeError(
            f"Duplicate PNG file names were found in "
            f"{directory_name} directory: "
            f"{sorted(set(duplicate_names))[:10]}"
        )

    return path_map


def relative_display_path(
    path: Path,
    directory: Path,
) -> str:
    try:
        return path.relative_to(directory).as_posix()
    except ValueError:
        return str(path)


def validate_pairing(
    image_map: dict[str, Path],
    mask_map: dict[str, Path],
) -> tuple[list[str], list[str]]:
    image_names = set(image_map)
    mask_names = set(mask_map)

    only_images = sorted(
        image_names - mask_names
    )

    only_masks = sorted(
        mask_names - image_names
    )

    if only_images or only_masks:
        raise RuntimeError(
            "Image-mask file-name mismatch. "
            f"Images without masks: {only_images[:10]}; "
            f"Masks without images: {only_masks[:10]}"
        )

    return only_images, only_masks


def print_summary(
    image_directory: Path,
    mask_directory: Path,
    image_map: dict[str, Path],
    mask_map: dict[str, Path],
    example_count: int,
) -> None:
    paired_count = len(
        set(image_map) & set(mask_map)
    )

    print(
        "Image folder exists:",
        image_directory.exists(),
    )

    print(
        "Mask folder exists :",
        mask_directory.exists(),
    )

    print("Images:", len(image_map))
    print("Masks :", len(mask_map))
    print("Paired files:", paired_count)
    print("Only images: 0")
    print("Only masks : 0")

    print("\nImage example:")

    for path in list(image_map.values())[:example_count]:
        print(
            relative_display_path(
                path,
                image_directory,
            )
        )

    print("\nMask example:")

    for path in list(mask_map.values())[:example_count]:
        print(
            relative_display_path(
                path,
                mask_directory,
            )
        )

    print(
        "\n[PASS] RUGD image-mask structure "
        "inspection completed."
    )


def main() -> int:
    arguments = parse_arguments()

    if arguments.example_count < 0:
        print(
            "[ERROR] --example-count must be "
            "zero or greater.",
            file=sys.stderr,
        )

        return 1

    input_root = (
        arguments.input_root
        .expanduser()
        .resolve()
    )

    image_directory = resolve_directory(
        explicit_directory=arguments.image_dir,
        input_root=input_root,
        default_relative_path=(
            DEFAULT_IMAGE_RELATIVE_PATH
        ),
    )

    mask_directory = resolve_directory(
        explicit_directory=arguments.mask_dir,
        input_root=input_root,
        default_relative_path=(
            DEFAULT_MASK_RELATIVE_PATH
        ),
    )

    try:
        image_map = collect_png_map(
            directory=image_directory,
            directory_name="Image",
        )

        mask_map = collect_png_map(
            directory=mask_directory,
            directory_name="Mask",
        )

        validate_pairing(
            image_map=image_map,
            mask_map=mask_map,
        )
    except (
        FileNotFoundError,
        NotADirectoryError,
        RuntimeError,
    ) as error:
        print(
            f"[ERROR] {error}",
            file=sys.stderr,
        )

        return 1

    print_summary(
        image_directory=image_directory,
        mask_directory=mask_directory,
        image_map=image_map,
        mask_map=mask_map,
        example_count=arguments.example_count,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())