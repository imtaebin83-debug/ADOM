from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import yaml
from PIL import Image


RGB_DIRECTORY_NAME = "pylon_camera_node"
MASK_DIRECTORY_NAME = "pylon_camera_node_label_id"

RGB_SUFFIXES = {".jpg", ".jpeg", ".png"}
MASK_SUFFIXES = {".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert RELLIS-3D raw semantic masks "
            "to contiguous Semantic20 train IDs."
        )
    )

    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--mapping",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


def collect_files(
    directory: Path,
    suffixes: set[str],
) -> dict[str, Path]:
    if not directory.is_dir():
        return {}

    files: dict[str, Path] = {}

    for path in sorted(directory.iterdir()):
        if (
            path.is_file()
            and path.suffix.lower() in suffixes
        ):
            if path.stem in files:
                raise RuntimeError(
                    f"Duplicate file stem: {path.stem}"
                )

            files[path.stem] = path

    return files


def read_id_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        image.load()
        mask = np.asarray(image)

    if mask.ndim == 2:
        result = mask
    elif mask.ndim == 3:
        channels_equal = all(
            np.array_equal(
                mask[:, :, 0],
                mask[:, :, index],
            )
            for index in range(1, mask.shape[2])
        )

        if not channels_equal:
            raise ValueError(
                f"Mask channels differ: {path}"
            )

        result = mask[:, :, 0]
    else:
        raise ValueError(
            f"Unsupported mask shape {mask.shape}: {path}"
        )

    if not np.issubdtype(
        result.dtype,
        np.integer,
    ):
        raise ValueError(
            f"Mask must use an integer dtype: {path}"
        )

    return result


def load_mapping(
    mapping_path: Path,
) -> tuple[dict[int, int], int]:
    with mapping_path.open(
        "r",
        encoding="utf-8-sig",
    ) as file:
        config = yaml.safe_load(file)

    mapping = {
        int(source_id): int(target_id)
        for source_id, target_id
        in config["rellis_to_target"].items()
    }

    num_classes = int(config["num_classes"])
    ignore_index = int(config["ignore_index"])

    expected_target_ids = (
        set(range(num_classes))
        | {ignore_index}
    )

    actual_target_ids = set(mapping.values())

    if actual_target_ids != expected_target_ids:
        raise ValueError(
            "Mapping target IDs do not match "
            f"0..{num_classes - 1} plus "
            f"{ignore_index}: "
            f"{sorted(actual_target_ids)}"
        )

    return mapping, ignore_index


def remap_mask(
    source_mask: np.ndarray,
    mapping: dict[int, int],
    ignore_index: int,
) -> np.ndarray:
    observed_ids = np.unique(source_mask)

    unknown_ids = {
        int(value)
        for value in observed_ids
        if int(value) not in mapping
    }

    if unknown_ids:
        raise ValueError(
            "Unknown raw mask IDs: "
            f"{sorted(unknown_ids)}"
        )

    maximum_source_id = max(mapping)

    lookup_table = np.full(
        maximum_source_id + 1,
        fill_value=ignore_index,
        dtype=np.uint8,
    )

    for source_id, target_id in mapping.items():
        lookup_table[source_id] = target_id

    return lookup_table[source_mask]


def main() -> None:
    args = parse_args()

    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    mapping_path = args.mapping.resolve()

    if not input_root.is_dir():
        raise FileNotFoundError(
            f"Input root does not exist: {input_root}"
        )

    if not mapping_path.is_file():
        raise FileNotFoundError(
            f"Mapping file does not exist: {mapping_path}"
        )

    if (
        args.limit is not None
        and args.limit <= 0
    ):
        raise ValueError(
            "--limit must be greater than zero."
        )

    mapping, ignore_index = load_mapping(
        mapping_path
    )

    samples: list[
        tuple[str, Path, Path]
    ] = []

    sequence_directories = sorted(
        path
        for path in input_root.iterdir()
        if path.is_dir()
    )

    for sequence_directory in sequence_directories:
        rgb_files = collect_files(
            sequence_directory
            / RGB_DIRECTORY_NAME,
            RGB_SUFFIXES,
        )

        mask_files = collect_files(
            sequence_directory
            / MASK_DIRECTORY_NAME,
            MASK_SUFFIXES,
        )

        if not rgb_files and not mask_files:
            continue

        rgb_only = (
            set(rgb_files) - set(mask_files)
        )

        mask_only = (
            set(mask_files) - set(rgb_files)
        )

        if mask_only:
            raise RuntimeError(
                f"Mask-only files in "
                f"{sequence_directory.name}: "
                f"{len(mask_only)}"
            )

        paired_stems = sorted(
            set(rgb_files) & set(mask_files)
        )

        print(
            f"[{sequence_directory.name}] "
            f"RGB={len(rgb_files)}, "
            f"mask={len(mask_files)}, "
            f"paired={len(paired_stems)}, "
            f"RGB-only={len(rgb_only)}"
        )

        for stem in paired_stems:
            samples.append(
                (
                    sequence_directory.name,
                    rgb_files[stem],
                    mask_files[stem],
                )
            )

    if not samples:
        raise RuntimeError(
            "No paired RELLIS-3D samples were found."
        )

    if args.limit is not None:
        samples = samples[: args.limit]

    converted_count = 0

    for sequence_name, rgb_path, mask_path in samples:
        output_image_directory = (
            output_root
            / "images"
            / sequence_name
        )

        output_mask_directory = (
            output_root
            / "masks"
            / sequence_name
        )

        output_image_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_mask_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_image_path = (
            output_image_directory
            / rgb_path.name
        )

        output_mask_path = (
            output_mask_directory
            / f"{mask_path.stem}.png"
        )

        if (
            not args.overwrite
            and (
                output_image_path.exists()
                or output_mask_path.exists()
            )
        ):
            raise FileExistsError(
                f"Output already exists: "
                f"{output_mask_path}"
            )

        source_mask = read_id_mask(mask_path)

        target_mask = remap_mask(
            source_mask,
            mapping,
            ignore_index,
        )

        shutil.copy2(
            rgb_path,
            output_image_path,
        )

        Image.fromarray(
            target_mask,
            mode="L",
        ).save(output_mask_path)

        converted_count += 1

    print(
        f"Converted samples: {converted_count}"
    )

    print(
        f"Output root: {output_root}"
    )


if __name__ == "__main__":
    main()
