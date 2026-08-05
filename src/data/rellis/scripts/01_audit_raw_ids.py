from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image


MASK_DIRECTORY_NAME = "pylon_camera_node_label_id"
MASK_SUFFIXES = {".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit all raw RELLIS-3D image-label IDs "
            "before Semantic20 mapping."
        )
    )

    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help=(
            "RELLIS-3D root containing sequence folders "
            "such as 00000, 00001, and 00002."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for generated audit reports.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of masks to inspect.",
    )

    return parser.parse_args()


def read_id_mask(mask_path: Path) -> np.ndarray:
    with Image.open(mask_path) as image:
        image.load()
        mask = np.asarray(image)

    if mask.ndim == 2:
        result = mask
    elif mask.ndim == 3:
        channels_equal = all(
            np.array_equal(mask[:, :, 0], mask[:, :, index])
            for index in range(1, mask.shape[2])
        )

        if not channels_equal:
            raise ValueError(
                "Multi-channel ID mask contains unequal channels."
            )

        result = mask[:, :, 0]
    else:
        raise ValueError(
            f"Unsupported mask shape: {mask.shape}"
        )

    if not np.issubdtype(result.dtype, np.integer):
        raise ValueError(
            f"Mask dtype must be integer: {result.dtype}"
        )

    return result


def discover_sequence_directories(
    input_root: Path,
) -> list[Path]:
    if (
        input_root / MASK_DIRECTORY_NAME
    ).is_dir():
        return [input_root]

    sequence_directories = sorted(
        path
        for path in input_root.iterdir()
        if (
            path.is_dir()
            and (
                path / MASK_DIRECTORY_NAME
            ).is_dir()
        )
    )

    return sequence_directories


def write_global_statistics(
    output_path: Path,
    pixel_counts: Counter[int],
    image_counts: Counter[int],
    sequence_counts: Counter[int],
) -> None:
    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.writer(file)

        writer.writerow(
            [
                "raw_id",
                "pixel_count",
                "image_presence_count",
                "sequence_count",
            ]
        )

        for raw_id in sorted(pixel_counts):
            writer.writerow(
                [
                    raw_id,
                    pixel_counts[raw_id],
                    image_counts[raw_id],
                    sequence_counts[raw_id],
                ]
            )


def write_sequence_statistics(
    output_path: Path,
    sequence_pixel_counts: dict[str, Counter[int]],
    sequence_image_counts: dict[str, Counter[int]],
) -> None:
    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.writer(file)

        writer.writerow(
            [
                "sequence",
                "raw_id",
                "pixel_count",
                "image_presence_count",
            ]
        )

        for sequence_name in sorted(
            sequence_pixel_counts
        ):
            pixel_counts = (
                sequence_pixel_counts[
                    sequence_name
                ]
            )

            image_counts = (
                sequence_image_counts[
                    sequence_name
                ]
            )

            for raw_id in sorted(pixel_counts):
                writer.writerow(
                    [
                        sequence_name,
                        raw_id,
                        pixel_counts[raw_id],
                        image_counts[raw_id],
                    ]
                )


def main() -> None:
    args = parse_args()

    input_root = args.input_root.resolve()
    output_dir = args.output_dir.resolve()

    if not input_root.is_dir():
        raise FileNotFoundError(
            f"Input root does not exist: {input_root}"
        )

    if (
        args.limit is not None
        and args.limit <= 0
    ):
        raise ValueError(
            "--limit must be greater than zero."
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    sequence_directories = (
        discover_sequence_directories(
            input_root
        )
    )

    if not sequence_directories:
        raise RuntimeError(
            "No RELLIS sequence directories containing "
            f"{MASK_DIRECTORY_NAME!r} were found."
        )

    global_pixel_counts: Counter[int] = Counter()
    global_image_counts: Counter[int] = Counter()
    global_sequence_counts: Counter[int] = Counter()

    sequence_pixel_counts: dict[
        str,
        Counter[int],
    ] = {}

    sequence_image_counts: dict[
        str,
        Counter[int],
    ] = {}

    errors: list[dict[str, str]] = []
    inspected_mask_count = 0
    stop_requested = False

    for sequence_directory in sequence_directories:
        sequence_name = sequence_directory.name
        mask_directory = (
            sequence_directory
            / MASK_DIRECTORY_NAME
        )

        current_pixel_counts: Counter[int] = (
            Counter()
        )

        current_image_counts: Counter[int] = (
            Counter()
        )

        mask_paths = sorted(
            path
            for path in mask_directory.rglob("*")
            if (
                path.is_file()
                and path.suffix.lower()
                in MASK_SUFFIXES
            )
        )

        for mask_path in mask_paths:
            if (
                args.limit is not None
                and inspected_mask_count
                >= args.limit
            ):
                stop_requested = True
                break

            relative_path = mask_path.relative_to(
                input_root
            )

            try:
                mask = read_id_mask(mask_path)

                raw_ids, counts = np.unique(
                    mask,
                    return_counts=True,
                )

                image_raw_ids: set[int] = set()

                for raw_id_value, count_value in zip(
                    raw_ids,
                    counts,
                    strict=True,
                ):
                    raw_id = int(raw_id_value)
                    pixel_count = int(count_value)

                    global_pixel_counts[
                        raw_id
                    ] += pixel_count

                    current_pixel_counts[
                        raw_id
                    ] += pixel_count

                    image_raw_ids.add(raw_id)

                for raw_id in image_raw_ids:
                    global_image_counts[
                        raw_id
                    ] += 1

                    current_image_counts[
                        raw_id
                    ] += 1

                inspected_mask_count += 1

            except Exception as error:
                errors.append(
                    {
                        "mask": relative_path.as_posix(),
                        "error": str(error),
                    }
                )

        sequence_pixel_counts[
            sequence_name
        ] = current_pixel_counts

        sequence_image_counts[
            sequence_name
        ] = current_image_counts

        for raw_id in current_pixel_counts:
            global_sequence_counts[
                raw_id
            ] += 1

        if stop_requested:
            break

    if inspected_mask_count == 0:
        errors.append(
            {
                "mask": "",
                "error": (
                    "No readable mask files were inspected."
                ),
            }
        )

    observed_raw_ids = sorted(
        global_pixel_counts
    )

    write_global_statistics(
        output_dir / "raw_id_statistics.csv",
        global_pixel_counts,
        global_image_counts,
        global_sequence_counts,
    )

    write_sequence_statistics(
        output_dir
        / "sequence_raw_id_statistics.csv",
        sequence_pixel_counts,
        sequence_image_counts,
    )

    summary = {
        "dataset": "rellis3d_semantic20_v1",
        "inspection_scope": (
            "raw RELLIS-3D ID masks"
        ),
        "sequence_count": len(
            sequence_pixel_counts
        ),
        "inspected_mask_count": (
            inspected_mask_count
        ),
        "observed_raw_ids": observed_raw_ids,
        "observed_raw_id_count": len(
            observed_raw_ids
        ),
        "error_count": len(errors),
        "errors": errors,
    }

    summary_path = (
        output_dir
        / "raw_id_audit_summary.json"
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
            ensure_ascii=False,
        )

        file.write("\n")

    print(
        f"Sequences inspected: "
        f"{summary['sequence_count']}"
    )

    print(
        f"Masks inspected: "
        f"{inspected_mask_count}"
    )

    print(
        "Observed raw IDs: "
        + ", ".join(
            str(raw_id)
            for raw_id in observed_raw_ids
        )
    )

    print(
        f"Errors: {len(errors)}"
    )

    print(
        f"Reports written to: "
        f"{output_dir}"
    )

    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
