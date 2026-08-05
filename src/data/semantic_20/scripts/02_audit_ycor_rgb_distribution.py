from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze YCOR semantic class distribution "
            "using RGB palette values."
        )
    )
    parser.add_argument(
        "--ycor-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--source-map",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    return parser.parse_args()


def load_source_mapping(
    path: Path,
) -> tuple[
    dict[int, dict],
    dict[int, dict],
]:
    with path.open(
        "r",
        encoding="utf-8-sig",
    ) as file:
        config = json.load(file)

    entries = config.get("source_palette_rgb")

    if not isinstance(entries, list):
        raise KeyError(
            "source_palette_rgb not found."
        )

    rgb_key_to_entry = {}
    source_id_to_entry = {}

    for entry in entries:
        source_id = int(
            entry["source_index_if_indexed"]
        )
        rgb = tuple(
            int(value)
            for value in entry["rgb"]
        )

        if len(rgb) != 3:
            raise ValueError(
                f"Invalid RGB value: {rgb}"
            )

        key = (
            (rgb[0] << 16)
            | (rgb[1] << 8)
            | rgb[2]
        )

        if key in rgb_key_to_entry:
            raise ValueError(
                f"Duplicate RGB value: {rgb}"
            )

        rgb_key_to_entry[key] = entry
        source_id_to_entry[source_id] = entry

    return rgb_key_to_entry, source_id_to_entry


def get_class_name(entry: dict) -> str:
    for key in (
        "source_name",
        "class_name",
        "name",
        "label",
    ):
        value = entry.get(key)

        if value is not None:
            return str(value)

    return "UNKNOWN_NAME"


def load_source_ids(
    mask_path: Path,
    rgb_key_to_entry: dict[int, dict],
) -> np.ndarray:
    with Image.open(mask_path) as image:
        image.load()

        if image.mode in ("P", "RGB", "RGBA"):
            rgb = np.asarray(
                image.convert("RGB"),
                dtype=np.uint8,
            )

            packed = (
                (
                    rgb[:, :, 0].astype(np.uint32)
                    << 16
                )
                | (
                    rgb[:, :, 1].astype(np.uint32)
                    << 8
                )
                | rgb[:, :, 2].astype(np.uint32)
            )

            unique_keys, inverse = np.unique(
                packed,
                return_inverse=True,
            )

            unknown_keys = [
                int(key)
                for key in unique_keys
                if int(key) not in rgb_key_to_entry
            ]

            if unknown_keys:
                unknown_rgb = [
                    (
                        (key >> 16) & 255,
                        (key >> 8) & 255,
                        key & 255,
                    )
                    for key in unknown_keys[:20]
                ]

                raise ValueError(
                    f"Unknown RGB labels in "
                    f"{mask_path}: {unknown_rgb}"
                )

            source_ids = np.asarray(
                [
                    int(
                        rgb_key_to_entry[
                            int(key)
                        ][
                            "source_index_if_indexed"
                        ]
                    )
                    for key in unique_keys
                ],
                dtype=np.int32,
            )

            return source_ids[inverse].reshape(
                packed.shape
            )

        array = np.asarray(image)

    if array.ndim != 2:
        raise ValueError(
            f"Unsupported mask shape: "
            f"{array.shape} {mask_path}"
        )

    return array.astype(
        np.int32,
        copy=False,
    )


def main() -> None:
    args = parse_args()

    if not args.ycor_root.is_dir():
        raise FileNotFoundError(
            args.ycor_root
        )

    if not args.source_map.is_file():
        raise FileNotFoundError(
            args.source_map
        )

    (
        rgb_key_to_entry,
        source_id_to_entry,
    ) = load_source_mapping(
        args.source_map
    )

    pixel_counts: Counter[int] = Counter()
    presence_counts: Counter[int] = Counter()
    split_presence: dict[
        tuple[str, int],
        int,
    ] = defaultdict(int)

    mask_count = 0
    total_pixels = 0

    for split_name in ("train", "valid"):
        split_root = (
            args.ycor_root
            / split_name
        )

        if not split_root.is_dir():
            raise FileNotFoundError(
                split_root
            )

        sample_dirs = sorted(
            path
            for path in split_root.iterdir()
            if path.is_dir()
        )

        for sample_dir in sample_dirs:
            mask_path = (
                sample_dir
                / "labels.png"
            )

            if not mask_path.is_file():
                raise FileNotFoundError(
                    mask_path
                )

            source_mask = load_source_ids(
                mask_path,
                rgb_key_to_entry,
            )

            values, counts = np.unique(
                source_mask,
                return_counts=True,
            )

            total_pixels += int(
                source_mask.size
            )
            mask_count += 1

            for value, count in zip(
                values,
                counts,
            ):
                source_id = int(value)

                pixel_counts[source_id] += int(
                    count
                )
                presence_counts[source_id] += 1
                split_presence[
                    (split_name, source_id)
                ] += 1

            if mask_count % 100 == 0:
                print(
                    f"Processed: {mask_count}"
                )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "source_id",
        "source_class",
        "rgb",
        "pixel_count",
        "pixel_percent",
        "image_presence_count",
        "train_presence_count",
        "valid_presence_count",
        "mask_count",
    ]

    rows = []

    for source_id in sorted(pixel_counts):
        entry = source_id_to_entry[source_id]
        rgb = entry["rgb"]
        count = pixel_counts[source_id]

        rows.append(
            {
                "source_id": source_id,
                "source_class": get_class_name(
                    entry
                ),
                "rgb": ",".join(
                    str(value)
                    for value in rgb
                ),
                "pixel_count": count,
                "pixel_percent": (
                    100.0
                    * count
                    / total_pixels
                ),
                "image_presence_count": (
                    presence_counts[source_id]
                ),
                "train_presence_count": (
                    split_presence[
                        ("train", source_id)
                    ]
                ),
                "valid_presence_count": (
                    split_presence[
                        ("valid", source_id)
                    ]
                ),
                "mask_count": mask_count,
            }
        )

    with args.output.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)

    print()
    print("YCOR RGB distribution completed.")
    print(f"Masks: {mask_count}")
    print(f"Total pixels: {total_pixels}")
    print(f"Output: {args.output}")

    print()
    for row in sorted(
        rows,
        key=lambda item: item["pixel_count"],
        reverse=True,
    ):
        print(
            f"id={row['source_id']}, "
            f"class={row['source_class']}, "
            f"percent="
            f"{row['pixel_percent']:.4f}%, "
            f"presence="
            f"{row['image_presence_count']}"
        )


if __name__ == "__main__":
    main()
