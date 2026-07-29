from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from common import (
    OUTPUT_SPLITS,
    PROCESSED_ROOT,
    TARGET_PALETTE,
)


SAMPLES_PER_SPLIT = 36
THUMB_SIZE = (480, 270)
OVERLAY_ALPHA = 0.45
RANDOM_SEED = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate deterministic YCOR preview sheets "
            "from a processed dataset."
        )
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROCESSED_ROOT,
        help=(
            "Processed YCOR_ADOM directory containing "
            "images/, masks/, and metadata/."
        ),
    )

    return parser.parse_args()


def colorize_mask(
    mask: np.ndarray,
) -> Image.Image:
    color = np.zeros(
        (*mask.shape, 3),
        dtype=np.uint8,
    )

    for class_id, rgb in TARGET_PALETTE.items():
        color[
            mask == class_id
        ] = rgb

    return Image.fromarray(
        color,
        mode="RGB",
    )


def make_overlay(
    image: Image.Image,
    mask: np.ndarray,
) -> Image.Image:
    image = image.convert("RGB")
    color_mask = colorize_mask(mask)

    overlay = Image.blend(
        image,
        color_mask,
        OVERLAY_ALPHA,
    )

    overlay_array = np.asarray(
        overlay
    ).copy()

    image_array = np.asarray(
        image
    )

    overlay_array[
        mask == 255
    ] = image_array[
        mask == 255
    ]

    return Image.fromarray(
        overlay_array,
        mode="RGB",
    )


def add_caption(
    image: Image.Image,
    text: str,
) -> Image.Image:
    canvas = Image.new(
        "RGB",
        (
            image.width,
            image.height + 28,
        ),
        "white",
    )

    canvas.paste(
        image,
        (0, 28),
    )

    draw = ImageDraw.Draw(
        canvas
    )

    draw.text(
        (8, 7),
        text,
        fill="black",
    )

    return canvas


def make_contact_sheet(
    images: list[Image.Image],
    columns: int = 3,
) -> Image.Image:
    if not images:
        raise ValueError(
            "No preview images were generated."
        )

    rows = (
        len(images)
        + columns
        - 1
    ) // columns

    cell_width = max(
        image.width
        for image in images
    )

    cell_height = max(
        image.height
        for image in images
    )

    sheet = Image.new(
        "RGB",
        (
            columns * cell_width,
            rows * cell_height,
        ),
        "white",
    )

    for index, image in enumerate(
        images
    ):
        x = (
            index % columns
        ) * cell_width

        y = (
            index // columns
        ) * cell_height

        sheet.paste(
            image,
            (x, y),
        )

    return sheet


def main() -> None:
    args = parse_args()

    processed_root = (
        args.output_root
        .expanduser()
        .resolve()
    )

    images_root = (
        processed_root
        / "images"
    )

    masks_root = (
        processed_root
        / "masks"
    )

    metadata_root = (
        processed_root
        / "metadata"
    )

    preview_root = (
        processed_root
        / "qc"
        / "previews"
    )

    preview_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    rng = random.Random(
        RANDOM_SEED
    )

    for split in OUTPUT_SPLITS:
        metadata_path = (
            metadata_root
            / f"{split}.csv"
        )

        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"Metadata not found: {metadata_path}"
            )

        metadata = pd.read_csv(
            metadata_path,
            dtype=str,
        )

        sample_count = min(
            SAMPLES_PER_SPLIT,
            len(metadata),
        )

        if sample_count == 0:
            raise RuntimeError(
                f"No metadata samples found for {split}."
            )

        sampled_indices = rng.sample(
            range(
                len(metadata)
            ),
            sample_count,
        )

        previews = []

        for index in sampled_indices:
            row = metadata.iloc[
                index
            ]

            image_path = (
                images_root
                / split
                / row["image_filename"]
            )

            mask_path = (
                masks_root
                / split
                / row["mask_filename"]
            )

            if not image_path.is_file():
                raise FileNotFoundError(
                    f"Preview image not found: {image_path}"
                )

            if not mask_path.is_file():
                raise FileNotFoundError(
                    f"Preview mask not found: {mask_path}"
                )

            with Image.open(
                image_path
            ) as source_image:
                image = source_image.convert(
                    "RGB"
                )

            with Image.open(
                mask_path
            ) as source_mask:
                mask = np.asarray(
                    source_mask,
                    dtype=np.uint8,
                )

            if image.size != (
                mask.shape[1],
                mask.shape[0],
            ):
                raise ValueError(
                    "Preview image-mask size mismatch: "
                    f"{row['sample_id']}"
                )

            overlay = make_overlay(
                image,
                mask,
            )

            overlay.thumbnail(
                THUMB_SIZE,
                Image.Resampling.LANCZOS,
            )

            caption = (
                f"{split} | "
                f"{row['sample_id']} | "
                f"{row['source_sample_name']}"
            )

            previews.append(
                add_caption(
                    overlay,
                    caption,
                )
            )

        sheet = make_contact_sheet(
            previews,
            columns=3,
        )

        output_path = (
            preview_root
            / f"{split}_preview.jpg"
        )

        sheet.save(
            output_path,
            quality=92,
        )

        print(
            f"[{split}] preview: {output_path}"
        )

    class_names = {
        0: "paved_low_cost (not present in YCOR)",
        1: "natural_low_cost",
        2: "medium_cost",
        3: "high_cost_or_obstacle",
        255: "ignore",
    }

    legend = Image.new(
        "RGB",
        (
            500,
            46 * len(
                TARGET_PALETTE
            ),
        ),
        "white",
    )

    draw = ImageDraw.Draw(
        legend
    )

    for index, (
        class_id,
        rgb,
    ) in enumerate(
        TARGET_PALETTE.items()
    ):
        y = index * 46

        draw.rectangle(
            (
                10,
                y + 8,
                42,
                y + 38,
            ),
            fill=rgb,
        )

        draw.text(
            (
                54,
                y + 14,
            ),
            (
                f"{class_id}: "
                f"{class_names[class_id]}"
            ),
            fill="black",
        )

    legend_path = (
        preview_root
        / "legend.png"
    )

    legend.save(
        legend_path
    )

    print(
        f"[legend] {legend_path}"
    )

    print(
        "07_make_previews.py: PASS"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(
            f"\nERROR: {exc}",
            file=sys.stderr,
        )
        raise