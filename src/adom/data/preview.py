from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .schema import COST4_CLASSES, COST4_PALETTE


def colorize(mask: np.ndarray) -> Image.Image:
    output = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_id, color in COST4_PALETTE.items():
        output[mask == class_id] = color
    return Image.fromarray(output, mode="RGB")


def save_preview(
    image_path: Path,
    mask_path: Path,
    output_path: Path,
    sample_id: str,
) -> None:
    with Image.open(image_path) as source_image:
        image = source_image.convert("RGB")
    with Image.open(mask_path) as source_mask:
        mask = np.asarray(source_mask)
    color = colorize(mask)
    overlay = Image.blend(image, color, alpha=0.42)
    width = 420

    def resized(value: Image.Image, nearest: bool = False) -> Image.Image:
        ratio = width / value.width
        size = (width, max(1, round(value.height * ratio)))
        method = Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR
        return value.resize(size, method)

    panels = [resized(image), resized(color, True), resized(overlay)]
    header = 30
    canvas = Image.new(
        "RGB",
        (width * 3, panels[0].height + header),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), sample_id, fill="black")
    for index, panel in enumerate(panels):
        canvas.paste(panel, (index * width, header))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="JPEG", quality=88, optimize=False)


def write_legend(path: Path) -> None:
    rows = [(class_id, name) for class_id, name in COST4_CLASSES.items()]
    rows.append((255, "ignore"))
    image = Image.new("RGB", (480, 36 * len(rows)), "white")
    draw = ImageDraw.Draw(image)
    for index, (class_id, name) in enumerate(rows):
        y = index * 36
        draw.rectangle((8, y + 8, 28, y + 28), fill=COST4_PALETTE[class_id])
        draw.text((40, y + 10), f"{class_id}: {name}", fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
