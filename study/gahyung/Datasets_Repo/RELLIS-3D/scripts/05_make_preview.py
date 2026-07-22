from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[1]

METADATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "rellis_cost4_standard"
    / "metadata.csv"
)

DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "rellis3d_preview.png"
)

# ADOM Cost4 시각화 색상
PALETTE = {
    0: (128, 128, 128),   # paved_low_cost
    1: (60, 180, 75),     # natural_low_cost
    2: (255, 165, 0),     # medium_cost
    3: (220, 20, 60),     # high_cost_or_obstacle
    255: (0, 0, 0),       # ignore
}

CLASS_NAMES = {
    0: "0 paved_low_cost",
    1: "1 natural_low_cost",
    2: "2 medium_cost",
    3: "3 high_cost_or_obstacle",
    255: "255 ignore",
}


def load_font(size: int) -> ImageFont.ImageFont:
    """Windows Arial을 우선 사용하고, 없으면 기본 글꼴을 사용한다."""
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def resolve_project_path(path_text: str) -> Path:
    """metadata.csv의 상대 경로를 실제 프로젝트 경로로 변환한다."""
    path = Path(path_text)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def load_metadata() -> list[dict[str, str]]:
    """정상 변환된 metadata 행만 읽는다."""
    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"metadata.csv가 없습니다: {METADATA_PATH}"
        )

    with METADATA_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        rows = list(csv.DictReader(file))

    valid_rows = [
        row
        for row in rows
        if row.get("status", "").strip() == "ok"
    ]

    if not valid_rows:
        raise RuntimeError(
            "metadata.csv에서 status=ok인 샘플을 찾지 못했습니다."
        )

    return valid_rows


def select_samples(
    rows: list[dict[str, str]],
    per_sequence: int,
    max_ignore_ratio: float,
) -> list[dict[str, str]]:
    """
    sequence별 대표 샘플을 선택한다.

    ignore 비율이 너무 높은 샘플은 우선 제외하고,
    파일 순서상 고르게 떨어진 샘플을 선택한다.
    """
    grouped: dict[str, list[dict[str, str]]] = {}

    for row in rows:
        sequence = row["sequence"]
        grouped.setdefault(sequence, []).append(row)

    selected: list[dict[str, str]] = []

    for sequence in sorted(grouped):
        sequence_rows = grouped[sequence]

        candidates = []

        for row in sequence_rows:
            try:
                ignore_ratio = float(row["ignore_ratio"])
            except (TypeError, ValueError):
                ignore_ratio = 1.0

            if ignore_ratio <= max_ignore_ratio:
                candidates.append(row)

        # 조건을 만족하는 샘플이 없으면 전체 샘플 사용
        if not candidates:
            candidates = sequence_rows

        candidates.sort(
            key=lambda row: row["original_stem"]
        )

        sample_count = min(
            per_sequence,
            len(candidates),
        )

        if sample_count == 1:
            indexes = [len(candidates) // 2]
        else:
            indexes = np.linspace(
                0,
                len(candidates) - 1,
                sample_count,
                dtype=int,
            ).tolist()

        for index in indexes:
            selected.append(candidates[index])

    return selected


def resize_with_padding(
    image: Image.Image,
    size: tuple[int, int],
    resample: Image.Resampling,
) -> Image.Image:
    """비율을 유지하면서 지정 크기에 맞추고 남는 영역은 검정색으로 채운다."""
    contained = ImageOps.contain(
        image,
        size,
        method=resample,
    )

    canvas = Image.new(
        "RGB",
        size,
        (0, 0, 0),
    )

    x = (size[0] - contained.width) // 2
    y = (size[1] - contained.height) // 2

    canvas.paste(
        contained.convert("RGB"),
        (x, y),
    )

    return canvas


def resize_mask_with_padding(
    mask: Image.Image,
    size: tuple[int, int],
) -> Image.Image:
    """클래스 ID가 변하지 않도록 nearest-neighbor 방식으로 resize한다."""
    contained = ImageOps.contain(
        mask,
        size,
        method=Image.Resampling.NEAREST,
    )

    canvas = Image.new(
        "L",
        size,
        color=255,
    )

    x = (size[0] - contained.width) // 2
    y = (size[1] - contained.height) // 2

    canvas.paste(
        contained,
        (x, y),
    )

    return canvas


def colorize_mask(mask_array: np.ndarray) -> np.ndarray:
    """단일 채널 Cost4 mask를 RGB 컬러 mask로 변환한다."""
    height, width = mask_array.shape

    color_mask = np.zeros(
        (height, width, 3),
        dtype=np.uint8,
    )

    for class_id, color in PALETTE.items():
        color_mask[mask_array == class_id] = color

    return color_mask


def make_overlay(
    rgb_array: np.ndarray,
    mask_array: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """ignore 영역을 제외하고 mask 색상을 RGB 위에 겹친다."""
    color_mask = colorize_mask(mask_array)

    overlay = rgb_array.astype(np.float32).copy()

    valid_pixels = mask_array != 255

    overlay[valid_pixels] = (
        (1.0 - alpha) * overlay[valid_pixels]
        + alpha * color_mask[valid_pixels]
    )

    return np.clip(
        overlay,
        0,
        255,
    ).astype(np.uint8)


def create_preview(
    selected_rows: list[dict[str, str]],
    output_path: Path,
    panel_width: int,
    panel_height: int,
    alpha: float,
) -> None:
    """RGB, Cost4 mask, overlay를 한 장의 preview 이미지로 만든다."""
    title_height = 70
    row_label_height = 42
    column_label_height = 42
    row_gap = 18
    outer_margin = 25
    legend_height = 95

    columns = 3
    preview_width = (
        outer_margin * 2
        + panel_width * columns
    )

    row_height = (
        row_label_height
        + column_label_height
        + panel_height
    )

    preview_height = (
        title_height
        + outer_margin
        + len(selected_rows) * row_height
        + max(0, len(selected_rows) - 1) * row_gap
        + legend_height
        + outer_margin
    )

    preview = Image.new(
        "RGB",
        (preview_width, preview_height),
        (245, 245, 245),
    )

    draw = ImageDraw.Draw(preview)

    title_font = load_font(28)
    label_font = load_font(20)
    small_font = load_font(16)

    draw.text(
        (outer_margin, 22),
        "RELLIS-3D ADOM Cost4 Preprocessing Preview",
        fill=(20, 20, 20),
        font=title_font,
    )

    current_y = title_height + outer_margin

    column_labels = [
        "RGB",
        "Cost4 Mask",
        "Overlay",
    ]

    for row in selected_rows:
        sequence = row["sequence"]
        sample_id = row["sample_id"]

        rgb_path = resolve_project_path(
            row["rgb_path"]
        )

        mask_path = resolve_project_path(
            row["converted_mask_path"]
        )

        if not rgb_path.exists():
            raise FileNotFoundError(
                f"RGB 파일이 없습니다: {rgb_path}"
            )

        if not mask_path.exists():
            raise FileNotFoundError(
                f"변환 mask가 없습니다: {mask_path}"
            )

        with Image.open(rgb_path) as image:
            rgb = image.convert("RGB")

        with Image.open(mask_path) as image:
            mask = image.convert("L")

        rgb_resized = resize_with_padding(
            rgb,
            (panel_width, panel_height),
            Image.Resampling.BILINEAR,
        )

        mask_resized = resize_mask_with_padding(
            mask,
            (panel_width, panel_height),
        )

        rgb_array = np.asarray(
            rgb_resized,
            dtype=np.uint8,
        )

        mask_array = np.asarray(
            mask_resized,
            dtype=np.uint8,
        )

        color_mask_array = colorize_mask(
            mask_array
        )

        overlay_array = make_overlay(
            rgb_array,
            mask_array,
            alpha,
        )

        color_mask = Image.fromarray(
            color_mask_array,
            mode="RGB",
        )

        overlay = Image.fromarray(
            overlay_array,
            mode="RGB",
        )

        draw.text(
            (outer_margin, current_y),
            f"Sequence {sequence} | {sample_id}",
            fill=(25, 25, 25),
            font=label_font,
        )

        label_y = current_y + row_label_height

        for column_index, label in enumerate(column_labels):
            x = (
                outer_margin
                + column_index * panel_width
            )

            draw.rectangle(
                (
                    x,
                    label_y,
                    x + panel_width,
                    label_y + column_label_height,
                ),
                fill=(225, 225, 225),
            )

            draw.text(
                (x + 12, label_y + 10),
                label,
                fill=(20, 20, 20),
                font=label_font,
            )

        image_y = label_y + column_label_height

        preview.paste(
            rgb_resized,
            (outer_margin, image_y),
        )

        preview.paste(
            color_mask,
            (
                outer_margin + panel_width,
                image_y,
            ),
        )

        preview.paste(
            overlay,
            (
                outer_margin + panel_width * 2,
                image_y,
            ),
        )

        current_y += row_height + row_gap

    legend_y = preview_height - legend_height

    draw.text(
        (outer_margin, legend_y),
        "Legend",
        fill=(20, 20, 20),
        font=label_font,
    )

    legend_x = outer_margin
    legend_item_y = legend_y + 35

    for class_id in [0, 1, 2, 3, 255]:
        color = PALETTE[class_id]

        draw.rectangle(
            (
                legend_x,
                legend_item_y,
                legend_x + 22,
                legend_item_y + 22,
            ),
            fill=color,
            outline=(40, 40, 40),
        )

        draw.text(
            (
                legend_x + 30,
                legend_item_y + 3,
            ),
            CLASS_NAMES[class_id],
            fill=(20, 20, 20),
            font=small_font,
        )

        legend_x += 270

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    preview.save(
        output_path,
        format="PNG",
        optimize=True,
    )

    print(f"선택한 preview sample 수: {len(selected_rows)}")
    print(f"저장 위치: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--per-sequence",
        type=int,
        default=1,
        help="sequence별 대표 샘플 수",
    )

    parser.add_argument(
        "--max-ignore-ratio",
        type=float,
        default=0.6,
        help="대표 샘플 선택 시 허용할 최대 ignore 비율",
    )

    parser.add_argument(
        "--panel-width",
        type=int,
        default=480,
    )

    parser.add_argument(
        "--panel-height",
        type=int,
        default=300,
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="overlay mask 투명도",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
    )

    args = parser.parse_args()

    if args.per_sequence < 1:
        raise ValueError(
            "--per-sequence는 1 이상이어야 합니다."
        )

    if not 0.0 <= args.max_ignore_ratio <= 1.0:
        raise ValueError(
            "--max-ignore-ratio는 0~1 범위여야 합니다."
        )

    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError(
            "--alpha는 0~1 범위여야 합니다."
        )

    rows = load_metadata()

    selected_rows = select_samples(
        rows,
        per_sequence=args.per_sequence,
        max_ignore_ratio=args.max_ignore_ratio,
    )

    create_preview(
        selected_rows,
        output_path=args.output,
        panel_width=args.panel_width,
        panel_height=args.panel_height,
        alpha=args.alpha,
    )


if __name__ == "__main__":
    main()