from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[1]

## 0727 RGB 원본과 처리 결과 경로를 분리하고 metadata 필수 열 정의
DEFAULT_INPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "Rellis-3D"
)

DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "rellis_cost4_standard"
)

DEFAULT_PREVIEW_RELATIVE_PATH = (
    Path("previews")
    / "rellis3d_preview.png"
)

REQUIRED_METADATA_COLUMNS = {
    "sequence",
    "sample_id",
    "original_stem",
    "rgb_path",
    "converted_mask_path",
    "status",
    "ignore_ratio",
}
##

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

## 0727 CLI → 환경변수 → 기본값 순서로 경로를 결정하는 함수 추가
def resolve_path(
    cli_value: Path | None,
    env_name: str,
    default: Path,
) -> Path:
    """
    Resolve a path in this order:

    CLI argument -> environment variable -> default.
    """
    value = cli_value or os.getenv(env_name)

    if value is None:
        return default.resolve()

    return Path(value).expanduser().resolve()
##


## 0727 metadata의 상대경로를 지정된 데이터 root 기준으로 변환
def resolve_data_path(
    base_root: Path,
    path_text: str,
) -> Path:
    path = Path(path_text)

    if path.is_absolute():
        return path.resolve()

    return (base_root / path).resolve()
##

def load_font(size: int) -> ImageFont.ImageFont:
    ## 0727 깨진 글꼴 설명을 정상 문자열로 수정
    """Use Arial when available and otherwise use PIL's default font."""
    ##
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()

## 0727 metadata 경로를 외부에서 받고 필수 열과 정상 행을 검증하도록 수정
def load_metadata(
    metadata_path: Path,
) -> list[dict[str, str]]:
    """Load successfully converted rows from metadata.csv."""
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"metadata.csv does not exist: {metadata_path}"
        )

    with metadata_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        metadata_columns = set(
            reader.fieldnames or []
        )

        missing_columns = sorted(
            REQUIRED_METADATA_COLUMNS
            - metadata_columns
        )

        if missing_columns:
            raise RuntimeError(
                "metadata.csv is missing required columns: "
                f"{missing_columns}"
            )

        rows = list(reader)

    if not rows:
        raise RuntimeError(
            "metadata.csv contains no data rows."
        )

    valid_rows = [
        row
        for row in rows
        if row.get("status", "").strip() == "ok"
    ]

    if not valid_rows:
        raise RuntimeError(
            "No status=ok samples were found in metadata.csv."
        )

    return valid_rows
##


## 0727 sequence별 preview 샘플 선택과 ignore 비율 처리를 명확히 수정
def select_samples(
    rows: list[dict[str, str]],
    per_sequence: int,
    max_ignore_ratio: float,
) -> list[dict[str, str]]:
    """
    Select evenly distributed preview samples for each sequence.

    Samples above max_ignore_ratio are excluded first. If no sample
    satisfies the threshold for a sequence, all valid rows from that
    sequence are used as fallback candidates.
    """
    grouped: dict[str, list[dict[str, str]]] = {}

    for row in rows:
        sequence = row["sequence"]
        grouped.setdefault(sequence, []).append(row)

    selected: list[dict[str, str]] = []

    for sequence in sorted(grouped):
        sequence_rows = grouped[sequence]

        candidates: list[dict[str, str]] = []

        for row in sequence_rows:
            try:
                ignore_ratio = float(
                    row.get("ignore_ratio", "")
                )
            except (TypeError, ValueError):
                ignore_ratio = 1.0

            if ignore_ratio <= max_ignore_ratio:
                candidates.append(row)

        # Fall back to all rows when no sample satisfies the threshold.
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

    if not selected:
        raise RuntimeError(
            "No preview samples were selected."
        )

    return selected
##


def resize_with_padding(
    image: Image.Image,
    size: tuple[int, int],
    resample: Image.Resampling,
) -> Image.Image:
    ## 0727 RGB resize 동작 설명 수정
    """Resize while preserving aspect ratio and pad unused areas."""
    ##
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
    ## 0727 mask ID 보존을 위한 nearest-neighbor resize 설명 수정
    """Resize a mask with nearest-neighbor interpolation and padding."""
    ##
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
    ## 0727 단일 채널 Cost4 mask 색상 변환 설명 수정
    """Convert a single-channel Cost4 mask into an RGB color mask."""
    ##
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
    ## 0727 ignore 영역을 제외한 RGB overlay 동작 설명 수정
    """Overlay valid mask pixels on RGB while excluding ignore pixels."""
    ##
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


## 0727 RGB root와 처리 결과 root를 분리하고 기존 preview 덮어쓰기 방지
def create_preview(
    selected_rows: list[dict[str, str]],
    input_root: Path,
    output_root: Path,
    output_path: Path,
    panel_width: int,
    panel_height: int,
    alpha: float,
    overwrite: bool,
) -> None:
    """Create one preview image containing RGB, mask, and overlay panels."""
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            "Preview output already exists. "
            "Use --overwrite to replace it: "
            f"{output_path}"
        )
##
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

        ## 0727 RGB는 input_root, 변환 mask는 output_root 기준으로 경로 해석
        rgb_path = resolve_data_path(
            input_root,
            row["rgb_path"],
        )

        mask_path = resolve_data_path(
            output_root,
            row["converted_mask_path"],
        )

        if not rgb_path.is_file():
            raise FileNotFoundError(
                f"RGB file does not exist: {rgb_path}"
            )

        if not mask_path.is_file():
            raise FileNotFoundError(
                f"Converted mask does not exist: {mask_path}"
            )
        ##

        with Image.open(rgb_path) as image:
            rgb = image.convert("RGB")

        with Image.open(mask_path) as image:
            mask = image.convert("L")

        ## 0727 preview 생성 전 RGB와 mask 원본 크기 일치 여부 검사
        if rgb.size != mask.size:
            raise ValueError(
                "RGB and mask dimensions do not match: "
                f"sample_id={sample_id}, "
                f"rgb_size={rgb.size}, "
                f"mask_size={mask.size}"
            )
        ##

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

    ## 0727 preview 생성 결과를 명확한 출력 형식으로 수정
    print(
        f"Selected preview samples: "
        f"{len(selected_rows)}"
    )
    print(f"Preview output: {output_path}")
    ##


## 0727 원본 RGB, 처리 결과, metadata, preview 출력 경로를 CLI로 지정하도록 수정
def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-root",
        type=Path,
        default=None,
        help=(
            "Raw RELLIS-3D root used to resolve rgb_path. "
            "Environment variable: RELLIS_INPUT_ROOT"
        ),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Processed dataset root used to resolve "
            "converted_mask_path. Environment variable: "
            "RELLIS_OUTPUT_ROOT"
        ),
    )

    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help=(
            "Path to metadata.csv. The default is "
            "output-root/metadata.csv. Environment variable: "
            "RELLIS_METADATA_PATH"
        ),
    )

    parser.add_argument(
        "--per-sequence",
        type=int,
        default=1,
        help="Number of preview samples selected per sequence.",
    )

    parser.add_argument(
        "--max-ignore-ratio",
        type=float,
        default=0.6,
        help=(
            "Maximum ignore_ratio used for preferred "
            "preview sample selection."
        ),
    )

    parser.add_argument(
        "--panel-width",
        type=int,
        default=480,
        help="Width of each RGB, mask, and overlay panel.",
    )

    parser.add_argument(
        "--panel-height",
        type=int,
        default=300,
        help="Height of each RGB, mask, and overlay panel.",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Mask opacity used for the RGB overlay.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Preview PNG output path. The default is "
            "output-root/previews/rellis3d_preview.png. "
            "Environment variable: RELLIS_PREVIEW_PATH"
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing preview PNG.",
    )

    args = parser.parse_args()

    input_root = resolve_path(
        args.input_root,
        "RELLIS_INPUT_ROOT",
        DEFAULT_INPUT_ROOT,
    )

    output_root = resolve_path(
        args.output_root,
        "RELLIS_OUTPUT_ROOT",
        DEFAULT_OUTPUT_ROOT,
    )

    metadata_path = resolve_path(
        args.metadata,
        "RELLIS_METADATA_PATH",
        output_root / "metadata.csv",
    )

    preview_path = resolve_path(
        args.output,
        "RELLIS_PREVIEW_PATH",
        output_root / DEFAULT_PREVIEW_RELATIVE_PATH,
    )

    if not input_root.is_dir():
        raise FileNotFoundError(
            f"Input root does not exist: {input_root}"
        )

    if not output_root.is_dir():
        raise FileNotFoundError(
            f"Output root does not exist: {output_root}"
        )

    mask_root = output_root / "masks"

    if not mask_root.is_dir():
        raise FileNotFoundError(
            f"Converted mask directory does not exist: {mask_root}"
        )

    if args.per_sequence < 1:
        raise ValueError(
            "--per-sequence must be at least 1."
        )

    if not 0.0 <= args.max_ignore_ratio <= 1.0:
        raise ValueError(
            "--max-ignore-ratio must be between 0 and 1."
        )

    if args.panel_width < 1:
        raise ValueError(
            "--panel-width must be at least 1."
        )

    if args.panel_height < 1:
        raise ValueError(
            "--panel-height must be at least 1."
        )

    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError(
            "--alpha must be between 0 and 1."
        )

    rows = load_metadata(
        metadata_path
    )

    selected_rows = select_samples(
        rows,
        per_sequence=args.per_sequence,
        max_ignore_ratio=args.max_ignore_ratio,
    )

    create_preview(
        selected_rows=selected_rows,
        input_root=input_root,
        output_root=output_root,
        output_path=preview_path,
        panel_width=args.panel_width,
        panel_height=args.panel_height,
        alpha=args.alpha,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
##
