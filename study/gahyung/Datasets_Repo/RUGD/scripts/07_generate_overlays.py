from pathlib import Path
from PIL import Image
import numpy as np

ROOT = Path(r"C:\Users\gahyu\RUGD")

IMAGE_ROOT = ROOT / "processed" / "images"
MASK_ROOT = ROOT / "processed" / "annotations"
OUTPUT_ROOT = ROOT / "processed" / "qc" / "overlays"

# ADOM 클래스 시각화 색상
CLASS_COLORS = {
    0: np.array([128, 64, 128], dtype=np.uint8),   # paved_low_cost
    1: np.array([107, 142, 35], dtype=np.uint8),   # natural_low_cost
    2: np.array([255, 165, 0], dtype=np.uint8),    # medium_cost
    3: np.array([220, 20, 60], dtype=np.uint8),    # high_cost_or_obstacle
}

SAMPLES_PER_SPLIT = 100
ALPHA = 0.45


def select_evenly(paths, sample_count):
    """전체 sequence에서 최대한 고르게 샘플을 선택한다."""

    if len(paths) <= sample_count:
        return paths

    indices = np.linspace(
        0,
        len(paths) - 1,
        sample_count,
        dtype=int,
    )

    return [paths[index] for index in indices]


def make_overlay(image_path, mask_path, output_path):
    image = np.array(
        Image.open(image_path).convert("RGB"),
        dtype=np.uint8,
    )

    mask = np.array(Image.open(mask_path))

    if mask.ndim != 2:
        raise RuntimeError(
            f"단일 채널 마스크가 아닙니다: "
            f"{mask_path.name}, shape={mask.shape}"
        )

    if image.shape[:2] != mask.shape:
        raise RuntimeError(
            f"이미지와 마스크 크기 불일치: {image_path.name}"
        )

    color_mask = np.zeros_like(image)
    valid_pixels = np.zeros(mask.shape, dtype=bool)

    for class_id, color in CLASS_COLORS.items():
        class_pixels = mask == class_id
        color_mask[class_pixels] = color
        valid_pixels |= class_pixels

    overlay = image.copy()

    blended = (
        image.astype(np.float32) * (1.0 - ALPHA)
        + color_mask.astype(np.float32) * ALPHA
    ).clip(0, 255).astype(np.uint8)

    # 0~3 클래스 영역만 색상을 겹치고,
    # 255 ignore 영역은 원본 RGB 그대로 유지
    overlay[valid_pixels] = blended[valid_pixels]

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    Image.fromarray(overlay).save(output_path)


for split in ("train", "val", "test"):
    image_dir = IMAGE_ROOT / split
    mask_dir = MASK_ROOT / split
    output_dir = OUTPUT_ROOT / split

    image_paths = sorted(image_dir.glob("*.png"))
    selected_paths = select_evenly(
        image_paths,
        SAMPLES_PER_SPLIT,
    )

    print(f"{split}: overlay {len(selected_paths)}개 생성")

    for image_path in selected_paths:
        mask_path = mask_dir / image_path.name

        if not mask_path.exists():
            raise FileNotFoundError(
                f"마스크 없음: {mask_path}"
            )

        output_path = (
            output_dir
            / f"{image_path.stem}_overlay.png"
        )

        make_overlay(
            image_path,
            mask_path,
            output_path,
        )

print("\nOverlay 생성 완료")
print("저장 위치:", OUTPUT_ROOT)