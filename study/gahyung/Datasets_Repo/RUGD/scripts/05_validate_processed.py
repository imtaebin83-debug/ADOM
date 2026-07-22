from pathlib import Path
from PIL import Image
import numpy as np

ROOT = Path(r"C:\Users\gahyu\RUGD")

IMAGE_DIR = ROOT / "processed" / "images" / "all"
MASK_DIR = ROOT / "processed" / "annotations" / "all"

VALID_TARGET_IDS = {0, 1, 2, 3, 255}

image_paths = sorted(IMAGE_DIR.glob("*.png"))
mask_paths = sorted(MASK_DIR.glob("*.png"))

if len(image_paths) != len(mask_paths):
    raise RuntimeError(
        f"개수 불일치: "
        f"images={len(image_paths)}, "
        f"masks={len(mask_paths)}"
    )

for image_path in image_paths:
    mask_path = MASK_DIR / image_path.name

    if not mask_path.exists():
        raise FileNotFoundError(mask_path)

    with Image.open(image_path) as image:
        image_size = image.size

    mask = np.array(Image.open(mask_path))

    if mask.ndim != 2:
        raise RuntimeError(
            f"다중 채널 마스크: {mask_path.name}"
        )

    with Image.open(mask_path) as mask_image:
        mask_size = mask_image.size

    if image_size != mask_size:
        raise RuntimeError(
            f"크기 불일치: {image_path.name}, "
            f"image={image_size}, mask={mask_size}"
        )

    unique_ids = {
        int(value)
        for value in np.unique(mask)
    }

    invalid_ids = unique_ids - VALID_TARGET_IDS

    if invalid_ids:
        raise RuntimeError(
            f"잘못된 ID: {mask_path.name}, "
            f"{sorted(invalid_ids)}"
        )

print("검사 완료")
print("정상 이미지–마스크 쌍:", len(image_paths))