from pathlib import Path
from PIL import Image
import numpy as np

ROOT = Path(r"C:\Users\gahyu\RUGD")

IMAGE_ROOT = ROOT / "processed" / "images"
MASK_ROOT = ROOT / "processed" / "annotations"

VALID_IDS = {0, 1, 2, 3, 255}

all_names = set()
total_count = 0

for split in ("train", "val", "test"):
    image_dir = IMAGE_ROOT / split
    mask_dir = MASK_ROOT / split

    image_paths = sorted(image_dir.glob("*.png"))
    mask_paths = sorted(mask_dir.glob("*.png"))

    image_names = {
        path.name for path in image_paths
    }

    mask_names = {
        path.name for path in mask_paths
    }

    if image_names != mask_names:
        only_images = image_names - mask_names
        only_masks = mask_names - image_names

        raise RuntimeError(
            f"{split} 파일명 불일치\n"
            f"마스크 없는 이미지: "
            f"{list(only_images)[:10]}\n"
            f"이미지 없는 마스크: "
            f"{list(only_masks)[:10]}"
        )

    duplicated = all_names & image_names

    if duplicated:
        raise RuntimeError(
            f"split 중복 파일 발견: "
            f"{list(duplicated)[:10]}"
        )

    all_names.update(image_names)
    total_count += len(image_names)

    split_ids = set()

    for image_path in image_paths:
        mask_path = mask_dir / image_path.name

        with Image.open(image_path) as image:
            image_size = image.size

        with Image.open(mask_path) as mask_image:
            mask_size = mask_image.size
            mask = np.array(mask_image)

        if image_size != mask_size:
            raise RuntimeError(
                f"크기 불일치: {image_path.name}\n"
                f"image={image_size}, mask={mask_size}"
            )

        if mask.ndim != 2:
            raise RuntimeError(
                f"다중 채널 마스크: "
                f"{mask_path.name}, shape={mask.shape}"
            )

        current_ids = {
            int(value)
            for value in np.unique(mask)
        }

        invalid_ids = current_ids - VALID_IDS

        if invalid_ids:
            raise RuntimeError(
                f"허용되지 않은 ID: "
                f"{mask_path.name}, "
                f"{sorted(invalid_ids)}"
            )

        split_ids.update(current_ids)

    print(f"\n[{split}]")
    print("이미지 수:", len(image_paths))
    print("마스크 수:", len(mask_paths))
    print("사용된 ID:", sorted(split_ids))

print("\n전체 파일 수:", total_count)

if total_count != 7436:
    raise RuntimeError(
        f"전체 파일이 7436개가 아닙니다: "
        f"{total_count}"
    )

print("\n최종 검사 통과")
print("RUGD 전처리 데이터가 준비되었습니다.")