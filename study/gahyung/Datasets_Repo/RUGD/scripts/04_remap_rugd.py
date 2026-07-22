from pathlib import Path
from PIL import Image
import numpy as np
import shutil
from tqdm import tqdm


ROOT = Path(r"C:\Users\gahyu\RUGD")
RAW_ROOT = ROOT / "raw" / "RUGD"

IMAGE_DIR = (
    RAW_ROOT
    / "3.after join creek"
    / "image"
)

OUTPUT_IMAGE_DIR = (
    ROOT / "processed" / "images" / "all"
)

OUTPUT_MASK_DIR = (
    ROOT / "processed" / "annotations" / "all"
)

OUTPUT_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_MASK_DIR.mkdir(parents=True, exist_ok=True)


# RUGD 공식 RGB 색상 → semantic class
RGB_TO_NAME = {
    (108, 64, 20): "dirt",
    (255, 229, 204): "sand",
    (0, 102, 0): "grass",
    (0, 255, 0): "tree",
    (0, 153, 153): "pole",
    (0, 128, 255): "water",
    (0, 0, 255): "sky",
    (255, 255, 0): "vehicle",
    (255, 0, 127): "container",
    (64, 64, 64): "asphalt",
    (255, 128, 0): "gravel",
    (255, 0, 0): "building",
    (153, 76, 0): "mulch",
    (102, 102, 0): "rock-bed",
    (102, 0, 0): "log",
    (0, 255, 128): "bicycle",
    (204, 153, 255): "person",
    (102, 0, 204): "fence",
    (255, 153, 204): "bush",
    (0, 102, 102): "sign",
    (153, 204, 255): "rock",
    (102, 255, 255): "bridge",
    (101, 101, 11): "concrete",
    (114, 85, 47): "picnic-table",
    (0, 0, 0): "unlabeled",
}


RUGD_TO_ADOM = {
    "asphalt": 0,
    "concrete": 0,

    "dirt": 1,
    "grass": 1,
    "gravel": 1,

    "sand": 2,
    "mulch": 2,
    "rock-bed": 2,
    "bush": 2,

    "tree": 3,
    "pole": 3,
    "water": 3,
    "vehicle": 3,
    "container": 3,
    "building": 3,
    "log": 3,
    "bicycle": 3,
    "person": 3,
    "fence": 3,
    "sign": 3,
    "rock": 3,
    "bridge": 3,
    "picnic-table": 3,

    "sky": 255,
    "unlabeled": 255,
}


# 두 색상 annotation 폴더 자동 탐색
COLOR_DIRS = [
    path
    for path in RAW_ROOT.iterdir()
    if path.is_dir()
    and "indexlabel" in path.name.lower()
    and "color" in path.name.lower()
]

if not COLOR_DIRS:
    raise RuntimeError("indexLabel-color 폴더를 찾지 못했습니다.")


# 파일 이름 → 색상 마스크 경로
color_mask_map = {}

for color_dir in COLOR_DIRS:
    for mask_path in color_dir.rglob("*.png"):
        if mask_path.name in color_mask_map:
            raise RuntimeError(
                f"중복 색상 마스크: {mask_path.name}"
            )

        color_mask_map[mask_path.name] = mask_path


print("RGB 이미지 수:", len(list(IMAGE_DIR.glob("*.png"))))
print("색상 마스크 수:", len(color_mask_map))

if len(color_mask_map) != 7436:
    raise RuntimeError(
        f"색상 마스크가 7436개가 아닙니다: "
        f"{len(color_mask_map)}"
    )


known_colors = set(RGB_TO_NAME)


for file_name, color_mask_path in tqdm(
    sorted(color_mask_map.items())
):
    image_path = IMAGE_DIR / file_name

    if not image_path.exists():
        raise FileNotFoundError(
            f"대응 RGB 이미지 없음: {file_name}"
        )

    color_mask = np.array(
        Image.open(color_mask_path).convert("RGB")
    )

    target_mask = np.full(
        color_mask.shape[:2],
        fill_value=255,
        dtype=np.uint8,
    )

    unique_colors = {
        tuple(int(value) for value in color)
        for color in np.unique(
            color_mask.reshape(-1, 3),
            axis=0,
        )
    }

    unknown_colors = unique_colors - known_colors

    if unknown_colors:
        raise RuntimeError(
            f"알 수 없는 RGB 색상 발견\n"
            f"파일: {file_name}\n"
            f"색상: {sorted(unknown_colors)}"
        )

    for rgb, class_name in RGB_TO_NAME.items():
        target_id = RUGD_TO_ADOM[class_name]

        pixels = np.all(
            color_mask == np.array(rgb, dtype=np.uint8),
            axis=2,
        )

        target_mask[pixels] = target_id

    Image.fromarray(target_mask).save(
        OUTPUT_MASK_DIR / file_name
    )

    shutil.copy2(
        image_path,
        OUTPUT_IMAGE_DIR / file_name,
    )


print("\n변환 완료")
print(
    "Images:",
    len(list(OUTPUT_IMAGE_DIR.glob("*.png"))),
)
print(
    "Masks:",
    len(list(OUTPUT_MASK_DIR.glob("*.png"))),
)