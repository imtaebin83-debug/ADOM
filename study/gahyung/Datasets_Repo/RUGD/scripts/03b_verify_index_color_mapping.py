from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(r"C:\Users\gahyu\RUGD\raw\RUGD")

INDEX_DIR = (
    ROOT
    / "3.after join creek"
    / "indexLabel"
)

# 폴더명에 공백이 포함돼 있으므로 자동 탐색
COLOR_DIRS = [
    path
    for path in ROOT.iterdir()
    if path.is_dir()
    and "indexlabel" in path.name.lower()
    and "color" in path.name.lower()
]

OFFICIAL_RGB_TO_NAME = {
    (108, 64, 20): "dirt",
    (255, 229, 204): "sand",
    (0, 102, 0): "grass",
    (0, 255, 0): "tree",
    (0, 153, 153): "pole",
    (0, 128, 255): "water",
    (0, 0, 255): "sky",
    (255, 255, 0): "vehicle",
    (255, 0, 127): "container/generic-object",
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

    # 공식 클래스 색상이 아닌 미라벨 영역
    (0, 0, 0): "unlabeled/void",
}

# 색상 마스크를 파일명으로 검색
color_map = {}

for color_dir in COLOR_DIRS:
    for path in color_dir.rglob("*.png"):
        if path.name in color_map:
            raise RuntimeError(
                f"중복 색상 마스크 파일명: {path.name}\n"
                f"{color_map[path.name]}\n{path}"
            )

        color_map[path.name] = path

index_paths = sorted(INDEX_DIR.glob("*.png"))

if not index_paths:
    raise RuntimeError(
        f"indexLabel 파일을 찾지 못했습니다: {INDEX_DIR}"
    )

print("indexLabel 수:", len(index_paths))
print("색상 마스크 수:", len(color_map))
print("색상 마스크 폴더:")

for path in COLOR_DIRS:
    print(" -", path)

# index ID별 대응 RGB 색상 개수
id_color_counts = defaultdict(Counter)

for number, index_path in enumerate(index_paths, start=1):
    color_path = color_map.get(index_path.name)

    if color_path is None:
        raise FileNotFoundError(
            f"대응 색상 마스크 없음: {index_path.name}"
        )

    index_mask = np.array(Image.open(index_path))
    color_mask = np.array(
        Image.open(color_path).convert("RGB")
    )

    if index_mask.ndim != 2:
        raise RuntimeError(
            f"indexLabel이 단일 채널이 아닙니다: "
            f"{index_path.name}, {index_mask.shape}"
        )

    if index_mask.shape != color_mask.shape[:2]:
        raise RuntimeError(
            f"크기 불일치: {index_path.name}\n"
            f"index={index_mask.shape}, "
            f"color={color_mask.shape[:2]}"
        )

    # 같은 이미지 안의 각 index ID를 색상 마스크와 비교
    for index_id in np.unique(index_mask):
        pixels = color_mask[index_mask == index_id]

        colors, counts = np.unique(
            pixels,
            axis=0,
            return_counts=True,
        )

        for color, count in zip(colors, counts):
            rgb = tuple(int(value) for value in color)
            id_color_counts[int(index_id)][rgb] += int(count)

    if number % 500 == 0:
        print(f"진행: {number}/{len(index_paths)}")

print("\n=== index ID와 RGB 클래스 대응 결과 ===")

for index_id in sorted(id_color_counts):
    counter = id_color_counts[index_id]
    total = sum(counter.values())

    print(f"\n[index ID {index_id}]")

    for rgb, count in counter.most_common(5):
        class_name = OFFICIAL_RGB_TO_NAME.get(
            rgb,
            "unknown RGB",
        )

        ratio = count / total * 100

        print(
            f"  {rgb} → {class_name}: "
            f"{count:,} pixels ({ratio:.4f}%)"
        )