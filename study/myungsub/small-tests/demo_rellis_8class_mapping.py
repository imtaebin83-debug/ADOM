#!/usr/bin/env python3
"""
demo_rellis_8class_mapping.py
====================================================================
ADOM 프로젝트: 탄약 보급 비전투차량(아리온스맷 / HR-셰르파) 시나리오
RELLIS-3D 원본 클래스 및 레퍼런스 이미지를 8개 시맨틱 클래스로 변환하는 소형 테스트

[ 사용 방법 ]
  1) 기본 합성 레퍼런스 씬 테스트:
     python3 demo_rellis_8class_mapping.py --out result.png

  2) 커스텀 외부 PNG 이미지(마스크 또는 컬러 이미지) 입력 변환:
     python3 demo_rellis_8class_mapping.py --image input_mask.png --out my_output.png
====================================================================
"""

import os
import sys
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ==============================================================================
# 1. ADOM 8-Class 온톨로지 정의 (클래스 ID, 명칭, 표준 RGB 색상, ROS2 Cost)
# ==============================================================================
ADOM_CLASSES = {
    0: {"name": "0: Void / Sky",         "rgb": (210, 210, 210), "cost": 0},    # 회색
    1: {"name": "1: Traversable-Safe",   "rgb": ( 46, 204, 113), "cost": 0},    # 안전 녹색
    2: {"name": "2: Traversable-Rough",  "rgb": (241, 196,  15), "cost": 128},  # 주의 노랑
    3: {"name": "3: High-Risk-Terrain",  "rgb": (230, 126,  34), "cost": 210},  # 주황 (Mud)
    4: {"name": "4: Water / Puddle",     "rgb": ( 52, 152, 219), "cost": 254},  # 파랑 (Lethal)
    5: {"name": "5: Static-Obstacle",    "rgb": ( 44,  62,  80), "cost": 254},  # 짙은 남색/숲
    6: {"name": "6: Rare-Obstacle(Log)", "rgb": (231,  76,  60), "cost": 254},  # 빨강 (Log/Boulder 강조!)
    7: {"name": "7: Dynamic-Object",     "rgb": (155,  89, 182), "cost": 254},  # 보라
}

# ==============================================================================
# 2. 레퍼런스 이미지(1~8번 영역) -> ADOM 8-Class ID 매핑 규칙
# ==============================================================================
REF_TO_ADOM_MAPPING = {
    1: 0, # 하늘 -> Void/Sky
    2: 1, # 도로 -> Traversable-Safe
    3: 1, # 왼쪽 지면 -> Traversable-Safe
    4: 6, # 쓰러진 통나무 -> Rare-Obstacle (Log)
    5: 2, # 오른쪽 진경 초목 -> Traversable-Rough
    6: 5, # 오른쪽 중경 초목 -> Static-Obstacle
    7: 5, # 왼쪽 중경 초목 -> Static-Obstacle
    8: 5, # 원경 숲 -> Static-Obstacle
}

REF_PALETTE = {
    1: (225, 225, 225), # 하늘
    2: (138,  82,  48), # 도로
    3: (198, 140,  55), # 왼쪽 지면
    4: ( 82,  83,  88), # 쓰러진 통나무 (Rare Obstacle)
    5: ( 85, 160,  70), # 오른쪽 진경 초목
    6: ( 35, 110,  55), # 오른쪽 중경 초목
    7: ( 45, 125,  65), # 왼쪽 중경 초목
    8: ( 20,  75,  35), # 원경 숲
}


def generate_reference_scene(height=480, width=640):
    """
    첨부된 오프로드 레퍼런스 이미지의 구도를 모사한 2D 씬 라벨(1~8)을 생성합니다.
    """
    ref_map = np.ones((height, width), dtype=np.uint8) * 1  # 하늘

    y_grid, x_grid = np.mgrid[0:height, 0:width]
    y_norm = y_grid / height
    x_norm = x_grid / width

    forest_mask = (y_norm > 0.35) & ((x_norm < 0.35) | (x_norm > 0.55))
    ref_map[forest_mask] = 8

    mg_left = (y_norm > 0.45) & (x_norm < 0.30)
    mg_right = (y_norm > 0.45) & (x_norm > 0.50)
    ref_map[mg_left] = 7
    ref_map[mg_right] = 6

    fg_right = (y_norm > 0.60) & (x_norm > 0.48)
    ref_map[fg_right] = 5

    left_ground = (y_norm > 0.55) & (x_norm < 0.45)
    ref_map[left_ground] = 3

    road_mask = (y_norm > 0.50) & (x_norm > (0.42 - (y_norm - 0.5) * 0.55)) & (x_norm < (0.48 + (y_norm - 0.5) * 0.75))
    ref_map[road_mask] = 2

    log_mask = (y_norm > 0.51) & (y_norm < 0.61) & (x_norm < 0.38)
    ref_map[log_mask] = 4

    return ref_map


def load_and_parse_custom_png(image_path):
    """
    사용자가 지정한 외부 PNG 이미지 파일을 읽어 레퍼런스 라벨 맵(1~8) 또는 8-Class ID로 파싱
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"[ERROR] 지정한 PNG 파일을 찾을 수 없습니다: {image_path}")

    img = Image.open(image_path)
    img_arr = np.array(img)

    # 1. 2D 인덱스 라벨 마스크인 경우 (그레이스케일 또는 라벨 PNG)
    if img_arr.ndim == 2:
        return img_arr.astype(np.uint8), img.convert("RGB")

    # 2. RGB 컬러 이미지인 경우 -> 각 픽셀 색상과 가장 가까운 REF_PALETTE ID(1~8)로 근사 분류
    rgb_arr = img_arr[:, :, :3].astype(np.int32)
    h, w, _ = rgb_arr.shape
    ref_map = np.ones((h, w), dtype=np.uint8)
    min_dist = np.full((h, w), 1e9, dtype=np.float32)

    for ref_id, palette_color in REF_PALETTE.items():
        c = np.array(palette_color, dtype=np.int32)
        dist = np.sum((rgb_arr - c) ** 2, axis=-1)
        mask = dist < min_dist
        min_dist[mask] = dist[mask]
        ref_map[mask] = ref_id

    return ref_map, img.convert("RGB")


def map_ref_to_adom8(ref_map):
    """
    레퍼런스 라벨(1~8)을 ADOM 8-Class ID(0~7)로 매핑
    """
    adom_map = np.zeros_like(ref_map, dtype=np.uint8)
    for ref_id, adom_id in REF_TO_ADOM_MAPPING.items():
        adom_map[ref_map == ref_id] = adom_id
    return adom_map


def render_color_mask(class_map, color_dict):
    """
    클래스 ID 배열 -> RGB PIL Image 변환
    """
    h, w = class_map.shape
    rgb_img = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, cls_info in color_dict.items():
        color = cls_info["rgb"] if isinstance(cls_info, dict) else cls_info
        rgb_img[class_map == cls_id] = color
    return Image.fromarray(rgb_img)


def render_costmap(adom_map):
    """
    ADOM 8-Class 라벨 -> ROS2 Nav2 주행 비용(Costmap: 0~254) 컬러 매핑
    """
    h, w = adom_map.shape
    cost_map = np.zeros((h, w), dtype=np.uint8)
    for adom_id, cls_info in ADOM_CLASSES.items():
        cost_map[adom_map == adom_id] = cls_info["cost"]

    heatmap = np.zeros((h, w, 3), dtype=np.uint8)
    heatmap[cost_map == 0] = (46, 204, 113)
    heatmap[cost_map == 128] = (241, 196, 15)
    heatmap[cost_map == 210] = (230, 126, 34)
    heatmap[cost_map == 254] = (192, 57, 43)

    return Image.fromarray(heatmap), cost_map


def print_class_statistics(adom_map):
    """
    8개 시맨틱 클래스별 픽셀 점유율 리포트 출력
    """
    total_pixels = adom_map.size
    print("\n" + "="*68)
    print(" [ADOM 8-Class Semantic Segmentation 분포 분석 리포트]")
    print("="*68)
    print(f"{'ID':<4} | {'Class Name':<24} | {'Pixel Count':<12} | {'Ratio (%)':<10} | {'ROS2 Cost'}")
    print("-" * 68)

    for cls_id, cls_info in ADOM_CLASSES.items():
        count = np.sum(adom_map == cls_id)
        ratio = (count / total_pixels) * 100.0
        print(f"{cls_id:<4} | {cls_info['name']:<24} | {count:<12,d} | {ratio:>7.2f} %  | {cls_info['cost']}")

    print("="*68 + "\n")


def make_comparison_canvas(img_ref, img_adom, img_cost, save_path):
    """
    3개의 렌더링 결과(원본, 8-Class 시맨틱, ROS2 Costmap)를 나란히 연결하여 PNG 저장
    """
    w, h = img_ref.size
    border = 10
    title_h = 50
    footer_h = 130

    canvas_w = w * 3 + border * 4
    canvas_h = h + title_h + footer_h + border * 2
    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    font = ImageFont.load_default()

    draw.text((canvas_w // 2 - 190, 15), "ADOM Off-Road 8-Class Semantic Mapping & Costmap Demo",
              fill=(30, 30, 30), font=font)

    y_offset = title_h
    canvas.paste(img_ref, (border, y_offset))
    canvas.paste(img_adom, (border * 2 + w, y_offset))
    canvas.paste(img_cost, (border * 3 + w * 2, y_offset))

    draw.text((border + 10, y_offset - 18), "1. Input Scene / Reference Mask", fill=(0,0,0))
    draw.text((border * 2 + w + 10, y_offset - 18), "2. ADOM 8-Class Mapped Semantic Mask", fill=(0,0,0))
    draw.text((border * 3 + w * 2 + 10, y_offset - 18), "3. ROS2 Nav2 Traversability Costmap", fill=(0,0,0))

    legend_y = y_offset + h + 20
    draw.text((border + 10, legend_y), "[ ADOM 8-Class Ontology Palette ]", fill=(40, 40, 40))

    x_col1 = border + 10
    x_col2 = border + 320
    row_h = 22

    items = list(ADOM_CLASSES.items())
    for i, (cls_id, info) in enumerate(items):
        col_x = x_col1 if i < 4 else x_col2
        row_y = legend_y + 24 + (i % 4) * row_h
        draw.rectangle([col_x, row_y, col_x + 16, row_y + 14], fill=info["rgb"], outline=(0,0,0))
        draw.text((col_x + 24, row_y), f"{info['name']} (Cost: {info['cost']})", fill=(30, 30, 30))

    canvas.save(save_path)
    print(f"[SUCCESS] 시각화 결과가 저장되었습니다 -> {os.path.abspath(save_path)}")


def main():
    parser = argparse.ArgumentParser(description="ADOM RELLIS-3D 8-Class Semantic Mapping Demo")
    parser.add_argument("--image", "-i", type=str, default=None,
                        help="외부 입력 PNG 이미지/마스크 경로 (지정하지 않으면 레퍼런스 합성 씬 사용)")
    parser.add_argument("--out", "-o", type=str, default="output_8class_comparison.png",
                        help="출력 결과 PNG 이미지 저장 경로")
    args = parser.parse_args()

    if args.image:
        print(f"[INFO] 외부 PNG 이미지 파일 읽는 중: {args.image}")
        ref_map, img_ref = load_and_parse_custom_png(args.image)
    else:
        print("[INFO] 기본 레퍼런스 오프로드 주행 씬 라벨 생성 중...")
        ref_map = generate_reference_scene(480, 640)
        img_ref = render_color_mask(ref_map, REF_PALETTE)

    print("[INFO] RELLIS-3D 기준 -> ADOM 8-Class 온톨로지 매핑 수행 중...")
    adom_map = map_ref_to_adom8(ref_map)

    print("[INFO] ROS2 Nav2 주행 비용(Costmap) 변환 중...")
    img_adom = render_color_mask(adom_map, ADOM_CLASSES)
    img_cost, _ = render_costmap(adom_map)

    print_class_statistics(adom_map)

    make_comparison_canvas(img_ref, img_adom, img_cost, args.out)


if __name__ == "__main__":
    main()
