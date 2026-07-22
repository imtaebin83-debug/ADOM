from __future__ import annotations

import json
import sys

from common import (
    PROCESSED_ROOT,
    TARGET_CLASSES,
    TARGET_PALETTE,
)


def main() -> None:
    info = {
        "dataset_name": "YCOR_ADOM",
        "task": "2D traversability semantic segmentation",
        "official_source_split": {
            "train": "931 images",
            "val": "145 images from source folder 'valid'",
        },
        "official_test_split": None,
        "image_format": "JPEG RGB",
        "mask_format": "PNG uint8 single-channel indexed mask",
        "ignore_index": 255,
        "reduce_zero_label": False,
        "num_model_classes": 4,
        "classes": [
            {
                "id": class_id,
                "name": TARGET_CLASSES[class_id],
                "palette_rgb": list(TARGET_PALETTE[class_id]),
            }
            for class_id in (0, 1, 2, 3)
        ],
        "splits": {
            "train": {
                "image_dir": "images/train",
                "mask_dir": "masks/train",
                "metadata": "metadata/train.csv",
            },
            "val": {
                "image_dir": "images/val",
                "mask_dir": "masks/val",
                "metadata": "metadata/val.csv",
            },
        },
        "notes": [
            "YCOR has no paved/asphalt class, so target ID 0 is expected to have zero pixels.",
            "Use YCOR together with RELLIS-3D/RUGD for the full ADOM 4-class model.",
            "Do not set reduce_zero_label=True.",
            "Use ignore_index=255.",
            "No arbitrary test split is generated because the official session-separated train/valid split is preserved.",
        ],
    }

    info_path = PROCESSED_ROOT / "dataset_info.json"
    with info_path.open("w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    snippet = """# MMSegmentation dataset fields for YCOR_ADOM
dataset_type = 'BaseSegDataset'
data_root = '/ABSOLUTE/PATH/YCOR_ADOM'

metainfo = dict(
    classes=(
        'paved_low_cost',
        'natural_low_cost',
        'medium_cost',
        'high_cost_or_obstacle',
    ),
    palette=[
        [128, 128, 128],
        [60, 180, 75],
        [255, 225, 25],
        [230, 25, 75],
    ],
)

train_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='images/train',
            seg_map_path='masks/train',
        ),
        img_suffix='.jpg',
        seg_map_suffix='.png',
        metainfo=metainfo,
        reduce_zero_label=False,
    )
)

val_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='images/val',
            seg_map_path='masks/val',
        ),
        img_suffix='.jpg',
        seg_map_suffix='.png',
        metainfo=metainfo,
        reduce_zero_label=False,
    )
)

# decode_head.num_classes = 4
# auxiliary_head.num_classes = 4  # only if an auxiliary head exists
# loss_decode.ignore_index = 255
"""
    snippet_path = PROCESSED_ROOT / "mmseg_dataset_snippet.py"
    snippet_path.write_text(snippet, encoding="utf-8")

    print(f"[saved] {info_path}")
    print(f"[saved] {snippet_path}")
    print("08_write_training_info.py: PASS")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise
