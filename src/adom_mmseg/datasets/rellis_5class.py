"""RELLIS-3D semantic-cost dataset for ADOM SegFormer experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from mmseg.datasets import BaseSegDataset
from mmseg.registry import DATASETS


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MAPPING_FILE = (
    REPO_ROOT / 'configs' / 'datasets' / 'rellis3d_semantic_cost_5class.yaml'
)


def load_cost_mapping(mapping_file: Optional[str | Path] = None) -> Dict[str, Any]:
    """Load and validate the RELLIS raw-id to ADOM semantic-cost mapping."""
    mapping_path = Path(mapping_file) if mapping_file else DEFAULT_MAPPING_FILE
    with mapping_path.open('r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    ignore_index = int(cfg.get('ignore_index', 255))
    classes = cfg['classes']
    valid_meta_ids = {int(item['id']) for item in classes}
    valid_targets = valid_meta_ids | {ignore_index}

    raw_to_meta = {}
    raw_label_names = {}
    for item in cfg['raw_labels']:
        raw_id = int(item['raw_id'])
        meta_id = int(item['meta_id'])
        if meta_id not in valid_targets:
            raise ValueError(
                f'Invalid meta_id={meta_id} for raw_id={raw_id} in {mapping_path}'
            )
        raw_to_meta[raw_id] = meta_id
        raw_label_names[raw_id] = item['name']

    metainfo = dict(
        classes=tuple(item['name'] for item in classes),
        palette=[item['color'] for item in classes],
    )

    return dict(
        mapping_path=mapping_path,
        ignore_index=ignore_index,
        metainfo=metainfo,
        raw_to_meta=raw_to_meta,
        raw_label_names=raw_label_names,
    )


def build_dense_label_map(
    raw_to_meta: Dict[int, int],
    ignore_index: int = 255,
    max_label_id: int = 255,
) -> Dict[int, int]:
    """Build a dense MMSeg label_map so unknown raw ids become ignore."""
    label_map = {raw_id: ignore_index for raw_id in range(max_label_id + 1)}
    label_map.update({int(k): int(v) for k, v in raw_to_meta.items()})
    return label_map


@DATASETS.register_module()
class RELLIS5ClassDataset(BaseSegDataset):
    """RELLIS-3D remapped to ADOM semantic cost classes.

    The model predicts four logits:
        0: paved_low_cost
        1: natural_low_cost
        2: medium_cost
        3: high_cost_or_obstacle

    Pixels mapped to 255 are ignored by the segmentation loss.
    """

    METAINFO = dict(
        classes=(
            'paved_low_cost',
            'natural_low_cost',
            'medium_cost',
            'high_cost_or_obstacle',
        ),
        palette=[
            [128, 64, 128],
            [0, 180, 0],
            [255, 190, 0],
            [220, 20, 60],
        ],
    )

    def __init__(
        self,
        mapping_file: Optional[str] = None,
        img_suffix: str = '.jpg',
        seg_map_suffix: str = '.png',
        reduce_zero_label: bool = False,
        ignore_index: int = 255,
        **kwargs,
    ) -> None:
        mapping = load_cost_mapping(mapping_file)
        if ignore_index != mapping['ignore_index']:
            raise ValueError(
                'Dataset ignore_index must match the mapping file: '
                f'{ignore_index} != {mapping["ignore_index"]}'
            )

        self.mapping_file = str(mapping['mapping_path'])
        self.raw_to_meta = mapping['raw_to_meta']
        self.rellis_label_map = build_dense_label_map(
            self.raw_to_meta,
            ignore_index=mapping['ignore_index'],
        )

        metainfo = kwargs.pop('metainfo', None)
        if metainfo is None:
            metainfo = mapping['metainfo']

        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            metainfo=metainfo,
            reduce_zero_label=reduce_zero_label,
            ignore_index=ignore_index,
            **kwargs,
        )

    def load_data_list(self):
        """Attach the RELLIS raw-id mapping to each segmentation sample."""
        data_list = super().load_data_list()
        for data_info in data_list:
            if 'seg_map_path' in data_info:
                data_info['label_map'] = self.rellis_label_map
                data_info['reduce_zero_label'] = False
        return data_list
