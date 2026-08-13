from __future__ import annotations

import itertools
from collections.abc import Iterator, Sized
from typing import Any

from mmengine.dist import get_dist_info, sync_random_seed
from mmengine.registry import DATA_SAMPLERS
from torch.utils.data import Sampler

from adom.runtime.source_sampling import RareClassSourceSchedule, WeightedSourceSchedule


@DATA_SAMPLERS.register_module()
class SourceWeightedInfiniteSampler(Sampler[int]):
    """Deterministic exact-ratio infinite sampler keyed by sample ID prefix."""

    def __init__(
        self,
        dataset: Sized,
        source_weights: dict[str, float],
        seed: int | None = None,
        start_index: int = 0,
    ) -> None:
        rank, world_size = get_dist_info()
        if seed is None:
            seed = sync_random_seed()
        if start_index < 0:
            raise ValueError("start_index must be non-negative")
        self.dataset = dataset
        self.source_weights = {
            source: float(value) for source, value in source_weights.items()
        }
        self.seed = int(seed)
        self.start_index = int(start_index)
        self.rank = rank
        self.world_size = world_size
        if hasattr(dataset, "full_init"):
            dataset.full_init()  # type: ignore[attr-defined]
        source_to_indices: dict[str, list[int]] = {}
        for index in range(len(dataset)):  # type: ignore[arg-type]
            info: dict[str, Any]
            if hasattr(dataset, "get_data_info"):
                info = dataset.get_data_info(index)  # type: ignore[attr-defined]
            else:
                info = dataset[index]  # type: ignore[index]
            sample_id = str(info.get("sample_id", ""))
            if "/" not in sample_id:
                raise ValueError(
                    "Weighted sampler requires source-prefixed sample_id: "
                    f"{sample_id}"
                )
            source_to_indices.setdefault(sample_id.split("/", 1)[0], []).append(index)
        self.schedule = WeightedSourceSchedule(
            source_to_indices, self.source_weights, self.seed
        )

    def __iter__(self) -> Iterator[int]:
        start = self.start_index + self.rank
        yield from itertools.islice(iter(self.schedule), start, None, self.world_size)

    def __len__(self) -> int:
        return len(self.dataset)  # type: ignore[arg-type]

    def set_epoch(self, epoch: int) -> None:
        return None


@DATA_SAMPLERS.register_module()
class SourceRareClassInfiniteSampler(Sampler[int]):
    """Keep exact source quotas while reweighting rare images within each quota."""

    def __init__(
        self,
        dataset: Sized,
        source_weights: dict[str, float],
        rare_class_ids: list[int] | tuple[int, ...],
        seed: int | None = None,
        start_index: int = 0,
        rare_probability: float = 0.5,
        temperature: float = 0.01,
        minimum_pixels: int = 1,
        ignore_index: int = 255,
    ) -> None:
        rank, world_size = get_dist_info()
        if seed is None:
            seed = sync_random_seed()
        if start_index < 0:
            raise ValueError("start_index must be non-negative")
        if minimum_pixels < 1:
            raise ValueError("minimum_pixels must be positive")
        self.dataset = dataset
        self.source_weights = {
            source: float(value) for source, value in source_weights.items()
        }
        self.rare_class_ids = tuple(sorted({int(value) for value in rare_class_ids}))
        self.seed = int(seed)
        self.start_index = int(start_index)
        self.rare_probability = float(rare_probability)
        self.temperature = float(temperature)
        self.minimum_pixels = int(minimum_pixels)
        self.ignore_index = int(ignore_index)
        self.rank = rank
        self.world_size = world_size
        if hasattr(dataset, "full_init"):
            dataset.full_init()  # type: ignore[attr-defined]

        import numpy as np
        from PIL import Image

        source_to_indices: dict[str, list[int]] = {}
        index_to_classes: dict[int, set[int]] = {}
        for index in range(len(dataset)):  # type: ignore[arg-type]
            info: dict[str, Any]
            if hasattr(dataset, "get_data_info"):
                info = dataset.get_data_info(index)  # type: ignore[attr-defined]
            else:
                info = dataset[index]  # type: ignore[index]
            sample_id = str(info.get("sample_id", ""))
            if "/" not in sample_id:
                raise ValueError(
                    "Rare-class sampler requires source-prefixed sample_id: "
                    f"{sample_id}"
                )
            source_to_indices.setdefault(sample_id.split("/", 1)[0], []).append(index)
            mask_path = info.get("seg_map_path")
            if not mask_path:
                raise ValueError(f"Rare-class sampler found no seg_map_path: {sample_id}")
            with Image.open(str(mask_path)) as image:
                mask = np.asarray(image.convert("L"))
            classes = {
                class_id
                for class_id in self.rare_class_ids
                if int(np.count_nonzero(mask == class_id)) >= self.minimum_pixels
            }
            index_to_classes[index] = classes
        self.schedule = RareClassSourceSchedule(
            source_to_indices,
            self.source_weights,
            index_to_classes,
            self.rare_class_ids,
            self.seed,
            rare_probability=self.rare_probability,
            temperature=self.temperature,
        )

    def __iter__(self) -> Iterator[int]:
        start = self.start_index + self.rank
        yield from itertools.islice(iter(self.schedule), start, None, self.world_size)

    def __len__(self) -> int:
        return len(self.dataset)  # type: ignore[arg-type]

    def set_epoch(self, epoch: int) -> None:
        return None
