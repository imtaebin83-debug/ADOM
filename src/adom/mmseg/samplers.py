from __future__ import annotations

import itertools
from collections.abc import Iterator, Sized
from typing import Any

from mmengine.dist import get_dist_info, sync_random_seed
from mmengine.registry import DATA_SAMPLERS
from torch.utils.data import Sampler

from adom.runtime.source_sampling import WeightedSourceSchedule


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
