from __future__ import annotations

import itertools
import math
import random
from collections import Counter
from fractions import Fraction
from typing import Iterator


def integer_source_slots(weights: dict[str, float]) -> list[str]:
    if not weights or any(float(value) <= 0 for value in weights.values()):
        raise ValueError("Source weights must be non-empty and positive")
    fractions = {
        source: Fraction(str(float(weight))).limit_denominator(10_000)
        for source, weight in weights.items()
    }
    denominator = math.lcm(*(value.denominator for value in fractions.values()))
    counts = {
        source: value.numerator * (denominator // value.denominator)
        for source, value in fractions.items()
    }
    divisor = math.gcd(*counts.values())
    slots = [
        source
        for source in sorted(counts)
        for _ in range(counts[source] // divisor)
    ]
    if len(slots) > 10_000:
        raise ValueError("Source weights require an impractically large exact cycle")
    return slots


class WeightedSourceSchedule:
    """Infinite exact-ratio source schedule with per-source shuffled sampling."""

    def __init__(
        self,
        source_to_indices: dict[str, list[int]],
        source_weights: dict[str, float],
        seed: int,
    ) -> None:
        if set(source_to_indices) != set(source_weights):
            raise ValueError(
                "Sampler source groups and weights differ: "
                f"groups={sorted(source_to_indices)}, weights={sorted(source_weights)}"
            )
        if any(not values for values in source_to_indices.values()):
            raise ValueError("Every weighted source must contain at least one sample")
        self.source_to_indices = {
            source: list(values) for source, values in source_to_indices.items()
        }
        self.source_weights = {
            source: float(value) for source, value in source_weights.items()
        }
        self.seed = int(seed)
        self.source_slots = integer_source_slots(self.source_weights)

    def __iter__(self) -> Iterator[int]:
        rng = random.Random(self.seed)
        source_orders = {
            source: list(indices)
            for source, indices in self.source_to_indices.items()
        }
        source_positions = {source: len(values) for source, values in source_orders.items()}
        while True:
            slots = list(self.source_slots)
            rng.shuffle(slots)
            for source in slots:
                order = source_orders[source]
                position = source_positions[source]
                if position >= len(order):
                    rng.shuffle(order)
                    position = 0
                yield order[position]
                source_positions[source] = position + 1

    def source_counts(self, sample_count: int) -> Counter[str]:
        index_to_source = {
            index: source
            for source, indices in self.source_to_indices.items()
            for index in indices
        }
        return Counter(
            index_to_source[index]
            for index in itertools.islice(iter(self), int(sample_count))
        )
