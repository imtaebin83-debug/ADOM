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


class RareClassSourceSchedule:
    """Exact source quotas with deterministic rare-class sampling inside each source.

    A source slot is chosen by the same exact-ratio cycle as
    :class:`WeightedSourceSchedule`. Only the image choice inside that slot is
    reweighted, so rare-class sampling can never alter source exposure.
    """

    def __init__(
        self,
        source_to_indices: dict[str, list[int]],
        source_weights: dict[str, float],
        index_to_classes: dict[int, set[int]],
        rare_class_ids: list[int] | tuple[int, ...],
        seed: int,
        *,
        rare_probability: float = 0.5,
        temperature: float = 0.01,
    ) -> None:
        if not 0.0 <= float(rare_probability) <= 1.0:
            raise ValueError("Rare sampling probability must be between 0 and 1")
        if float(temperature) <= 0.0:
            raise ValueError("Rare sampling temperature must be positive")
        if set(source_to_indices) != set(source_weights):
            raise ValueError(
                "Sampler source groups and weights differ: "
                f"groups={sorted(source_to_indices)}, weights={sorted(source_weights)}"
            )
        if any(not values for values in source_to_indices.values()):
            raise ValueError("Every weighted source must contain at least one sample")
        all_indices = {index for values in source_to_indices.values() for index in values}
        if set(index_to_classes) != all_indices:
            raise ValueError("Class-presence statistics must cover every sampler index")
        self.source_to_indices = {
            source: list(values) for source, values in source_to_indices.items()
        }
        self.source_weights = {
            source: float(value) for source, value in source_weights.items()
        }
        self.index_to_classes = {
            int(index): {int(value) for value in classes}
            for index, classes in index_to_classes.items()
        }
        self.rare_class_ids = tuple(sorted({int(value) for value in rare_class_ids}))
        if not self.rare_class_ids:
            raise ValueError("At least one rare class ID is required")
        self.seed = int(seed)
        self.rare_probability = float(rare_probability)
        self.temperature = float(temperature)
        self.source_slots = integer_source_slots(self.source_weights)
        self.source_class_indices: dict[str, dict[int, list[int]]] = {}
        self.source_class_probabilities: dict[str, dict[int, float]] = {}
        for source, indices in self.source_to_indices.items():
            class_indices = {
                class_id: [
                    index
                    for index in indices
                    if class_id in self.index_to_classes[index]
                ]
                for class_id in self.rare_class_ids
            }
            class_indices = {
                class_id: values for class_id, values in class_indices.items() if values
            }
            self.source_class_indices[source] = class_indices
            if not class_indices:
                self.source_class_probabilities[source] = {}
                continue
            frequencies = {
                class_id: len(values) / len(indices)
                for class_id, values in class_indices.items()
            }
            scores = {class_id: 1.0 - value for class_id, value in frequencies.items()}
            maximum = max(scores.values())
            unnormalized = {
                class_id: math.exp((score - maximum) / self.temperature)
                for class_id, score in scores.items()
            }
            total = sum(unnormalized.values())
            self.source_class_probabilities[source] = {
                class_id: value / total for class_id, value in unnormalized.items()
            }

    def __iter__(self) -> Iterator[int]:
        rng = random.Random(self.seed)
        source_orders = {
            source: list(indices) for source, indices in self.source_to_indices.items()
        }
        source_positions = {source: len(values) for source, values in source_orders.items()}
        while True:
            slots = list(self.source_slots)
            rng.shuffle(slots)
            for source in slots:
                probabilities = self.source_class_probabilities[source]
                if probabilities and rng.random() < self.rare_probability:
                    class_ids = sorted(probabilities)
                    class_id = rng.choices(
                        class_ids,
                        weights=[probabilities[value] for value in class_ids],
                        k=1,
                    )[0]
                    yield rng.choice(self.source_class_indices[source][class_id])
                    continue
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
