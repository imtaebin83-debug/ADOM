from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from ..models import SampleRecord, ValidationReport


class DatasetAdapter(ABC):
    name: str

    def __init__(self, input_root: Path):
        self.input_root = input_root

    @abstractmethod
    def discover(self, report: ValidationReport) -> list[SampleRecord]:
        """Discover every complete sample and report incomplete pairs."""

    @abstractmethod
    def read_source_mask(self, path: Path) -> np.ndarray:
        """Read a source semantic mask as a two-dimensional integer array."""
