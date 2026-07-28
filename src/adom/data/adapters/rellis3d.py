from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from ..models import DatasetError, SampleRecord, ValidationReport
from .base import DatasetAdapter


class Rellis3DAdapter(DatasetAdapter):
    name = "rellis3d"
    image_suffixes = frozenset({".jpg", ".jpeg", ".png"})
    mask_suffixes = frozenset({".png"})

    def _resolve_root(self) -> Path:
        root = self.input_root
        if any(path.is_dir() and path.name.isdigit() for path in root.iterdir()):
            return root
        candidates = [
            root / "Rellis-3D",
            root / "RELLIS-3D",
            root / "rellis3d",
            root / "raw" / "Rellis-3D",
        ]
        valid_by_resolved_path: dict[str, Path] = {}
        for path in candidates:
            if path.is_dir() and any(
                child.is_dir() and child.name.isdigit()
                for child in path.iterdir()
            ):
                # Rellis-3D and RELLIS-3D resolve to the same directory on
                # case-insensitive filesystems. Do not report that as ambiguity.
                valid_by_resolved_path[str(path.resolve()).casefold()] = path
        valid = list(valid_by_resolved_path.values())
        if len(valid) != 1:
            raise DatasetError(
                f"Could not resolve one RELLIS sequence root below {self.input_root}"
            )
        return valid[0]

    @staticmethod
    def _collect(
        directory: Path,
        suffixes: frozenset[str],
        report: ValidationReport,
        sequence: str,
        kind: str,
    ) -> dict[str, Path]:
        if not directory.is_dir():
            report.error("missing_directory", str(directory), sequence)
            return {}
        files: dict[str, Path] = {}
        for path in sorted(directory.iterdir()):
            if not path.is_file() or path.suffix.lower() not in suffixes:
                continue
            if path.stem in files:
                report.error(
                    "duplicate_stem",
                    f"{files[path.stem]} and {path}",
                    f"{sequence}_{path.stem}",
                )
                continue
            files[path.stem] = path
        if not files:
            report.error("empty_directory", f"No {kind} files in {directory}", sequence)
        return files

    def discover(self, report: ValidationReport) -> list[SampleRecord]:
        root = self._resolve_root()
        sequences = sorted(
            path for path in root.iterdir() if path.is_dir() and path.name.isdigit()
        )
        if not sequences:
            report.error("no_sequences", f"No numeric sequence directories in {root}")
            return []
        samples: list[SampleRecord] = []
        seen: set[str] = set()
        for sequence_dir in sequences:
            sequence = sequence_dir.name
            images = self._collect(
                sequence_dir / "pylon_camera_node",
                self.image_suffixes,
                report,
                sequence,
                "image",
            )
            masks = self._collect(
                sequence_dir / "pylon_camera_node_label_id",
                self.mask_suffixes,
                report,
                sequence,
                "mask",
            )
            for stem in sorted(set(images) - set(masks)):
                report.error(
                    "image_without_mask",
                    str(images[stem]),
                    f"{sequence}_{stem}",
                )
            for stem in sorted(set(masks) - set(images)):
                report.error(
                    "mask_without_image",
                    str(masks[stem]),
                    f"{sequence}_{stem}",
                )
            for stem in sorted(set(images) & set(masks)):
                sample_id = f"{sequence}_{stem}"
                if sample_id in seen:
                    report.error("duplicate_sample_id", sample_id, sample_id)
                    continue
                seen.add(sample_id)
                samples.append(
                    SampleRecord(
                        sample_id=sample_id,
                        sequence=sequence,
                        image_path=images[stem],
                        source_mask_path=masks[stem],
                    )
                )
        if not samples:
            report.error("no_samples", f"No complete pairs found below {root}")
        return samples

    def read_source_mask(self, path: Path) -> np.ndarray:
        with Image.open(path) as image:
            image.load()
            mask = np.asarray(image)
        if mask.ndim == 2:
            return mask
        if mask.ndim == 3 and mask.shape[2] in {3, 4}:
            first = mask[:, :, 0]
            if all(np.array_equal(first, mask[:, :, index]) for index in range(1, 3)):
                return first
        raise DatasetError(f"RELLIS source mask is not an indexed ID mask: {path}")
