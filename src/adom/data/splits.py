from __future__ import annotations

from pathlib import Path

from .models import DatasetError, ValidationReport


SPLIT_NAMES = ("train", "val", "test")


def load_splits(
    split_root: Path,
    report: ValidationReport,
) -> dict[str, tuple[str, ...]]:
    result: dict[str, tuple[str, ...]] = {}
    owners: dict[str, str] = {}
    for split in SPLIT_NAMES:
        path = split_root / f"{split}.txt"
        if not path.is_file():
            report.error("missing_split_file", str(path), split)
            result[split] = ()
            continue
        raw_ids = [
            line.strip()
            for line in path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        seen: set[str] = set()
        unique: list[str] = []
        for sample_id in raw_ids:
            if any(character.isspace() for character in sample_id):
                report.error(
                    "invalid_split_id",
                    f"Whitespace is not allowed: {sample_id!r}",
                    sample_id,
                )
                continue
            if sample_id in seen:
                report.error("duplicate_split_id", split, sample_id)
                continue
            seen.add(sample_id)
            unique.append(sample_id)
            previous = owners.get(sample_id)
            if previous is not None:
                report.error(
                    "split_overlap",
                    f"{previous} and {split}",
                    sample_id,
                )
            else:
                owners[sample_id] = split
        if not unique:
            report.error("empty_split", str(path), split)
        result[split] = tuple(unique)
    return result


def split_owner_map(splits: dict[str, tuple[str, ...]]) -> dict[str, str]:
    owners: dict[str, str] = {}
    for split, sample_ids in splits.items():
        for sample_id in sample_ids:
            if sample_id in owners:
                raise DatasetError(f"Split overlap for {sample_id}")
            owners[sample_id] = split
    return owners
