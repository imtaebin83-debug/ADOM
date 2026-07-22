from __future__ import annotations

import sys
from pathlib import Path

from common import (
    EXPECTED_COUNTS,
    OUTPUT_SPLITS,
    SOURCE_SPLITS,
    discover_dataset_root,
    ensure_directories,
    find_pair,
    load_source_mask,
)


def main() -> None:
    ensure_directories()
    dataset_root = discover_dataset_root()

    print(f"[dataset root] {dataset_root}")

    total = 0
    first_pair = None

    for output_split in OUTPUT_SPLITS:
        source_split = SOURCE_SPLITS[output_split]
        split_dir = dataset_root / source_split

        sample_dirs = sorted(
            path for path in split_dir.iterdir()
            if path.is_dir()
        )

        print(
            f"[{output_split}] source='{source_split}', "
            f"sample folders={len(sample_dirs):,}"
        )

        expected = EXPECTED_COUNTS[output_split]
        if len(sample_dirs) != expected:
            print(
                f"WARNING: official expected count is {expected:,}, "
                f"but found {len(sample_dirs):,}."
            )

        total += len(sample_dirs)

        if first_pair is None and sample_dirs:
            first_pair = find_pair(sample_dirs[0])

    if first_pair is None:
        raise RuntimeError("No image-mask pair was found.")

    image_path, mask_path = first_pair
    print(f"[sample image] {image_path}")
    print(f"[sample mask]  {mask_path}")

    source_mask, encoding = load_source_mask(mask_path)
    print(f"[mask encoding] {encoding}")
    print(f"[mask shape]    {source_mask.shape}")
    print(f"[all samples]   {total:,}")

    print("\n01_check_raw_structure.py: PASS")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise
