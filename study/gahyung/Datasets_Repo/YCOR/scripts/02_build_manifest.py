from __future__ import annotations

import sys

import pandas as pd

from common import (
    MANIFEST_DIR,
    OUTPUT_SPLITS,
    SOURCE_SPLITS,
    discover_dataset_root,
    ensure_directories,
    find_pair,
)


def main() -> None:
    ensure_directories()
    dataset_root = discover_dataset_root()
    all_frames = []

    for output_split in OUTPUT_SPLITS:
        source_split = SOURCE_SPLITS[output_split]
        split_dir = dataset_root / source_split

        sample_dirs = sorted(
            path for path in split_dir.iterdir()
            if path.is_dir()
        )

        rows = []
        for index, sample_dir in enumerate(sample_dirs):
            image_path, mask_path = find_pair(sample_dir)
            sample_id = f"ycor_{output_split}_{index:06d}"

            rows.append(
                {
                    "dataset": "YCOR",
                    "split": output_split,
                    "source_split": source_split,
                    "sample_index": index,
                    "sample_id": sample_id,
                    "source_sample_name": sample_dir.name,
                    "source_sample_dir": str(sample_dir.resolve()),
                    "source_image": str(image_path.resolve()),
                    "source_mask": str(mask_path.resolve()),
                    "output_image_filename": sample_id + ".jpg",
                    "output_mask_filename": sample_id + ".png",
                }
            )

        manifest = pd.DataFrame(rows)

        if manifest["sample_id"].duplicated().any():
            raise ValueError(f"Duplicate generated IDs in {output_split}")

        out_path = MANIFEST_DIR / f"{output_split}.csv"
        manifest.to_csv(out_path, index=False, encoding="utf-8-sig")
        all_frames.append(manifest)

        print(f"[{output_split}] {len(manifest):,} pairs")
        print(f"  saved: {out_path}")

    all_manifest = pd.concat(all_frames, ignore_index=True)

    if all_manifest["sample_id"].duplicated().any():
        raise ValueError("Duplicate sample IDs across splits.")

    all_path = MANIFEST_DIR / "all_samples.csv"
    all_manifest.to_csv(all_path, index=False, encoding="utf-8-sig")

    print(f"\n[all] {len(all_manifest):,} pairs")
    print(f"[saved] {all_path}")
    print("02_build_manifest.py: PASS")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise
