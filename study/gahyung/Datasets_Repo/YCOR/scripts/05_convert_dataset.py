from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from common import (
    ALLOWED_TARGET_IDS,
    IMAGES_ROOT,
    MANIFEST_DIR,
    MASKS_ROOT,
    METADATA_ROOT,
    OUTPUT_SPLITS,
    ensure_directories,
    remap_mask,
    save_uint8_mask,
)


def main() -> None:
    ensure_directories()

    for split in OUTPUT_SPLITS:
        manifest_path = MANIFEST_DIR / f"{split}.csv"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest not found: {manifest_path}\n"
                "Run 02_build_manifest.py first."
            )

        manifest = pd.read_csv(manifest_path, dtype=str)
        metadata_rows = []

        image_dir = IMAGES_ROOT / split
        mask_dir = MASKS_ROOT / split

        for _, row in tqdm(
            manifest.iterrows(),
            total=len(manifest),
            desc=f"convert {split}",
        ):
            source_image = Path(row["source_image"])
            source_mask = Path(row["source_mask"])

            output_image = image_dir / row["output_image_filename"]
            output_mask = mask_dir / row["output_mask_filename"]

            with Image.open(source_image) as image:
                image.load()
                image_size = image.size
                if image.mode != "RGB":
                    image = image.convert("RGB")
                    image.save(output_image, format="JPEG", quality=95)
                else:
                    shutil.copy2(source_image, output_image)

            target_mask, source_encoding = remap_mask(source_mask)
            mask_size = (target_mask.shape[1], target_mask.shape[0])

            if image_size != mask_size:
                raise ValueError(
                    f"Size mismatch after load: image={image_size}, "
                    f"mask={mask_size}, sample={row['sample_id']}"
                )

            used_ids = set(int(v) for v in np.unique(target_mask))
            invalid_ids = used_ids - ALLOWED_TARGET_IDS
            if invalid_ids:
                raise ValueError(
                    f"Invalid target IDs {sorted(invalid_ids)}: {source_mask}"
                )

            save_uint8_mask(target_mask, output_mask)

            metadata_rows.append(
                {
                    "dataset": "YCOR",
                    "split": split,
                    "sample_id": row["sample_id"],
                    "source_split": row["source_split"],
                    "source_sample_name": row["source_sample_name"],
                    "image_filename": row["output_image_filename"],
                    "mask_filename": row["output_mask_filename"],
                    "image_relpath": output_image.relative_to(
                        PROCESSED_ROOT
                    ).as_posix(),
                    "mask_relpath": output_mask.relative_to(
                        PROCESSED_ROOT
                    ).as_posix(),
                    "width": image_size[0],
                    "height": image_size[1],
                    "source_encoding": source_encoding,
                    "source_image": row["source_image"],
                    "source_mask": row["source_mask"],
                }
            )

        metadata = pd.DataFrame(metadata_rows)
        metadata_path = METADATA_ROOT / f"{split}.csv"
        metadata.to_csv(metadata_path, index=False, encoding="utf-8-sig")

        print(f"[{split}] exported {len(metadata):,} pairs")
        print(f"  images:   {image_dir}")
        print(f"  masks:    {mask_dir}")
        print(f"  metadata: {metadata_path}")

    print("\n05_convert_dataset.py: PASS")


# Imported here to keep the path reference explicit in generated metadata.
from common import PROCESSED_ROOT


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise
