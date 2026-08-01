from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import common
from common import (
    ALLOWED_TARGET_IDS,
    EXPECTED_COUNTS,
    OUTPUT_SPLITS,
    discover_dataset_root,
    is_dataset_root,
    save_uint8_mask,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert YCOR into the ADOM Cost4 format."
    )

    parser.add_argument(
        "--input-root",
        type=Path,
        default=None,
        help="YCOR dataset root containing train/ and valid/.",
    )

    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=common.MANIFEST_DIR,
        help="Directory containing train.csv and val.csv manifests.",
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=common.PROCESSED_ROOT,
        help="Processed YCOR_ADOM output directory.",
    )

    parser.add_argument(
        "--mapping",
        type=Path,
        default=common.MAPPING_FILE,
        help="YCOR label_mapping.json path.",
    )

    return parser.parse_args()


def resolve_dataset_root(
    input_root: Path | None,
) -> Path:
    if input_root is None:
        return discover_dataset_root()

    dataset_root = input_root.expanduser().resolve()

    if not is_dataset_root(dataset_root):
        raise FileNotFoundError(
            "The selected input root must contain "
            f"non-empty train/ and valid/ folders: {dataset_root}"
        )

    return dataset_root


def resolve_manifest_source(
    dataset_root: Path,
    stored_value: str,
    field_name: str,
) -> Path:
    stored_path = Path(stored_value)

    if stored_path.is_absolute():
        raise ValueError(
            f"Absolute path found in manifest column "
            f"{field_name}: {stored_value}"
        )

    resolved_path = (
        dataset_root
        / stored_path
    ).resolve()

    try:
        resolved_path.relative_to(dataset_root)
    except ValueError as exc:
        raise ValueError(
            f"Manifest path escapes the dataset root: {stored_value}"
        ) from exc

    return resolved_path


def main() -> None:
    args = parse_args()

    dataset_root = resolve_dataset_root(
        args.input_root
    )

    manifest_dir = (
        args.manifest_dir
        .expanduser()
        .resolve()
    )

    processed_root = (
        args.output_root
        .expanduser()
        .resolve()
    )

    mapping_file = (
        args.mapping
        .expanduser()
        .resolve()
    )

    if not mapping_file.is_file():
        raise FileNotFoundError(
            f"Mapping file not found: {mapping_file}"
        )

    # Preserve the original common.py and select
    # the mapping only for this script process.
    common.MAPPING_FILE = mapping_file

    images_root = (
        processed_root
        / "images"
    )

    masks_root = (
        processed_root
        / "masks"
    )

    metadata_root = (
        processed_root
        / "metadata"
    )

    metadata_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    for split in OUTPUT_SPLITS:
        manifest_path = (
            manifest_dir
            / f"{split}.csv"
        )

        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest not found: {manifest_path}\n"
                "Run 02_build_manifest.py first."
            )

        manifest = pd.read_csv(
            manifest_path,
            dtype=str,
        )

        expected_count = EXPECTED_COUNTS[
            split
        ]

        if len(manifest) != expected_count:
            raise RuntimeError(
                f"{split} manifest count mismatch: "
                f"expected={expected_count}, "
                f"actual={len(manifest)}"
            )

        metadata_rows = []

        image_dir = (
            images_root
            / split
        )

        mask_dir = (
            masks_root
            / split
        )

        image_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        mask_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        for _, row in tqdm(
            manifest.iterrows(),
            total=len(manifest),
            desc=f"convert {split}",
        ):
            source_image = resolve_manifest_source(
                dataset_root,
                row["source_image"],
                "source_image",
            )

            source_mask = resolve_manifest_source(
                dataset_root,
                row["source_mask"],
                "source_mask",
            )

            output_image = (
                image_dir
                / row["output_image_filename"]
            )

            output_mask = (
                mask_dir
                / row["output_mask_filename"]
            )

            with Image.open(
                source_image
            ) as image:
                image.load()
                image_size = image.size

                if image.mode != "RGB":
                    image = image.convert(
                        "RGB"
                    )

                    image.save(
                        output_image,
                        format="JPEG",
                        quality=95,
                    )
                else:
                    shutil.copy2(
                        source_image,
                        output_image,
                    )

            target_mask, source_encoding = (
                common.remap_mask(
                    source_mask
                )
            )

            mask_size = (
                target_mask.shape[1],
                target_mask.shape[0],
            )

            if image_size != mask_size:
                raise ValueError(
                    "Size mismatch after load: "
                    f"image={image_size}, "
                    f"mask={mask_size}, "
                    f"sample={row['sample_id']}"
                )

            used_ids = {
                int(value)
                for value
                in np.unique(
                    target_mask
                )
            }

            invalid_ids = (
                used_ids
                - ALLOWED_TARGET_IDS
            )

            if invalid_ids:
                raise ValueError(
                    "Invalid target IDs "
                    f"{sorted(invalid_ids)}: "
                    f"{row['source_mask']}"
                )

            save_uint8_mask(
                target_mask,
                output_mask,
            )

            metadata_rows.append(
                {
                    "dataset": "YCOR",
                    "split": split,
                    "sample_id": (
                        row["sample_id"]
                    ),
                    "source_split": (
                        row["source_split"]
                    ),
                    "source_sample_name": (
                        row[
                            "source_sample_name"
                        ]
                    ),
                    "image_filename": (
                        row[
                            "output_image_filename"
                        ]
                    ),
                    "mask_filename": (
                        row[
                            "output_mask_filename"
                        ]
                    ),
                    "image_relpath": (
                        output_image
                        .relative_to(
                            processed_root
                        )
                        .as_posix()
                    ),
                    "mask_relpath": (
                        output_mask
                        .relative_to(
                            processed_root
                        )
                        .as_posix()
                    ),
                    "width": image_size[0],
                    "height": image_size[1],
                    "source_encoding": (
                        source_encoding
                    ),
                    "source_image": (
                        Path(
                            row["source_image"]
                        ).as_posix()
                    ),
                    "source_mask": (
                        Path(
                            row["source_mask"]
                        ).as_posix()
                    ),
                }
            )

        metadata = pd.DataFrame(
            metadata_rows
        )

        metadata_path = (
            metadata_root
            / f"{split}.csv"
        )

        metadata.to_csv(
            metadata_path,
            index=False,
            encoding="utf-8-sig",
        )

        print(
            f"[{split}] exported "
            f"{len(metadata):,} pairs"
        )

        print(
            f"  images:   {image_dir}"
        )

        print(
            f"  masks:    {mask_dir}"
        )

        print(
            f"  metadata: {metadata_path}"
        )

    print(
        "\n05_convert_dataset.py: PASS"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(
            f"\nERROR: {exc}",
            file=sys.stderr,
        )
        raise