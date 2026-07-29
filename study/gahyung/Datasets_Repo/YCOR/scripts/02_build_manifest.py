from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from common import (
    EXPECTED_COUNTS,
    MANIFEST_DIR,
    OUTPUT_SPLITS,
    SOURCE_SPLITS,
    discover_dataset_root,
    find_pair,
    is_dataset_root,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build YCOR manifests using source paths "
            "relative to the dataset root."
        )
    )

    parser.add_argument(
        "--input-root",
        type=Path,
        default=None,
        help=(
            "YCOR dataset root containing train/ and valid/. "
            "When omitted, the original repository discovery logic is used."
        ),
    )

    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=MANIFEST_DIR,
        help="Manifest CSV output directory.",
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
            "The selected YCOR input root must contain "
            f"non-empty train/ and valid/ folders: {dataset_root}"
        )

    return dataset_root


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

    all_frames = []

    for output_split in OUTPUT_SPLITS:
        source_split = SOURCE_SPLITS[
            output_split
        ]

        split_dir = (
            dataset_root
            / source_split
        )

        sample_dirs = sorted(
            path
            for path in split_dir.iterdir()
            if path.is_dir()
        )

        expected = EXPECTED_COUNTS[
            output_split
        ]

        if len(sample_dirs) != expected:
            raise RuntimeError(
                f"{output_split} sample count mismatch: "
                f"expected={expected}, "
                f"actual={len(sample_dirs)}"
            )

        rows = []

        for index, sample_dir in enumerate(
            sample_dirs
        ):
            image_path, mask_path = find_pair(
                sample_dir
            )

            sample_id = (
                f"ycor_{output_split}_{index:06d}"
            )

            rows.append(
                {
                    "dataset": "YCOR",
                    "split": output_split,
                    "source_split": source_split,
                    "sample_index": index,
                    "sample_id": sample_id,
                    "source_sample_name": (
                        sample_dir.name
                    ),
                    "source_sample_dir": (
                        sample_dir.relative_to(
                            dataset_root
                        ).as_posix()
                    ),
                    "source_image": (
                        image_path.relative_to(
                            dataset_root
                        ).as_posix()
                    ),
                    "source_mask": (
                        mask_path.relative_to(
                            dataset_root
                        ).as_posix()
                    ),
                    "output_image_filename": (
                        sample_id + ".jpg"
                    ),
                    "output_mask_filename": (
                        sample_id + ".png"
                    ),
                }
            )

        manifest = pd.DataFrame(rows)

        if manifest[
            "sample_id"
        ].duplicated().any():
            raise ValueError(
                f"Duplicate generated IDs in {output_split}"
            )

        for column in (
            "source_sample_dir",
            "source_image",
            "source_mask",
        ):
            absolute_rows = manifest[
                column
            ].map(
                lambda value: Path(
                    str(value)
                ).is_absolute()
            )

            if absolute_rows.any():
                raise ValueError(
                    "Absolute source paths were generated "
                    f"in column: {column}"
                )

        all_frames.append(
            manifest
        )

    all_manifest = pd.concat(
        all_frames,
        ignore_index=True,
    )

    if all_manifest[
        "sample_id"
    ].duplicated().any():
        raise ValueError(
            "Duplicate sample IDs across splits."
        )

    expected_total = sum(
        EXPECTED_COUNTS.values()
    )

    if len(all_manifest) != expected_total:
        raise RuntimeError(
            "Combined manifest count mismatch: "
            f"expected={expected_total}, "
            f"actual={len(all_manifest)}"
        )

    manifest_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for manifest in all_frames:
        output_split = manifest[
            "split"
        ].iloc[0]

        output_path = (
            manifest_dir
            / f"{output_split}.csv"
        )

        manifest.to_csv(
            output_path,
            index=False,
            encoding="utf-8-sig",
        )

        print(
            f"[{output_split}] "
            f"{len(manifest):,} pairs"
        )

        print(
            f"  saved: {output_path}"
        )

    all_path = (
        manifest_dir
        / "all_samples.csv"
    )

    all_manifest.to_csv(
        all_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"\n[all] {len(all_manifest):,} pairs"
    )

    print(
        f"[saved] {all_path}"
    )

    print(
        "02_build_manifest.py: PASS"
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