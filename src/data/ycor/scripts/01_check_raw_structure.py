from __future__ import annotations

import argparse
import sys
from pathlib import Path

from common import (
    EXPECTED_COUNTS,
    OUTPUT_SPLITS,
    SOURCE_SPLITS,
    discover_dataset_root,
    find_pair,
    is_dataset_root,
    load_source_mask,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check the raw YCOR dataset structure."
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
        "--output",
        type=Path,
        default=None,
        help="Optional portable text report output path.",
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

    report_lines = [
        "[dataset root] <YCOR_DATASET_ROOT>",
    ]

    total = 0
    first_pair = None

    for output_split in OUTPUT_SPLITS:
        source_split = SOURCE_SPLITS[output_split]
        split_dir = dataset_root / source_split

        sample_dirs = sorted(
            path
            for path in split_dir.iterdir()
            if path.is_dir()
        )

        expected = EXPECTED_COUNTS[output_split]

        if len(sample_dirs) != expected:
            raise RuntimeError(
                f"{output_split} sample count mismatch: "
                f"expected={expected}, "
                f"actual={len(sample_dirs)}"
            )

        pair_count = 0

        for sample_dir in sample_dirs:
            image_path, mask_path = find_pair(
                sample_dir
            )

            pair_count += 1

            if first_pair is None:
                first_pair = (
                    image_path,
                    mask_path,
                )

        if pair_count != len(sample_dirs):
            raise RuntimeError(
                f"{output_split} pair count mismatch: "
                f"samples={len(sample_dirs)}, "
                f"pairs={pair_count}"
            )

        report_lines.append(
            f"[{output_split}] "
            f"source='{source_split}', "
            f"sample folders={len(sample_dirs):,}, "
            f"pairs={pair_count:,}"
        )

        total += len(sample_dirs)

    if first_pair is None:
        raise RuntimeError(
            "No image-mask pair was found."
        )

    image_path, mask_path = first_pair
    source_mask, encoding = load_source_mask(
        mask_path
    )

    report_lines.extend(
        [
            (
                "[sample image] "
                + image_path.relative_to(
                    dataset_root
                ).as_posix()
            ),
            (
                "[sample mask]  "
                + mask_path.relative_to(
                    dataset_root
                ).as_posix()
            ),
            f"[mask encoding] {encoding}",
            f"[mask shape]    {source_mask.shape}",
            f"[all samples]   {total:,}",
            "",
            "01_check_raw_structure.py: PASS",
        ]
    )

    report_text = (
        "\n".join(report_lines)
        + "\n"
    )

    print(
        report_text,
        end="",
    )

    if args.output is not None:
        output_path = (
            args.output
            .expanduser()
            .resolve()
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path.write_text(
            report_text,
            encoding="utf-8",
        )

        print(
            f"[saved] {output_path}"
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