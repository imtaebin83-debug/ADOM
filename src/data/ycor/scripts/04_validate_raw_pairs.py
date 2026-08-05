from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

import common
from common import (
    EXPECTED_COUNTS,
    OUTPUT_SPLITS,
    discover_dataset_root,
    is_dataset_root,
    load_source_mask,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate every raw YCOR image-mask pair."
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
        "--report-dir",
        type=Path,
        default=common.REPORT_DIR,
        help="Directory for raw validation reports.",
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


def validate_row(
    row: pd.Series,
    dataset_root: Path,
) -> dict:
    image_path = resolve_manifest_source(
        dataset_root,
        row["source_image"],
        "source_image",
    )

    mask_path = resolve_manifest_source(
        dataset_root,
        row["source_mask"],
        "source_mask",
    )

    result = {
        "split": row["split"],
        "sample_id": row["sample_id"],
        "source_sample_name": (
            row["source_sample_name"]
        ),
        "image_exists": image_path.exists(),
        "mask_exists": mask_path.exists(),
        "image_width": "",
        "image_height": "",
        "mask_width": "",
        "mask_height": "",
        "mask_encoding": "",
        "status": "ok",
        "error": "",
    }

    try:
        if not image_path.exists():
            raise FileNotFoundError(
                "Missing image: "
                + Path(
                    row["source_image"]
                ).as_posix()
            )

        if not mask_path.exists():
            raise FileNotFoundError(
                "Missing mask: "
                + Path(
                    row["source_mask"]
                ).as_posix()
            )

        with Image.open(image_path) as image:
            image.load()
            image_width, image_height = (
                image.size
            )
            image.convert("RGB")

        source_mask, encoding = (
            load_source_mask(
                mask_path
            )
        )

        mask_height, mask_width = (
            source_mask.shape[:2]
        )

        result.update(
            {
                "image_width": image_width,
                "image_height": image_height,
                "mask_width": mask_width,
                "mask_height": mask_height,
                "mask_encoding": encoding,
            }
        )

        if (
            image_width,
            image_height,
        ) != (
            mask_width,
            mask_height,
        ):
            raise ValueError(
                "Size mismatch: "
                f"image={(image_width, image_height)}, "
                f"mask={(mask_width, mask_height)}"
            )

    except Exception as exc:
        result["status"] = "error"

        error_text = str(exc).replace(
            str(dataset_root),
            "<YCOR_DATASET_ROOT>",
        )

        result["error"] = error_text

    return result


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

    report_dir = (
        args.report_dir
        .expanduser()
        .resolve()
    )

    report_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summaries = []

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

        rows = []

        for _, row in tqdm(
            manifest.iterrows(),
            total=len(manifest),
            desc=f"validate {split}",
        ):
            rows.append(
                validate_row(
                    row,
                    dataset_root,
                )
            )

        report = pd.DataFrame(rows)

        report_path = (
            report_dir
            / f"raw_validation_{split}.csv"
        )

        report.to_csv(
            report_path,
            index=False,
            encoding="utf-8-sig",
        )

        error_count = int(
            (
                report["status"]
                != "ok"
            ).sum()
        )

        summaries.append(
            {
                "split": split,
                "samples": len(report),
                "errors": error_count,
            }
        )

        print(
            f"[{split}] "
            f"samples={len(report):,}, "
            f"errors={error_count:,}"
        )

        print(
            f"  report: {report_path}"
        )

    summary = pd.DataFrame(
        summaries
    )

    summary_path = (
        report_dir
        / "raw_validation_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    total_errors = int(
        summary["errors"].sum()
    )

    if total_errors:
        raise RuntimeError(
            "Raw validation found "
            f"{total_errors} invalid pairs."
        )

    print(f"\n[saved] {summary_path}")
    print(
        "04_validate_raw_pairs.py: PASS"
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