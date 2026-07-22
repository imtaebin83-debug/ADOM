from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

from common import (
    MANIFEST_DIR,
    OUTPUT_SPLITS,
    REPORT_DIR,
    load_source_mask,
)


def validate_row(row: pd.Series) -> dict:
    image_path = Path(row["source_image"])
    mask_path = Path(row["source_mask"])

    result = {
        "split": row["split"],
        "sample_id": row["sample_id"],
        "source_sample_name": row["source_sample_name"],
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
            raise FileNotFoundError(f"Missing image: {image_path}")
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing mask: {mask_path}")

        with Image.open(image_path) as image:
            image.load()
            image_width, image_height = image.size
            image.convert("RGB")

        source_mask, encoding = load_source_mask(mask_path)
        mask_height, mask_width = source_mask.shape[:2]

        result.update(
            {
                "image_width": image_width,
                "image_height": image_height,
                "mask_width": mask_width,
                "mask_height": mask_height,
                "mask_encoding": encoding,
            }
        )

        if (image_width, image_height) != (mask_width, mask_height):
            raise ValueError(
                f"Size mismatch: image={(image_width, image_height)}, "
                f"mask={(mask_width, mask_height)}"
            )

    except Exception as exc:
        result["status"] = "error"
        result["error"] = str(exc)

    return result


def main() -> None:
    summaries = []

    for split in OUTPUT_SPLITS:
        manifest_path = MANIFEST_DIR / f"{split}.csv"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest not found: {manifest_path}\n"
                "Run 02_build_manifest.py first."
            )

        manifest = pd.read_csv(manifest_path, dtype=str)
        rows = []

        for _, row in tqdm(
            manifest.iterrows(),
            total=len(manifest),
            desc=f"validate {split}",
        ):
            rows.append(validate_row(row))

        report = pd.DataFrame(rows)
        report_path = REPORT_DIR / f"raw_validation_{split}.csv"
        report.to_csv(report_path, index=False, encoding="utf-8-sig")

        error_count = int((report["status"] != "ok").sum())
        summaries.append(
            {
                "split": split,
                "samples": len(report),
                "errors": error_count,
            }
        )

        print(f"[{split}] samples={len(report):,}, errors={error_count:,}")
        print(f"  report: {report_path}")

    summary = pd.DataFrame(summaries)
    summary_path = REPORT_DIR / "raw_validation_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    total_errors = int(summary["errors"].sum())
    if total_errors:
        raise RuntimeError(
            f"Raw validation found {total_errors} invalid pairs."
        )

    print(f"\n[saved] {summary_path}")
    print("04_validate_raw_pairs.py: PASS")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise
