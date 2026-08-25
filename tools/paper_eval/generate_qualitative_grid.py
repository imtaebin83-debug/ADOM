from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from _common import (
    IGNORE_INDEX,
    SEMANTIC20_PALETTE,
    read_manifest,
    safe_prediction_name,
    write_json,
)


def _number(value: str) -> float | None:
    if value in {"", "N/A", "None", "null"}:
        return None
    return float(value)


def _per_image(path: Path) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            parsed: dict[str, Any] = dict(row)
            for key, value in row.items():
                if key.endswith(("_iou", "_recall", "mIoU")):
                    parsed[key] = _number(value)
                elif key.endswith("_gt_pixels"):
                    parsed[key] = int(value)
            output[row["sample_id"]] = parsed
    return output


def _paired_rows(root: Path, dataset: str) -> list[dict[str, Any]]:
    metrics = root / "metrics"
    baseline = _per_image(metrics / f"{dataset}__b0_e0__per_image.csv")
    eadom = _per_image(metrics / f"{dataset}__eadom__per_image.csv")
    if list(baseline) != list(eadom):
        raise RuntimeError(f"{dataset} qualitative inputs are not identically ordered")
    output: list[dict[str, Any]] = []
    for sample_id, left in baseline.items():
        right = eadom[sample_id]
        row: dict[str, Any] = {
            "dataset": dataset,
            "sample_id": sample_id,
            "sequence": left["sequence"],
            "b0_prediction": left["prediction_path"],
            "eadom_prediction": right["prediction_path"],
        }
        for metric in ("common_supported_mIoU", "log_iou", "rubble_iou"):
            b0_value = left[metric]
            eadom_value = right[metric]
            row[f"b0_{metric}"] = b0_value
            row[f"eadom_{metric}"] = eadom_value
            row[f"delta_{metric}"] = (
                None
                if b0_value is None or eadom_value is None
                else eadom_value - b0_value
            )
        for class_name in ("log", "rubble"):
            row[f"{class_name}_gt_pixels"] = left[f"{class_name}_gt_pixels"]
        output.append(row)
    return output


def _unique_sequence_first(
    rows: list[dict[str, Any]], count: int
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        sequence = row["sequence"]
        if sequence not in seen:
            selected.append(row)
            seen.add(sequence)
        if len(selected) == count:
            return selected
    for row in rows:
        if row not in selected:
            selected.append(row)
        if len(selected) == count:
            break
    return selected


def _extreme_selection(
    rows: list[dict[str, Any]], metric: str, count: int
) -> list[dict[str, Any]]:
    defined = [row for row in rows if row[metric] is not None]
    if not defined:
        return []
    ordered = sorted(defined, key=lambda row: (row[metric], row["sample_id"]))
    candidates: list[dict[str, Any]] = [ordered[0]]
    if len(ordered) > 1:
        candidates.append(ordered[-1])
    if len(ordered) > 2:
        candidates.append(ordered[len(ordered) // 2])
    remaining = sorted(
        (row for row in ordered if row not in candidates),
        key=lambda row: (row["sequence"], row["sample_id"]),
    )
    return _unique_sequence_first(candidates + remaining, min(count, len(defined)))


def _negative_selection(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    negatives = [
        row
        for row in rows
        if row["log_gt_pixels"] == 0 and row["rubble_gt_pixels"] == 0
    ]
    # This category is deliberately selected without looking at model performance.
    negatives.sort(key=lambda row: (row["sequence"], row["sample_id"]))
    return _unique_sequence_first(negatives, min(count, len(negatives)))


def _colorize(mask: np.ndarray) -> Image.Image:
    if mask.ndim != 2:
        raise ValueError(f"Expected 2-D mask, got {mask.shape}")
    output = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for index, color in enumerate(SEMANTIC20_PALETTE):
        output[mask == index] = color
    output[mask == IGNORE_INDEX] = (0, 0, 0)
    invalid = ~(((mask >= 0) & (mask < len(SEMANTIC20_PALETTE))) | (mask == IGNORE_INDEX))
    if np.any(invalid):
        raise ValueError(f"Invalid mask IDs: {np.unique(mask[invalid]).tolist()}")
    return Image.fromarray(output, mode="RGB")


def _open_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        value = np.asarray(image)
    if value.ndim != 2:
        raise ValueError(f"Expected train-ID mask: {path}")
    return value


def _fmt(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.2f}"


def _render(
    row: dict[str, Any],
    image_path: Path,
    annotation_path: Path,
    output_path: Path,
    panel_width: int,
) -> None:
    with Image.open(image_path) as source:
        rgb = source.convert("RGB")
    gt = _colorize(_open_mask(annotation_path))
    b0 = _colorize(_open_mask(Path(row["b0_prediction"])))
    eadom = _colorize(_open_mask(Path(row["eadom_prediction"])))
    if not (rgb.size == gt.size == b0.size == eadom.size):
        raise ValueError(f"Qualitative image/mask shape mismatch: {row['sample_id']}")
    panel_height = max(1, round(rgb.height * panel_width / rgb.width))
    panels = [
        image.resize((panel_width, panel_height), Image.Resampling.BILINEAR)
        for image in (rgb, gt, b0, eadom)
    ]
    header = 28
    footer = 82
    canvas = Image.new("RGB", (panel_width * 4, header + panel_height + footer), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for index, (title, panel) in enumerate(
        zip(("RGB", "Ground Truth", "B0-E0", "E-ADOM"), panels)
    ):
        x = index * panel_width
        draw.text((x + 6, 8), title, fill="black", font=font)
        canvas.paste(panel, (x, header))
    y = header + panel_height + 5
    lines = (
        f"sample={row['sample_id']}  sequence={row['sequence']}",
        f"log GT={row['log_gt_pixels']}  IoU B0={_fmt(row['b0_log_iou'])}  E={_fmt(row['eadom_log_iou'])}  delta={_fmt(row['delta_log_iou'])}",
        f"rubble GT={row['rubble_gt_pixels']}  IoU B0={_fmt(row['b0_rubble_iou'])}  E={_fmt(row['eadom_rubble_iou'])}  delta={_fmt(row['delta_rubble_iou'])}",
        f"common image mIoU B0={_fmt(row['b0_common_supported_mIoU'])}  E={_fmt(row['eadom_common_supported_mIoU'])}  delta={_fmt(row['delta_common_supported_mIoU'])}",
    )
    for line in lines:
        draw.text((6, y), line, fill="black", font=font)
        y += 17
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def generate(args: argparse.Namespace) -> dict[str, Any]:
    root = args.output_dir.resolve()
    qualitative = root / "qualitative"
    if qualitative.exists() and any(qualitative.rglob("*.png")):
        raise FileExistsError(f"Refusing to overwrite qualitative outputs: {qualitative}")
    manifest_paths = {
        "rellis": root / "manifests" / "rellis_test_manifest.csv",
        "korean": root / "manifests" / "korean_test_manifest.csv",
    }
    manifests = {
        dataset: {row.sample_id: row for row in read_manifest(path)}
        for dataset, path in manifest_paths.items()
    }
    all_rows = {
        dataset: _paired_rows(root, dataset) for dataset in ("rellis", "korean")
    }
    selections: list[dict[str, Any]] = []
    for dataset, rows in all_rows.items():
        categories = {
            "log": _extreme_selection(
                [row for row in rows if row["log_gt_pixels"] > 0],
                "delta_log_iou",
                args.per_category,
            ),
            "rubble": _extreme_selection(
                [row for row in rows if row["rubble_gt_pixels"] > 0],
                "delta_rubble_iou",
                args.per_category,
            ),
            "regression_cases": _unique_sequence_first(
                sorted(
                    (
                        row
                        for row in rows
                        if row["delta_common_supported_mIoU"] is not None
                    ),
                    key=lambda row: (
                        row["delta_common_supported_mIoU"],
                        row["sample_id"],
                    ),
                ),
                min(args.per_category, len(rows)),
            ),
            "negative_cases": _negative_selection(rows, args.per_category),
        }
        for category, selected in categories.items():
            for rank, row in enumerate(selected, start=1):
                manifest = manifests[dataset][row["sample_id"]]
                filename = f"{dataset}__{rank:02d}__{safe_prediction_name(row['sample_id'])}"
                output_path = qualitative / category / filename
                _render(
                    row,
                    manifest.image_path,
                    manifest.annotation_path,
                    output_path,
                    args.panel_width,
                )
                selections.append(
                    {
                        **row,
                        "category": category,
                        "rank": rank,
                        "output_path": str(output_path.resolve()),
                        "selection_rule": (
                            "performance-blind first-per-sequence"
                            if category == "negative_cases"
                            else "deterministic extremes/median with unique-sequence priority"
                        ),
                    }
                )
    payload = {
        "schema_version": "adom-paper-eval-qualitative-v1",
        "per_category_per_dataset": args.per_category,
        "columns": ["RGB", "Ground Truth", "B0-E0", "E-ADOM"],
        "selection_policy": {
            "positive_classes": "best, worst and median paired IoU deltas; unique sequences first",
            "regression": "lowest paired common-supported image mIoU deltas",
            "negative": "log/rubble absent; sorted by sequence/sample without model scores",
        },
        "selections": selections,
    }
    write_json(qualitative / "selection_manifest.json", payload)
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate deterministic RGB/GT/B0-E0/E-ADOM qualitative grids"
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--per-category", type=int, default=4)
    parser.add_argument("--panel-width", type=int, default=480)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    result = generate(parse_args(argv))
    print(json.dumps({"selection_count": len(result["selections"])}, indent=2))


if __name__ == "__main__":
    main()
