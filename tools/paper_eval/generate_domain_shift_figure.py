from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from _common import (
    CLASS_INDEX,
    IGNORE_INDEX,
    SEMANTIC20_CLASSES,
    SEMANTIC20_PALETTE,
    read_manifest,
    write_json,
)


DISPLAY_CLASSES = (
    "grass",
    "tree",
    "log",
    "person",
    "bush",
    "barrier",
    "puddle",
    "mud",
    "rubble",
)


def _font(size: int) -> ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    if path.is_file():
        try:
            return ImageFont.truetype(str(path), size=size)
        except (ImportError, OSError):
            pass
    return ImageFont.load_default()


def _colorize(path: Path) -> Image.Image:
    with Image.open(path) as source:
        mask = np.asarray(source)
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2-D Semantic20 mask: {path}")
    output = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for index, color in enumerate(SEMANTIC20_PALETTE):
        output[mask == index] = color
    output[mask == IGNORE_INDEX] = (0, 0, 0)
    return Image.fromarray(output, mode="RGB")


def _image(path: Path) -> Image.Image:
    with Image.open(path) as source:
        return source.convert("RGB")


def _fit(
    image: Image.Image, size: tuple[int, int], *, is_mask: bool = False
) -> Image.Image:
    method = Image.Resampling.NEAREST if is_mask else Image.Resampling.BILINEAR
    return ImageOps.pad(image, size, method=method, color=(30, 30, 30))


def _load_confusions(
    manifest_path: Path, confusion_path: Path
) -> dict[str, dict[str, Any]]:
    records = read_manifest(manifest_path)
    payload = np.load(confusion_path, allow_pickle=False)
    ids = payload["sample_ids"].astype(str).tolist()
    if ids != [record.sample_id for record in records]:
        raise RuntimeError("Manifest and per-image confusion order differ")
    return {
        record.sample_id: {
            "record": record,
            "confusion": payload["confusions"][index].astype(np.int64, copy=False),
        }
        for index, record in enumerate(records)
    }


def _iou(confusion: np.ndarray, class_name: str) -> float | None:
    index = CLASS_INDEX[class_name]
    gt = int(confusion[index].sum())
    if gt == 0:
        return None
    tp = int(confusion[index, index])
    fp = int(confusion[:, index].sum()) - tp
    fn = gt - tp
    return 100.0 * tp / (tp + fp + fn)


def _gt_pixels(entry: dict[str, Any], class_name: str) -> int:
    return int(entry["confusion"][CLASS_INDEX[class_name]].sum())


def _median_positive(
    entries: dict[str, dict[str, Any]], class_name: str
) -> dict[str, Any]:
    positive = [
        entry for entry in entries.values() if _gt_pixels(entry, class_name) > 0
    ]
    positive.sort(
        key=lambda entry: (
            _iou(entry["confusion"], class_name),
            entry["record"].sample_id,
        )
    )
    if not positive:
        raise RuntimeError(f"No positive images for {class_name}")
    return positive[len(positive) // 2]


def _largest_gt(
    entries: dict[str, dict[str, Any]], class_name: str
) -> dict[str, Any]:
    positive = [
        entry for entry in entries.values() if _gt_pixels(entry, class_name) > 0
    ]
    if not positive:
        raise RuntimeError(f"No positive images for {class_name}")
    return max(
        positive,
        key=lambda entry: (
            _gt_pixels(entry, class_name),
            entry["record"].sample_id,
        ),
    )


def _top_predictions(
    confusion: np.ndarray, class_name: str, count: int = 3
) -> list[tuple[str, float]]:
    row = confusion[CLASS_INDEX[class_name]]
    total = int(row.sum())
    if total == 0:
        return []
    indices = np.argsort(row)[::-1]
    return [
        (SEMANTIC20_CLASSES[index], 100.0 * int(row[index]) / total)
        for index in indices[:count]
        if row[index] > 0
    ]


def _prediction_path(entry: dict[str, Any], prediction_root: Path) -> Path:
    from _common import safe_prediction_name

    return prediction_root / safe_prediction_name(entry["record"].sample_id)


def _legend(draw: ImageDraw.ImageDraw, y: int, width: int) -> None:
    font = _font(15)
    x = 12
    for name in DISPLAY_CLASSES:
        color = SEMANTIC20_PALETTE[CLASS_INDEX[name]]
        draw.rectangle((x, y, x + 14, y + 14), fill=color, outline="white")
        draw.text((x + 19, y - 2), name, fill="white", font=font)
        x += 19 + max(50, int(draw.textlength(name, font=font)))
        if x > width - 90:
            y += 21
            x = 12


def _domain_figure(
    output: Path,
    pairs: list[tuple[str, dict[str, Any], dict[str, Any], Path, Path]],
) -> None:
    panel_size = (320, 210)
    header = 34
    footer = 76
    legend = 52
    width = panel_size[0] * 6
    height = (header + panel_size[1] + footer) * len(pairs) + legend
    canvas = Image.new("RGB", (width, height), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    title_font = _font(17)
    text_font = _font(14)
    headers = (
        "RELLIS RGB",
        "RELLIS GT",
        "RELLIS B0-E0",
        "Korean RGB",
        "Korean GT",
        "Korean B0-E0",
    )
    for row_index, (class_name, source, target, source_pred_root, target_pred_root) in enumerate(pairs):
        y = row_index * (header + panel_size[1] + footer)
        source_record = source["record"]
        target_record = target["record"]
        panels = (
            (_image(source_record.image_path), False),
            (_colorize(source_record.annotation_path), True),
            (_colorize(_prediction_path(source, source_pred_root)), True),
            (_image(target_record.image_path), False),
            (_colorize(target_record.annotation_path), True),
            (_colorize(_prediction_path(target, target_pred_root)), True),
        )
        for index, (heading, panel_info) in enumerate(zip(headers, panels)):
            panel, is_mask = panel_info
            x = index * panel_size[0]
            draw.text((x + 8, y + 7), heading, fill="white", font=title_font)
            canvas.paste(_fit(panel, panel_size, is_mask=is_mask), (x, y + header))
        source_iou = _iou(source["confusion"], class_name)
        target_iou = _iou(target["confusion"], class_name)
        source_top = ", ".join(
            f"{name} {value:.1f}%"
            for name, value in _top_predictions(source["confusion"], class_name)
        )
        target_top = ", ".join(
            f"{name} {value:.1f}%"
            for name, value in _top_predictions(target["confusion"], class_name)
        )
        text_y = y + header + panel_size[1] + 7
        draw.text(
            (8, text_y),
            f"{class_name.upper()} | RELLIS median-positive | GT={_gt_pixels(source, class_name):,} px | IoU={source_iou:.2f}% | {source_top}",
            fill="white",
            font=text_font,
        )
        draw.text(
            (8, text_y + 23),
            f"Korean largest-GT | GT={_gt_pixels(target, class_name):,} px | IoU={target_iou:.2f}% | {target_top}",
            fill="white",
            font=text_font,
        )
        draw.text(
            (8, text_y + 46),
            f"RELLIS: {source_record.sequence} | Korean: {target_record.sequence}",
            fill=(185, 185, 185),
            font=text_font,
        )
    _legend(draw, height - legend + 10, width)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def _person_figure(
    output: Path, entry: dict[str, Any], prediction_root: Path
) -> None:
    panel_size = (500, 300)
    header = 38
    footer = 110
    width = panel_size[0] * 3
    height = header + panel_size[1] + footer
    canvas = Image.new("RGB", (width, height), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    title_font = _font(18)
    text_font = _font(16)
    record = entry["record"]
    panels = (
        ("Korean RGB", _image(record.image_path), False),
        ("Partial GT: person", _colorize(record.annotation_path), True),
        ("B0-E0 prediction", _colorize(_prediction_path(entry, prediction_root)), True),
    )
    for index, (title, panel, is_mask) in enumerate(panels):
        x = index * panel_size[0]
        draw.text((x + 8, 8), title, fill="white", font=title_font)
        canvas.paste(_fit(panel, panel_size, is_mask=is_mask), (x, header))
    top = ", ".join(
        f"{name} {value:.1f}%"
        for name, value in _top_predictions(entry["confusion"], "person", count=5)
    )
    y = header + panel_size[1] + 8
    draw.text(
        (8, y),
        f"Median person-positive frame | GT={_gt_pixels(entry, 'person'):,} px | IoU={_iou(entry['confusion'], 'person'):.2f}%",
        fill="white",
        font=text_font,
    )
    draw.text((8, y + 26), f"Predictions on person GT: {top}", fill="white", font=text_font)
    draw.text((8, y + 52), f"sequence={record.sequence}", fill=(185, 185, 185), font=text_font)
    _legend(draw, y + 78, width)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def _conflict_figure(
    output: Path,
    train_entry: dict[str, Any],
    val_entry: dict[str, Any],
    prediction_root: Path,
    conflict_row: dict[str, str],
) -> None:
    panel_size = (400, 260)
    header = 38
    footer = 112
    width = panel_size[0] * 4
    height = header + panel_size[1] + footer
    canvas = Image.new("RGB", (width, height), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    title_font = _font(18)
    text_font = _font(15)
    record = train_entry["record"]
    panels = (
        ("Same RGB", _image(record.image_path), False),
        ("Train GT: rubble", _colorize(record.annotation_path), True),
        ("Val GT: log", _colorize(val_entry["record"].annotation_path), True),
        ("B0-E0 prediction", _colorize(_prediction_path(train_entry, prediction_root)), True),
    )
    for index, (title, panel, is_mask) in enumerate(panels):
        x = index * panel_size[0]
        draw.text((x + 8, 8), title, fill="white", font=title_font)
        canvas.paste(_fit(panel, panel_size, is_mask=is_mask), (x, header))
    y = header + panel_size[1] + 8
    draw.text(
        (8, y),
        f"Identical RGB SHA-256; conflicting non-ignore pixels={int(conflict_row['conflicting_pixels']):,}",
        fill="white",
        font=text_font,
    )
    draw.text(
        (8, y + 24),
        f"label pair={conflict_row['label_pairs']} | train={record.sample_id}",
        fill="white",
        font=text_font,
    )
    draw.text(
        (8, y + 48),
        f"val={val_entry['record'].sample_id}",
        fill=(185, 185, 185),
        font=text_font,
    )
    _legend(draw, y + 77, width)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def generate(args: argparse.Namespace) -> dict[str, Any]:
    root = args.paper_eval_root.resolve()
    supplemental = args.supplemental_dir.resolve()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite domain-shift figures: {output}")

    rellis = _load_confusions(
        root / "manifests/rellis_test_manifest.csv",
        root / "metrics/rellis__b0_e0__per_image_confusions.npz",
    )
    korean_test = _load_confusions(
        root / "manifests/korean_test_manifest.csv",
        root / "metrics/korean__b0_e0__per_image_confusions.npz",
    )
    korean_train = _load_confusions(
        root / "manifests/korean_train_manifest.csv",
        supplemental / "metrics/korean_train__b0_e0__per_image_confusions.npz",
    )
    korean_val = _load_confusions(
        root / "manifests/korean_val_manifest.csv",
        supplemental / "metrics/korean_val__b0_e0__per_image_confusions.npz",
    )
    rellis_pred = root / "predictions/rellis__b0_e0"
    korean_test_pred = root / "predictions/korean__b0_e0"
    korean_train_pred = supplemental / "predictions/korean_train__b0_e0"

    source_log = _median_positive(rellis, "log")
    target_log = _largest_gt(korean_test, "log")
    source_rubble = _median_positive(rellis, "rubble")
    target_rubble = _largest_gt(korean_test, "rubble")
    person = _median_positive(korean_train, "person")

    with (supplemental / "duplicate_label_conflicts.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as stream:
        conflicts = list(csv.DictReader(stream))
    conflict = max(conflicts, key=lambda row: int(row["conflicting_pixels"]))
    train_conflict = korean_train[conflict["first_sample_id"]]
    val_conflict = korean_val[conflict["second_sample_id"]]

    output.mkdir(parents=True, exist_ok=True)
    domain_path = output / "domain_shift_log_rubble.png"
    person_path = output / "person_partial_success.png"
    conflict_path = output / "duplicate_label_conflict.png"
    _domain_figure(
        domain_path,
        [
            ("log", source_log, target_log, rellis_pred, korean_test_pred),
            ("rubble", source_rubble, target_rubble, rellis_pred, korean_test_pred),
        ],
    )
    _person_figure(person_path, person, korean_train_pred)
    _conflict_figure(
        conflict_path,
        train_conflict,
        val_conflict,
        korean_train_pred,
        conflict,
    )
    selections = {
        "schema_version": "adom-domain-shift-figure-v1",
        "selection_rules": {
            "rellis_log_rubble": "median per-image class IoU among GT-positive images",
            "korean_log_rubble": "largest class GT pixel count in held-out test",
            "korean_person": "median per-image person IoU among GT-positive train images",
            "annotation_conflict": "largest conflicting-pixel duplicate group",
        },
        "selections": {
            "rellis_log": source_log["record"].sample_id,
            "korean_log": target_log["record"].sample_id,
            "rellis_rubble": source_rubble["record"].sample_id,
            "korean_rubble": target_rubble["record"].sample_id,
            "korean_person": person["record"].sample_id,
            "annotation_conflict_train": train_conflict["record"].sample_id,
            "annotation_conflict_val": val_conflict["record"].sample_id,
        },
        "outputs": {
            "domain_shift": str(domain_path),
            "person": str(person_path),
            "annotation_conflict": str(conflict_path),
        },
    }
    write_json(output / "selection_manifest.json", selections)
    return selections


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate deterministic figures for B0-E0 domain-shift analysis"
    )
    parser.add_argument("--paper-eval-root", required=True, type=Path)
    parser.add_argument("--supplemental-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    result = generate(parse_args(argv))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
