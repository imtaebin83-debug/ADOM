from __future__ import annotations

## 0729 RUGD final-check를 CLI·환경변수 기반으로 재구성
import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError
##

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed" / "rugd_cost4_standard"
SPLITS = ("train", "val", "test")
CLASS_IDS = (0, 1, 2, 3, 255)
VALID_IDS = set(CLASS_IDS)
CLASS_NAMES = {
    0: "paved_low_cost",
    1: "natural_low_cost",
    2: "medium_cost",
    3: "high_cost_or_obstacle",
    255: "ignore",
}

METADATA_COLUMNS = {
    "sample_id", "file_name", "rgb_path", "source_color_mask_path",
    "converted_image_path", "converted_mask_path", "width", "height",
    "status", "details",
}
METADATA_PATH_COLUMNS = (
    "rgb_path", "source_color_mask_path",
    "converted_image_path", "converted_mask_path",
)
QC_COLUMNS = {"sample_id", "status", "details"}
STATISTICS_COLUMNS = {
    "split", "class_id", "class_name", "pixel_count",
    "image_count_containing_class", "percentage_of_all_pixels",
    "percentage_excluding_ignore",
}
OVERLAY_COLUMNS = {
    "split", "sample_id", "source_image", "source_mask",
    "overlay_path", "width", "height", "mask_ids",
}


def resolve_path(cli_value: Path | None, env_name: str, default: Path) -> Path:
    value = cli_value or os.getenv(env_name)
    return Path(value or default).expanduser().resolve()


def read_csv(path: Path, required: set[str], label: str) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise RuntimeError(f"{label} is missing required columns: {missing}")
        rows = list(reader)

    if not rows:
        raise RuntimeError(f"{label} contains no rows: {path}")
    return rows


def collect_png(directory: Path, label: str) -> dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"{label} directory does not exist: {directory}")

    result = {path.name: path for path in sorted(directory.glob("*.png"))}
    if not result:
        raise RuntimeError(f"No PNG files were found in {label}: {directory}")
    return result


def validate_pairs(
    images: dict[str, Path], masks: dict[str, Path], label: str,
) -> list[str]:
    image_names = set(images)
    mask_names = set(masks)
    image_only = sorted(image_names - mask_names)
    mask_only = sorted(mask_names - image_names)

    if image_only or mask_only:
        raise RuntimeError(
            f"RGB-mask pair mismatch: {label}\n"
            f"RGB images without masks: {len(image_only)} {image_only[:10]}\n"
            f"Masks without RGB images: {len(mask_only)} {mask_only[:10]}"
        )
    return sorted(image_names)


def read_split_file(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Split file does not exist: {path}")

    stems = [
        line.strip()
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    if not stems:
        raise RuntimeError(f"Split file contains no samples: {path}")
    if len(stems) != len(set(stems)):
        raise RuntimeError(f"Duplicate sample IDs inside split file: {path}")
    return stems


## 0729 실제 split 파일·이미지·mask·ID를 전수 검사
def validate_processed_data(
    all_root: Path,
    split_dataset_root: Path,
) -> tuple[
    dict[str, int], dict[str, set[int]], dict[str, set[str]], set[str]
]:
    all_image_root = all_root / "images"
    all_mask_root = all_root / "annotations"
    split_image_root = split_dataset_root / "images"
    split_mask_root = split_dataset_root / "annotations"
    split_manifest_root = split_dataset_root / "splits"

    all_images = collect_png(
        all_image_root / "all",
        "all processed RGB images",
    )
    all_masks = collect_png(
        all_mask_root / "all",
        "all processed masks",
    )
    all_names = set(validate_pairs(all_images, all_masks, "all samples"))

    split_counts: dict[str, int] = {}
    split_ids: dict[str, set[int]] = {}
    split_sample_ids: dict[str, set[str]] = {}
    assigned_names: set[str] = set()

    for split in SPLITS:
        images = collect_png(
            split_image_root / split,
            f"{split} RGB images",
        )
        masks = collect_png(
            split_mask_root / split,
            f"{split} masks",
        )
        paired_names = validate_pairs(images, masks, f"split={split}")
        txt_stems = read_split_file(
            split_manifest_root / f"{split}.txt"
        )
        actual_stems = [Path(name).stem for name in paired_names]

        if txt_stems != actual_stems:
            raise RuntimeError(
                f"Split TXT content or order does not match generated files: {split}"
            )

        overlap = sorted(assigned_names & set(paired_names))
        if overlap:
            raise RuntimeError(
                f"Cross-split duplicate files were found: {split}, {overlap[:10]}"
            )

        assigned_names.update(paired_names)
        split_sample_ids[split] = set(txt_stems)
        ids: set[int] = set()
        valid_pixels = 0

        for name in paired_names:
            try:
                with Image.open(images[name]) as image_file:
                    image_file.load()
                    image_size = image_file.size

                with Image.open(masks[name]) as mask_file:
                    mask_format = mask_file.format
                    mask_mode = mask_file.mode
                    mask_size = mask_file.size
                    mask = np.asarray(mask_file, dtype=np.uint8)
            except (UnidentifiedImageError, OSError) as error:
                raise RuntimeError(
                    f"Failed to read RGB or mask: split={split}, file={name}"
                ) from error

            if mask_format != "PNG":
                raise RuntimeError(f"Mask is not PNG: split={split}, file={name}")
            if mask_mode != "L":
                raise RuntimeError(
                    f"Mask mode is not L: split={split}, file={name}, mode={mask_mode}"
                )
            if mask.ndim != 2:
                raise RuntimeError(
                    f"Mask is not single-channel: split={split}, file={name}, "
                    f"shape={mask.shape}"
                )
            if image_size != mask_size:
                raise RuntimeError(
                    f"RGB-mask size mismatch: split={split}, file={name}, "
                    f"image={image_size}, mask={mask_size}"
                )

            current_ids = {int(value) for value in np.unique(mask).tolist()}
            invalid_ids = sorted(current_ids - VALID_IDS)
            if invalid_ids:
                raise RuntimeError(
                    f"Mask contains invalid IDs: split={split}, file={name}, "
                    f"invalid={invalid_ids}"
                )

            ids.update(current_ids)
            valid_pixels += int(np.count_nonzero(mask != 255))

        if valid_pixels == 0:
            raise RuntimeError(f"Split contains no valid Cost4 pixels: {split}")

        split_counts[split] = len(paired_names)
        split_ids[split] = ids

    if assigned_names != all_names:
        unassigned = sorted(all_names - assigned_names)
        extra = sorted(assigned_names - all_names)
        raise RuntimeError(
            "Split assignments do not cover all samples.\n"
            f"Unassigned: {unassigned[:10]}\nExtra: {extra[:10]}"
        )

    return split_counts, split_ids, split_sample_ids, all_names
##


def validate_metadata(
    path: Path, expected_names: set[str], expected_ids: set[str]
) -> int:
    rows = read_csv(path, METADATA_COLUMNS, "RUGD metadata CSV")
    sample_ids: set[str] = set()
    file_names: set[str] = set()

    for row in rows:
        sample_id = row["sample_id"].strip()
        file_name = row["file_name"].strip()
        status = row["status"].strip().lower()

        if not sample_id or not file_name:
            raise RuntimeError("Metadata contains an empty sample_id or file_name.")
        if sample_id in sample_ids or file_name in file_names:
            raise RuntimeError(
                f"Metadata contains duplicates: {sample_id}, {file_name}"
            )
        if status != "ok":
            raise RuntimeError(
                f"Metadata contains a non-ok status: {sample_id}, {status}"
            )

        for column in METADATA_PATH_COLUMNS:
            value = row[column].strip()
            if not value:
                raise RuntimeError(
                    f"Metadata contains an empty path: {sample_id}, {column}"
                )
            if Path(value).is_absolute():
                raise RuntimeError(
                    f"Metadata contains an absolute path: {sample_id}, {column}"
                )

        sample_ids.add(sample_id)
        file_names.add(file_name)

    if file_names != expected_names:
        raise RuntimeError("Metadata file names do not match processed samples.")
    if sample_ids != expected_ids:
        raise RuntimeError("Metadata sample IDs do not match split samples.")
    return len(rows)


def validate_qc(path: Path, expected_ids: set[str]) -> int:
    rows = read_csv(path, QC_COLUMNS, "RUGD QC report")
    actual_ids: set[str] = set()

    for row in rows:
        sample_id = row["sample_id"].strip()
        status = row["status"].strip().lower()

        if not sample_id:
            raise RuntimeError("QC report contains an empty sample_id.")
        if sample_id in actual_ids:
            raise RuntimeError(f"Duplicate QC sample_id: {sample_id}")
        if status != "ok":
            raise RuntimeError(
                f"QC report contains a failure status: {sample_id}, {status}"
            )
        actual_ids.add(sample_id)

    if actual_ids != expected_ids:
        raise RuntimeError("QC report sample IDs do not match split samples.")
    return len(rows)


def validate_statistics(json_path: Path, csv_path: Path, counts: dict[str, int]) -> None:
    if not json_path.is_file():
        raise FileNotFoundError(f"Statistics JSON does not exist: {json_path}")

    with json_path.open("r", encoding="utf-8") as file:
        statistics = json.load(file)

    if not isinstance(statistics, dict) or set(statistics) != set(SPLITS):
        raise RuntimeError("Statistics JSON split structure is invalid.")

    expected_class_keys = {str(value) for value in CLASS_IDS}
    for split in SPLITS:
        result = statistics[split]
        if not isinstance(result, dict):
            raise RuntimeError(f"Invalid statistics split structure: {split}")
        if int(result.get("image_count", -1)) != counts[split]:
            raise RuntimeError(f"Statistics image count mismatch: {split}")
        classes = result.get("classes")
        if not isinstance(classes, dict) or set(classes) != expected_class_keys:
            raise RuntimeError(f"Statistics class structure is invalid: {split}")

    rows = read_csv(csv_path, STATISTICS_COLUMNS, "RUGD statistics CSV")
    if len(rows) != len(SPLITS) * len(CLASS_IDS):
        raise RuntimeError(f"Statistics CSV row count mismatch: {len(rows)}")

    seen: set[tuple[str, int]] = set()
    for row in rows:
        split = row["split"].strip()
        class_id = int(row["class_id"])
        class_name = row["class_name"].strip()

        if split not in SPLITS or class_id not in VALID_IDS:
            raise RuntimeError(f"Invalid statistics CSV row: {split}, {class_id}")
        if class_name != CLASS_NAMES[class_id]:
            raise RuntimeError(f"Statistics class name mismatch: {split}, {class_id}")

        key = (split, class_id)
        if key in seen:
            raise RuntimeError(f"Duplicate statistics CSV row: {key}")
        seen.add(key)


def validate_overlays(
    manifest_path: Path,
    processed_root: Path,
    split_sample_ids: dict[str, set[str]],
) -> dict[str, int]:
    rows = read_csv(manifest_path, OVERLAY_COLUMNS, "RUGD overlay manifest")
    overlay_root = manifest_path.parent
    counts = {split: 0 for split in SPLITS}
    seen: set[tuple[str, str]] = set()
    manifest_outputs: set[Path] = set()

    for row in rows:
        split = row["split"].strip()
        sample_id = row["sample_id"].strip()

        if split not in SPLITS:
            raise RuntimeError(f"Overlay manifest contains an invalid split: {split}")
        if sample_id not in split_sample_ids[split]:
            raise RuntimeError(
                f"Overlay sample is not in its split: {split}, {sample_id}"
            )
        if (split, sample_id) in seen:
            raise RuntimeError(f"Duplicate overlay sample: {split}, {sample_id}")
        seen.add((split, sample_id))

        for column in ("source_image", "source_mask", "overlay_path"):
            value = row[column].strip()
            if not value or Path(value).is_absolute():
                raise RuntimeError(
                    f"Overlay manifest path is empty or absolute: {sample_id}, {column}"
                )

        image_path = processed_root / row["source_image"]
        mask_path = processed_root / row["source_mask"]
        overlay_path = overlay_root / row["overlay_path"]

        if not image_path.is_file():
            raise FileNotFoundError(f"Overlay source image does not exist: {image_path}")
        if not mask_path.is_file():
            raise FileNotFoundError(f"Overlay source mask does not exist: {mask_path}")
        if not overlay_path.is_file():
            raise FileNotFoundError(f"Overlay output does not exist: {overlay_path}")
        if image_path.stem != sample_id or mask_path.stem != sample_id:
            raise RuntimeError(f"Overlay sample_id mismatch: {split}, {sample_id}")

        manifest_outputs.add(overlay_path.resolve())
        counts[split] += 1

    if any(counts[split] == 0 for split in SPLITS):
        raise RuntimeError(f"Overlay manifest has an empty split: {counts}")

    actual_outputs = {
        path.resolve() for path in overlay_root.rglob("*_overlay.png")
    }
    if actual_outputs != manifest_outputs:
        raise RuntimeError("Overlay manifest does not match generated overlay files.")
    return counts


def write_final_check(path: Path, lines: list[str], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(
            "RUGD final-check output already exists. "
            f"Use --overwrite or a new --output: {path}"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")

    try:
        temporary.write_text(
            "\n".join(lines) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


## 0729 모든 검사가 성공한 뒤에만 final_check.txt에 PASS 기록
def run_final_check(
    processed_root: Path,
    split_root: Path,
    metadata_path: Path,
    qc_path: Path,
    statistics_json: Path,
    statistics_csv: Path,
    overlay_manifest: Path,
    output_path: Path,
    expected_total: int | None,
    overwrite: bool,
) -> None:
    counts, used_ids, split_ids, all_names = (
        validate_processed_data(
            processed_root,
            split_root,
        )
    )
    total = len(all_names)

    if expected_total is not None and total != expected_total:
        raise RuntimeError(
            f"Total sample count mismatch: expected={expected_total}, actual={total}"
        )

    all_sample_ids = set().union(*(split_ids[split] for split in SPLITS))
    metadata_rows = validate_metadata(
        metadata_path, all_names, all_sample_ids
    )
    qc_rows = validate_qc(qc_path, all_sample_ids)
    validate_statistics(statistics_json, statistics_csv, counts)
    overlay_counts = validate_overlays(
        overlay_manifest,
        split_root,
        split_ids,
    )

    all_used_ids = sorted(set().union(*(used_ids[split] for split in SPLITS)))
    lines = [
        "RUGD FINAL CHECK",
        "status=PASS",
        f"total_samples={total}",
        f"expected_total={expected_total if expected_total is not None else 'not_set'}",
        f"metadata_rows={metadata_rows}",
        f"qc_rows={qc_rows}",
        "split_overlap=0",
        "unassigned_samples=0",
        "used_ids=" + ",".join(str(value) for value in all_used_ids),
        "statistics_status=PASS",
        "overlay_status=PASS",
    ]

    for split in SPLITS:
        lines.extend([
            f"{split}_samples={counts[split]}",
            f"{split}_images={counts[split]}",
            f"{split}_masks={counts[split]}",
            f"{split}_used_ids=" + ",".join(
                str(value) for value in sorted(used_ids[split])
            ),
            f"{split}_overlays={overlay_counts[split]}",
        ])

    lines.append("final_status=PASS")
    write_final_check(output_path, lines, overwrite)

    print("\nRUGD final-check summary")

    for split in SPLITS:
        print(f"\n[{split}]")
        print(f"이미지 수: {counts[split]}")
        print(f"마스크 수: {counts[split]}")
        print(f"사용된 ID: {sorted(used_ids[split])}")
        print(f"오버레이 수: {overlay_counts[split]}")

    print(f"\n전체 파일 수: {total}")
    print(f"Metadata 행 수: {metadata_rows}")
    print(f"QC 행 수: {qc_rows}")
    print("통계 검사: PASS")
    print("오버레이 검사: PASS")
    print(f"final-check 저장 위치: {output_path}")
    print("\n최종 검사 통과")
    print("RUGD 전처리 데이터가 준비되었습니다.")
    print("[PASS] RUGD final check completed.")
##


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the final integrity check for processed RUGD Cost4 data."
    )
    parser.add_argument("--processed-root", type=Path, default=None)
    parser.add_argument("--split-root", type=Path, default=None)
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--qc-report", type=Path, default=None)
    parser.add_argument("--statistics-json", type=Path, default=None)
    parser.add_argument("--statistics-csv", type=Path, default=None)
    parser.add_argument("--overlay-manifest", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--expected-total", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    processed_root = resolve_path(
        args.processed_root,
        "RUGD_OUTPUT_ROOT",
        DEFAULT_PROCESSED_ROOT,
    )
    split_root = resolve_path(
        args.split_root,
        "RUGD_SPLIT_ROOT",
        processed_root,
    )
    metadata_path = resolve_path(
        args.metadata, "RUGD_METADATA_PATH", processed_root / "metadata.csv"
    )
    qc_path = resolve_path(
        args.qc_report,
        "RUGD_QC_REPORT_PATH",
        processed_root / "qc" / "qc_report.csv",
    )
    statistics_json = resolve_path(
        args.statistics_json,
        "RUGD_STATISTICS_JSON_PATH",
        processed_root / "metadata" / "class_statistics.json",
    )
    statistics_csv = resolve_path(
        args.statistics_csv,
        "RUGD_STATISTICS_CSV_PATH",
        processed_root / "metadata" / "class_statistics.csv",
    )
    overlay_manifest = resolve_path(
        args.overlay_manifest,
        "RUGD_OVERLAY_MANIFEST_PATH",
        processed_root / "qc" / "overlays" / "overlay_manifest.csv",
    )
    output_path = resolve_path(
        args.output,
        "RUGD_FINAL_CHECK_OUTPUT",
        PROJECT_ROOT / "results" / "final_check.txt",
    )

    if not processed_root.is_dir():
        raise FileNotFoundError(
            f"Processed root does not exist: {processed_root}"
        )
    if not split_root.is_dir():
        raise FileNotFoundError(
            f"Split root does not exist: {split_root}"
        )
    if args.expected_total is not None and args.expected_total <= 0:
        raise ValueError(f"expected-total must be positive: {args.expected_total}")

    run_final_check(
        processed_root,
        split_root,
        metadata_path,
        qc_path,
        statistics_json,
        statistics_csv,
        overlay_manifest,
        output_path,
        args.expected_total,
        args.overwrite,
    )


if __name__ == "__main__":
    main()
##
