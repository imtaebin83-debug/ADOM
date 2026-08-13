from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image

from adom.evaluation_semantic20 import RARE_RISK4, SEMANTIC20_CLASSES


IGNORE_INDEX = 255
RARE_RISK_IDS = tuple(SEMANTIC20_CLASSES.index(name) for name in RARE_RISK4)
@dataclass(frozen=True)
class Sample:
    sample_id: str
    source: str
    mask_path: Path


@dataclass(frozen=True)
class Geometry:
    resized_wh: tuple[int, int]
    crop_xywh: tuple[int, int, int, int]
    pad_ltrb: tuple[int, int, int, int]
    output_wh: tuple[int, int]


CANDIDATES: dict[str, dict[str, Any]] = {
    "i0_crop512": {
        "policy": "RandomResize(1024x512, ratio=0.5..2.0) + RandomCrop(512x512)",
        "output_wh": (512, 512),
        "random_crop": True,
    },
    "i1_nocrop_640x384": {
        "policy": "keep-ratio resize + right/bottom ignore pad, no crop",
        "output_wh": (640, 384),
        "random_crop": False,
    },
    "i2_nocrop_640x480": {
        "policy": "keep-ratio resize + right/bottom ignore pad, no crop",
        "output_wh": (640, 480),
        "random_crop": False,
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_path(root: Path, stored: str) -> Path:
    relative = Path(stored)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Non-portable manifest path: {stored}")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"Manifest path escapes dataset root: {stored}") from error
    return resolved


def load_samples(
    root: Path,
    split: Path,
    *,
    manifest: Path | None = None,
    default_source: str = "rellis3d",
) -> list[Sample]:
    root = root.resolve()
    split_path = split if split.is_absolute() else root / split
    keys = [
        line.strip()
        for line in split_path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    if not keys or len(keys) != len(set(keys)):
        raise ValueError(f"Split is empty or contains duplicates: {split_path}")

    manifest_rows: dict[str, dict[str, str]] = {}
    manifest_path: Path | None = None
    if manifest is not None:
        manifest_path = manifest if manifest.is_absolute() else root / manifest
        with manifest_path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            required = {"sample_key", "mask_path"}
            missing = required - set(reader.fieldnames or ())
            if missing:
                raise ValueError(
                    f"Manifest {manifest_path} is missing fields: {sorted(missing)}"
                )
            for row in reader:
                key = row["sample_key"]
                if key in manifest_rows:
                    raise ValueError(f"Duplicate manifest sample: {key}")
                manifest_rows[key] = row

    samples: list[Sample] = []
    for key in keys:
        if manifest_path is None:
            mask_path = root / "masks" / f"{key}.png"
            source = key.split("/", 1)[0] if key.startswith("rellis3d/") else default_source
        else:
            if key not in manifest_rows:
                raise ValueError(f"Split sample is absent from manifest: {key}")
            row = manifest_rows[key]
            mask_path = _safe_path(root, row["mask_path"])
            source = row.get("source") or (
                key.split("/", 1)[0] if "/" in key else default_source
            )
        if not mask_path.is_file():
            raise FileNotFoundError(mask_path)
        samples.append(Sample(key, source, mask_path))
    return samples


def _keep_ratio_size(source_wh: tuple[int, int], target_wh: tuple[int, int]) -> tuple[int, int]:
    width, height = source_wh
    target_width, target_height = target_wh
    scale = min(target_width / width, target_height / height)
    return max(1, round(width * scale)), max(1, round(height * scale))


def _random_resize_size(source_wh: tuple[int, int], ratio: float) -> tuple[int, int]:
    width, height = source_wh
    max_edge = 1024.0 * ratio
    min_edge = 512.0 * ratio
    scale = min(max_edge / max(width, height), min_edge / min(width, height))
    return max(1, round(width * scale)), max(1, round(height * scale))


def _resize_mask(mask: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    image = Image.fromarray(mask, mode="L")
    return np.asarray(image.resize(size_wh, resample=Image.Resampling.NEAREST))


def _pad_mask(mask: np.ndarray, output_wh: tuple[int, int]) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    output_width, output_height = output_wh
    height, width = mask.shape
    if width > output_width or height > output_height:
        raise ValueError(f"Cannot pad {width}x{height} into {output_width}x{output_height}")
    output = np.full((output_height, output_width), IGNORE_INDEX, dtype=np.uint8)
    output[:height, :width] = mask
    return output, (0, 0, output_width - width, output_height - height)


def _crop_bbox(
    mask: np.ndarray,
    rng: random.Random,
    *,
    crop_wh: tuple[int, int] = (512, 512),
    cat_max_ratio: float = 0.75,
    attempts: int = 10,
) -> tuple[int, int, int, int]:
    height, width = mask.shape
    crop_width, crop_height = crop_wh
    max_x = max(width - crop_width, 0)
    max_y = max(height - crop_height, 0)
    bbox = (0, 0, min(crop_width, width), min(crop_height, height))
    for _ in range(attempts):
        x = rng.randint(0, max_x) if max_x else 0
        y = rng.randint(0, max_y) if max_y else 0
        actual_width = min(crop_width, width - x)
        actual_height = min(crop_height, height - y)
        bbox = (x, y, actual_width, actual_height)
        crop = mask[y : y + actual_height, x : x + actual_width]
        labels, counts = np.unique(crop, return_counts=True)
        valid = counts[labels != IGNORE_INDEX]
        if valid.size > 1 and int(valid.max()) / int(valid.sum()) < cat_max_ratio:
            break
    return bbox


def transform_mask(
    mask: np.ndarray,
    candidate: str,
    rng: random.Random,
) -> tuple[np.ndarray, np.ndarray, Geometry]:
    if candidate not in CANDIDATES:
        raise ValueError(f"Unknown transform candidate: {candidate}")
    source_wh = (mask.shape[1], mask.shape[0])
    output_wh = tuple(CANDIDATES[candidate]["output_wh"])
    if candidate == "i0_crop512":
        resized_wh = _random_resize_size(source_wh, rng.uniform(0.5, 2.0))
        resized = _resize_mask(mask, resized_wh)
        x, y, width, height = _crop_bbox(resized, rng)
        cropped = resized[y : y + height, x : x + width]
        transformed, pad = _pad_mask(cropped, output_wh)
        return transformed, resized, Geometry(resized_wh, (x, y, width, height), pad, output_wh)
    resized_wh = _keep_ratio_size(source_wh, output_wh)
    resized = _resize_mask(mask, resized_wh)
    transformed, pad = _pad_mask(resized, output_wh)
    return transformed, resized, Geometry(
        resized_wh,
        (0, 0, resized_wh[0], resized_wh[1]),
        pad,
        output_wh,
    )


def _component_stats(binary: np.ndarray) -> tuple[int, int]:
    try:
        import cv2

        count, _, stats, _ = cv2.connectedComponentsWithStats(
            binary.astype(np.uint8), connectivity=8
        )
        if count <= 1:
            return 0, 0
        return count - 1, int(stats[1:, cv2.CC_STAT_AREA].max())
    except ImportError:
        pass

    # Run-length connected components avoids a Python operation per pixel and
    # keeps the no-OpenCV fallback practical for large uniform terrain masks.
    parents: list[int] = []
    areas: list[int] = []

    def find(label: int) -> int:
        while parents[label] != label:
            parents[label] = parents[parents[label]]
            label = parents[label]
        return label

    def union(left: int, right: int) -> int:
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            return left_root
        if areas[left_root] < areas[right_root]:
            left_root, right_root = right_root, left_root
        parents[right_root] = left_root
        areas[left_root] += areas[right_root]
        return left_root

    previous: list[tuple[int, int, int]] = []
    for row in binary:
        padded = np.pad(row.astype(np.int8), (1, 1))
        changes = np.flatnonzero(np.diff(padded))
        current: list[tuple[int, int, int]] = []
        for start, stop in zip(changes[0::2], changes[1::2]):
            end = int(stop) - 1
            label = len(parents)
            parents.append(label)
            areas.append(end - int(start) + 1)
            overlaps = [
                previous_label
                for previous_start, previous_end, previous_label in previous
                if previous_end >= int(start) - 1 and previous_start <= end + 1
            ]
            for previous_label in overlaps:
                label = union(label, previous_label)
            current.append((int(start), end, label))
        previous = current
    roots = {find(label) for label in range(len(parents))}
    return len(roots), max((areas[root] for root in roots), default=0)


def _mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return float(sum(values) / len(values)) if values else None


def audit_transforms(
    samples: list[Sample],
    *,
    draws: int = 20,
    seed: int = 42,
) -> dict[str, Any]:
    if draws < 20:
        raise ValueError("Transform audit requires at least 20 Monte Carlo draws")
    rng = random.Random(seed)
    aggregates: dict[tuple[str, str, int], dict[str, Any]] = defaultdict(
        lambda: {
            "original_image_draws": 0,
            "retained_image_draws": 0,
            "pixel_ratios": [],
            "component_count_ratios": [],
            "largest_component_pixel_ratios": [],
        }
    )
    candidate_totals: dict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {
            "draws": 0,
            "non_ignore_ratios": [],
            "pad_ratios": [],
            "aspect_distortions": [],
            "geometry_counts": Counter(),
        }
    )

    mask_digests: dict[str, str] = {}
    for sample in samples:
        with Image.open(sample.mask_path) as image:
            mask = np.asarray(image.convert("L"))
        if mask.dtype != np.uint8 or mask.ndim != 2:
            raise ValueError(f"Mask must be uint8 HxW: {sample.mask_path}")
        invalid = set(int(value) for value in np.unique(mask)) - (set(range(19)) | {255})
        if invalid:
            raise ValueError(f"Invalid Semantic20 IDs in {sample.sample_id}: {sorted(invalid)}")
        mask_digests[sample.sample_id] = _sha256(sample.mask_path)
        original_ids = [int(value) for value in np.unique(mask) if value != IGNORE_INDEX]

        for candidate, candidate_definition in CANDIDATES.items():
            # No-crop candidates are deterministic. Compute their geometry and
            # class statistics once, then apply the requested audit weight so
            # summaries remain directly comparable with the random-crop draws.
            evaluated_draws = draws if candidate_definition["random_crop"] else 1
            draw_weight = 1 if candidate_definition["random_crop"] else draws
            for _ in range(evaluated_draws):
                transformed, resized, geometry = transform_mask(mask, candidate, rng)
                total_key = (candidate, sample.source)
                totals = candidate_totals[total_key]
                totals["draws"] += draw_weight
                totals["non_ignore_ratios"].extend(
                    [float(np.mean(transformed != IGNORE_INDEX))] * draw_weight
                )
                pad_ratio = 1.0 - (
                    geometry.crop_xywh[2]
                    * geometry.crop_xywh[3]
                    / float(geometry.output_wh[0] * geometry.output_wh[1])
                )
                totals["pad_ratios"].extend([pad_ratio] * draw_weight)
                source_aspect = mask.shape[1] / mask.shape[0]
                resized_aspect = geometry.resized_wh[0] / geometry.resized_wh[1]
                aspect_distortion = abs(resized_aspect / source_aspect - 1.0)
                totals["aspect_distortions"].extend(
                    [aspect_distortion] * draw_weight
                )
                geometry_key = json.dumps(
                    {
                        "resized_wh": geometry.resized_wh,
                        "crop_xywh": geometry.crop_xywh,
                        "pad_ltrb": geometry.pad_ltrb,
                        "output_wh": geometry.output_wh,
                    },
                    sort_keys=True,
                )
                totals["geometry_counts"][geometry_key] += draw_weight

                for class_id in original_ids:
                    aggregate = aggregates[(candidate, sample.source, class_id)]
                    aggregate["original_image_draws"] += draw_weight
                    resized_pixels = int(np.count_nonzero(resized == class_id))
                    transformed_pixels = int(np.count_nonzero(transformed == class_id))
                    if transformed_pixels:
                        aggregate["retained_image_draws"] += draw_weight
                    pixel_ratio = (
                        transformed_pixels / resized_pixels if resized_pixels else 0.0
                    )
                    aggregate["pixel_ratios"].extend([pixel_ratio] * draw_weight)
                    before_count, before_largest = _component_stats(resized == class_id)
                    after_count, after_largest = _component_stats(transformed == class_id)
                    component_ratio = (
                        after_count / before_count if before_count else 0.0
                    )
                    aggregate["component_count_ratios"].extend(
                        [component_ratio] * draw_weight
                    )
                    largest_component_ratio = (
                        after_largest / before_largest if before_largest else 0.0
                    )
                    aggregate["largest_component_pixel_ratios"].extend(
                        [largest_component_ratio] * draw_weight
                    )

    class_rows: list[dict[str, Any]] = []
    for (candidate, source, class_id), values in sorted(aggregates.items()):
        original_draws = int(values["original_image_draws"])
        retained_draws = int(values["retained_image_draws"])
        class_rows.append(
            {
                "candidate": candidate,
                "source": source,
                "class_id": class_id,
                "class_name": SEMANTIC20_CLASSES[class_id],
                "original_image_draws": original_draws,
                "retained_image_draws": retained_draws,
                "retention_probability": retained_draws / original_draws,
                "crop_miss_rate": (
                    (original_draws - retained_draws) / original_draws
                    if candidate == "i0_crop512"
                    else 0.0
                ),
                "mean_retained_pixel_ratio": _mean(values["pixel_ratios"]),
                "mean_component_count_ratio": _mean(values["component_count_ratios"]),
                "mean_largest_component_pixel_ratio": _mean(
                    values["largest_component_pixel_ratios"]
                ),
                "rare_risk": class_id in RARE_RISK_IDS,
            }
        )

    candidate_rows: list[dict[str, Any]] = []
    for (candidate, source), values in sorted(candidate_totals.items()):
        candidate_rows.append(
            {
                "candidate": candidate,
                "source": source,
                "draws": values["draws"],
                "mean_non_ignore_ratio": _mean(values["non_ignore_ratios"]),
                "mean_pad_ratio": _mean(values["pad_ratios"]),
                "mean_aspect_distortion": _mean(values["aspect_distortions"]),
                "geometry_counts": [
                    {**json.loads(key), "count": count}
                    for key, count in sorted(values["geometry_counts"].items())
                ],
            }
        )

    digest = hashlib.sha256()
    for sample_id, value in sorted(mask_digests.items()):
        digest.update(sample_id.encode("utf-8"))
        digest.update(value.encode("ascii"))
    return {
        "schema_version": "adom-ta0-transform-audit-v1",
        "seed": seed,
        "monte_carlo_draws_per_sample": draws,
        "sample_count": len(samples),
        "source_sample_counts": dict(Counter(sample.source for sample in samples)),
        "semantic_contract": {"class_ids": list(range(19)), "ignore_index": 255},
        "rare_risk_ids": list(RARE_RISK_IDS),
        "mask_set_sha256": digest.hexdigest(),
        "candidates": {
            name: {
                **value,
                "mask_interpolation": "nearest",
                "image_interpolation": "bilinear",
                "pad_direction": "right_bottom",
                "seg_pad_value": 255,
            }
            for name, value in CANDIDATES.items()
        },
        "candidate_source_summary": candidate_rows,
        "class_retention": class_rows,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Audit TA0 crop/no-crop mask retention without training"
    )
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--split", required=True, type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--default-source", default="rellis3d")
    parser.add_argument("--draws", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        samples = load_samples(
            args.dataset,
            args.split,
            manifest=args.manifest,
            default_source=args.default_source,
        )
        report = audit_transforms(samples, draws=args.draws, seed=args.seed)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(report, indent=2, sort_keys=True))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.exit(2, f"ERROR: {error}\n")


if __name__ == "__main__":
    main()
