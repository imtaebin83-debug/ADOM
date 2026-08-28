from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

from adom.evaluation_semantic20 import SEMANTIC20_CLASSES


NUM_CLASSES = 19
IGNORE_INDEX = 255
SPLITS = ("reference", "validation", "test")
SCORE_NAMES = (
    "sml_uncertainty",
    "entropy",
    "msp_uncertainty",
    "margin_uncertainty",
    "energy",
    "negative_softmax_variance",
    "negative_logit_variance",
)


@dataclass(frozen=True)
class Sample:
    sample_id: str
    sequence_id: str
    split: str
    logits_path: Path
    label_path: Path
    image_path: Path | None = None


def _resolve_file(root: Path, value: str, field: str) -> Path:
    path = Path(value)
    path = path if path.is_absolute() else root / path
    if not path.is_file():
        raise FileNotFoundError(f"{field} does not exist: {path}")
    return path


def load_manifest(path: Path) -> list[Sample]:
    root = path.parent
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "sample_id",
            "sequence_id",
            "split",
            "logits_path",
            "label_path",
        }
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Manifest is missing fields: {sorted(missing)}")
        samples: list[Sample] = []
        seen_samples: set[str] = set()
        sequence_splits: dict[str, str] = {}
        for row in reader:
            sample_id = row["sample_id"].strip()
            sequence_id = row["sequence_id"].strip()
            split = row["split"].strip()
            if not sample_id or not sequence_id:
                raise ValueError("sample_id and sequence_id must be non-empty")
            if sample_id in seen_samples:
                raise ValueError(f"Duplicate sample_id: {sample_id}")
            if split not in SPLITS:
                raise ValueError(f"Unknown split for {sample_id}: {split}")
            previous = sequence_splits.setdefault(sequence_id, split)
            if previous != split:
                raise ValueError(
                    f"Sequence leakage: {sequence_id} appears in {previous} and {split}"
                )
            image_value = (row.get("image_path") or "").strip()
            samples.append(
                Sample(
                    sample_id=sample_id,
                    sequence_id=sequence_id,
                    split=split,
                    logits_path=_resolve_file(root, row["logits_path"], "logits_path"),
                    label_path=_resolve_file(root, row["label_path"], "label_path"),
                    image_path=(
                        _resolve_file(root, image_value, "image_path")
                        if image_value
                        else None
                    ),
                )
            )
            seen_samples.add(sample_id)
    if not samples:
        raise ValueError("Manifest contains no samples")
    present = {sample.split for sample in samples}
    missing_splits = set(SPLITS) - present
    if missing_splits:
        raise ValueError(f"Manifest is missing splits: {sorted(missing_splits)}")
    return samples


def load_logits(path: Path) -> np.ndarray:
    logits = np.load(path, allow_pickle=False)
    if logits.ndim == 4 and logits.shape[0] == 1:
        logits = logits[0]
    if logits.ndim != 3 or logits.shape[0] != NUM_CLASSES:
        raise ValueError(
            f"Expected logits shaped (19,H,W) or (1,19,H,W), got {logits.shape}: {path}"
        )
    logits = np.asarray(logits, dtype=np.float64)
    if not np.isfinite(logits).all():
        raise ValueError(f"Non-finite logits: {path}")
    return logits


def load_label(path: Path, shape: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as source:
        label = np.asarray(source, dtype=np.uint8)
    if label.ndim != 2 or label.shape != shape:
        raise ValueError(f"Label shape {label.shape} does not match logits {shape}: {path}")
    invalid = set(int(value) for value in np.unique(label)) - set(range(NUM_CLASSES)) - {
        IGNORE_INDEX
    }
    if invalid:
        raise ValueError(f"Label contains invalid Semantic20 IDs {sorted(invalid)}: {path}")
    return label


def stable_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=0, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / np.sum(exponent, axis=0, keepdims=True)


def logsumexp(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    scaled = logits / temperature
    maximum = np.max(scaled, axis=0)
    return maximum + np.log(np.sum(np.exp(scaled - maximum), axis=0))


def fit_sml_statistics(
    samples: Sequence[Sample], minimum_pixels: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    sums = np.zeros(NUM_CLASSES, dtype=np.float64)
    sum_squares = np.zeros(NUM_CLASSES, dtype=np.float64)
    gt_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    for sample in samples:
        logits = load_logits(sample.logits_path)
        label = load_label(sample.label_path, tuple(logits.shape[1:]))
        valid = label != IGNORE_INDEX
        prediction = np.argmax(logits, axis=0)
        maximum = np.max(logits, axis=0)
        gt_counts += np.bincount(label[valid], minlength=NUM_CLASSES)[:NUM_CLASSES]
        for class_id in range(NUM_CLASSES):
            values = maximum[valid & (prediction == class_id)]
            counts[class_id] += values.size
            sums[class_id] += float(np.sum(values))
            sum_squares[class_id] += float(np.sum(values * values))
    supported = counts >= minimum_pixels
    total = int(np.sum(counts))
    if total < minimum_pixels:
        raise ValueError("Reference split has too few valid predicted pixels")
    global_mean = float(np.sum(sums) / total)
    global_variance = max(float(np.sum(sum_squares) / total - global_mean**2), 0.0)
    global_standard_deviation = math.sqrt(global_variance)
    if global_standard_deviation <= 1e-12:
        raise ValueError("Near-zero global SML reference standard deviation")
    means = np.full(NUM_CLASSES, global_mean, dtype=np.float64)
    variances = np.full(NUM_CLASSES, global_variance, dtype=np.float64)
    means[supported] = sums[supported] / counts[supported]
    variances[supported] = np.maximum(
        sum_squares[supported] / counts[supported] - means[supported] ** 2, 0.0
    )
    standard_deviations = np.sqrt(variances)
    if np.any(standard_deviations <= 1e-12):
        bad = np.flatnonzero(standard_deviations <= 1e-12).tolist()
        raise ValueError(f"Near-zero SML reference standard deviation for IDs: {bad}")
    return means, standard_deviations, counts, gt_counts, supported


def spearman_permutation_test(
    x: np.ndarray,
    y: np.ndarray,
    *,
    permutations: int = 5000,
    seed: int = 20260813,
) -> dict[str, float | int | None]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if (
        x.size < 3
        or np.ptp(x) <= np.finfo(np.float64).eps
        or np.ptp(y) <= np.finfo(np.float64).eps
    ):
        return {"n_classes": int(x.size), "rho": None, "permutation_p_two_sided": None}
    x_rank = _average_ranks(x)
    y_rank = _average_ranks(y)
    rho = float(np.corrcoef(x_rank, y_rank)[0, 1])
    generator = np.random.default_rng(seed)
    extreme = 0
    for _ in range(permutations):
        permuted = generator.permutation(y_rank)
        candidate = float(np.corrcoef(x_rank, permuted)[0, 1])
        if abs(candidate) >= abs(rho) - 1e-15:
            extreme += 1
    return {
        "n_classes": int(x.size),
        "rho": rho,
        "permutation_p_two_sided": (extreme + 1) / (permutations + 1),
        "permutations": permutations,
    }


def score_maps(
    logits: np.ndarray,
    sml_means: np.ndarray,
    sml_standard_deviations: np.ndarray,
    *,
    temperature: float = 1.0,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    probability = stable_softmax(logits / temperature)
    prediction = np.argmax(logits, axis=0)
    maximum_logit = np.max(logits, axis=0)
    sorted_probability = np.sort(probability, axis=0)
    maximum_probability = sorted_probability[-1]
    margin = sorted_probability[-1] - sorted_probability[-2]
    entropy = -np.sum(probability * np.log(np.clip(probability, 1e-12, 1.0)), axis=0)
    entropy /= math.log(NUM_CLASSES)
    sml = (maximum_logit - sml_means[prediction]) / sml_standard_deviations[
        prediction
    ]
    scores = {
        # Every map is oriented so that a larger value means "more suspicious".
        "sml_uncertainty": -sml,
        "entropy": entropy,
        "msp_uncertainty": 1.0 - maximum_probability,
        "margin_uncertainty": 1.0 - margin,
        "energy": -temperature * logsumexp(logits, temperature),
        "negative_softmax_variance": -np.var(probability, axis=0),
        "negative_logit_variance": -np.var(logits, axis=0),
    }
    return prediction, scores


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def binary_auroc(scores: np.ndarray, positive: np.ndarray) -> float:
    positive = np.asarray(positive, dtype=bool)
    positives = int(np.count_nonzero(positive))
    negatives = int(positive.size - positives)
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = _average_ranks(np.asarray(scores, dtype=np.float64))
    return float(
        (np.sum(ranks[positive]) - positives * (positives + 1) / 2)
        / (positives * negatives)
    )


def average_precision(scores: np.ndarray, positive: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    positive = np.asarray(positive, dtype=bool)
    positives = int(np.count_nonzero(positive))
    if positives == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    scores = scores[order]
    positive = positive[order]
    true_positive = false_positive = 0
    previous_recall = 0.0
    result = 0.0
    start = 0
    while start < scores.size:
        end = start + 1
        while end < scores.size and scores[end] == scores[start]:
            end += 1
        true_positive += int(np.count_nonzero(positive[start:end]))
        false_positive += int(end - start - np.count_nonzero(positive[start:end]))
        recall = true_positive / positives
        precision = true_positive / (true_positive + false_positive)
        result += (recall - previous_recall) * precision
        previous_recall = recall
        start = end
    return float(result)


def select_f1_threshold(scores: np.ndarray, positive: np.ndarray) -> dict[str, float]:
    scores = np.asarray(scores, dtype=np.float64)
    positive = np.asarray(positive, dtype=bool)
    positives = int(np.count_nonzero(positive))
    negatives = int(positive.size - positives)
    if positives == 0 or negatives == 0:
        raise ValueError("Threshold selection requires positive and negative validation pixels")
    order = np.argsort(-scores, kind="mergesort")
    scores = scores[order]
    positive = positive[order]
    true_positive = false_positive = 0
    best = {"threshold": float("inf"), "f1": -1.0, "precision": 0.0, "recall": 0.0}
    start = 0
    while start < scores.size:
        end = start + 1
        while end < scores.size and scores[end] == scores[start]:
            end += 1
        true_positive += int(np.count_nonzero(positive[start:end]))
        false_positive += int(end - start - np.count_nonzero(positive[start:end]))
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / positives
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        if f1 > best["f1"]:
            best = {
                "threshold": float(scores[start]),
                "f1": float(f1),
                "precision": float(precision),
                "recall": float(recall),
            }
        start = end
    return best


def evaluate_threshold(
    scores: np.ndarray, positive: np.ndarray, threshold: float
) -> dict[str, float | int]:
    positive = np.asarray(positive, dtype=bool)
    detected = np.asarray(scores) >= threshold
    tp = int(np.count_nonzero(detected & positive))
    fp = int(np.count_nonzero(detected & ~positive))
    fn = int(np.count_nonzero(~detected & positive))
    tn = int(np.count_nonzero(~detected & ~positive))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
    }


def _sample_stratum(values: np.ndarray, limit: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size <= limit:
        return values
    indices = np.linspace(0, values.size - 1, num=limit, dtype=np.int64)
    return values[indices]


def _task_masks(
    label: np.ndarray, prediction: np.ndarray, true_class: int, predicted_class: int
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    valid = label != IGNORE_INDEX
    error_positive = valid & (label != prediction)
    error_control = valid & (label == prediction)
    pair_positive = valid & (label == true_class) & (prediction == predicted_class)
    pair_control = valid & (label == predicted_class) & (prediction == predicted_class)
    return {
        "all_error": (error_positive, error_control),
        f"{SEMANTIC20_CLASSES[true_class]}_to_{SEMANTIC20_CLASSES[predicted_class]}": (
            pair_positive,
            pair_control,
        ),
    }


def _write_visualization(
    sample: Sample,
    score: np.ndarray,
    valid: np.ndarray,
    threshold: float,
    output_dir: Path,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    finite = score[np.isfinite(score)]
    low, high = np.percentile(finite, [1, 99]) if finite.size else (0.0, 1.0)
    scaled = np.clip((score - low) / max(high - low, 1e-12), 0.0, 1.0)
    heatmap = np.zeros((*score.shape, 3), dtype=np.uint8)
    heatmap[..., 0] = np.rint(255 * scaled).astype(np.uint8)
    heatmap[..., 2] = np.rint(255 * (1.0 - scaled)).astype(np.uint8)
    heatmap[~valid] = 0
    review = np.where(valid & (score >= threshold), 255, 0).astype(np.uint8)
    heatmap_path = output_dir / f"{sample.sample_id}_sml_heatmap.png"
    review_path = output_dir / f"{sample.sample_id}_review_mask.png"
    Image.fromarray(heatmap, mode="RGB").save(heatmap_path)
    Image.fromarray(review, mode="L").save(review_path)
    generated = {"heatmap": heatmap_path.name, "review_mask": review_path.name}
    if sample.image_path is not None:
        with Image.open(sample.image_path) as source:
            rgb = source.convert("RGB")
        resized = Image.fromarray(heatmap, mode="RGB").resize(rgb.size, Image.BILINEAR)
        overlay = Image.blend(rgb, resized, 0.4)
        overlay_path = output_dir / f"{sample.sample_id}_sml_overlay.png"
        overlay.save(overlay_path)
        generated["overlay"] = overlay_path.name
    return generated


def run_analysis(args: argparse.Namespace) -> dict[str, Any]:
    samples = load_manifest(args.manifest)
    reference = [sample for sample in samples if sample.split == "reference"]
    means, standard_deviations, reference_counts, reference_gt_counts, sml_supported = (
        fit_sml_statistics(reference, args.minimum_reference_pixels)
    )
    if not sml_supported[args.predicted_class]:
        raise ValueError(
            "Primary predicted class lacks enough reference pixels for class-wise SML: "
            f"{SEMANTIC20_CLASSES[args.predicted_class]} has "
            f"{reference_counts[args.predicted_class]}, requires "
            f"{args.minimum_reference_pixels}"
        )
    pair_name = (
        f"{SEMANTIC20_CLASSES[args.true_class]}_to_"
        f"{SEMANTIC20_CLASSES[args.predicted_class]}"
    )
    collected: dict[str, dict[str, dict[str, list[np.ndarray]]]] = {
        split: {
            task: {
                score: []
                for score in SCORE_NAMES
            }
            for task in ("all_error", pair_name)
        }
        for split in ("validation", "test")
    }
    controls: dict[str, dict[str, dict[str, list[np.ndarray]]]] = {
        split: {
            task: {score: [] for score in SCORE_NAMES}
            for task in ("all_error", pair_name)
        }
        for split in ("validation", "test")
    }
    confusion = {
        split: np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
        for split in SPLITS
    }
    cache: dict[str, tuple[Sample, np.ndarray, np.ndarray]] = {}
    frame_rows: list[dict[str, Any]] = []
    for sample in samples:
        logits = load_logits(sample.logits_path)
        label = load_label(sample.label_path, tuple(logits.shape[1:]))
        prediction, scores = score_maps(
            logits, means, standard_deviations, temperature=args.temperature
        )
        valid = label != IGNORE_INDEX
        flat_pair = label[valid].astype(np.int64) * NUM_CLASSES + prediction[valid]
        confusion[sample.split] += np.bincount(
            flat_pair, minlength=NUM_CLASSES * NUM_CLASSES
        ).reshape(NUM_CLASSES, NUM_CLASSES)
        if sample.split == "reference":
            continue
        tasks = _task_masks(label, prediction, args.true_class, args.predicted_class)
        for task_name, (positive_mask, control_mask) in tasks.items():
            for score_name, score in scores.items():
                positive_values = _sample_stratum(
                    score[positive_mask], args.max_pixels_per_stratum_per_frame
                )
                control_values = _sample_stratum(
                    score[control_mask], args.max_pixels_per_stratum_per_frame
                )
                if positive_values.size:
                    collected[sample.split][task_name][score_name].append(
                        positive_values
                    )
                if control_values.size:
                    controls[sample.split][task_name][score_name].append(control_values)
                frame_rows.append(
                    {
                        "sample_id": sample.sample_id,
                        "sequence_id": sample.sequence_id,
                        "split": sample.split,
                        "task": task_name,
                        "score": score_name,
                        "positive_pixels": int(np.count_nonzero(positive_mask)),
                        "control_pixels": int(np.count_nonzero(control_mask)),
                        "positive_median": (
                            float(np.median(score[positive_mask]))
                            if np.any(positive_mask)
                            else ""
                        ),
                        "control_median": (
                            float(np.median(score[control_mask]))
                            if np.any(control_mask)
                            else ""
                        ),
                    }
                )
        cache[sample.sample_id] = (sample, scores["sml_uncertainty"], valid)

    results: dict[str, Any] = {}
    thresholds: dict[str, dict[str, float]] = {}
    for task_name in ("all_error", pair_name):
        results[task_name] = {}
        thresholds[task_name] = {}
        for score_name in SCORE_NAMES:
            split_payload: dict[str, Any] = {}
            validation_threshold: float | None = None
            for split in ("validation", "test"):
                positive_parts = collected[split][task_name][score_name]
                control_parts = controls[split][task_name][score_name]
                if not positive_parts or not control_parts:
                    split_payload[split] = {
                        "status": "INSUFFICIENT_STRATA",
                        "positive_pixels_sampled": int(
                            sum(item.size for item in positive_parts)
                        ),
                        "control_pixels_sampled": int(
                            sum(item.size for item in control_parts)
                        ),
                    }
                    continue
                positive_values = np.concatenate(positive_parts)
                control_values = np.concatenate(control_parts)
                values = np.concatenate((positive_values, control_values))
                target = np.concatenate(
                    (
                        np.ones(positive_values.size, dtype=bool),
                        np.zeros(control_values.size, dtype=bool),
                    )
                )
                payload: dict[str, Any] = {
                    "status": "EVALUATED",
                    "positive_pixels_sampled": int(positive_values.size),
                    "control_pixels_sampled": int(control_values.size),
                    "positive_median": float(np.median(positive_values)),
                    "control_median": float(np.median(control_values)),
                    "auroc": binary_auroc(values, target),
                    "average_precision": average_precision(values, target),
                }
                if split == "validation":
                    selected = select_f1_threshold(values, target)
                    validation_threshold = selected["threshold"]
                    thresholds[task_name][score_name] = validation_threshold
                    payload["threshold_selection"] = selected
                elif validation_threshold is not None:
                    payload["fixed_validation_threshold"] = validation_threshold
                    payload["fixed_threshold_metrics"] = evaluate_threshold(
                        values, target, validation_threshold
                    )
                split_payload[split] = payload
            results[task_name][score_name] = split_payload

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame_path = args.output_dir / "frame-strata-summary.csv"
    with frame_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(frame_rows[0]))
        writer.writeheader()
        writer.writerows(frame_rows)

    visualizations: list[dict[str, Any]] = []
    threshold = thresholds.get(pair_name, {}).get("sml_uncertainty")
    if threshold is not None:
        ranked = sorted(
            (
                (
                    float(np.mean(score[valid] >= threshold)),
                    sample_id,
                    sample,
                    score,
                    valid,
                )
                for sample_id, (sample, score, valid) in cache.items()
                if sample.split == "test"
            ),
            reverse=True,
            key=lambda item: item[0],
        )
        for ratio, sample_id, sample, score, valid in ranked[
            : args.visualization_count
        ]:
            generated = _write_visualization(
                sample, score, valid, threshold, args.output_dir / "visualizations"
            )
            visualizations.append(
                {"sample_id": sample_id, "review_area_ratio": ratio, **generated}
            )

    class_rows = []
    test_matrix = confusion["test"]
    for class_id, name in enumerate(SEMANTIC20_CLASSES):
        tp = int(test_matrix[class_id, class_id])
        fp = int(np.sum(test_matrix[:, class_id]) - tp)
        negatives = int(np.sum(test_matrix) - np.sum(test_matrix[class_id, :]))
        predictions = tp + fp
        false_discovery_rate = fp / predictions if predictions else None
        class_rows.append(
            {
                "id": class_id,
                "name": name,
                "reference_gt_pixels": int(reference_gt_counts[class_id]),
                "reference_gt_share": float(
                    reference_gt_counts[class_id] / max(np.sum(reference_gt_counts), 1)
                ),
                "test_false_positive_pixels": fp,
                "test_false_positive_rate": fp / negatives if negatives else 0.0,
                "test_false_discovery_rate": false_discovery_rate,
            }
        )

    shares = np.asarray([row["reference_gt_share"] for row in class_rows])
    false_positive_rates = np.asarray(
        [row["test_false_positive_rate"] for row in class_rows]
    )
    fdr_values = np.asarray(
        [
            float(row["test_false_discovery_rate"])
            if row["test_false_discovery_rate"] is not None
            else float("nan")
            for row in class_rows
        ]
    )
    prior_association = {
        "log_reference_share_vs_test_false_positive_rate": spearman_permutation_test(
            np.log(shares + 1e-12), false_positive_rates
        ),
        "log_reference_share_vs_test_false_discovery_rate": spearman_permutation_test(
            np.log(shares + 1e-12), fdr_values
        ),
        "interpretation": (
            "Exploratory class-level association only; it cannot establish that "
            "training pixel frequency caused field false positives."
        ),
    }

    report = {
        "schema_version": "adom-semantic20-field-uncertainty-v1",
        "status": "EXPLORATORY_NOT_SAFETY_CERTIFIED",
        "contract": {
            "ontology": "Semantic20 IDs 0..18; ignore 255",
            "score_direction": "larger means more suspicious for every score",
            "primary_score": "sml_uncertainty",
            "primary_pair": pair_name,
            "threshold_policy": "select on validation sequences; apply unchanged to test",
            "pixel_metrics_warning": (
                "Pixels are spatially correlated. Use frame-strata-summary.csv and "
                "sequence-level replication for inferential claims."
            ),
        },
        "reference_sml": {
            "definition": "ICCV 2021 SML class-wise predicted max-logit standardization",
            "means": means.tolist(),
            "standard_deviations": standard_deviations.tolist(),
            "predicted_pixel_counts": reference_counts.tolist(),
            "minimum_pixels_per_supported_class": args.minimum_reference_pixels,
            "class_wise_supported": sml_supported.tolist(),
            "fallback": (
                "Unsupported non-primary classes use global reference max-logit "
                "mean/std and must not be interpreted as exact class-wise SML."
            ),
        },
        "sampling": {
            "method": "deterministic evenly spaced within each frame and stratum",
            "maximum_pixels_per_stratum_per_frame": (
                args.max_pixels_per_stratum_per_frame
            ),
        },
        "results": results,
        "confusion_matrices_gt_rows_predicted_columns": {
            split: matrix.tolist() for split, matrix in confusion.items()
        },
        "class_prior_fp_diagnostics": class_rows,
        "class_prior_fp_association": prior_association,
        "visualizations": visualizations,
        "limitations": [
            "Raw-logit or softmax dispersion alone does not identify an object's semantic name.",
            "A high softmax-class variance usually indicates a peaked, confident prediction; its negative is used as an uncertainty score.",
            "Softmax-derived scores may miss overconfident domain-shift errors.",
            "Class pixel frequency association is observational and does not establish causality.",
            "No threshold from this report may directly command the RC car; keep watchdog and manual reset contracts.",
        ],
    }
    report_path = args.output_dir / "uncertainty-report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"report": str(report_path), "primary_pair": pair_name}, indent=2))
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Semantic20 field errors with logit-derived uncertainty scores"
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--true-class", type=int, default=0)
    parser.add_argument("--predicted-class", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--minimum-reference-pixels", type=int, default=100)
    parser.add_argument("--max-pixels-per-stratum-per-frame", type=int, default=5000)
    parser.add_argument("--visualization-count", type=int, default=10)
    args = parser.parse_args(argv)
    if not args.manifest.is_file():
        parser.error(f"manifest does not exist: {args.manifest}")
    if not 0 <= args.true_class < NUM_CLASSES:
        parser.error("true-class must be in 0..18")
    if not 0 <= args.predicted_class < NUM_CLASSES:
        parser.error("predicted-class must be in 0..18")
    if args.true_class == args.predicted_class:
        parser.error("true-class and predicted-class must differ")
    if args.temperature <= 0:
        parser.error("temperature must be positive")
    if args.minimum_reference_pixels < 2:
        parser.error("minimum-reference-pixels must be at least 2")
    if args.max_pixels_per_stratum_per_frame < 1:
        parser.error("max-pixels-per-stratum-per-frame must be positive")
    if args.visualization_count < 0:
        parser.error("visualization-count must be non-negative")
    run_analysis(args)


if __name__ == "__main__":
    main()
