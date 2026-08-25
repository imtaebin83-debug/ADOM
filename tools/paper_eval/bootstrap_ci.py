from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any

import numpy as np

from _common import CLASS_INDEX, metric_value, write_dict_csv, write_json


PAIRED_METRICS = (
    "common_supported_mIoU",
    "log/IoU",
    "log/Recall",
    "rubble/IoU",
    "rubble/Recall",
    "barrier/IoU",
    "mud/IoU",
)


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def _metric_full(
    confusion: np.ndarray,
    metric: str,
    common_classes: list[str],
) -> float | None:
    if metric == "common_supported_mIoU":
        supported = [
            name
            for name in common_classes
            if confusion[CLASS_INDEX[name]].sum() > 0
        ]
        if not supported:
            return None
        return metric_value(confusion, metric, common_classes=supported)
    return metric_value(confusion, metric, common_classes=common_classes)


def _batch_metric(
    confusions: np.ndarray,
    metric: str,
    common_classes: list[str],
) -> np.ndarray:
    confusions = confusions.astype(np.float64, copy=False)
    gt = confusions.sum(axis=2)
    pred = confusions.sum(axis=1)
    tp = np.diagonal(confusions, axis1=1, axis2=2)
    if metric == "common_supported_mIoU":
        indices = [CLASS_INDEX[name] for name in common_classes]
        union = gt[:, indices] + pred[:, indices] - tp[:, indices]
        values = np.divide(
            tp[:, indices],
            union,
            out=np.full_like(union, np.nan),
            where=(gt[:, indices] > 0) & (union > 0),
        )
        return np.nanmean(values, axis=1) * 100.0
    class_name, metric_name = metric.split("/", 1)
    index = CLASS_INDEX[class_name]
    if metric_name == "IoU":
        denominator = gt[:, index] + pred[:, index] - tp[:, index]
    elif metric_name == "Recall":
        denominator = gt[:, index]
    else:  # pragma: no cover - PAIRED_METRICS is fixed
        raise ValueError(metric)
    return np.divide(
        tp[:, index],
        denominator,
        out=np.full(confusions.shape[0], np.nan, dtype=np.float64),
        where=(gt[:, index] > 0) & (denominator > 0),
    ) * 100.0


def _units(sequences: np.ndarray) -> tuple[str, list[str], list[np.ndarray], str | None]:
    values = [str(value) for value in sequences.tolist()]
    if values and all(value and value != "unknown" for value in values):
        grouped: defaultdict[str, list[int]] = defaultdict(list)
        for index, value in enumerate(values):
            grouped[value].append(index)
        names = sorted(grouped)
        return (
            "sequence",
            names,
            [np.asarray(grouped[name], dtype=np.int64) for name in names],
            None,
        )
    return (
        "image",
        [str(index) for index in range(len(values))],
        [np.asarray([index], dtype=np.int64) for index in range(len(values))],
        "At least one sequence ID was missing; images are the resampling units.",
    )


def bootstrap(args: argparse.Namespace) -> dict[str, Any]:
    if args.samples < 10_000:
        raise ValueError("Paper evaluation requires at least 10,000 bootstrap resamples")
    baseline = _load(args.baseline.resolve())
    eadom = _load(args.eadom.resolve())
    for field in ("sample_ids", "sequences", "confusions", "common_classes"):
        if field not in baseline or field not in eadom:
            raise ValueError(f"Per-image confusion artifact is missing {field}")
    if not np.array_equal(baseline["sample_ids"], eadom["sample_ids"]):
        raise RuntimeError("Paired artifacts do not have identical ordered sample IDs")
    if not np.array_equal(baseline["sequences"], eadom["sequences"]):
        raise RuntimeError("Paired artifacts do not have identical sequence IDs")
    if not np.array_equal(baseline["common_classes"], eadom["common_classes"]):
        raise RuntimeError("Paired artifacts do not use the same common class set")
    baseline_confusions = np.asarray(baseline["confusions"], dtype=np.int64)
    eadom_confusions = np.asarray(eadom["confusions"], dtype=np.int64)
    if baseline_confusions.shape != eadom_confusions.shape:
        raise RuntimeError("Paired confusion arrays have different shapes")
    if not np.array_equal(
        baseline_confusions.sum(axis=2), eadom_confusions.sum(axis=2)
    ):
        raise RuntimeError("Paired artifacts do not contain identical per-image GT counts")

    unit_type, unit_names, unit_indices, limitation = _units(baseline["sequences"])
    unit_baseline = np.stack(
        [baseline_confusions[indices].sum(axis=0) for indices in unit_indices]
    )
    unit_eadom = np.stack(
        [eadom_confusions[indices].sum(axis=0) for indices in unit_indices]
    )
    unit_count = len(unit_names)
    if unit_count == 0:
        raise ValueError("No paired resampling units")
    rng = np.random.default_rng(args.seed)
    delta_values: dict[str, list[np.ndarray]] = {metric: [] for metric in PAIRED_METRICS}
    for start in range(0, args.samples, args.batch_size):
        count = min(args.batch_size, args.samples - start)
        draws = rng.integers(0, unit_count, size=(count, unit_count))
        frequencies = np.zeros((count, unit_count), dtype=np.int64)
        for row_index in range(count):
            frequencies[row_index] = np.bincount(
                draws[row_index], minlength=unit_count
            )
        sampled_baseline = np.einsum(
            "bu,uij->bij", frequencies, unit_baseline, optimize=True
        )
        sampled_eadom = np.einsum(
            "bu,uij->bij", frequencies, unit_eadom, optimize=True
        )
        for metric in PAIRED_METRICS:
            baseline_values = _batch_metric(
                sampled_baseline, metric, baseline["common_classes"].tolist()
            )
            eadom_values = _batch_metric(
                sampled_eadom, metric, baseline["common_classes"].tolist()
            )
            delta_values[metric].append(eadom_values - baseline_values)

    full_baseline = baseline_confusions.sum(axis=0)
    full_eadom = eadom_confusions.sum(axis=0)
    common_classes = [str(value) for value in baseline["common_classes"].tolist()]
    results: list[dict[str, Any]] = []
    for metric in PAIRED_METRICS:
        baseline_value = _metric_full(full_baseline, metric, common_classes)
        eadom_value = _metric_full(full_eadom, metric, common_classes)
        point_delta = (
            None
            if baseline_value is None or eadom_value is None
            else float(eadom_value - baseline_value)
        )
        if metric == "common_supported_mIoU":
            positive_units = unit_count
        else:
            class_name = metric.split("/", 1)[0]
            index = CLASS_INDEX[class_name]
            positive_units = int(np.count_nonzero(unit_baseline[:, index, :].sum(axis=1)))
        combined = np.concatenate(delta_values[metric])
        finite = combined[np.isfinite(combined)]
        status = "PASS"
        reason = None
        if unit_count < 2:
            status = "INSUFFICIENT_SUPPORT"
            reason = f"only {unit_count} independent {unit_type} unit"
        elif positive_units < 2:
            status = "INSUFFICIENT_SUPPORT"
            reason = f"class GT occurs in only {positive_units} {unit_type} units"
        elif finite.size < args.samples * 0.9:
            status = "INSUFFICIENT_SUPPORT"
            reason = f"only {finite.size}/{args.samples} bootstrap deltas were defined"
        lower = upper = None
        if status == "PASS":
            lower, upper = (float(value) for value in np.percentile(finite, [2.5, 97.5]))
        results.append(
            {
                "dataset": args.dataset,
                "metric": metric,
                "b0_e0": baseline_value,
                "eadom": eadom_value,
                "delta_eadom_minus_b0_e0": point_delta,
                "ci95_lower": lower,
                "ci95_upper": upper,
                "status": status,
                "reason": reason,
                "resampling_unit": unit_type,
                "unit_count": unit_count,
                "positive_unit_count": positive_units,
                "bootstrap_samples": args.samples,
                "bootstrap_seed": args.seed,
            }
        )

    output = {
        "schema_version": "adom-paper-eval-paired-bootstrap-v1",
        "dataset": args.dataset,
        "resampling_unit": unit_type,
        "unit_count": unit_count,
        "unit_names": unit_names,
        "bootstrap_samples": args.samples,
        "bootstrap_seed": args.seed,
        "limitation": limitation,
        "common_classes": common_classes,
        "results": results,
        "paired_sample_count": int(baseline_confusions.shape[0]),
        "baseline_artifact": str(args.baseline.resolve()),
        "eadom_artifact": str(args.eadom.resolve()),
    }
    output_dir = args.output_dir.resolve() / "metrics"
    output_json = output_dir / f"{args.dataset}__paired_bootstrap.json"
    output_csv = output_dir / f"{args.dataset}__paired_bootstrap.csv"
    if output_json.exists() or output_csv.exists():
        raise FileExistsError(f"Refusing to overwrite paired bootstrap: {args.dataset}")
    write_json(output_json, output)
    write_dict_csv(
        output_csv,
        results,
        (
            "dataset",
            "metric",
            "b0_e0",
            "eadom",
            "delta_eadom_minus_b0_e0",
            "ci95_lower",
            "ci95_upper",
            "status",
            "reason",
            "resampling_unit",
            "unit_count",
            "positive_unit_count",
            "bootstrap_samples",
            "bootstrap_seed",
        ),
    )
    return output


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sequence-aware paired bootstrap for E-ADOM minus B0-E0"
    )
    parser.add_argument("--dataset", required=True, choices=("rellis", "korean"))
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--eadom", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--batch-size", type=int, default=250)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    result = bootstrap(parse_args(argv))
    print(json.dumps(result["results"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
