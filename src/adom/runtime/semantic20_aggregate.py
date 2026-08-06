from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from adom.data.io import write_json


REQUIRED_PAIRED_SEEDS = (42, 43, 44)


def _load_summary(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "experiment" not in payload or "seed" not in payload or "models" not in payload:
        raise ValueError(f"Invalid Semantic20 summary: {path}")
    return payload


def _by_seed(paths: list[Path]) -> dict[int, dict[str, Any]]:
    values: dict[int, dict[str, Any]] = {}
    for path in paths:
        payload = _load_summary(path)
        seed = int(payload["seed"])
        if seed in values:
            raise ValueError(f"Duplicate seed {seed} in paired summaries")
        values[seed] = payload
    if tuple(sorted(values)) != REQUIRED_PAIRED_SEEDS:
        raise ValueError(
            f"Expected paired seeds {REQUIRED_PAIRED_SEEDS}, got {tuple(sorted(values))}"
        )
    dataset_digests = {
        value.get("dataset_contract", {}).get("dataset_content_sha256")
        for value in values.values()
    }
    if None in dataset_digests or len(dataset_digests) != 1:
        raise ValueError(
            "All seeds on one side must use one recorded dataset content digest"
        )
    for seed, value in values.items():
        if value.get("gate") != "full":
            raise ValueError(f"Seed {seed} is not a full training summary")
        model_names = {str(model["model"]) for model in value["models"]}
        if model_names != {"b0", "b2"}:
            raise ValueError(
                f"Seed {seed} must contain exactly paired B0/B2, got {model_names}"
            )
    return values


def _model_metrics(summary: dict[str, Any]) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for model in summary["models"]:
        selected = model.get("validation_selection", {})
        metrics = selected.get("metrics", {})
        output[str(model["model"])] = {
            str(key): float(value)
            for key, value in metrics.items()
            if isinstance(value, (int, float))
        }
    return output


def _stats(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "values": [float(value) for value in array],
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
    }


def _one_metric_ending_with(metrics: dict[str, Any], suffix: str) -> dict[str, Any]:
    matches = [value for key, value in metrics.items() if key.endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one paired metric ending with {suffix!r}, found {len(matches)}"
        )
    return matches[0]


def aggregate_paired_runs(
    baseline_paths: list[Path],
    candidate_paths: list[Path],
) -> dict[str, Any]:
    baseline = _by_seed(baseline_paths)
    candidate = _by_seed(candidate_paths)
    baseline_experiments = {value["experiment"] for value in baseline.values()}
    candidate_experiments = {value["experiment"] for value in candidate.values()}
    if len(baseline_experiments) != 1 or len(candidate_experiments) != 1:
        raise ValueError("Each side of a paired comparison must use one experiment")

    result: dict[str, Any] = {
        "schema_version": "semantic20-paired-seeds-v1",
        "seeds": list(REQUIRED_PAIRED_SEEDS),
        "baseline_experiment": next(iter(baseline_experiments)),
        "candidate_experiment": next(iter(candidate_experiments)),
        "models": {},
    }
    baseline_metrics = {seed: _model_metrics(value) for seed, value in baseline.items()}
    candidate_metrics = {seed: _model_metrics(value) for seed, value in candidate.items()}
    model_names = set.intersection(
        *(set(value) for value in baseline_metrics.values()),
        *(set(value) for value in candidate_metrics.values()),
    )
    if not model_names:
        raise ValueError("Paired summaries have no common model")

    for model in sorted(model_names):
        common_metrics = set.intersection(
            *(set(value[model]) for value in baseline_metrics.values()),
            *(set(value[model]) for value in candidate_metrics.values()),
        )
        metric_rows: dict[str, Any] = {}
        for metric in sorted(common_metrics):
            baseline_values = [
                baseline_metrics[seed][model][metric] for seed in REQUIRED_PAIRED_SEEDS
            ]
            candidate_values = [
                candidate_metrics[seed][model][metric] for seed in REQUIRED_PAIRED_SEEDS
            ]
            deltas = [
                candidate_value - baseline_value
                for baseline_value, candidate_value in zip(
                    baseline_values, candidate_values, strict=True
                )
            ]
            metric_rows[metric] = {
                "baseline": _stats(baseline_values),
                "candidate": _stats(candidate_values),
                "paired_delta": _stats(deltas),
            }
        augmented = _one_metric_ending_with(metric_rows, "mIoU/AugmentedRisk2")
        overall = _one_metric_ending_with(metric_rows, "mIoU/ValSupported13")
        pole = _one_metric_ending_with(metric_rows, "IoU/pole")
        augmented_pass = augmented["paired_delta"]["mean"] >= 3.0
        overall_pass = overall["paired_delta"]["mean"] >= -1.0
        pole_positive_count = sum(
            value > 0.0 for value in pole["paired_delta"]["values"]
        )
        pole_pass = pole_positive_count >= 2
        result["models"][model] = {
            "metrics": metric_rows,
            "success_gates": {
                "AugmentedRisk2_delta_mean_at_least_3pp": augmented_pass,
                "ValSupported13_delta_mean_at_least_minus_1pp": overall_pass,
                "pole_improved_in_at_least_2_seeds": pole_pass,
                "pole_positive_seed_count": pole_positive_count,
                "primary_gates_passed": augmented_pass and overall_pass and pole_pass,
                "note": "Inspect log and barrier deltas for non-degradation before GO.",
            },
        }
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate paired Semantic20 seeds as mean, std, and paired delta"
    )
    parser.add_argument("--baseline", nargs=3, type=Path, required=True)
    parser.add_argument("--candidate", nargs=3, type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = aggregate_paired_runs(args.baseline, args.candidate)
    write_json(args.output.resolve(), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
