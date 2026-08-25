from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any

from _common import (
    parse_bool,
    parse_optional_bool,
    parse_optional_float,
    read_json,
    utc_now,
    validate_metadata,
    wilson_interval,
    write_csv,
    write_json,
)


def _percent(value: float | None) -> float | None:
    return None if value is None else 100.0 * value


def _rate(successes: int, total: int) -> dict[str, Any]:
    lower, upper = wilson_interval(successes, total)
    return {
        "numerator": successes,
        "denominator": total,
        "percent": None if total == 0 else 100.0 * successes / total,
        "wilson95_lower_percent": _percent(lower),
        "wilson95_upper_percent": _percent(upper),
    }


def _latency_summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean_s": None, "median_s": None, "min_s": None, "max_s": None}
    return {
        "count": len(values),
        "mean_s": mean(values),
        "median_s": median(values),
        "min_s": min(values),
        "max_s": max(values),
    }


def _numeric_summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": mean(values),
        "median": median(values),
        "min": min(values),
        "max": max(values),
    }


def analyze(experiment_root: Path, output_dir: Path) -> dict[str, Any]:
    trials_root = experiment_root / "trials"
    trial_rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    counts: Counter[str] = Counter()
    latencies: dict[str, list[float]] = {
        "decision": [],
        "detection_to_stop": [],
        "braking": [],
    }
    event_times: dict[str, list[float]] = {
        "first_hazard_detection": [],
        "first_stop_command": [],
        "physical_stop": [],
    }
    perception_detection_successes = 0
    perception_detection_total = 0
    hazard_present_trial_count = 0
    dropped_frame_counts: list[float] = []
    for trial_dir in sorted(path for path in trials_root.glob("*") if path.is_dir()):
        metadata_path = trial_dir / "metadata.json"
        annotation_path = trial_dir / "human_annotation.json"
        if not metadata_path.is_file() or not annotation_path.is_file():
            errors.append({"trial_id": trial_dir.name, "error": "missing metadata.json or human_annotation.json"})
            continue
        try:
            metadata = read_json(metadata_path)
            annotation = read_json(annotation_path)
            validation_errors = validate_metadata(metadata)
        except Exception as error:
            errors.append({"trial_id": trial_dir.name, "error": f"invalid JSON: {error}"})
            continue
        if validation_errors:
            errors.extend({"trial_id": trial_dir.name, "error": value} for value in validation_errors)
            continue
        try:
            excluded = bool(metadata.get("exclusion_reason") or annotation.get("exclusion_reason"))
            hazard_present = parse_bool(metadata["hazard_present"], field="hazard_present")
            physical = parse_optional_bool(annotation.get("physical_stop_before_boundary"), field="physical_stop_before_boundary")
            stop_decision = parse_optional_bool(annotation.get("stop_decision_observed"), field="stop_decision_observed")
            trial_completed = parse_optional_bool(annotation.get("trial_completed"), field="trial_completed")
            intervention = parse_optional_bool(
                annotation.get("emergency_intervention", metadata.get("emergency_intervention")),
                field="emergency_intervention",
            )
            detection = parse_optional_bool(annotation.get("hazard_detection_observed"), field="hazard_detection_observed")
        except ValueError as error:
            errors.append({"trial_id": trial_dir.name, "error": str(error)})
            continue
        if hazard_present:
            hazard_present_trial_count += 1
            if detection is not None:
                perception_detection_total += 1
                perception_detection_successes += int(detection)
        if physical is not None:
            stopped = physical
            basis = "physical_stop_before_boundary"
        elif stop_decision is not None:
            stopped = stop_decision
            basis = "stop_command_proxy"
        else:
            stopped = None
            basis = "N/A"
        if excluded:
            outcome = "EXCLUDED"
        elif trial_completed is False:
            outcome = "INTERRUPTED"
        elif intervention is True:
            outcome = "INTERVENTION"
        elif stopped is None:
            outcome = "INSUFFICIENT_DATA"
        elif hazard_present and stopped:
            outcome = "TP"
        elif hazard_present and not stopped:
            outcome = "FN"
        elif not hazard_present and stopped:
            outcome = "FP"
        else:
            outcome = "TN"
        counts[outcome] += 1
        for key, target in (
            ("decision_latency_s", "decision"),
            ("detection_to_stop_latency_s", "detection_to_stop"),
            ("braking_latency_s", "braking"),
        ):
            value = parse_optional_float(annotation.get(key), field=key)
            if value is not None:
                latencies[target].append(value)
        for key, target in (
            ("first_hazard_detection_s", "first_hazard_detection"),
            ("first_stop_command_s", "first_stop_command"),
            ("physical_stop_time_s", "physical_stop"),
        ):
            value = parse_optional_float(annotation.get(key), field=key)
            if value is not None:
                event_times[target].append(value)
        dropped_frames = parse_optional_float(annotation.get("dropped_frame_count"), field="dropped_frame_count")
        if dropped_frames is not None:
            dropped_frame_counts.append(dropped_frames)
        trial_rows.append(
            {
                "trial_id": metadata["trial_id"],
                "model": metadata["model"],
                "scene_id": metadata["scene_id"],
                "hazard_type": metadata["hazard_type"],
                "hazard_present": hazard_present,
                "outcome": outcome,
                "outcome_basis": basis,
                "trial_completed": trial_completed,
                "human_intervention": intervention,
                "hazard_detection_observed": detection,
                "decision_latency_s": annotation.get("decision_latency_s"),
                "detection_to_stop_latency_s": annotation.get("detection_to_stop_latency_s"),
                "braking_latency_s": annotation.get("braking_latency_s"),
                "first_hazard_detection_s": annotation.get("first_hazard_detection_s"),
                "first_stop_command_s": annotation.get("first_stop_command_s"),
                "physical_stop_time_s": annotation.get("physical_stop_time_s"),
                "dropped_frame_count": annotation.get("dropped_frame_count"),
            }
        )
    tp, fn, fp, tn = (counts[name] for name in ("TP", "FN", "FP", "TN"))
    analyzed = tp + fn + fp + tn
    completion_count = sum(
        1 for row in trial_rows if str(row["trial_completed"]).lower() == "true"
    )
    intervention_count = sum(
        1 for row in trial_rows if str(row["human_intervention"]).lower() == "true"
    )
    summary = {
        "schema_version": "adom-rc-eval-analysis-v1",
        "generated_at_utc": utc_now(),
        "experiment_root": str(experiment_root.resolve()),
        "trial_counts": dict(counts),
        "analyzed_trial_count": analyzed,
        "stop_success_rate": _rate(tp, tp + fn),
        "miss_rate": _rate(fn, tp + fn),
        "false_stop_rate": _rate(fp, fp + tn),
        "trial_completion_rate": _rate(completion_count, len(trial_rows)),
        "human_intervention_rate": _rate(intervention_count, len(trial_rows)),
        "perception_hazard_detection_rate": _rate(perception_detection_successes, perception_detection_total),
        "perception_detection_label_coverage": _rate(perception_detection_total, hazard_present_trial_count),
        "latencies": {name: _latency_summary(values) for name, values in latencies.items()},
        "event_times_from_trial_start": {name: _latency_summary(values) for name, values in event_times.items()},
        "dropped_frame_count": _numeric_summary(dropped_frame_counts),
        "limitations": [
            "Wilson intervals treat trials as independent; repeated trials in one scene may be correlated.",
            "When physical stop is unavailable, stop-command success is reported only as a proxy.",
            "This logger does not publish motor, servo, autonomy, or emergency-stop commands.",
        ],
        "errors": errors,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = list(trial_rows[0]) if trial_rows else ["trial_id", "outcome"]
    write_csv(output_dir / "trial_outcomes.csv", trial_rows, fields)
    write_csv(output_dir / "validation_errors.csv", errors, ("trial_id", "error"))
    write_json(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze human-annotated RC Go/Stop trials")
    parser.add_argument("--experiment-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    summary = analyze(args.experiment_root, args.output_dir)
    print(f"analyzed {summary['analyzed_trial_count']} trials")


if __name__ == "__main__":
    main()
