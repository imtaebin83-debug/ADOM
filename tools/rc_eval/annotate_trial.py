from __future__ import annotations

import argparse
from pathlib import Path

from _common import utc_now, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a human RC trial annotation")
    parser.add_argument("--trial-dir", required=True, type=Path)
    parser.add_argument("--stop-decision-observed", choices=("true", "false", "unknown"), required=True)
    parser.add_argument("--hazard-detection-observed", choices=("true", "false", "unknown"), default="unknown")
    parser.add_argument("--physical-stop-before-boundary", choices=("true", "false", "unknown"), default="unknown")
    parser.add_argument("--trial-completed", choices=("true", "false"), required=True)
    parser.add_argument("--emergency-intervention", choices=("true", "false"), required=True)
    parser.add_argument("--decision-latency-s", type=float)
    parser.add_argument("--detection-to-stop-latency-s", type=float)
    parser.add_argument("--braking-latency-s", type=float)
    parser.add_argument("--first-hazard-detection-s", type=float)
    parser.add_argument("--first-stop-command-s", type=float)
    parser.add_argument("--physical-stop-time-s", type=float)
    parser.add_argument("--dropped-frame-count", type=int)
    parser.add_argument("--exclusion-reason", default="")
    parser.add_argument("--notes", default="")
    args = parser.parse_args()
    convert = lambda value: None if value == "unknown" else value == "true"
    payload = {
        "schema_version": "adom-rc-human-annotation-v1",
        "annotated_at_utc": utc_now(),
        "stop_decision_observed": convert(args.stop_decision_observed),
        "hazard_detection_observed": convert(args.hazard_detection_observed),
        "physical_stop_before_boundary": convert(args.physical_stop_before_boundary),
        "trial_completed": convert(args.trial_completed),
        "emergency_intervention": convert(args.emergency_intervention),
        "decision_latency_s": args.decision_latency_s,
        "detection_to_stop_latency_s": args.detection_to_stop_latency_s,
        "braking_latency_s": args.braking_latency_s,
        "first_hazard_detection_s": args.first_hazard_detection_s,
        "first_stop_command_s": args.first_stop_command_s,
        "physical_stop_time_s": args.physical_stop_time_s,
        "dropped_frame_count": args.dropped_frame_count,
        "exclusion_reason": args.exclusion_reason,
        "notes": args.notes,
    }
    write_json(args.trial_dir / "human_annotation.json", payload)
    print(args.trial_dir / "human_annotation.json")


if __name__ == "__main__":
    main()
