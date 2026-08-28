from __future__ import annotations

import argparse
import random
from pathlib import Path

from _common import TRIAL_FIELDS, write_csv


def build_plan(seed: int, repetitions: int) -> list[dict[str, object]]:
    rng = random.Random(seed)
    conditions = [
        (model, present)
        for model in ("b0-e0", "eadom")
        for present in (True, False)
        for _ in range(repetitions)
    ]
    rng.shuffle(conditions)
    rows: list[dict[str, object]] = []
    counters = {(model, present): 0 for model, present in conditions}
    for order, (model, present) in enumerate(conditions, start=1):
        counters[(model, present)] += 1
        label = "present" if present else "absent"
        rows.append(
            {
                "trial_id": f"T{order:03d}_{model.replace('-', '')}_{label}",
                "order": order,
                "operator": "",
                "model": model,
                "checkpoint_sha256": "",
                "scene_id": "",
                "hazard_type": "log" if present else "none",
                "hazard_present": str(present).lower(),
                "start_position_marker": "",
                "obstacle_position_marker": "" if not present else "",
                "commanded_speed_mps": "",
                "battery_voltage": "",
                "lighting_weather_note": "",
                "rosbag_path": "",
                "video_path": "",
                "emergency_intervention": "false",
                "human_final_outcome": "",
                "exclusion_reason": "",
                "condition_repetition": counters[(model, present)],
                "randomization_seed": seed,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a randomized, balanced RC evaluation trial plan")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260825)
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("--repetitions must be positive")
    rows = build_plan(args.seed, args.repetitions)
    write_csv(args.output, rows, (*TRIAL_FIELDS, "condition_repetition", "randomization_seed"))
    print(f"wrote {len(rows)} trials to {args.output}")


if __name__ == "__main__":
    main()
