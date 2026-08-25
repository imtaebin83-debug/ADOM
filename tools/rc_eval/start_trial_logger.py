from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

from _common import read_json, utc_now, validate_metadata, write_json


def _load_topics(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml

        config = yaml.safe_load(text)
    except ImportError:
        config = json.loads(text)
    topics = config.get("topics", {})
    required = ("camera", "segmentation_mask", "hazard_detection", "go_stop_decision", "motor_command", "emergency_stop")
    missing = [name for name in required if not topics.get(name)]
    if missing:
        raise RuntimeError(f"BLOCKED_UNVERIFIED_TOPICS: {', '.join(missing)}")
    return sorted({str(topics[name]) for name in required})


def main() -> None:
    parser = argparse.ArgumentParser(description="Start a subscribe-only rosbag logger; dry-run by default")
    parser.add_argument("--trial-dir", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--execute-read-only", action="store_true")
    parser.add_argument("--acknowledge-no-publish", action="store_true")
    args = parser.parse_args()
    metadata = read_json(args.metadata)
    errors = validate_metadata(metadata)
    if errors:
        raise SystemExit("invalid metadata: " + "; ".join(errors))
    topics: list[str] = []
    topic_validation = "PASS"
    try:
        topics = _load_topics(args.config)
    except RuntimeError as error:
        topic_validation = str(error)
    if args.execute_read_only:
        if not args.acknowledge_no_publish:
            raise SystemExit("--execute-read-only requires --acknowledge-no-publish")
        if topic_validation != "PASS":
            raise SystemExit(topic_validation)
        if shutil.which("ros2") is None:
            raise SystemExit("BLOCKED_ROS2_NOT_INSTALLED")
    args.trial_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.trial_dir / "metadata.json", {**metadata, "logger_started_at_utc": utc_now()})
    state = {
        "schema_version": "adom-rc-logger-state-v1",
        "status": "DRY_RUN",
        "publish_commands_executed": 0,
        "pid": None,
        "command": None,
        "topic_validation": topic_validation,
    }
    if args.execute_read_only:
        bag_path = args.trial_dir / "rosbag"
        command = ["ros2", "bag", "record", "-o", str(bag_path), *topics]
        process = subprocess.Popen(command, stdout=(args.trial_dir / "rosbag.stdout.log").open("w"), stderr=subprocess.STDOUT)
        state.update({"status": "RECORDING_READ_ONLY", "pid": process.pid, "command": command, "started_at_utc": utc_now()})
    write_json(args.trial_dir / "logger_state.json", state)
    print(json.dumps(state, indent=2))


if __name__ == "__main__":
    main()
