from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess

from _common import utc_now, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only ROS 2 graph inspection; never publishes")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--execute-read-only", action="store_true", help="run ros2 list/info commands")
    args = parser.parse_args()
    ros2 = shutil.which("ros2")
    commands = [
        ["ros2", "node", "list"],
        ["ros2", "topic", "list", "-t"],
    ]
    results = []
    if args.execute_read_only and ros2:
        for command in commands:
            run = subprocess.run(command, capture_output=True, text=True, timeout=30, check=False)
            results.append({"command": command, "returncode": run.returncode, "stdout": run.stdout, "stderr": run.stderr})
        status = "PASS" if all(row["returncode"] == 0 for row in results) else "BLOCKED_ROS_GRAPH_COMMAND_FAILED"
    elif args.execute_read_only:
        status = "BLOCKED_ROS2_NOT_INSTALLED"
    else:
        status = "DRY_RUN"
    payload = {
        "schema_version": "adom-rc-ros-graph-v1",
        "captured_at_utc": utc_now(),
        "status": status,
        "ros2_executable": ros2,
        "publish_commands_executed": 0,
        "commands": results,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "ros_graph.json", payload)
    lines = [
        "# ROS graph inspection",
        "",
        f"- Status: `{status}`",
        f"- ros2 executable: `{ros2 or 'N/A'}`",
        "- Publisher or motor commands executed: **0**",
        "",
    ]
    for result in results:
        lines.extend([f"## `{' '.join(result['command'])}`", "", "```text", result["stdout"].rstrip(), "```", ""])
    (args.output_dir / "ros_graph_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
