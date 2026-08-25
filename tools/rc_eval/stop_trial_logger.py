from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal

from _common import read_json, utc_now, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Stop a previously started read-only rosbag logger")
    parser.add_argument("--trial-dir", required=True, type=Path)
    parser.add_argument("--execute-read-only", action="store_true")
    args = parser.parse_args()
    path = args.trial_dir / "logger_state.json"
    state = read_json(path)
    if not args.execute_read_only:
        print(json.dumps({**state, "requested_action": "DRY_RUN_STOP"}, indent=2))
        return
    pid = state.get("pid")
    if state.get("status") != "RECORDING_READ_ONLY" or not isinstance(pid, int):
        raise SystemExit("no active read-only logger PID")
    os.kill(pid, signal.SIGINT)
    state.update({"status": "STOP_SIGNAL_SENT", "stopped_at_utc": utc_now()})
    write_json(path, state)
    print(json.dumps(state, indent=2))


if __name__ == "__main__":
    main()
