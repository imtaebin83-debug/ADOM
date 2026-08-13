import json
import os
import re
import shutil
import signal
import subprocess
import threading
from datetime import datetime
from pathlib import Path

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from std_msgs.msg import String


DEFAULT_TOPIC_REGEX = r"^/zed(/.*)?/rgb(/.*)?$"


def directory_size(path):
    """Return the apparent size of all regular files below path."""
    total = 0
    try:
        entries = path.rglob("*")
        for entry in entries:
            try:
                if entry.is_file():
                    total += entry.stat().st_size
            except OSError:
                continue
    except OSError:
        return total
    return total


class DataRecorder(Node):
    """Toggle a bounded rosbag recording from a gamepad button."""

    def __init__(self):
        super().__init__("data_recorder")
        defaults = {
            "joy_topic": "/joy",
            "record_button": 4,
            "capture_root": "data/captures",
            "topic_regex": DEFAULT_TOPIC_REGEX,
            "record_mask": False,
            "mask_topic": "/adom/perception/semantic20_mask_evidence",
            "record_evidence": False,
            "evidence_image_topic": "/zed/zed_node/rgb/color/rect/image",
            "max_size_gb": 10.0,
            "size_check_period_sec": 0.5,
            "bag_split_size_mb": 1024,
            "status_topic": "/adom/recording/status",
            "auto_start": False,
            "session_prefix": "",
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        self.p = {name: self.get_parameter(name).value for name in defaults}
        self._validate_parameters()
        self._topic_regex = self._effective_topic_regex()

        configured_root = Path(str(self.p["capture_root"]).strip() or "data/captures")
        repo_root = Path(os.environ.get("ADOM_REPO_ROOT", "."))
        self._capture_root = (
            configured_root
            if configured_root.is_absolute()
            else repo_root / configured_root
        ).resolve()
        self._capture_root.mkdir(parents=True, exist_ok=True)

        self._lock = threading.RLock()
        self._process = None
        self._session_dir = None
        self._started_at = None
        self._previous_buttons = []
        self._layout_error_reported = False
        self._stop_reason = "not_started"
        self._max_bytes = int(float(self.p["max_size_gb"]) * 1_000_000_000)

        self._status_pub = self.create_publisher(
            String, str(self.p["status_topic"]), 10
        )
        self.create_subscription(
            Joy, str(self.p["joy_topic"]), self._on_joy, 10
        )
        self.create_timer(float(self.p["size_check_period_sec"]), self._monitor)
        self.create_timer(1.0, self._publish_status)
        self.get_logger().info(
            f"Data recorder ready: Y/button {self.p['record_button']} toggles recording; "
            f"limit={self.p['max_size_gb']} GB, root={self._capture_root}, "
            f"record_mask={bool(self.p['record_mask'])}, "
            f"record_evidence={bool(self.p['record_evidence'])}"
        )
        if bool(self.p["auto_start"]):
            self.start_recording()

    def _validate_parameters(self):
        if int(self.p["record_button"]) < 0:
            raise ValueError("record_button must be non-negative")
        if float(self.p["max_size_gb"]) <= 0.0:
            raise ValueError("max_size_gb must be positive")
        if float(self.p["size_check_period_sec"]) <= 0.0:
            raise ValueError("size_check_period_sec must be positive")
        if int(self.p["bag_split_size_mb"]) <= 0:
            raise ValueError("bag_split_size_mb must be positive")
        if not str(self.p["topic_regex"]).strip():
            raise ValueError("topic_regex must not be empty")
        if bool(self.p["record_mask"]) and not str(self.p["mask_topic"]).strip():
            raise ValueError("mask_topic must not be empty when record_mask is true")
        if bool(self.p["record_evidence"]):
            if not bool(self.p["record_mask"]):
                raise ValueError("record_evidence requires record_mask=true")
            if not str(self.p["evidence_image_topic"]).strip():
                raise ValueError("evidence_image_topic must not be empty")

    def _effective_topic_regex(self):
        base_regex = str(self.p["topic_regex"]).strip()
        extra_topics = []
        if bool(self.p["record_mask"]):
            extra_topics.append(str(self.p["mask_topic"]).strip())
        if bool(self.p["record_evidence"]):
            extra_topics.append(str(self.p["evidence_image_topic"]).strip())
        regex = base_regex
        for topic in extra_topics:
            regex = rf"(?:{regex})|(?:^{re.escape(topic)}$)"
        return regex

    @staticmethod
    def _rising(buttons, previous, index):
        old = previous[index] if len(previous) > index else 0
        return bool(buttons[index]) and not bool(old)

    def _on_joy(self, msg):
        index = int(self.p["record_button"])
        if len(msg.buttons) <= index:
            if not self._layout_error_reported:
                self.get_logger().error(
                    f"Joy has {len(msg.buttons)} buttons, but record_button is {index}"
                )
                self._layout_error_reported = True
            self._previous_buttons = list(msg.buttons)
            return
        self._layout_error_reported = False

        pressed = self._rising(msg.buttons, self._previous_buttons, index)
        self._previous_buttons = list(msg.buttons)
        if not pressed:
            return

        with self._lock:
            recording = self._process is not None
        if recording:
            self.stop_recording("button")
        else:
            self.start_recording()

    def start_recording(self):
        ros2 = shutil.which("ros2")
        if ros2 is None:
            self.get_logger().error("Cannot start recording: ros2 executable not found")
            return

        timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%z")
        prefix = str(self.p["session_prefix"]).strip()
        if prefix:
            timestamp = f"{prefix}_{timestamp}"
        session_dir = self._unique_session_dir(timestamp)
        bag_path = session_dir / "rosbag"
        session_dir.mkdir(parents=True)
        command = [
            ros2,
            "bag",
            "record",
            "--output",
            str(bag_path),
            "--regex",
            self._topic_regex,
            "--max-bag-size",
            str(int(self.p["bag_split_size_mb"]) * 1_000_000),
            "--disable-keyboard-controls",
        ]

        try:
            process = subprocess.Popen(command, start_new_session=True)
        except OSError as exc:
            self.get_logger().error(f"Cannot start rosbag: {exc}")
            try:
                session_dir.rmdir()
            except OSError:
                pass
            return

        with self._lock:
            self._process = process
            self._session_dir = session_dir
            self._started_at = datetime.now().astimezone()
            self._stop_reason = "recording"
        self._write_session_info("recording")
        self.get_logger().warning(f"RECORDING STARTED: {session_dir}")
        self._publish_status()

    def _unique_session_dir(self, timestamp):
        candidate = self._capture_root / timestamp
        suffix = 1
        while candidate.exists():
            candidate = self._capture_root / f"{timestamp}_{suffix:02d}"
            suffix += 1
        return candidate

    def stop_recording(self, reason):
        with self._lock:
            process = self._process
        if process is None:
            return

        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGINT)
                process.wait(timeout=15.0)
            except subprocess.TimeoutExpired:
                self.get_logger().error(
                    "rosbag did not stop after SIGINT; sending SIGTERM"
                )
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    self.get_logger().error(
                        "rosbag did not stop after SIGTERM; sending SIGKILL"
                    )
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5.0)
            except ProcessLookupError:
                pass

        with self._lock:
            self._stop_reason = reason
        self._write_session_info("stopped")
        size_gb = self._current_size() / 1_000_000_000
        self.get_logger().warning(
            f"RECORDING STOPPED ({reason}): {size_gb:.3f} GB in {self._session_dir}"
        )
        with self._lock:
            self._process = None
            self._session_dir = None
            self._started_at = None
        self._publish_status()

    def _monitor(self):
        with self._lock:
            process = self._process
        if process is None:
            return
        return_code = process.poll()
        if return_code is not None:
            self.get_logger().error(
                f"rosbag exited unexpectedly with code {return_code}"
            )
            self.stop_recording(f"rosbag_exit_{return_code}")
            return
        if self._current_size() >= self._max_bytes:
            self.get_logger().warning(
                f"{self.p['max_size_gb']} GB session limit reached; stopping recording"
            )
            self.stop_recording("size_limit")

    def _current_size(self):
        with self._lock:
            session_dir = self._session_dir
        return 0 if session_dir is None else directory_size(session_dir)

    def _write_session_info(self, state):
        with self._lock:
            session_dir = self._session_dir
            started_at = self._started_at
            reason = self._stop_reason
        if session_dir is None:
            return
        info = {
            "state": state,
            "started_at": started_at.isoformat() if started_at else None,
            "updated_at": datetime.now().astimezone().isoformat(),
            "stop_reason": None if state == "recording" else reason,
            "topic_regex": self._topic_regex,
            "record_mask": bool(self.p["record_mask"]),
            "record_evidence": bool(self.p["record_evidence"]),
            "mask_topic": (
                str(self.p["mask_topic"]) if bool(self.p["record_mask"]) else None
            ),
            "evidence_image_topic": (
                str(self.p["evidence_image_topic"])
                if bool(self.p["record_evidence"])
                else None
            ),
            "max_size_bytes": self._max_bytes,
            "size_bytes": directory_size(session_dir),
        }
        try:
            (session_dir / "session.json").write_text(
                json.dumps(info, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        except OSError as exc:
            self.get_logger().error(f"Failed to write session metadata: {exc}")

    def _publish_status(self):
        with self._lock:
            recording = self._process is not None
            session_dir = self._session_dir
            reason = self._stop_reason
        msg = String()
        msg.data = json.dumps(
            {
                "recording": recording,
                "session": str(session_dir) if session_dir else None,
                "size_bytes": self._current_size(),
                "max_size_bytes": self._max_bytes,
                "reason": reason,
                "record_mask": bool(self.p["record_mask"]),
                "record_evidence": bool(self.p["record_evidence"]),
            },
            separators=(",", ":"),
        )
        self._status_pub.publish(msg)

    def destroy_node(self):
        self.stop_recording("node_shutdown")
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = DataRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
