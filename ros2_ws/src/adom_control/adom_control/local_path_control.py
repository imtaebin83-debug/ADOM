from __future__ import annotations

import json
import math

from geometry_msgs.msg import Twist
from nav_msgs.msg import Path
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu
from std_msgs.msg import String

from adom.autonomy import PathControlConfig, control_local_path


def message_stamp_ns(message, fallback_ns: int) -> int:
    stamp_ns = int(message.header.stamp.sec) * 1_000_000_000 + int(
        message.header.stamp.nanosec
    )
    return stamp_ns if stamp_ns > 0 else fallback_ns


class LocalPathControlNode(Node):
    """Track a robot-frame local path without using GPS as a control input."""

    def __init__(self) -> None:
        super().__init__("local_path_control")
        defaults = {
            "local_path_topic": "/adom/navigation/local_path",
            "imu_topic": "/zed/zed_node/imu/data",
            "cmd_vel_topic": "/cmd_vel",
            "status_topic": "/adom/control/local_path_status",
            "base_frame": "base_link",
            "control_rate_hz": 20.0,
            "path_timeout_sec": 0.25,
            "imu_timeout_sec": 0.25,
            "wheelbase_m": 0.33,
            "lookahead_m": 0.80,
            "max_steering_deg": 20.0,
            "max_speed_mps": 0.25,
            "min_speed_mps": 0.06,
            "curvature_speed_gain": 1.5,
            "speed_kp": 0.0,
            "path_stop_distance_m": 0.50,
            "path_slow_distance_m": 2.0,
            "imu_accel_bias_x_mps2": 0.0,
            "imu_max_abs_accel_mps2": 5.0,
            "imu_max_integration_dt_sec": 0.10,
            "imu_estimated_speed_limit_mps": 1.0,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        self.p = {name: self.get_parameter(name).value for name in defaults}
        if float(self.p["max_speed_mps"]) > 0.30:
            raise ValueError("local path controller max_speed_mps must be <= 0.30")
        self._config = PathControlConfig(
            wheelbase_m=float(self.p["wheelbase_m"]),
            lookahead_m=float(self.p["lookahead_m"]),
            max_steering_deg=float(self.p["max_steering_deg"]),
            max_speed_mps=float(self.p["max_speed_mps"]),
            min_speed_mps=float(self.p["min_speed_mps"]),
            curvature_speed_gain=float(self.p["curvature_speed_gain"]),
            speed_kp=float(self.p["speed_kp"]),
            path_stop_distance_m=float(self.p["path_stop_distance_m"]),
            path_slow_distance_m=float(self.p["path_slow_distance_m"]),
        )
        self._path_xy: np.ndarray | None = None
        self._path_source_ns: int | None = None
        self._last_path_ns: int | None = None
        self._last_imu_ns: int | None = None
        self._last_imu_stamp_ns: int | None = None
        self._estimated_speed_mps = 0.0
        self._acceleration_x_mps2 = 0.0
        self._yaw_rate_rps = 0.0

        self._cmd_pub = self.create_publisher(
            Twist, str(self.p["cmd_vel_topic"]), 10
        )
        self._status_pub = self.create_publisher(
            String, str(self.p["status_topic"]), 10
        )
        self.create_subscription(
            Path, str(self.p["local_path_topic"]), self._on_path, 1
        )
        self.create_subscription(
            Imu, str(self.p["imu_topic"]), self._on_imu, qos_profile_sensor_data
        )
        self.create_timer(1.0 / float(self.p["control_rate_hz"]), self._update)
        self.get_logger().warning(
            "Local path control starts fail-safe at zero until path and IMU arrive; GPS is logging-only."
        )

    def _on_path(self, message: Path) -> None:
        now_ns = self.get_clock().now().nanoseconds
        if message.header.frame_id not in ("", str(self.p["base_frame"])):
            self.get_logger().error(
                f"Rejected local path in {message.header.frame_id!r}; "
                f"expected {self.p['base_frame']!r}"
            )
            return
        self._path_xy = np.asarray(
            [[pose.pose.position.x, pose.pose.position.y] for pose in message.poses],
            dtype=np.float64,
        ).reshape(-1, 2)
        self._path_source_ns = message_stamp_ns(message, now_ns)
        self._last_path_ns = now_ns

    def _on_imu(self, message: Imu) -> None:
        now_ns = self.get_clock().now().nanoseconds
        stamp_ns = message_stamp_ns(message, now_ns)
        acceleration = float(message.linear_acceleration.x) - float(
            self.p["imu_accel_bias_x_mps2"]
        )
        acceleration_limit = float(self.p["imu_max_abs_accel_mps2"])
        acceleration = max(-acceleration_limit, min(acceleration_limit, acceleration))
        if self._last_imu_stamp_ns is not None:
            dt = (stamp_ns - self._last_imu_stamp_ns) / 1e9
            if 0.0 < dt <= float(self.p["imu_max_integration_dt_sec"]):
                self._estimated_speed_mps = max(
                    0.0,
                    min(
                        float(self.p["imu_estimated_speed_limit_mps"]),
                        self._estimated_speed_mps + acceleration * dt,
                    ),
                )
        self._acceleration_x_mps2 = acceleration
        self._yaw_rate_rps = float(message.angular_velocity.z)
        self._last_imu_stamp_ns = stamp_ns
        self._last_imu_ns = now_ns

    @staticmethod
    def _age(now_ns: int, stamp_ns: int | None) -> float:
        return math.inf if stamp_ns is None else (now_ns - stamp_ns) / 1e9

    def _publish(self, speed_mps: float, steering_rad: float) -> None:
        message = Twist()
        message.linear.x = float(speed_mps)
        message.angular.z = (
            0.0
            if speed_mps <= 0.0
            else speed_mps
            * math.tan(steering_rad)
            / float(self.p["wheelbase_m"])
        )
        self._cmd_pub.publish(message)

    def _publish_status(self, state: str, **fields) -> None:
        message = String()
        message.data = json.dumps({"state": state, **fields}, sort_keys=True)
        self._status_pub.publish(message)

    def _stop(self, reason: str, **fields) -> None:
        self._publish(0.0, 0.0)
        self._publish_status("stopped", reason=reason, **fields)

    def _update(self) -> None:
        now_ns = self.get_clock().now().nanoseconds
        path_age = self._age(now_ns, self._last_path_ns)
        imu_age = self._age(now_ns, self._last_imu_ns)
        common = {
            "path_age_sec": None if not math.isfinite(path_age) else round(path_age, 3),
            "imu_age_sec": None if not math.isfinite(imu_age) else round(imu_age, 3),
        }
        if self._path_xy is None or path_age > float(self.p["path_timeout_sec"]):
            self._stop("path_watchdog", **common)
            return
        if len(self._path_xy) == 0:
            self._stop("empty_path", **common)
            return
        if imu_age > float(self.p["imu_timeout_sec"]):
            self._stop("imu_watchdog", **common)
            return
        try:
            command = control_local_path(
                self._path_xy, self._estimated_speed_mps, self._config
            )
        except Exception as error:
            self.get_logger().error(f"Local path control failed: {error}")
            self._stop("controller_error", message=str(error), **common)
            return
        self._publish(command.speed_mps, command.steering_rad)
        source_to_command_ms = (
            None
            if self._path_source_ns is None or now_ns < self._path_source_ns
            else round((now_ns - self._path_source_ns) / 1e6, 2)
        )
        self._publish_status(
            "tracking",
            **common,
            speed_command_mps=round(command.speed_mps, 3),
            estimated_speed_mps=round(self._estimated_speed_mps, 3),
            acceleration_x_mps2=round(self._acceleration_x_mps2, 3),
            yaw_rate_rps=round(self._yaw_rate_rps, 3),
            steering_deg=round(math.degrees(command.steering_rad), 2),
            target_x_m=round(command.target_x_m, 3),
            target_y_m=round(command.target_y_m, 3),
            available_path_m=round(command.available_path_m, 3),
            source_to_command_ms=source_to_command_ms,
        )

    def destroy_node(self):
        if rclpy.ok(context=self.context):
            self._publish(0.0, 0.0)
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LocalPathControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
