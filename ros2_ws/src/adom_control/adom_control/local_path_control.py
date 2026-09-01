from __future__ import annotations

import json
import math

from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import Twist
from nav_msgs.msg import Path
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32, String

from adom.autonomy import (
    ImuSpeedEstimator,
    ImuSpeedEstimatorConfig,
    PathControlConfig,
    StuckRecoveryConfig,
    StuckRecoveryDecision,
    StuckRecoveryGate,
    control_local_path,
)


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
            "planned_speed_topic": "/adom/navigation/planned_speed",
            "imu_topic": "/zed/zed_node/imu/data",
            "cmd_vel_topic": "/cmd_vel",
            "status_topic": "/adom/control/local_path_status",
            "drive_topic": "/drive",
            "base_frame": "base_link",
            "control_rate_hz": 50.0,
            "path_timeout_sec": 0.25,
            "planned_speed_timeout_sec": 0.25,
            "imu_timeout_sec": 0.25,
            "wheelbase_m": 0.33,
            "lookahead_m": 0.80,
            "max_steering_deg": 24.0,
            "max_speed_mps": 1.0,
            "min_speed_mps": 0.10,
            "downstream_max_speed_mps": 12.0,
            "curvature_speed_gain": 1.5,
            "speed_kp": 0.0,
            "path_stop_distance_m": 0.50,
            "path_slow_distance_m": 2.0,
            "imu_accel_bias_x_mps2": 0.0,
            "imu_bias_learning_rate": 0.02,
            "imu_stationary_accel_threshold_mps2": 0.35,
            "imu_max_abs_accel_mps2": 5.0,
            "imu_max_integration_dt_sec": 0.10,
            "imu_estimated_speed_limit_mps": 4.0,
            "imu_velocity_leak_per_sec": 0.02,
            "imu_stationary_command_threshold_mps": 0.02,
            "imu_stationary_hold_sec": 0.50,
            "drive_feedback_timeout_sec": 0.25,
            "stuck_recovery_enabled": True,
            "stuck_command_min_mps": 0.12,
            "stuck_max_estimated_speed_mps": 0.08,
            "stuck_max_abs_yaw_rate_rps": 0.08,
            "stuck_min_abs_steering_deg": 12.0,
            "stuck_steering_stability_deg": 5.0,
            "stuck_hold_sec": 1.50,
            "stuck_recovery_duration_sec": 0.75,
            "stuck_recovery_speed_mps": 1.0,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        self.p = {name: self.get_parameter(name).value for name in defaults}
        if float(self.p["max_speed_mps"]) <= 0.0:
            raise ValueError("local path controller max_speed_mps must be positive")
        if float(self.p["max_speed_mps"]) > float(
            self.p["downstream_max_speed_mps"]
        ):
            raise ValueError(
                "local path controller max_speed_mps must not exceed the downstream control limit"
            )
        if not 0.0 < float(self.p["stuck_recovery_speed_mps"]) <= float(
            self.p["max_speed_mps"]
        ):
            raise ValueError(
                "stuck_recovery_speed_mps must be positive and not exceed max_speed_mps"
            )
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
        self._imu_estimator = ImuSpeedEstimator(
            ImuSpeedEstimatorConfig(
                initial_bias_mps2=float(self.p["imu_accel_bias_x_mps2"]),
                bias_learning_rate=float(self.p["imu_bias_learning_rate"]),
                stationary_accel_threshold_mps2=float(
                    self.p["imu_stationary_accel_threshold_mps2"]
                ),
                max_abs_accel_mps2=float(self.p["imu_max_abs_accel_mps2"]),
                max_integration_dt_sec=float(self.p["imu_max_integration_dt_sec"]),
                speed_limit_mps=float(self.p["imu_estimated_speed_limit_mps"]),
                velocity_leak_per_sec=float(self.p["imu_velocity_leak_per_sec"]),
            )
        )
        self._stuck_recovery = StuckRecoveryGate(
            StuckRecoveryConfig(
                command_min_mps=float(self.p["stuck_command_min_mps"]),
                max_estimated_speed_mps=float(
                    self.p["stuck_max_estimated_speed_mps"]
                ),
                max_abs_yaw_rate_rps=float(
                    self.p["stuck_max_abs_yaw_rate_rps"]
                ),
                min_abs_steering_deg=float(self.p["stuck_min_abs_steering_deg"]),
                steering_stability_deg=float(
                    self.p["stuck_steering_stability_deg"]
                ),
                hold_sec=float(self.p["stuck_hold_sec"]),
                duration_sec=float(self.p["stuck_recovery_duration_sec"]),
            )
        )
        self._path_xy: np.ndarray | None = None
        self._path_source_ns: int | None = None
        self._last_path_ns: int | None = None
        self._planned_speed_mps: float | None = None
        self._last_planned_speed_ns: int | None = None
        self._last_imu_ns: int | None = None
        self._last_imu_stamp_ns: int | None = None
        self._actuator_speed_mps = 0.0
        self._last_drive_ns: int | None = None
        self._zero_drive_since_ns: int | None = None
        self._estimated_speed_mps = 0.0
        self._imu_bias_mps2 = float(self.p["imu_accel_bias_x_mps2"])
        self._acceleration_x_mps2 = 0.0
        self._imu_stationary_update = False
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
            Float32,
            str(self.p["planned_speed_topic"]),
            self._on_planned_speed,
            10,
        )
        self.create_subscription(
            Imu, str(self.p["imu_topic"]), self._on_imu, qos_profile_sensor_data
        )
        self.create_subscription(
            AckermannDriveStamped,
            str(self.p["drive_topic"]),
            self._on_drive,
            10,
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

    def _on_planned_speed(self, message: Float32) -> None:
        speed_mps = float(message.data)
        if not math.isfinite(speed_mps) or speed_mps < 0.0:
            self.get_logger().error(
                f"Rejected invalid planned speed {speed_mps!r} m/s"
            )
            return
        self._planned_speed_mps = min(speed_mps, self._config.max_speed_mps)
        self._last_planned_speed_ns = self.get_clock().now().nanoseconds

    def _on_drive(self, message: AckermannDriveStamped) -> None:
        now_ns = self.get_clock().now().nanoseconds
        self._actuator_speed_mps = float(message.drive.speed)
        self._last_drive_ns = now_ns
        if abs(self._actuator_speed_mps) <= float(
            self.p["imu_stationary_command_threshold_mps"]
        ):
            if self._zero_drive_since_ns is None:
                self._zero_drive_since_ns = now_ns
        else:
            self._zero_drive_since_ns = None

    def _on_imu(self, message: Imu) -> None:
        now_ns = self.get_clock().now().nanoseconds
        stamp_ns = message_stamp_ns(message, now_ns)
        drive_age = self._age(now_ns, self._last_drive_ns)
        drive_fresh = drive_age <= float(self.p["drive_feedback_timeout_sec"])
        stationary = (
            drive_fresh
            and self._zero_drive_since_ns is not None
            and (now_ns - self._zero_drive_since_ns) / 1e9
            >= float(self.p["imu_stationary_hold_sec"])
        )
        estimate = self._imu_estimator.update(
            float(message.linear_acceleration.x), stamp_ns, stationary=stationary
        )
        self._estimated_speed_mps = estimate.speed_mps
        self._imu_bias_mps2 = estimate.bias_mps2
        self._acceleration_x_mps2 = estimate.corrected_accel_mps2
        self._imu_stationary_update = estimate.stationary_update
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
        self._stuck_recovery.reset()
        self._publish(0.0, 0.0)
        self._publish_status(
            "stopped", reason=reason, stuck_recovery_state="inactive", **fields
        )

    def _update(self) -> None:
        now_ns = self.get_clock().now().nanoseconds
        path_age = self._age(now_ns, self._last_path_ns)
        speed_age = self._age(now_ns, self._last_planned_speed_ns)
        imu_age = self._age(now_ns, self._last_imu_ns)
        common = {
            "path_age_sec": None if not math.isfinite(path_age) else round(path_age, 3),
            "planned_speed_age_sec": (
                None if not math.isfinite(speed_age) else round(speed_age, 3)
            ),
            "imu_age_sec": None if not math.isfinite(imu_age) else round(imu_age, 3),
        }
        if self._path_xy is None or path_age > float(self.p["path_timeout_sec"]):
            self._stop("path_watchdog", **common)
            return
        if len(self._path_xy) == 0:
            self._stop("empty_path", **common)
            return
        if (
            self._planned_speed_mps is None
            or speed_age > float(self.p["planned_speed_timeout_sec"])
        ):
            self._stop("planned_speed_watchdog", **common)
            return
        if imu_age > float(self.p["imu_timeout_sec"]):
            self._stop("imu_watchdog", **common)
            return
        try:
            command = control_local_path(
                self._path_xy,
                self._estimated_speed_mps,
                self._config,
                planned_speed_mps=self._planned_speed_mps,
            )
        except Exception as error:
            self.get_logger().error(f"Local path control failed: {error}")
            self._stop("controller_error", message=str(error), **common)
            return
        recovery = StuckRecoveryDecision("disabled", False, 0.0)
        if bool(self.p["stuck_recovery_enabled"]):
            recovery = self._stuck_recovery.update(
                now_ns,
                path_valid=(
                    command.speed_mps > 0.0
                    and command.available_path_m > self._config.path_stop_distance_m
                ),
                commanded_speed_mps=command.speed_mps,
                steering_rad=command.steering_rad,
                estimated_speed_mps=self._estimated_speed_mps,
                yaw_rate_rps=self._yaw_rate_rps,
            )
        published_speed_mps = (
            float(self.p["stuck_recovery_speed_mps"])
            if recovery.active
            else command.speed_mps
        )
        self._publish(published_speed_mps, command.steering_rad)
        source_to_command_ms = (
            None
            if self._path_source_ns is None or now_ns < self._path_source_ns
            else round((now_ns - self._path_source_ns) / 1e6, 2)
        )
        self._publish_status(
            "tracking",
            **common,
            speed_command_mps=round(published_speed_mps, 3),
            nominal_speed_command_mps=round(command.speed_mps, 3),
            planned_speed_mps=round(self._planned_speed_mps, 3),
            estimated_speed_mps=round(self._estimated_speed_mps, 3),
            speed_feedback_error_mps=round(
                self._planned_speed_mps - self._estimated_speed_mps, 3
            ),
            imu_bias_x_mps2=round(self._imu_bias_mps2, 4),
            imu_stationary_update=self._imu_stationary_update,
            actuator_speed_command_mps=round(self._actuator_speed_mps, 3),
            acceleration_x_mps2=round(self._acceleration_x_mps2, 3),
            yaw_rate_rps=round(self._yaw_rate_rps, 3),
            steering_deg=round(math.degrees(command.steering_rad), 2),
            target_x_m=round(command.target_x_m, 3),
            target_y_m=round(command.target_y_m, 3),
            available_path_m=round(command.available_path_m, 3),
            source_to_command_ms=source_to_command_ms,
            stuck_recovery_state=recovery.state,
            stuck_recovery_active=recovery.active,
            stuck_candidate_age_sec=round(recovery.candidate_age_sec, 3),
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
