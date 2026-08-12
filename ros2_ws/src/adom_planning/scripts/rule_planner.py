#!/usr/bin/env python3
from __future__ import annotations

from collections import deque
import json
import math

from geometry_msgs.msg import Point, PoseStamped, Twist
from nav_msgs.msg import OccupancyGrid, Path
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from visualization_msgs.msg import Marker

from adom.autonomy import CostmapConfig, PlannerConfig, plan_corridor


class RulePlannerNode(Node):
    def __init__(self) -> None:
        super().__init__("rule_planner")
        defaults = {
            "costmap_topic": "/adom/navigation/semantic_costmap",
            "cmd_vel_topic": "/cmd_vel",
            "publish_cmd_vel": False,
            "local_path_topic": "/adom/navigation/local_path",
            "path_marker_topic": "/adom/navigation/rule_path",
            "status_topic": "/adom/navigation/rule_status",
            "action_latency_topic": "/adom/navigation/action_latency",
            "action_latency_window": 300,
            "publish_rate_hz": 20.0,
            "costmap_timeout_sec": 0.20,
            "max_source_age_sec": 0.80,
            "wheelbase_m": 0.33,
            "max_steering_deg": 20.0,
            "lookahead_m": 3.0,
            "path_step_m": 0.10,
            "tree_depth": 3,
            "tree_branch_steering_deg": 10.0,
            "steering_change_penalty": 4.0,
            "corridor_half_width_m": 0.18,
            "unknown_cost": 70.0,
            "lethal_cost": 90,
            "stop_distance_m": 0.75,
            "max_speed_mps": 0.25,
            "min_speed_mps": 0.08,
            "steering_penalty": 8.0,
            "distance_decay_m": 1.25,
            "clearance_penalty": 35.0,
            "slow_distance_m": 2.0,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        self.p = {name: self.get_parameter(name).value for name in defaults}
        if float(self.p["max_speed_mps"]) > 0.30:
            raise ValueError(
                "rule planner max_speed_mps must not exceed ADOM's 0.30 m/s limit"
            )
        self._planner = PlannerConfig(
            wheelbase_m=float(self.p["wheelbase_m"]),
            max_steering_deg=float(self.p["max_steering_deg"]),
            lookahead_m=float(self.p["lookahead_m"]),
            path_step_m=float(self.p["path_step_m"]),
            tree_depth=int(self.p["tree_depth"]),
            tree_branch_steering_deg=float(self.p["tree_branch_steering_deg"]),
            steering_change_penalty=float(self.p["steering_change_penalty"]),
            corridor_half_width_m=float(self.p["corridor_half_width_m"]),
            unknown_cost=float(self.p["unknown_cost"]),
            lethal_cost=int(self.p["lethal_cost"]),
            stop_distance_m=float(self.p["stop_distance_m"]),
            max_speed_mps=float(self.p["max_speed_mps"]),
            min_speed_mps=float(self.p["min_speed_mps"]),
            steering_penalty=float(self.p["steering_penalty"]),
            distance_decay_m=float(self.p["distance_decay_m"]),
            clearance_penalty=float(self.p["clearance_penalty"]),
            slow_distance_m=float(self.p["slow_distance_m"]),
        )
        self._grid: np.ndarray | None = None
        self._costmap_config: CostmapConfig | None = None
        self._frame_id = "base_link"
        self._last_grid_ns: int | None = None
        self._source_stamp_ns: int | None = None
        self._source_stamp = None
        self._source_action_reported = False
        self._action_latencies_ms: deque[float] = deque(
            maxlen=max(1, int(self.p["action_latency_window"]))
        )
        self._cmd_pub = self.create_publisher(Twist, str(self.p["cmd_vel_topic"]), 10)
        self._path_pub = self.create_publisher(
            Path, str(self.p["local_path_topic"]), 1
        )
        self._marker_pub = self.create_publisher(
            Marker, str(self.p["path_marker_topic"]), 10
        )
        self._status_pub = self.create_publisher(String, str(self.p["status_topic"]), 10)
        self._latency_pub = self.create_publisher(
            String, str(self.p["action_latency_topic"]), 10
        )
        self.create_subscription(
            OccupancyGrid, str(self.p["costmap_topic"]), self._on_costmap, 1
        )
        self.create_timer(1.0 / float(self.p["publish_rate_hz"]), self._update)
        self.get_logger().warning(
            "Rule planner starts with zero command; gamepad A must arm autonomous control."
        )

    def _on_costmap(self, message: OccupancyGrid) -> None:
        expected = int(message.info.width) * int(message.info.height)
        if expected <= 0 or len(message.data) != expected:
            self.get_logger().error("Rejected malformed semantic OccupancyGrid")
            return
        source_ns = int(message.header.stamp.sec) * 1_000_000_000 + int(
            message.header.stamp.nanosec
        )
        now_ns = self.get_clock().now().nanoseconds
        if source_ns > 0:
            source_age = (now_ns - source_ns) / 1e9
            if source_age > float(self.p["max_source_age_sec"]) or source_age < -0.10:
                self.get_logger().warning(
                    f"Rejected stale or future semantic costmap (age={source_age:.3f}s)"
                )
                return
        resolution = float(message.info.resolution)
        width_m = int(message.info.height) * resolution
        length_m = int(message.info.width) * resolution
        self._costmap_config = CostmapConfig(
            resolution_m=resolution,
            length_m=length_m,
            width_m=width_m,
            inflation_radius_m=0.0,
        )
        self._grid = np.asarray(message.data, dtype=np.int8).reshape(
            int(message.info.height), int(message.info.width)
        )
        self._frame_id = message.header.frame_id or "base_link"
        self._last_grid_ns = now_ns
        self._source_stamp_ns = source_ns if source_ns > 0 else None
        self._source_stamp = message.header.stamp
        self._source_action_reported = False

    def _publish_action_latency(self, state: str) -> None:
        if self._source_action_reported or self._source_stamp_ns is None:
            return
        action_ns = self.get_clock().now().nanoseconds
        latency_ms = (action_ns - self._source_stamp_ns) / 1e6
        if latency_ms < 0.0:
            return
        self._source_action_reported = True
        self._action_latencies_ms.append(latency_ms)
        values = np.asarray(self._action_latencies_ms, dtype=np.float64)
        message = String()
        message.data = json.dumps(
            {
                "state": state,
                "action_topic": str(self.p["cmd_vel_topic"]),
                "source_stamp_ns": self._source_stamp_ns,
                "action_stamp_ns": action_ns,
                "camera_to_action_ms": round(latency_ms, 2),
                "window_samples": len(values),
                "camera_to_action_p50_ms": round(float(np.percentile(values, 50)), 2),
                "camera_to_action_p95_ms": round(float(np.percentile(values, 95)), 2),
            },
            sort_keys=True,
        )
        self._latency_pub.publish(message)

    def _publish_stop(self, reason: str) -> None:
        if bool(self.p["publish_cmd_vel"]):
            self._cmd_pub.publish(Twist())
        self._publish_path(np.empty((0, 2), dtype=np.float64))
        status = String()
        status.data = json.dumps({"state": "stopped", "reason": reason}, sort_keys=True)
        self._status_pub.publish(status)

    def _publish_path(self, path_xy: np.ndarray) -> None:
        message = Path()
        message.header.frame_id = self._frame_id
        message.header.stamp = (
            self._source_stamp
            if self._source_stamp is not None
            else self.get_clock().now().to_msg()
        )
        if len(path_xy):
            headings = (
                np.zeros(1, dtype=np.float64)
                if len(path_xy) == 1
                else np.arctan2(
                    np.gradient(path_xy[:, 1]), np.gradient(path_xy[:, 0])
                )
            )
            for (x, y), heading in zip(path_xy, headings):
                pose = PoseStamped()
                pose.header = message.header
                pose.pose.position.x = float(x)
                pose.pose.position.y = float(y)
                pose.pose.orientation.z = math.sin(float(heading) / 2.0)
                pose.pose.orientation.w = math.cos(float(heading) / 2.0)
                message.poses.append(pose)
        self._path_pub.publish(message)

    def _publish_marker(self, path: np.ndarray, blocked: bool) -> None:
        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = self._frame_id
        marker.ns = "adom_rule_path"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.05
        marker.color.a = 1.0
        marker.color.r = 1.0 if blocked else 0.1
        marker.color.g = 0.1 if blocked else 1.0
        marker.color.b = 0.1
        marker.lifetime.sec = 0
        marker.lifetime.nanosec = 200_000_000
        marker.points = [Point(x=float(x), y=float(y), z=0.05) for x, y in path]
        self._marker_pub.publish(marker)

    def _update(self) -> None:
        if (
            self._grid is None
            or self._costmap_config is None
            or self._last_grid_ns is None
        ):
            self._publish_stop("no_costmap")
            return
        age = (self.get_clock().now().nanoseconds - self._last_grid_ns) / 1e9
        if age > float(self.p["costmap_timeout_sec"]):
            self._publish_stop("costmap_watchdog")
            return
        if not np.any(self._grid >= 0):
            self._publish_stop("empty_costmap")
            return
        try:
            plan = plan_corridor(self._grid, self._costmap_config, self._planner)
        except Exception as error:
            self.get_logger().error(f"Rule planning failed: {error}")
            self._publish_stop("planner_error")
            return

        command = Twist()
        command.linear.x = float(plan.speed_mps)
        command.angular.z = (
            0.0
            if plan.speed_mps <= 0.0
            else plan.speed_mps
            * math.tan(plan.steering_rad)
            / float(self.p["wheelbase_m"])
        )
        if bool(self.p["publish_cmd_vel"]):
            self._cmd_pub.publish(command)
        self._publish_path(plan.path_xy)
        if bool(self.p["publish_cmd_vel"]):
            self._publish_action_latency("blocked" if plan.blocked else "driving")
        self._publish_marker(plan.path_xy, plan.blocked)
        status = String()
        status.data = json.dumps(
            {
                "state": "blocked" if plan.blocked else "driving",
                "speed_mps": round(plan.speed_mps, 3),
                "steering_deg": round(math.degrees(plan.steering_rad), 2),
                "steering_sequence_deg": [
                    round(math.degrees(value), 2)
                    for value in plan.steering_sequence_rad
                ],
                "tree_depth": int(self.p["tree_depth"]),
                "score": round(plan.score, 2),
                "costmap_age_sec": round(age, 3),
                "local_path_points": int(len(plan.path_xy)),
                "lookahead_lateral_offset_m": (
                    None
                    if len(plan.path_xy) == 0
                    else round(float(plan.path_xy[-1, 1]), 3)
                ),
                "obstacle_clearance_m": round(plan.clearance_m, 3),
            },
            sort_keys=True,
        )
        self._status_pub.publish(status)

    def destroy_node(self):
        if rclpy.ok(context=self.context) and bool(self.p["publish_cmd_vel"]):
            self._cmd_pub.publish(Twist())
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RulePlannerNode()
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
