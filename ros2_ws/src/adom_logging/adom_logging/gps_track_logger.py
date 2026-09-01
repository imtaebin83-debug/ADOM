from __future__ import annotations

import json
import math

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import NavSatFix, NavSatStatus
from std_msgs.msg import String

from adom.autonomy import local_gps_xy_m



class GpsTrackLogger(Node):
    """Publish a visualization trail; never feed GPS into planning or control."""

    def __init__(self) -> None:
        super().__init__("gps_track_logger")
        defaults = {
            "fix_topic": "/fix",
            "path_topic": "/adom/logging/gps_path",
            "status_topic": "/adom/logging/gps_status",
            "frame_id": "gps_recording_origin",
            "minimum_step_m": 0.05,
            "maximum_covariance_m2": 25.0,
            "maximum_points": 20000,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        self.p = {name: self.get_parameter(name).value for name in defaults}
        self._origin: tuple[float, float] | None = None
        self._last_xy: tuple[float, float] | None = None
        self._path = Path()
        self._path.header.frame_id = str(self.p["frame_id"])
        self._path_pub = self.create_publisher(
            Path, str(self.p["path_topic"]), 1
        )
        self._status_pub = self.create_publisher(
            String, str(self.p["status_topic"]), 10
        )
        self.create_subscription(
            NavSatFix,
            str(self.p["fix_topic"]),
            self._on_fix,
            qos_profile_sensor_data,
        )
        self.get_logger().info(
            "GPS is enabled for route logging only and is not a planning/control input."
        )

    def _on_fix(self, message: NavSatFix) -> None:
        if message.status.status < NavSatStatus.STATUS_FIX:
            self._publish_status("rejected", "no_fix")
            return
        if not math.isfinite(message.latitude) or not math.isfinite(message.longitude):
            self._publish_status("rejected", "non_finite")
            return
        covariance = max(
            0.0,
            float(message.position_covariance[0])
            + float(message.position_covariance[4]),
        )
        if covariance > float(self.p["maximum_covariance_m2"]):
            self._publish_status("rejected", "covariance")
            return
        if self._origin is None:
            self._origin = (float(message.latitude), float(message.longitude))
        current_xy = local_gps_xy_m(
            self._origin[0],
            self._origin[1],
            float(message.latitude),
            float(message.longitude),
        )
        if self._last_xy is not None:
            step = math.hypot(
                current_xy[0] - self._last_xy[0],
                current_xy[1] - self._last_xy[1],
            )
            if step < float(self.p["minimum_step_m"]):
                return
        pose = PoseStamped()
        pose.header = message.header
        pose.header.frame_id = str(self.p["frame_id"])
        pose.pose.position.x = current_xy[0]
        pose.pose.position.y = current_xy[1]
        pose.pose.orientation.w = 1.0
        self._path.poses.append(pose)
        maximum_points = max(1, int(self.p["maximum_points"]))
        if len(self._path.poses) > maximum_points:
            self._path.poses = self._path.poses[-maximum_points:]
        self._last_xy = current_xy
        self._path.header.stamp = message.header.stamp
        self._path_pub.publish(self._path)
        self._publish_status("recording", None, covariance)

    def _publish_status(
        self, state: str, reason: str | None, covariance: float | None = None
    ) -> None:
        message = String()
        message.data = json.dumps(
            {
                "state": state,
                "reason": reason,
                "points": len(self._path.poses),
                "origin_latitude": None if self._origin is None else self._origin[0],
                "origin_longitude": None if self._origin is None else self._origin[1],
                "position_covariance_xy_m2": covariance,
                "usage": "logging_only",
            },
            sort_keys=True,
        )
        self._status_pub.publish(message)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = GpsTrackLogger()
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
