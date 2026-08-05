#!/usr/bin/env python3

import math
from pathlib import Path

import rclpy
import yaml
from action_msgs.msg import GoalStatus
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import Odometry
from rclpy.action import ActionClient
from rclpy.logging import get_logger
from rclpy.node import Node
from robot_localization.srv import FromLL
from sensor_msgs.msg import NavSatFix, NavSatStatus
from std_msgs.msg import Int32, String


class SequentialGpsWaypointExecutor(Node):
    """Convert WGS84 waypoints to map poses and send one Nav2 goal at a time."""

    def __init__(self):
        super().__init__("rtk_waypoint_executor")
        self.declare_parameter("waypoint_file", "")
        self.declare_parameter("from_ll_service", "/fromLL")
        self.declare_parameter("navigate_action", "/navigate_to_pose")
        self.declare_parameter("fix_topic", "/fix")
        self.declare_parameter("global_odometry_topic", "/odometry/global")
        self.declare_parameter("gps_odometry_topic", "/odometry/gps")
        self.declare_parameter("startup_timeout_sec", 60.0)
        self.declare_parameter("goal_timeout_sec", 120.0)
        self.declare_parameter("fix_timeout_sec", 2.0)
        self.declare_parameter("require_valid_fix", True)
        self.declare_parameter("cancel_on_fix_loss", True)
        self.declare_parameter("stop_on_failure", True)
        self.declare_parameter("loop", False)
        self.declare_parameter("start_index", 0)

        waypoint_file = str(self.get_parameter("waypoint_file").value)
        self._frame_id, self._raw_waypoints = self._load_waypoints(waypoint_file)
        start_index = int(self.get_parameter("start_index").value)
        if not 0 <= start_index < len(self._raw_waypoints):
            raise ValueError(f"start_index {start_index} is outside the waypoint list")

        self._converted = []
        self._convert_index = 0
        self._current_index = start_index
        self._phase = "WAITING_FOR_LOCALIZATION"
        self._started_ns = self.get_clock().now().nanoseconds
        self._last_fix_ns = None
        self._fix_valid = False
        self._odom_received = False
        self._gps_odom_received = False
        self._goal_handle = None
        self._goal_started_ns = None
        self._cancel_requested = False

        self._status_pub = self.create_publisher(String, "/adom/planning/waypoint_status", 10)
        self._index_pub = self.create_publisher(Int32, "/adom/planning/waypoint_index", 10)
        self._from_ll = self.create_client(
            FromLL, str(self.get_parameter("from_ll_service").value)
        )
        self._navigate = ActionClient(
            self, NavigateToPose, str(self.get_parameter("navigate_action").value)
        )
        self.create_subscription(
            NavSatFix, str(self.get_parameter("fix_topic").value), self._on_fix, 10
        )
        self.create_subscription(
            Odometry,
            str(self.get_parameter("global_odometry_topic").value),
            self._on_global_odometry,
            10,
        )
        self.create_subscription(
            Odometry,
            str(self.get_parameter("gps_odometry_topic").value),
            self._on_gps_odometry,
            10,
        )
        self.create_timer(0.2, self._tick)
        self._publish_status(
            "waiting for valid GNSS, global/GPS odometry, and /fromLL"
        )

    @staticmethod
    def _load_waypoints(filename):
        if not filename:
            raise ValueError("waypoint_file is required")
        path = Path(filename).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"waypoint file not found: {path}")
        with path.open("r", encoding="utf-8") as stream:
            document = yaml.safe_load(stream) or {}
        frame_id = str(document.get("frame_id", "map"))
        waypoints = document.get("waypoints")
        if not isinstance(waypoints, list) or not waypoints:
            raise ValueError("waypoint file must contain a non-empty 'waypoints' list")

        normalized = []
        for index, item in enumerate(waypoints):
            if not isinstance(item, dict):
                raise ValueError(f"waypoint {index} must be a mapping")
            try:
                latitude = float(item["latitude"])
                longitude = float(item["longitude"])
                altitude = float(item.get("altitude", 0.0))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid coordinates at waypoint {index}") from exc
            if not -90.0 <= latitude <= 90.0 or not -180.0 <= longitude <= 180.0:
                raise ValueError(f"coordinates out of range at waypoint {index}")
            heading_deg = item.get("heading_deg")
            heading_deg = None if heading_deg is None else float(heading_deg)
            if heading_deg is not None and not math.isfinite(heading_deg):
                raise ValueError(f"invalid heading_deg at waypoint {index}")
            normalized.append(
                {
                    "latitude": latitude,
                    "longitude": longitude,
                    "altitude": altitude,
                    "heading_deg": heading_deg,
                }
            )
        return frame_id, normalized

    def _on_fix(self, message):
        self._last_fix_ns = self.get_clock().now().nanoseconds
        self._fix_valid = (
            message.status.status >= NavSatStatus.STATUS_FIX
            and math.isfinite(message.latitude)
            and math.isfinite(message.longitude)
        )

    def _on_global_odometry(self, _message):
        self._odom_received = True

    def _on_gps_odometry(self, _message):
        # navsat_transform publishes this only after its geographic transform is ready.
        self._gps_odom_received = True

    def _fix_is_fresh(self, now_ns):
        if not bool(self.get_parameter("require_valid_fix").value):
            return True
        if self._last_fix_ns is None or not self._fix_valid:
            return False
        age = (now_ns - self._last_fix_ns) / 1e9
        return age <= float(self.get_parameter("fix_timeout_sec").value)

    def _tick(self):
        now_ns = self.get_clock().now().nanoseconds
        if self._phase == "WAITING_FOR_LOCALIZATION":
            elapsed = (now_ns - self._started_ns) / 1e9
            if elapsed > float(self.get_parameter("startup_timeout_sec").value):
                self._fail("startup timeout while waiting for localization")
                return
            if (
                self._fix_is_fresh(now_ns)
                and self._odom_received
                and self._gps_odom_received
                and self._from_ll.service_is_ready()
            ):
                self._phase = "CONVERTING"
                self._publish_status("localization ready; converting GPS waypoints")
                self._convert_next()
            return

        if self._phase == "READY" and self._navigate.server_is_ready():
            if self._fix_is_fresh(now_ns):
                self._send_current_goal()
            return

        if self._phase == "NAVIGATING":
            if (
                bool(self.get_parameter("cancel_on_fix_loss").value)
                and not self._fix_is_fresh(now_ns)
                and not self._cancel_requested
            ):
                self._cancel_goal("GNSS fix lost or stale")
                return
            elapsed = (now_ns - self._goal_started_ns) / 1e9
            if elapsed > float(self.get_parameter("goal_timeout_sec").value):
                self._cancel_goal("goal timeout")

    def _convert_next(self):
        if self._convert_index >= len(self._raw_waypoints):
            self._finish_conversion()
            return
        item = self._raw_waypoints[self._convert_index]
        request = FromLL.Request()
        request.ll_point.latitude = item["latitude"]
        request.ll_point.longitude = item["longitude"]
        request.ll_point.altitude = item["altitude"]
        future = self._from_ll.call_async(request)
        future.add_done_callback(self._on_converted)

    def _on_converted(self, future):
        try:
            response = future.result()
        except Exception as exc:
            self._fail(f"/fromLL failed at waypoint {self._convert_index}: {exc}")
            return
        point = response.map_point
        if not all(math.isfinite(value) for value in (point.x, point.y, point.z)):
            self._fail(f"/fromLL returned non-finite coordinates at waypoint {self._convert_index}")
            return
        self._converted.append((float(point.x), float(point.y), float(point.z)))
        self._convert_index += 1
        self._convert_next()

    def _finish_conversion(self):
        self._heading_degs = []
        for index, item in enumerate(self._raw_waypoints):
            if item["heading_deg"] is not None:
                heading_deg = item["heading_deg"]
            elif index + 1 < len(self._converted):
                current = self._converted[index]
                following = self._converted[index + 1]
                heading_deg = math.degrees(
                    math.atan2(
                        following[1] - current[1], following[0] - current[0]
                    )
                )
            elif self._heading_degs:
                heading_deg = self._heading_degs[-1]
            else:
                heading_deg = 0.0
            self._heading_degs.append(heading_deg)
        self._phase = "READY"
        self._publish_status(f"converted {len(self._converted)} waypoints; waiting for Nav2")

    def _send_current_goal(self):
        x, y, z = self._converted[self._current_index]
        heading_deg = self._heading_degs[self._current_index]
        heading_rad = math.radians(heading_deg)
        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = self._frame_id
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = x
        goal.pose.pose.position.y = y
        goal.pose.pose.position.z = z
        goal.pose.pose.orientation.z = math.sin(heading_rad / 2.0)
        goal.pose.pose.orientation.w = math.cos(heading_rad / 2.0)
        self._phase = "SENDING"
        self._publish_index()
        self._publish_status(
            f"sending waypoint {self._current_index + 1}/{len(self._converted)}: "
            f"x={x:.3f}, y={y:.3f}, heading_deg={heading_deg:.3f}"
        )
        future = self._navigate.send_goal_async(goal, feedback_callback=self._on_feedback)
        future.add_done_callback(self._on_goal_response)

    def _on_goal_response(self, future):
        try:
            goal_handle = future.result()
        except Exception as exc:
            self._handle_goal_failure(f"failed to send Nav2 goal: {exc}")
            return
        if not goal_handle.accepted:
            self._handle_goal_failure("Nav2 rejected the waypoint")
            return
        self._goal_handle = goal_handle
        self._goal_started_ns = self.get_clock().now().nanoseconds
        self._cancel_requested = False
        self._phase = "NAVIGATING"
        self._publish_status(
            f"navigating to waypoint {self._current_index + 1}/{len(self._converted)}"
        )
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._on_goal_result)

    def _on_feedback(self, feedback_message):
        feedback = feedback_message.feedback
        distance = getattr(feedback, "distance_remaining", math.nan)
        if math.isfinite(distance):
            self.get_logger().debug(
                f"waypoint {self._current_index + 1}: {distance:.2f} m remaining"
            )

    def _on_goal_result(self, future):
        try:
            status = future.result().status
        except Exception as exc:
            self._handle_goal_failure(f"failed to receive Nav2 result: {exc}")
            return
        self._goal_handle = None
        self._goal_started_ns = None
        self._cancel_requested = False
        if status == GoalStatus.STATUS_SUCCEEDED:
            self._publish_status(f"waypoint {self._current_index + 1} reached")
            self._advance()
        else:
            self._handle_goal_failure(
                f"waypoint {self._current_index + 1} ended with action status {status}"
            )

    def _cancel_goal(self, reason):
        if self._goal_handle is None:
            self._handle_goal_failure(reason)
            return
        self._cancel_requested = True
        self._phase = "CANCELLING"
        self._publish_status(f"cancelling waypoint: {reason}")
        self._goal_handle.cancel_goal_async()

    def _advance(self):
        self._current_index += 1
        if self._current_index >= len(self._converted):
            if bool(self.get_parameter("loop").value):
                self._current_index = 0
            else:
                self._phase = "COMPLETED"
                self._publish_status("all GPS waypoints completed")
                return
        self._phase = "READY"

    def _handle_goal_failure(self, reason):
        self._publish_status(reason)
        if bool(self.get_parameter("stop_on_failure").value):
            self._phase = "FAILED"
        else:
            self._advance()

    def _fail(self, reason):
        self._phase = "FAILED"
        self._publish_status(reason)
        self.get_logger().error(reason)

    def _publish_index(self):
        message = Int32()
        message.data = self._current_index
        self._index_pub.publish(message)

    def _publish_status(self, text):
        message = String()
        message.data = f"{self._phase}: {text}"
        self._status_pub.publish(message)
        self.get_logger().info(message.data)


def main(args=None):
    rclpy.init(args=args)
    try:
        node = SequentialGpsWaypointExecutor()
    except Exception as exc:
        get_logger("rtk_waypoint_executor").fatal(str(exc))
        rclpy.shutdown()
        return
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
