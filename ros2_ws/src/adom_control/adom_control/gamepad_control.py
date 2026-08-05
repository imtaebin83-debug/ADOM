import math
import threading

import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import Joy
from std_msgs.msg import String


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


def apply_deadzone(value, deadzone):
    if abs(value) <= deadzone:
        return 0.0
    return math.copysign((abs(value) - deadzone) / (1.0 - deadzone), value)


class GamepadControl(Node):
    """Select manual or autonomous input and publish one safe Ackermann command."""

    def __init__(self):
        super().__init__("gamepad_control")
        defaults = {
            "joy_topic": "/joy",
            "output_drive_topic": "/drive",
            "autonomous_drive_topic": "/drive/autonomous",
            "autonomous_cmd_vel_topic": "/cmd_vel",
            "autonomous_input_type": "twist",
            "mode_topic": "/adom/control/mode",
            "right_stick_x_axis": 3,
            "left_stick_y_axis": 1,
            "manual_button": 2,
            "autonomous_button": 0,
            "stop_button": 1,
            "steering_axis_scale": 1.0,
            "throttle_axis_scale": 1.0,
            "stick_deadzone": 0.10,
            "max_forward_speed_mps": 0.30,
            "max_reverse_speed_mps": 0.0,
            "max_steering_angle_rad": 0.35,
            "wheelbase_m": 0.33,
            "joy_timeout_sec": 0.50,
            "autonomous_timeout_sec": 0.25,
            "publish_rate_hz": 50.0,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        self.p = {name: self.get_parameter(name).value for name in defaults}

        self._validate_parameters()
        self._lock = threading.Lock()
        self._mode = "stopped"
        self._manual_armed = False
        self._manual_speed = 0.0
        self._manual_steering = 0.0
        self._auto_speed = 0.0
        self._auto_steering = 0.0
        self._last_joy_ns = None
        self._last_auto_ns = None
        self._previous_buttons = []

        self._drive_pub = self.create_publisher(
            AckermannDriveStamped, str(self.p["output_drive_topic"]), 10
        )
        self._mode_pub = self.create_publisher(String, str(self.p["mode_topic"]), 10)
        self.create_subscription(Joy, str(self.p["joy_topic"]), self._on_joy, 10)

        if str(self.p["autonomous_input_type"]) == "twist":
            self.create_subscription(
                Twist,
                str(self.p["autonomous_cmd_vel_topic"]),
                self._on_auto_twist,
                10,
            )
        else:
            self.create_subscription(
                AckermannDriveStamped,
                str(self.p["autonomous_drive_topic"]),
                self._on_auto_drive,
                10,
            )

        self.create_timer(1.0 / float(self.p["publish_rate_hz"]), self._update)
        self.create_timer(1.0, self._publish_mode)
        self.get_logger().warning(
            "Control starts STOPPED. Press X for manual or A for autonomous mode."
        )

    def _validate_parameters(self):
        if str(self.p["autonomous_input_type"]) not in ("twist", "ackermann"):
            raise ValueError("autonomous_input_type must be 'twist' or 'ackermann'")
        deadzone = float(self.p["stick_deadzone"])
        if not 0.0 <= deadzone < 1.0:
            raise ValueError("stick_deadzone must be in [0.0, 1.0)")
        indexes = (
            "right_stick_x_axis",
            "left_stick_y_axis",
            "manual_button",
            "autonomous_button",
            "stop_button",
        )
        if any(int(self.p[name]) < 0 for name in indexes):
            raise ValueError("axis and button indexes must be non-negative")

    def _on_joy(self, msg):
        required_axis = max(
            int(self.p["right_stick_x_axis"]), int(self.p["left_stick_y_axis"])
        )
        required_button = max(
            int(self.p["manual_button"]),
            int(self.p["autonomous_button"]),
            int(self.p["stop_button"]),
        )
        if len(msg.axes) <= required_axis or len(msg.buttons) <= required_button:
            self.get_logger().error(
                "Joy layout is smaller than configured axes/buttons; check /joy mapping."
            )
            return

        now_ns = self.get_clock().now().nanoseconds
        with self._lock:
            previous = self._previous_buttons
            manual_pressed = self._rising(msg.buttons, previous, int(self.p["manual_button"]))
            auto_pressed = self._rising(
                msg.buttons, previous, int(self.p["autonomous_button"])
            )
            stop_pressed = self._rising(msg.buttons, previous, int(self.p["stop_button"]))
            self._previous_buttons = list(msg.buttons)
            self._last_joy_ns = now_ns

            if stop_pressed:
                self._set_mode("stopped")
            elif manual_pressed:
                self._set_mode("manual")
            elif auto_pressed:
                self._set_mode("autonomous")

            steering_axis = apply_deadzone(
                float(msg.axes[int(self.p["right_stick_x_axis"])])
                * float(self.p["steering_axis_scale"]),
                float(self.p["stick_deadzone"]),
            )
            throttle_axis = apply_deadzone(
                float(msg.axes[int(self.p["left_stick_y_axis"])])
                * float(self.p["throttle_axis_scale"]),
                float(self.p["stick_deadzone"]),
            )

            if self._mode == "manual" and not self._manual_armed:
                self._manual_armed = steering_axis == 0.0 and throttle_axis == 0.0
                if self._manual_armed:
                    self.get_logger().info("Manual controls armed")

            if self._mode == "manual" and self._manual_armed:
                self._manual_steering = clamp(
                    steering_axis * float(self.p["max_steering_angle_rad"]),
                    -float(self.p["max_steering_angle_rad"]),
                    float(self.p["max_steering_angle_rad"]),
                )
                if throttle_axis >= 0.0:
                    self._manual_speed = (
                        throttle_axis * float(self.p["max_forward_speed_mps"])
                    )
                else:
                    self._manual_speed = (
                        throttle_axis * float(self.p["max_reverse_speed_mps"])
                    )

    @staticmethod
    def _rising(buttons, previous, index):
        old = previous[index] if len(previous) > index else 0
        return bool(buttons[index]) and not bool(old)

    def _set_mode(self, mode):
        if mode == self._mode:
            return
        self._mode = mode
        self._manual_speed = 0.0
        self._manual_steering = 0.0
        self._manual_armed = False
        if mode == "autonomous":
            self._last_auto_ns = None
        self.get_logger().warning(f"Control mode: {mode.upper()}")
        self._publish_mode()

    def _on_auto_twist(self, msg):
        speed = float(msg.linear.x)
        steering = (
            0.0
            if abs(speed) < 1e-3
            else math.atan(float(self.p["wheelbase_m"]) * float(msg.angular.z) / speed)
        )
        self._store_auto(speed, steering)

    def _on_auto_drive(self, msg):
        self._store_auto(float(msg.drive.speed), float(msg.drive.steering_angle))

    def _store_auto(self, speed, steering):
        with self._lock:
            self._auto_speed = clamp(
                speed,
                -float(self.p["max_reverse_speed_mps"]),
                float(self.p["max_forward_speed_mps"]),
            )
            self._auto_steering = clamp(
                steering,
                -float(self.p["max_steering_angle_rad"]),
                float(self.p["max_steering_angle_rad"]),
            )
            self._last_auto_ns = self.get_clock().now().nanoseconds

    def _update(self):
        now_ns = self.get_clock().now().nanoseconds
        with self._lock:
            mode = self._mode
            joy_age = self._age(now_ns, self._last_joy_ns)
            auto_age = self._age(now_ns, self._last_auto_ns)
            if (
                mode == "manual"
                and self._manual_armed
                and joy_age <= float(self.p["joy_timeout_sec"])
            ):
                speed, steering = self._manual_speed, self._manual_steering
            elif (
                mode == "autonomous"
                and auto_age <= float(self.p["autonomous_timeout_sec"])
            ):
                speed, steering = self._auto_speed, self._auto_steering
            else:
                speed, steering = 0.0, 0.0
        self._publish_drive(speed, steering)

    @staticmethod
    def _age(now_ns, then_ns):
        return math.inf if then_ns is None else (now_ns - then_ns) / 1e9

    def _publish_drive(self, speed, steering):
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.drive.speed = float(speed)
        msg.drive.steering_angle = float(steering)
        self._drive_pub.publish(msg)

    def _publish_mode(self):
        msg = String()
        msg.data = self._mode
        self._mode_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = GamepadControl()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._set_mode("stopped")
        node._publish_drive(0.0, 0.0)
        node.destroy_node()
        rclpy.shutdown()
