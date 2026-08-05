import math
import threading

import rclpy
from adom_control.pca9685_smbus import LinuxPca9685
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import Bool, Float64MultiArray


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


class Pca9685Control(Node):
    def __init__(self):
        super().__init__("pca9685_control")
        defaults = {
            "wheelbase_m": 0.33,
            "max_speed_mps": 0.3,
            "max_reverse_speed_mps": 0.0,
            "max_steering_angle_deg": 25.7831,
            "command_rate_hz": 50.0,
            "command_timeout_sec": 0.25,
            "drive_topic": "/drive",
            "cmd_vel_topic": "/cmd_vel",
            "enable_cmd_vel": False,
            "i2c_bus": 7,
            "i2c_address": 0x40,
            "pwm_frequency_hz": 50.0,
            "esc_channel": 0,
            "steering_channel": 1,
            "esc_neutral_us": 1500.0,
            "esc_forward_max_us": 1600.0,
            "esc_reverse_max_us": 1400.0,
            "steering_center_us": 1500.0,
            "steering_left_us": 1300.0,
            "steering_right_us": 1700.0,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        self.p = {name: self.get_parameter(name).value for name in defaults}
        self._lock = threading.Lock()
        self._last_command_ns = None
        self._linear = 0.0
        self._steering = 0.0
        self._estop = False
        self._pca = None
        self._pwm_state = self.create_publisher(
            Float64MultiArray, "/adom/control/pwm_us", 10
        )
        self.create_subscription(
            AckermannDriveStamped, str(self.p["drive_topic"]), self._on_drive, 10
        )
        if self.p["enable_cmd_vel"]:
            self.create_subscription(
                Twist, str(self.p["cmd_vel_topic"]), self._on_cmd_vel, 10
            )
        self.create_subscription(Bool, "/emergency_stop", self._on_estop, 10)
        self._initialize_hardware()
        self._write_neutral()
        self.create_timer(1.0 / float(self.p["command_rate_hz"]), self._update)

    def _initialize_hardware(self):
        try:
            self._pca = LinuxPca9685(
                bus_number=int(self.p["i2c_bus"]),
                address=int(self.p["i2c_address"]),
                frequency_hz=float(self.p["pwm_frequency_hz"]),
            )
            self.get_logger().info(
                "PCA9685 ready on /dev/i2c-%d at 0x%02x"
                % (int(self.p["i2c_bus"]), int(self.p["i2c_address"]))
            )
        except Exception as exc:
            self.get_logger().fatal(
                "PCA9685 initialization failed on /dev/i2c-%d address 0x%02x: %s. "
                "Check `i2cdetect -l`, `sudo i2cdetect -y -r BUS`, wiring, and i2c group access."
                % (int(self.p["i2c_bus"]), int(self.p["i2c_address"]), exc)
            )
            raise

    def _on_cmd_vel(self, msg):
        linear = float(msg.linear.x)
        angular = float(msg.angular.z)
        steering = (
            0.0
            if abs(linear) < 1e-3
            else math.degrees(
                math.atan(float(self.p["wheelbase_m"]) * angular / linear)
            )
        )
        with self._lock:
            self._linear = linear
            self._steering = steering
            self._last_command_ns = self.get_clock().now().nanoseconds

    def _on_drive(self, msg):
        with self._lock:
            self._linear = float(msg.drive.speed)
            self._steering = math.degrees(float(msg.drive.steering_angle))
            self._last_command_ns = self.get_clock().now().nanoseconds

    def _on_estop(self, msg):
        with self._lock:
            self._estop = bool(msg.data)

    def _update(self):
        now_ns = self.get_clock().now().nanoseconds
        with self._lock:
            age = (
                math.inf
                if self._last_command_ns is None
                else (now_ns - self._last_command_ns) / 1e9
            )
            linear, steering, estop = self._linear, self._steering, self._estop
        if estop or age > float(self.p["command_timeout_sec"]):
            self._write_neutral()
            return

        linear = clamp(
            linear,
            -float(self.p["max_reverse_speed_mps"]),
            float(self.p["max_speed_mps"]),
        )
        steering = clamp(
            steering,
            -float(self.p["max_steering_angle_deg"]),
            float(self.p["max_steering_angle_deg"]),
        )
        self._write(self._speed_to_pwm(linear), self._steering_to_pwm(steering))

    def _speed_to_pwm(self, speed):
        neutral = float(self.p["esc_neutral_us"])
        if speed >= 0.0:
            ratio = speed / max(float(self.p["max_speed_mps"]), 1e-6)
            return neutral + ratio * (float(self.p["esc_forward_max_us"]) - neutral)
        ratio = abs(speed) / max(float(self.p["max_reverse_speed_mps"]), 1e-6)
        return neutral + ratio * (float(self.p["esc_reverse_max_us"]) - neutral)

    def _steering_to_pwm(self, angle_deg):
        center = float(self.p["steering_center_us"])
        maximum = float(self.p["max_steering_angle_deg"])
        if angle_deg >= 0.0:
            return center + angle_deg / maximum * (
                float(self.p["steering_left_us"]) - center
            )
        return center + (-angle_deg) / maximum * (
            float(self.p["steering_right_us"]) - center
        )

    def _write_neutral(self):
        self._write(float(self.p["esc_neutral_us"]), float(self.p["steering_center_us"]))

    def _write(self, esc_us, steering_us):
        self._pca.set_pulse_us(int(self.p["esc_channel"]), esc_us)
        self._pca.set_pulse_us(int(self.p["steering_channel"]), steering_us)
        msg = Float64MultiArray()
        msg.data = [float(esc_us), float(steering_us)]
        self._pwm_state.publish(msg)

    def destroy_node(self):
        try:
            self._write_neutral()
            if self._pca is not None:
                self._pca.close()
        finally:
            super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = Pca9685Control()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
