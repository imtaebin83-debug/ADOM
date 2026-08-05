import select
import sys
import termios
import tty

import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from rclpy.node import Node
from std_msgs.msg import Bool


HELP = """
ADOM keyboard teleop (SSH terminal supported)

  w / s       speed up / slow down
  a / d       steer left / right
  c           center steering
  SPACE or x  latch emergency stop
  r           release E-stop (command remains neutral)
  q           E-stop and quit

Keep pressing w/s to satisfy the keyboard deadman timer.
"""


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


class KeyboardTeleop(Node):
    def __init__(self):
        super().__init__("keyboard_teleop")
        defaults = {
            "drive_topic": "/drive",
            "estop_topic": "/emergency_stop",
            "publish_rate_hz": 20.0,
            "key_timeout_sec": 0.35,
            "speed_step_mps": 0.05,
            "steering_step_rad": 0.05,
            "max_forward_speed_mps": 0.30,
            "max_reverse_speed_mps": 0.0,
            "max_steering_angle_rad": 0.35,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        self.p = {name: self.get_parameter(name).value for name in defaults}
        self._drive_pub = self.create_publisher(
            AckermannDriveStamped, str(self.p["drive_topic"]), 10
        )
        self._estop_pub = self.create_publisher(Bool, str(self.p["estop_topic"]), 10)
        self._speed = 0.0
        self._steering = 0.0
        self._last_motion_key_ns = None
        self._estopped = False
        self._quit = False
        self._old_terminal = None
        self.create_timer(1.0 / float(self.p["publish_rate_hz"]), self._tick)

    def start_terminal(self):
        if not sys.stdin.isatty():
            raise RuntimeError("keyboard_teleop requires an interactive TTY (use ssh -t)")
        self._old_terminal = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        print(HELP, flush=True)

    def restore_terminal(self):
        if self._old_terminal is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_terminal)
            self._old_terminal = None

    def _read_key(self):
        readable, _, _ = select.select([sys.stdin], [], [], 0.0)
        return sys.stdin.read(1).lower() if readable else None

    def _tick(self):
        key = self._read_key()
        if key is not None:
            self._handle_key(key)

        now_ns = self.get_clock().now().nanoseconds
        expired = (
            self._last_motion_key_ns is None
            or (now_ns - self._last_motion_key_ns) / 1e9
            > float(self.p["key_timeout_sec"])
        )
        if expired or self._estopped:
            self._speed = 0.0
        self._publish_drive()

    def _handle_key(self, key):
        now_ns = self.get_clock().now().nanoseconds
        if key == "w" and not self._estopped:
            self._speed = clamp(
                self._speed + float(self.p["speed_step_mps"]),
                -float(self.p["max_reverse_speed_mps"]),
                float(self.p["max_forward_speed_mps"]),
            )
            self._last_motion_key_ns = now_ns
        elif key == "s" and not self._estopped:
            self._speed = clamp(
                self._speed - float(self.p["speed_step_mps"]),
                -float(self.p["max_reverse_speed_mps"]),
                float(self.p["max_forward_speed_mps"]),
            )
            self._last_motion_key_ns = now_ns
        elif key == "a" and not self._estopped:
            self._steering = clamp(
                self._steering + float(self.p["steering_step_rad"]),
                -float(self.p["max_steering_angle_rad"]),
                float(self.p["max_steering_angle_rad"]),
            )
        elif key == "d" and not self._estopped:
            self._steering = clamp(
                self._steering - float(self.p["steering_step_rad"]),
                -float(self.p["max_steering_angle_rad"]),
                float(self.p["max_steering_angle_rad"]),
            )
        elif key == "c":
            self._steering = 0.0
        elif key in (" ", "x"):
            self.stop(latch_estop=True)
        elif key == "r":
            self.stop(latch_estop=False)
        elif key == "q":
            self.stop(latch_estop=True)
            self._quit = True

    def stop(self, latch_estop=True):
        self._speed = 0.0
        self._steering = 0.0
        self._last_motion_key_ns = None
        self._estopped = latch_estop
        self._publish_estop(latch_estop)
        self._publish_drive()

    def _publish_drive(self):
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.drive.speed = float(self._speed)
        msg.drive.steering_angle = float(self._steering)
        self._drive_pub.publish(msg)

    def _publish_estop(self, active):
        msg = Bool()
        msg.data = bool(active)
        self._estop_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = KeyboardTeleop()
    try:
        node.start_terminal()
        while rclpy.ok() and not node._quit:
            rclpy.spin_once(node, timeout_sec=0.05)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop(latch_estop=True)
        node.restore_terminal()
        node.destroy_node()
        rclpy.shutdown()
