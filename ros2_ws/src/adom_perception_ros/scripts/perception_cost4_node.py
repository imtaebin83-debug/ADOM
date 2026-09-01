#!/usr/bin/env python3
from __future__ import annotations

import json
import time

import cv2
from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String

from adom.perception import MmsegBackend, colorize_mask


class AdomCost4PerceptionNode(Node):
    """Retained Cost4 adapter; Semantic20 uses the separate default node."""

    def __init__(self) -> None:
        super().__init__("adom_perception")
        defaults = {
            "image_topic": "/zed/zed_node/rgb/color/rect/image",
            "mask_topic": "/adom/perception/semantic_mask",
            "confidence_topic": "/adom/perception/confidence",
            "overlay_topic": "/adom/perception/overlay",
            "status_topic": "/adom/perception/status",
            "config_path": "",
            "checkpoint_path": "",
            "device": "cuda:0",
            "target_fps": 15.0,
            "overlay_alpha": 0.45,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        self.p = {name: self.get_parameter(name).value for name in defaults}
        if not str(self.p["config_path"]) or not str(self.p["checkpoint_path"]):
            raise ValueError("config_path and checkpoint_path must point to trained artifacts")
        target_fps = float(self.p["target_fps"])
        if target_fps <= 0.0:
            raise ValueError("target_fps must be positive")

        self._minimum_period = 1.0 / target_fps
        self._last_started = -float("inf")
        self._frames = 0
        self._started_at = time.monotonic()
        self._bridge = CvBridge()
        self._backend = MmsegBackend(
            str(self.p["config_path"]),
            str(self.p["checkpoint_path"]),
            str(self.p["device"]),
        )
        self._mask_pub = self.create_publisher(Image, str(self.p["mask_topic"]), 1)
        self._confidence_pub = self.create_publisher(
            Image, str(self.p["confidence_topic"]), 1
        )
        self._overlay_pub = self.create_publisher(
            Image, str(self.p["overlay_topic"]), 1
        )
        self._status_pub = self.create_publisher(String, str(self.p["status_topic"]), 10)
        self.create_subscription(
            Image,
            str(self.p["image_topic"]),
            self._on_image,
            qos_profile_sensor_data,
        )

    def _on_image(self, message: Image) -> None:
        started = time.monotonic()
        if started - self._last_started < self._minimum_period:
            return
        self._last_started = started
        try:
            image = self._bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
            mask, confidence = self._backend.infer(image)
            colors = colorize_mask(mask)
            alpha = min(1.0, max(0.0, float(self.p["overlay_alpha"])))
            overlay = cv2.addWeighted(image, 1.0 - alpha, colors, alpha, 0.0)
            outputs = (
                self._bridge.cv2_to_imgmsg(mask, encoding="mono8"),
                self._bridge.cv2_to_imgmsg(confidence, encoding="mono8"),
                self._bridge.cv2_to_imgmsg(overlay, encoding="bgr8"),
            )
            for output in outputs:
                output.header = message.header
            self._mask_pub.publish(outputs[0])
            self._confidence_pub.publish(outputs[1])
            self._overlay_pub.publish(outputs[2])
            self._frames += 1
            elapsed = max(time.monotonic() - self._started_at, 1e-6)
            status = String()
            status.data = json.dumps(
                {
                    "state": "ok",
                    "ontology": "Cost4",
                    "latency_ms": round((time.monotonic() - started) * 1000.0, 2),
                    "average_fps": round(self._frames / elapsed, 2),
                    "frame_id": message.header.frame_id,
                },
                sort_keys=True,
            )
            self._status_pub.publish(status)
        except Exception as error:
            self.get_logger().error(f"Cost4 perception frame failed: {error}")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = AdomCost4PerceptionNode()
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
