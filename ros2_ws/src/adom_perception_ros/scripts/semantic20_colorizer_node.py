#!/usr/bin/env python3
from __future__ import annotations

import json

from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

from adom.perception import (
    SEMANTIC20_PALETTE_RGB,
    colorize_semantic20_mask,
    load_semantic20_ontology,
)


class Semantic20ColorizerNode(Node):
    """Colorize mono8 Semantic20 IDs without running model inference."""

    def __init__(self) -> None:
        super().__init__("semantic20_colorizer")
        defaults = {
            "mask_topic": "/adom/perception/semantic20_mask_evidence",
            "color_topic": "/adom/perception/semantic20_mask_color",
            "legend_topic": "/adom/perception/semantic20_legend",
            "bridge_mapping_path": "",
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        self.p = {name: self.get_parameter(name).value for name in defaults}

        mapping_path = str(self.p["bridge_mapping_path"]).strip() or None
        self._ontology = load_semantic20_ontology(mapping_path)
        self._bridge = CvBridge()
        self._color_pub = self.create_publisher(
            Image, str(self.p["color_topic"]), 1
        )
        self._legend_pub = self.create_publisher(
            String, str(self.p["legend_topic"]), 1
        )
        self.create_subscription(
            Image, str(self.p["mask_topic"]), self._on_mask, 1
        )
        self.create_timer(1.0, self._publish_legend)
        self._publish_legend()
        self.get_logger().info(
            f"Semantic20 colorizer: {self.p['mask_topic']} -> {self.p['color_topic']}"
        )

    def _publish_legend(self) -> None:
        entries = [
            {
                "id": class_id,
                "name": name,
                "rgb": SEMANTIC20_PALETTE_RGB[class_id].astype(int).tolist(),
            }
            for class_id, name in enumerate(self._ontology.classes)
        ]
        entries.append({"id": 255, "name": "ignore", "rgb": [0, 0, 0]})
        message = String()
        message.data = json.dumps(
            {
                "ontology": "Semantic20",
                "mapping_version": self._ontology.mapping_version,
                "classes": entries,
            },
            separators=(",", ":"),
        )
        self._legend_pub.publish(message)

    def _on_mask(self, message: Image) -> None:
        try:
            mask = self._bridge.imgmsg_to_cv2(message, desired_encoding="mono8")
            colors = colorize_semantic20_mask(mask, self._ontology)
            output = self._bridge.cv2_to_imgmsg(colors, encoding="bgr8")
            output.header = message.header
            self._color_pub.publish(output)
        except Exception as error:
            self.get_logger().error(f"Semantic20 mask colorization failed: {error}")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = Semantic20ColorizerNode()
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
