#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
import json
import threading
import time

import cv2
from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String

from adom.perception import (
    LatestItemMailbox,
    MmsegBackend,
    colorize_semantic20_mask,
    load_semantic20_ontology,
    semantic20_pixel_statistics,
)


@dataclass(frozen=True)
class ReceivedFrame:
    message: Image
    received_monotonic: float
    received_ros_ns: int


def stamp_ns(message: Image) -> int:
    return int(message.header.stamp.sec) * 1_000_000_000 + int(
        message.header.stamp.nanosec
    )


def elapsed_ms(later_ns: int, earlier_ns: int) -> float | None:
    if earlier_ns <= 0:
        return None
    elapsed = (later_ns - earlier_ns) / 1e6
    if elapsed < 0.0:
        return None
    return round(elapsed, 2)


class AdomPerceptionNode(Node):
    """Semantic20 inference with an explicit latest-frame-only worker."""

    def __init__(self) -> None:
        super().__init__("adom_perception")
        defaults = {
            "image_topic": "/zed/zed_node/rgb/color/rect/image",
            "mask_topic": "/adom/perception/semantic20_mask",
            "evidence_mask_topic": "/adom/perception/semantic20_mask_evidence",
            "evidence_image_topic": "/adom/perception/image_evidence",
            "evidence_mask_fps": 2.0,
            "confidence_topic": "/adom/perception/confidence",
            "overlay_topic": "/adom/perception/overlay",
            "status_topic": "/adom/perception/status",
            "bridge_mapping_path": "",
            "config_path": "",
            "checkpoint_path": "",
            "device": "cuda:0",
            "target_fps": 30.0,
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
        evidence_mask_fps = float(self.p["evidence_mask_fps"])
        if evidence_mask_fps < 0.0:
            raise ValueError("evidence_mask_fps must be non-negative")

        mapping_path = str(self.p["bridge_mapping_path"]).strip() or None
        self._ontology = load_semantic20_ontology(mapping_path)
        self._minimum_period = 1.0 / target_fps
        self._evidence_mask_period = (
            None if evidence_mask_fps == 0.0 else 1.0 / evidence_mask_fps
        )
        self._next_evidence_mask_publish = 0.0
        self._frames = 0
        self._started_at = time.monotonic()
        self._bridge = CvBridge()
        self._backend = MmsegBackend(
            str(self.p["config_path"]),
            str(self.p["checkpoint_path"]),
            str(self.p["device"]),
        )
        self._mailbox: LatestItemMailbox[ReceivedFrame] = LatestItemMailbox()
        self._shutdown = threading.Event()

        self._mask_pub = self.create_publisher(Image, str(self.p["mask_topic"]), 1)
        self._evidence_mask_pub = self.create_publisher(
            Image, str(self.p["evidence_mask_topic"]), 1
        )
        self._evidence_image_pub = self.create_publisher(
            Image, str(self.p["evidence_image_topic"]), 1
        )
        self._confidence_pub = self.create_publisher(
            Image, str(self.p["confidence_topic"]), 1
        )
        self._overlay_pub = self.create_publisher(
            Image, str(self.p["overlay_topic"]), 1
        )
        self._status_pub = self.create_publisher(String, str(self.p["status_topic"]), 10)
        image_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.create_subscription(
            Image,
            str(self.p["image_topic"]),
            self._on_image,
            image_qos,
        )
        self._worker = threading.Thread(
            target=self._inference_loop,
            name="adom-semantic20-inference",
            daemon=True,
        )
        self._worker.start()
        self.get_logger().info(
            "ADOM Semantic20 MMSeg inference ready on "
            f"{self.p['device']} at <= {target_fps:.1f} FPS; "
            f"mapping={self._ontology.dataset_name}:{self._ontology.mapping_version}"
        )

    def _on_image(self, message: Image) -> None:
        frame = ReceivedFrame(
            message=message,
            received_monotonic=time.monotonic(),
            received_ros_ns=self.get_clock().now().nanoseconds,
        )
        try:
            self._mailbox.put(frame)
        except RuntimeError:
            pass

    def _publish_status(self, **fields) -> None:
        status = String()
        status.data = json.dumps(fields, sort_keys=True)
        self._status_pub.publish(status)

    def _inference_loop(self) -> None:
        next_allowed_start = 0.0
        while not self._shutdown.is_set():
            delay = next_allowed_start - time.monotonic()
            if delay > 0.0 and self._shutdown.wait(delay):
                return
            item = self._mailbox.take()
            if item is None or self._shutdown.is_set():
                return
            started = time.monotonic()
            next_allowed_start = started + self._minimum_period
            self._process_frame(item.sequence, item.value, started)

    def _process_frame(
        self, sequence: int, frame: ReceivedFrame, started: float
    ) -> None:
        message = frame.message
        source_ns = stamp_ns(message)
        inference_started_ros_ns = self.get_clock().now().nanoseconds
        try:
            image = self._bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
            publish_confidence = self._confidence_pub.get_subscription_count() > 0
            publish_overlay = self._overlay_pub.get_subscription_count() > 0
            inference_started = time.monotonic()
            mask, confidence = self._backend.infer(
                image, include_confidence=publish_confidence
            )
            inference_finished = time.monotonic()
            pixel_statistics = semantic20_pixel_statistics(mask, self._ontology)

            mask_message = self._bridge.cv2_to_imgmsg(mask, encoding="mono8")
            mask_message.header = message.header
            self._mask_pub.publish(mask_message)
            evidence_mask_published = False
            evidence_image_published = False
            evidence_now = time.monotonic()
            if (
                self._evidence_mask_period is not None
                and (
                    self._evidence_mask_pub.get_subscription_count() > 0
                    or self._evidence_image_pub.get_subscription_count() > 0
                )
                and evidence_now >= self._next_evidence_mask_publish
            ):
                if self._evidence_mask_pub.get_subscription_count() > 0:
                    self._evidence_mask_pub.publish(mask_message)
                    evidence_mask_published = True
                if self._evidence_image_pub.get_subscription_count() > 0:
                    evidence_image_message = self._bridge.cv2_to_imgmsg(
                        image, encoding="bgr8"
                    )
                    evidence_image_message.header = message.header
                    self._evidence_image_pub.publish(evidence_image_message)
                    evidence_image_published = True
                self._next_evidence_mask_publish = (
                    evidence_now + self._evidence_mask_period
                )
            if publish_confidence:
                if confidence is None:
                    raise RuntimeError("backend did not return requested confidence")
                confidence_message = self._bridge.cv2_to_imgmsg(
                    confidence, encoding="mono8"
                )
                confidence_message.header = message.header
                self._confidence_pub.publish(confidence_message)
            if publish_overlay:
                colors = colorize_semantic20_mask(mask, self._ontology)
                alpha = min(1.0, max(0.0, float(self.p["overlay_alpha"])))
                overlay = cv2.addWeighted(image, 1.0 - alpha, colors, alpha, 0.0)
                overlay_message = self._bridge.cv2_to_imgmsg(
                    overlay, encoding="bgr8"
                )
                overlay_message.header = message.header
                self._overlay_pub.publish(overlay_message)

            output_ros_ns = self.get_clock().now().nanoseconds
            finished = time.monotonic()
            self._frames += 1
            elapsed = max(finished - self._started_at, 1e-6)
            self._publish_status(
                state="ok",
                ontology="Semantic20",
                mapping_version=self._ontology.mapping_version,
                source_sequence=sequence,
                source_stamp_ns=source_ns,
                frame_id=message.header.frame_id,
                target_fps=float(self.p["target_fps"]),
                average_fps=round(self._frames / elapsed, 2),
                queue_wait_ms=round((started - frame.received_monotonic) * 1000.0, 2),
                capture_to_receive_ms=elapsed_ms(frame.received_ros_ns, source_ns),
                capture_to_inference_start_ms=elapsed_ms(
                    inference_started_ros_ns, source_ns
                ),
                inference_ms=round(
                    (inference_finished - inference_started) * 1000.0, 2
                ),
                processing_ms=round((finished - started) * 1000.0, 2),
                capture_to_perception_output_ms=elapsed_ms(output_ros_ns, source_ns),
                received_frames=self._mailbox.received,
                overwritten_frames=self._mailbox.overwritten,
                confidence_subscribers=self._confidence_pub.get_subscription_count(),
                overlay_subscribers=self._overlay_pub.get_subscription_count(),
                evidence_mask_subscribers=(
                    self._evidence_mask_pub.get_subscription_count()
                ),
                evidence_mask_fps=float(self.p["evidence_mask_fps"]),
                evidence_mask_published=evidence_mask_published,
                evidence_image_subscribers=(
                    self._evidence_image_pub.get_subscription_count()
                ),
                evidence_image_published=evidence_image_published,
                **pixel_statistics,
            )
        except Exception as error:
            self.get_logger().error(f"Perception frame failed: {error}")
            self._publish_status(
                state="error",
                ontology="Semantic20",
                source_sequence=sequence,
                source_stamp_ns=source_ns,
                message=str(error),
            )

    def destroy_node(self):
        self._shutdown.set()
        self._mailbox.close()
        if self._worker.is_alive():
            self._worker.join(timeout=10.0)
        if self._worker.is_alive():
            self.get_logger().warning("Inference worker did not stop within 10 seconds")
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = AdomPerceptionNode()
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
