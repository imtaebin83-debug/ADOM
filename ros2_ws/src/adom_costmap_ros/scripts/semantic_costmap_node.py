#!/usr/bin/env python3
from __future__ import annotations

from collections import deque
import json
import time

import cv2
from cv_bridge import CvBridge
import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener

from adom.autonomy.costmap import (
    CostmapConfig,
    build_costmap,
    project_mask_depth_with_diagnostics,
    quaternion_matrix,
)
from adom.perception import load_semantic20_ontology


def stamp_ns(message) -> int:
    return int(message.header.stamp.sec) * 1_000_000_000 + int(
        message.header.stamp.nanosec
    )


def elapsed_ms_if_compatible(later_ns: int, earlier_ns: int) -> float | None:
    if earlier_ns <= 0:
        return None
    elapsed_ns = later_ns - earlier_ns
    # A live costmap cannot legitimately take a minute. Treat a larger delta
    # as evidence that the sensor and ROS clocks have different epochs.
    if elapsed_ns < 0 or elapsed_ns > 60_000_000_000:
        return None
    return round(elapsed_ns / 1e6, 2)


class SemanticCostmapNode(Node):
    def __init__(self) -> None:
        super().__init__("semantic_costmap")
        defaults = {
            "mask_topic": "/adom/perception/semantic_mask",
            "ontology": "cost4",
            "bridge_mapping_path": "",
            "depth_topic": "/zed/zed_node/depth/depth_registered",
            "camera_info_topic": "/zed/zed_node/rgb/color/rect/camera_info",
            "costmap_topic": "/adom/navigation/semantic_costmap",
            "status_topic": "/adom/navigation/costmap_status",
            "output_frame": "base_link",
            "resolution_m": 0.10,
            "length_m": 8.0,
            "width_m": 6.0,
            "min_range_m": 0.30,
            "max_range_m": 8.0,
            "sample_stride": 4,
            "min_height_m": -0.05,
            "max_height_m": 1.50,
            "class_costs": [0, 15, 60, 100],
            "geometric_obstacle_min_height_m": 0.10,
            "inflation_radius_m": 0.25,
            "inflation_seed_cost": 90,
            "inflation_min_cost": 60,
            "depth_queue_size": 15,
            "max_sync_error_sec": 0.35,
            "transform_timeout_sec": 0.10,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        self.p = {name: self.get_parameter(name).value for name in defaults}
        costs = tuple(int(value) for value in self.p["class_costs"])
        ontology = str(self.p["ontology"]).strip().lower()
        if ontology == "semantic20":
            mapping_path = str(self.p["bridge_mapping_path"]).strip() or None
            expected_classes = load_semantic20_ontology(mapping_path).num_classes
        elif ontology == "cost4":
            expected_classes = 4
        else:
            raise ValueError("ontology must be 'semantic20' or 'cost4'")
        if len(costs) != expected_classes or any(
            value < 0 or value > 100 for value in costs
        ):
            raise ValueError(
                f"class_costs must contain {expected_classes} values in [0,100]"
            )
        inflation_seed_cost = int(self.p["inflation_seed_cost"])
        inflation_min_cost = int(self.p["inflation_min_cost"])
        if not 0 <= inflation_min_cost <= inflation_seed_cost <= 100:
            raise ValueError(
                "inflation costs must satisfy "
                "0 <= inflation_min_cost <= inflation_seed_cost <= 100"
            )
        self._ontology = ontology
        self._config = CostmapConfig(
            resolution_m=float(self.p["resolution_m"]),
            length_m=float(self.p["length_m"]),
            width_m=float(self.p["width_m"]),
            min_range_m=float(self.p["min_range_m"]),
            max_range_m=float(self.p["max_range_m"]),
            sample_stride=int(self.p["sample_stride"]),
            min_height_m=float(self.p["min_height_m"]),
            max_height_m=float(self.p["max_height_m"]),
            class_costs=costs,
            geometric_obstacle_min_height_m=float(
                self.p["geometric_obstacle_min_height_m"]
            ),
            inflation_radius_m=float(self.p["inflation_radius_m"]),
            inflation_seed_cost=inflation_seed_cost,
            inflation_min_cost=inflation_min_cost,
        )
        self._bridge = CvBridge()
        self._camera_info: CameraInfo | None = None
        self._depth_messages: deque[Image] = deque(maxlen=int(self.p["depth_queue_size"]))
        self._tf_buffer = Buffer(cache_time=Duration(seconds=10.0))
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._costmap_pub = self.create_publisher(
            OccupancyGrid, str(self.p["costmap_topic"]), 1
        )
        self._status_pub = self.create_publisher(String, str(self.p["status_topic"]), 10)
        self.create_subscription(
            CameraInfo,
            str(self.p["camera_info_topic"]),
            self._on_camera_info,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Image,
            str(self.p["depth_topic"]),
            self._on_depth,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Image,
            str(self.p["mask_topic"]),
            self._on_mask,
            qos_profile_sensor_data,
        )

    def _on_camera_info(self, message: CameraInfo) -> None:
        self._camera_info = message

    def _on_depth(self, message: Image) -> None:
        self._depth_messages.append(message)

    def _closest_depth(self, mask: Image) -> tuple[Image | None, float]:
        if not self._depth_messages:
            return None, float("inf")
        target = stamp_ns(mask)
        closest = min(
            self._depth_messages, key=lambda item: abs(stamp_ns(item) - target)
        )
        return closest, abs(stamp_ns(closest) - target) / 1e9

    def _depth_metres(self, message: Image) -> np.ndarray:
        if message.encoding == "32FC1":
            return self._bridge.imgmsg_to_cv2(message, "32FC1").astype(
                np.float32, copy=False
            )
        if message.encoding in ("16UC1", "mono16"):
            millimetres = self._bridge.imgmsg_to_cv2(message, "16UC1")
            return millimetres.astype(np.float32) * 0.001
        raise ValueError(f"unsupported registered depth encoding: {message.encoding}")

    def _publish_status(self, state: str, **fields) -> None:
        message = String()
        message.data = json.dumps({"state": state, **fields}, sort_keys=True)
        self._status_pub.publish(message)

    def _on_mask(self, mask_message: Image) -> None:
        processing_started = time.monotonic()
        if self._camera_info is None:
            self._publish_status("waiting", reason="camera_info")
            return
        depth_message, sync_error = self._closest_depth(mask_message)
        if depth_message is None or sync_error > float(self.p["max_sync_error_sec"]):
            self._publish_status(
                "waiting",
                reason="synchronized_depth",
                sync_error_sec=round(sync_error, 3),
            )
            return
        try:
            mask = self._bridge.imgmsg_to_cv2(mask_message, "mono8")
            depth = self._depth_metres(depth_message)
            if depth.shape != mask.shape:
                depth = cv2.resize(
                    depth,
                    (mask.shape[1], mask.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )

            info = self._camera_info
            scale_x = mask.shape[1] / max(float(info.width), 1.0)
            scale_y = mask.shape[0] / max(float(info.height), 1.0)
            intrinsics = (
                float(info.k[0]) * scale_x,
                float(info.k[4]) * scale_y,
                float(info.k[2]) * scale_x,
                float(info.k[5]) * scale_y,
            )
            transform = self._tf_buffer.lookup_transform(
                str(self.p["output_frame"]),
                depth_message.header.frame_id,
                Time.from_msg(depth_message.header.stamp),
                timeout=Duration(seconds=float(self.p["transform_timeout_sec"])),
            )
            q = transform.transform.rotation
            t = transform.transform.translation
            rotation = quaternion_matrix(q.x, q.y, q.z, q.w)
            translation = np.asarray([t.x, t.y, t.z], dtype=np.float64)
            points, labels, projection = project_mask_depth_with_diagnostics(
                mask, depth, intrinsics, rotation, translation, self._config
            )
            grid = build_costmap(points, labels, self._config)
            if len(points):
                grid_in_bounds = (
                    (points[:, 0] >= 0.0)
                    & (points[:, 0] < self._config.length_m)
                    & (points[:, 1] >= -self._config.width_m / 2.0)
                    & (points[:, 1] < self._config.width_m / 2.0)
                )
                grid_in_bounds_points = int(np.count_nonzero(grid_in_bounds))
            else:
                grid_in_bounds_points = 0
            observed_cells = int(np.count_nonzero(grid >= 0))
            empty_reason = None
            if observed_cells == 0:
                if projection.in_range_depth_pixels == 0:
                    empty_reason = "no_depth_in_range"
                elif projection.depth_label_pixels == 0:
                    empty_reason = "no_depth_with_semantic_label"
                elif projection.height_valid_points == 0:
                    empty_reason = "height_filter"
                elif grid_in_bounds_points == 0:
                    empty_reason = "outside_costmap"
                else:
                    empty_reason = "rasterization"

            # Preserve camera timestamps through mask/depth synchronization and
            # TF lookup, then cross into the planner's ROS clock domain here.
            output_stamp = self.get_clock().now()
            output = OccupancyGrid()
            output.header.stamp = output_stamp.to_msg()
            output.header.frame_id = str(self.p["output_frame"])
            output.info.resolution = float(self._config.resolution_m)
            output.info.width = self._config.rows
            output.info.height = self._config.columns
            output.info.origin.position.x = 0.0
            output.info.origin.position.y = -self._config.width_m / 2.0
            output.info.origin.orientation.w = 1.0
            output.data = grid.reshape(-1).astype(np.int8).tolist()
            self._costmap_pub.publish(output)
            output_ns = output_stamp.nanoseconds
            source_ns = stamp_ns(mask_message)
            self._publish_status(
                "ok",
                ontology=self._ontology,
                projected_points=int(len(points)),
                observed_cells=observed_cells,
                empty_reason=empty_reason,
                sampled_pixels=projection.sampled_pixels,
                finite_depth_pixels=projection.finite_depth_pixels,
                in_range_depth_pixels=projection.in_range_depth_pixels,
                semantic_label_pixels=projection.semantic_label_pixels,
                depth_label_pixels=projection.depth_label_pixels,
                height_valid_points=projection.height_valid_points,
                transformed_z_min_m=projection.transformed_z_min_m,
                transformed_z_max_m=projection.transformed_z_max_m,
                grid_in_bounds_points=grid_in_bounds_points,
                sync_error_sec=round(sync_error, 3),
                processing_ms=round(
                    (time.monotonic() - processing_started) * 1000.0, 2
                ),
                source_to_costmap_output_ms=elapsed_ms_if_compatible(
                    output_ns, source_ns
                ),
            )
        except TransformException as error:
            self._publish_status("waiting", reason="transform", message=str(error))
        except Exception as error:
            self.get_logger().error(f"Semantic costmap update failed: {error}")
            self._publish_status("error", message=str(error))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SemanticCostmapNode()
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
