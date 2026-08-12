from pathlib import Path
import re
import unittest

import numpy as np
import yaml

from adom.perception import (
    LatestItemMailbox,
    SEMANTIC20_PALETTE_BGR,
    colorize_semantic20_mask,
    load_semantic20_ontology,
)


ROOT = Path(__file__).resolve().parents[1]
BRIDGE_MAPPING = (
    ROOT / "src" / "data" / "semantic_20" / "config" / "bridge_mapping.yaml"
)


class LatestFrameMailboxTests(unittest.TestCase):
    def test_pending_frame_is_replaced_by_most_recent_frame(self):
        mailbox: LatestItemMailbox[str] = LatestItemMailbox()
        mailbox.put("frame-01")
        mailbox.put("frame-04")

        item = mailbox.take()

        self.assertIsNotNone(item)
        self.assertEqual(item.value, "frame-04")
        self.assertEqual(item.sequence, 2)
        self.assertEqual(mailbox.received, 2)
        self.assertEqual(mailbox.overwritten, 1)

    def test_close_unblocks_empty_mailbox(self):
        mailbox: LatestItemMailbox[str] = LatestItemMailbox()
        mailbox.close()
        self.assertIsNone(mailbox.take())


class Semantic20PerceptionContractTests(unittest.TestCase):
    def setUp(self):
        self.ontology = load_semantic20_ontology(BRIDGE_MAPPING)

    def test_bridge_mapping_is_canonical_semantic20_contract(self):
        self.assertEqual(self.ontology.num_classes, 19)
        self.assertEqual(self.ontology.ignore_index, 255)
        self.assertEqual(self.ontology.classes[0], "dirt")
        self.assertEqual(self.ontology.classes[18], "rubble")

    def test_colorize_uses_semantic20_palette_and_black_ignore(self):
        mask = np.asarray([[0, 18, 255]], dtype=np.uint8)
        output = colorize_semantic20_mask(mask, self.ontology)
        np.testing.assert_array_equal(output[0, 0], SEMANTIC20_PALETTE_BGR[0])
        np.testing.assert_array_equal(output[0, 1], SEMANTIC20_PALETTE_BGR[18])
        np.testing.assert_array_equal(output[0, 2], [0, 0, 0])

    def test_unknown_id_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "invalid IDs"):
            self.ontology.validate_mask(np.asarray([[19]], dtype=np.uint8))

    def test_cost4_and_semantic20_ros_configs_are_separate(self):
        config_root = ROOT / "ros2_ws" / "src" / "adom_perception_ros" / "config"
        cost4 = yaml.safe_load((config_root / "perception.yaml").read_text())
        semantic20 = yaml.safe_load(
            (config_root / "perception_semantic20.yaml").read_text()
        )
        cost4_params = cost4["adom_perception"]["ros__parameters"]
        semantic20_params = semantic20["adom_perception"]["ros__parameters"]
        self.assertEqual(cost4_params["mask_topic"], "/adom/perception/semantic_mask")
        self.assertEqual(
            semantic20_params["mask_topic"], "/adom/perception/semantic20_mask"
        )
        self.assertNotIn("bridge_mapping_path", cost4_params)
        self.assertEqual(semantic20_params["target_fps"], 30.0)

    def test_live_autonomy_bag_is_numeric_status_only(self):
        config = yaml.safe_load(
            (
                ROOT
                / "ros2_ws"
                / "src"
                / "adom_logging"
                / "config"
                / "autonomy_logging.yaml"
            ).read_text()
        )
        topic_regex = config["autonomy_data_recorder"]["ros__parameters"][
            "topic_regex"
        ]
        matcher = re.compile(topic_regex)
        for required_topic in (
            "/adom/perception/status",
            "/adom/navigation/rule_status",
            "/adom/navigation/planned_speed",
            "/adom/control/local_path_status",
            "/adom/control/mode",
            "/adom/control/pwm_us",
            "/cmd_vel",
            "/drive/autonomous",
            "/drive",
            "/emergency_stop",
            "/fix",
        ):
            self.assertIsNotNone(matcher.fullmatch(required_topic))
        for high_load_topic in (
            "/adom/perception/semantic20_mask",
            "/adom/navigation/semantic_costmap",
            "/adom/navigation/local_path",
            "/adom/navigation/rule_path",
            "/adom/logging/gps_path",
            "/zed/zed_node/imu/data",
            "/tf",
            "/tf_static",
        ):
            self.assertIsNone(matcher.fullmatch(high_load_topic))
        self.assertNotIn("confidence", topic_regex)
        self.assertNotIn("overlay", topic_regex)


if __name__ == "__main__":
    unittest.main()
