import math
import unittest

import numpy as np

from adom.autonomy import (
    CostmapConfig,
    PathControlConfig,
    PlannerConfig,
    build_costmap,
    control_local_path,
    gps_speed_mps,
    plan_corridor,
)
from adom.autonomy.costmap import project_mask_depth
from adom.perception import COST4_PALETTE_BGR, colorize_mask


class PerceptionVisualizationTests(unittest.TestCase):
    def test_cost4_mask_uses_committed_bgr_palette(self):
        mask = np.asarray([[0, 1, 2, 3]], dtype=np.uint8)
        output = colorize_mask(mask)
        np.testing.assert_array_equal(output[0], COST4_PALETTE_BGR)

    def test_colorize_rejects_non_mask_input(self):
        with self.assertRaisesRegex(ValueError, "HxW"):
            colorize_mask(np.zeros((2, 2, 1), dtype=np.uint8))


class SemanticCostmapTests(unittest.TestCase):
    SEMANTIC20_COSTS = (
        20, 30, 100, 100, 100, 100, 100, 100, 0, 100,
        100, 100, 100, 75, 10, 100, 75, 85, 95,
    )

    def test_aligned_depth_projects_pixels_with_intrinsics(self):
        mask = np.asarray([[0, 3]], dtype=np.uint8)
        depth = np.asarray([[2.0, 2.0]], dtype=np.float32)
        config = CostmapConfig(
            sample_stride=1, inflation_radius_m=0.0, max_height_m=3.0
        )
        points, labels = project_mask_depth(
            mask,
            depth,
            (2.0, 2.0, 0.0, 0.0),
            np.eye(3),
            np.zeros(3),
            config,
        )
        np.testing.assert_allclose(points, [[0.0, 0.0, 2.0], [1.0, 0.0, 2.0]])
        np.testing.assert_array_equal(labels, [0, 3])

    def test_highest_semantic_cost_wins_in_shared_cell(self):
        config = CostmapConfig(inflation_radius_m=0.0)
        points = np.asarray([[1.0, 0.0, 0.0], [1.01, 0.01, 0.0]])
        grid = build_costmap(points, np.asarray([0, 3], dtype=np.uint8), config)
        row = int(config.width_m / 2.0 / config.resolution_m)
        column = int(1.0 / config.resolution_m)
        self.assertEqual(int(grid[row, column]), 100)

    def test_inflation_marks_vehicle_clearance_around_danger(self):
        config = CostmapConfig(inflation_radius_m=0.20, resolution_m=0.10)
        grid = build_costmap(
            np.asarray([[1.0, 0.0, 0.0]]), np.asarray([3], dtype=np.uint8), config
        )
        row = int(config.width_m / 2.0 / config.resolution_m)
        column = int(1.0 / config.resolution_m)
        self.assertEqual(int(grid[row, column]), 100)
        self.assertGreaterEqual(int(grid[row + 1, column]), 60)

    def test_vertical_geometry_is_lethal_even_if_semantically_traversable(self):
        config = CostmapConfig(inflation_radius_m=0.0)
        grid = build_costmap(
            np.asarray([[1.0, 0.0, 0.20]]), np.asarray([0], dtype=np.uint8), config
        )
        row = int(config.width_m / 2.0 / config.resolution_m)
        column = int(1.0 / config.resolution_m)
        self.assertEqual(int(grid[row, column]), 100)

    def test_semantic20_log_is_projected_and_rasterized_as_lethal(self):
        config = CostmapConfig(
            sample_stride=1,
            inflation_radius_m=0.0,
            max_height_m=3.0,
            class_costs=self.SEMANTIC20_COSTS,
        )
        points, labels = project_mask_depth(
            np.asarray([[10]], dtype=np.uint8),
            np.asarray([[1.0]], dtype=np.float32),
            (1.0, 1.0, 0.0, 0.0),
            np.asarray(
                [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]
            ),
            np.zeros(3),
            config,
        )
        grid = build_costmap(points, labels, config)
        row = int(config.width_m / 2.0 / config.resolution_m)
        column = int(1.0 / config.resolution_m)
        self.assertEqual(int(grid[row, column]), 100)

    def test_semantic20_ignore_is_not_projected(self):
        config = CostmapConfig(
            sample_stride=1,
            class_costs=self.SEMANTIC20_COSTS,
        )
        points, labels = project_mask_depth(
            np.asarray([[255]], dtype=np.uint8),
            np.asarray([[1.0]], dtype=np.float32),
            (1.0, 1.0, 0.0, 0.0),
            np.eye(3),
            np.zeros(3),
            config,
        )
        self.assertEqual(len(points), 0)
        self.assertEqual(len(labels), 0)


class RulePlannerTests(unittest.TestCase):
    def setUp(self):
        self.costmap = CostmapConfig(inflation_radius_m=0.0)
        self.planner = PlannerConfig()

    def test_clear_corridor_drives_straight_at_limited_speed(self):
        grid = np.zeros((self.costmap.columns, self.costmap.rows), dtype=np.int8)
        plan = plan_corridor(grid, self.costmap, self.planner)
        self.assertFalse(plan.blocked)
        self.assertAlmostEqual(plan.steering_rad, 0.0)
        self.assertLessEqual(plan.speed_mps, 0.25)

    def test_near_full_width_obstacle_stops(self):
        grid = np.zeros((self.costmap.columns, self.costmap.rows), dtype=np.int8)
        grid[:, 2:10] = 100
        plan = plan_corridor(grid, self.costmap, self.planner)
        self.assertTrue(plan.blocked)
        self.assertEqual(plan.speed_mps, 0.0)
        self.assertEqual(len(plan.path_xy), 0)

    def test_left_side_cost_selects_right_turn(self):
        grid = np.zeros((self.costmap.columns, self.costmap.rows), dtype=np.int8)
        center = self.costmap.columns // 2
        grid[center:, 10:31] = 100
        plan = plan_corridor(grid, self.costmap, self.planner)
        self.assertFalse(plan.blocked)
        self.assertLess(math.degrees(plan.steering_rad), 0.0)

    def test_near_depth_obstacle_slows_more_than_far_obstacle(self):
        distance_planner = PlannerConfig(
            max_steering_deg=1.0,
            steering_step_deg=1.0,
        )
        near = np.zeros((self.costmap.columns, self.costmap.rows), dtype=np.int8)
        far = near.copy()
        near[:, 10:12] = 100
        far[:, 25:27] = 100
        near_plan = plan_corridor(near, self.costmap, distance_planner)
        far_plan = plan_corridor(far, self.costmap, distance_planner)
        self.assertFalse(near_plan.blocked)
        self.assertFalse(far_plan.blocked)
        self.assertLess(near_plan.clearance_m, far_plan.clearance_m)
        self.assertLess(near_plan.speed_mps, far_plan.speed_mps)
        self.assertLess(len(near_plan.path_xy), len(far_plan.path_xy))


class LocalPathControlTests(unittest.TestCase):
    def test_short_safe_path_commands_stop(self):
        command = control_local_path(
            np.asarray([[0.2, 0.0], [0.4, 0.0]]),
            0.0,
            PathControlConfig(),
        )
        self.assertAlmostEqual(command.available_path_m, 0.4)
        self.assertEqual(command.speed_mps, 0.0)

    def test_path_clearance_scales_speed(self):
        config = PathControlConfig()
        short = control_local_path(
            np.asarray([[0.4, 0.0], [0.8, 0.0]]), 0.0, config
        )
        long = control_local_path(
            np.asarray([[0.5, 0.0], [1.0, 0.0], [2.0, 0.0]]), 0.0, config
        )
        self.assertLess(short.speed_mps, long.speed_mps)

    def test_pure_pursuit_turns_toward_path_and_slows_for_curvature(self):
        config = PathControlConfig()
        straight = control_local_path(
            np.asarray([[0.4, 0.0], [0.8, 0.0], [1.2, 0.0]]), 0.0, config
        )
        left = control_local_path(
            np.asarray([[0.4, 0.1], [0.8, 0.35], [1.2, 0.7]]), 0.0, config
        )
        self.assertGreater(left.steering_rad, 0.0)
        self.assertLess(left.speed_mps, straight.speed_mps)
        self.assertLessEqual(left.speed_mps, 0.25)

    def test_gps_speed_uses_position_and_timestamp_delta(self):
        # About 0.22 m east at the equator over one second.
        speed = gps_speed_mps((0.0, 0.0, 0), (0.0, 0.000002, 1_000_000_000))
        self.assertIsNotNone(speed)
        self.assertAlmostEqual(speed, 0.222, places=2)

    def test_gps_speed_rejects_implausible_jump(self):
        speed = gps_speed_mps((0.0, 0.0, 0), (0.0, 1.0, 1_000_000_000))
        self.assertIsNone(speed)


if __name__ == "__main__":
    unittest.main()
