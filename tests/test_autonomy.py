import math
import unittest

import numpy as np

from adom.autonomy import CostmapConfig, PlannerConfig, build_costmap, plan_corridor
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

    def test_left_side_cost_selects_right_turn(self):
        grid = np.zeros((self.costmap.columns, self.costmap.rows), dtype=np.int8)
        center = self.costmap.columns // 2
        grid[center:, 10:31] = 100
        plan = plan_corridor(grid, self.costmap, self.planner)
        self.assertFalse(plan.blocked)
        self.assertLess(math.degrees(plan.steering_rad), 0.0)


if __name__ == "__main__":
    unittest.main()
