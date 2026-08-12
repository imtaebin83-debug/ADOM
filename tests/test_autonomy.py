import math
from pathlib import Path
import unittest

import numpy as np
import yaml

from adom.autonomy import (
    CostmapConfig,
    ImuSpeedEstimator,
    ImuSpeedEstimatorConfig,
    PathControlConfig,
    PlannerConfig,
    StuckRecoveryConfig,
    StuckRecoveryGate,
    build_costmap,
    control_local_path,
    gps_speed_mps,
    local_gps_xy_m,
    plan_corridor,
    speed_to_pwm_us,
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

    def test_inflation_does_not_promote_nonlethal_semantic_cost(self):
        config = CostmapConfig(
            class_costs=(0, 85, 100),
            inflation_radius_m=0.20,
            resolution_m=0.10,
            inflation_seed_cost=90,
        )
        grid = build_costmap(
            np.asarray([[1.0, 0.0, 0.0]]), np.asarray([1], dtype=np.uint8), config
        )
        row = int(config.width_m / 2.0 / config.resolution_m)
        column = int(1.0 / config.resolution_m)
        self.assertEqual(int(grid[row, column]), 85)
        self.assertEqual(int(grid[row + 1, column]), -1)

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

    def test_below_ground_depth_overshoot_is_rejected_after_tf(self):
        config = CostmapConfig(sample_stride=1)
        points, labels = project_mask_depth(
            np.asarray([[3]], dtype=np.uint8),
            np.asarray([[1.0]], dtype=np.float32),
            (1.0, 1.0, 0.0, 0.0),
            # ROS optical (X right, Y down, Z forward) -> base_link
            # (X forward, Y left, Z up).
            np.asarray(
                [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]
            ),
            np.asarray([0.0, 0.0, -0.06]),
            config,
        )
        self.assertEqual(len(points), 0)
        self.assertEqual(len(labels), 0)

    def test_zed_depth_config_matches_costmap_range_and_mount_height(self):
        root = Path(__file__).resolve().parents[1]
        zed = yaml.safe_load(
            (root / "ros2_ws/src/adom_sensors/config/zed2i.yaml").read_text()
        )["/**"]["ros__parameters"]["depth"]
        costmap = yaml.safe_load(
            (
                root
                / "ros2_ws/src/adom_costmap_ros/config/semantic20_costs.yaml"
            ).read_text()
        )["semantic_costmap"]["ros__parameters"]
        urdf = (
            root / "ros2_ws/src/adom_description/urdf/adom_vehicle.urdf.xacro"
        ).read_text()

        self.assertEqual(zed["depth_mode"], "NEURAL_LIGHT")
        self.assertEqual(zed["min_depth"], costmap["min_range_m"])
        self.assertEqual(zed["max_depth"], costmap["max_range_m"])
        self.assertEqual(zed["depth_confidence"], 50)
        self.assertEqual(zed["depth_texture_conf"], 50)
        self.assertIn('name="zed_z" default="0.21"', urdf)


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
            tree_depth=1,
            tree_branch_steering_deg=1.0,
            side_cost_enabled=False,
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

    def test_tree_plan_exposes_one_direction_choice_per_depth(self):
        grid = np.zeros((self.costmap.columns, self.costmap.rows), dtype=np.int8)
        plan = plan_corridor(
            grid,
            self.costmap,
            PlannerConfig(tree_depth=3, tree_branch_steering_deg=10.0),
        )
        self.assertEqual(len(plan.steering_sequence_rad), 3)
        self.assertTrue(all(abs(value) < 1e-9 for value in plan.steering_sequence_rad))

    def test_lower_left_half_cost_fixes_first_action_and_reduces_tree_to_25(self):
        grid = np.zeros((self.costmap.columns, self.costmap.rows), dtype=np.int8)
        center = self.costmap.columns // 2
        grid[:center, :] = 100
        plan = plan_corridor(grid, self.costmap, self.planner)
        self.assertTrue(plan.side_cost.active)
        self.assertEqual(plan.side_cost.selected_side, 1)
        self.assertLess(plan.side_cost.left_cost, plan.side_cost.right_cost)
        self.assertGreater(plan.steering_rad, 0.0)
        self.assertEqual(plan.candidate_count, 25)

    def test_lower_right_half_cost_fixes_first_action_and_reduces_tree_to_25(self):
        grid = np.zeros((self.costmap.columns, self.costmap.rows), dtype=np.int8)
        center = self.costmap.columns // 2
        grid[center:, :] = 100
        plan = plan_corridor(grid, self.costmap, self.planner)
        self.assertTrue(plan.side_cost.active)
        self.assertEqual(plan.side_cost.selected_side, -1)
        self.assertLess(plan.side_cost.right_cost, plan.side_cost.left_cost)
        self.assertLess(plan.steering_rad, 0.0)
        self.assertEqual(plan.candidate_count, 25)

    def test_side_cost_helper_does_not_add_blocked_condition(self):
        grid = np.zeros((self.costmap.columns, self.costmap.rows), dtype=np.int8)
        center = self.costmap.columns // 2
        grid[center - 2 : center + 3, 5:7] = 100
        grid[:center, 20:30] = 75
        plan = plan_corridor(
            grid, self.costmap, PlannerConfig(stop_distance_m=0.30)
        )
        self.assertTrue(plan.side_cost.active)
        self.assertFalse(plan.blocked)
        self.assertEqual(plan.candidate_count, 25)

    def test_clear_scene_keeps_full_tree_and_straight_path(self):
        grid = np.zeros((self.costmap.columns, self.costmap.rows), dtype=np.int8)
        plan = plan_corridor(grid, self.costmap, self.planner)
        self.assertFalse(plan.side_cost.active)
        self.assertEqual(plan.candidate_count, 125)
        self.assertAlmostEqual(plan.steering_rad, 0.0)


class LocalPathControlTests(unittest.TestCase):
    def test_ros_speed_limits_are_consistent_across_pipeline(self):
        root = Path(__file__).resolve().parents[1]
        planner = yaml.safe_load(
            (root / "ros2_ws/src/adom_planning/config/rule_planner.yaml").read_text()
        )["rule_planner"]["ros__parameters"]
        local = yaml.safe_load(
            (root / "ros2_ws/src/adom_control/config/local_path_control.yaml").read_text()
        )["local_path_control"]["ros__parameters"]
        vehicle = yaml.safe_load(
            (root / "ros2_ws/src/adom_control/config/vehicle.yaml").read_text()
        )
        pca = vehicle["pca9685_control"]["ros__parameters"]
        gamepad = vehicle["gamepad_control"]["ros__parameters"]

        self.assertEqual(planner["min_speed_mps"], 0.10)
        self.assertEqual(planner["max_speed_mps"], 1.0)
        self.assertEqual(local["max_speed_mps"], planner["max_speed_mps"])
        self.assertEqual(planner["max_steering_deg"], 24.0)
        self.assertEqual(local["max_steering_deg"], planner["max_steering_deg"])
        self.assertEqual(planner["lookahead_m"], 4.0)
        self.assertEqual(planner["slow_distance_m"], 3.0)
        self.assertTrue(planner["side_cost_enabled"])
        self.assertEqual(
            planner["downstream_max_speed_mps"], gamepad["max_forward_speed_mps"]
        )
        self.assertEqual(
            local["downstream_max_speed_mps"], pca["max_speed_mps"]
        )
        self.assertEqual(pca["max_speed_mps"], 15.0)
        self.assertEqual(pca["esc_neutral_us"], 1500.0)
        self.assertEqual(pca["esc_forward_max_us"], 2000.0)

    def test_nominal_forward_speed_maps_linearly_to_pwm(self):
        def pulse(speed_mps):
            return speed_to_pwm_us(
                speed_mps,
                max_forward_speed_mps=12.0,
                max_reverse_speed_mps=3.0,
                neutral_us=1500.0,
                forward_max_us=2000.0,
                reverse_max_us=1000.0,
            )

        self.assertEqual(pulse(0.0), 1500.0)
        self.assertAlmostEqual(pulse(0.25), 1578.9583333)
        self.assertEqual(pulse(3.0), 1677.5)
        self.assertEqual(pulse(12.0), 2000.0)

    def test_planner_speed_profile_is_preserved(self):
        command = control_local_path(
            np.asarray([[0.5, 0.0], [1.5, 0.0], [3.0, 0.0]]),
            0.0,
            PathControlConfig(max_speed_mps=3.0, min_speed_mps=0.25, speed_kp=0.0),
            planned_speed_mps=2.4,
        )
        self.assertAlmostEqual(command.speed_mps, 2.4)

    def test_planner_stop_overrides_nonempty_path(self):
        command = control_local_path(
            np.asarray([[0.5, 0.0], [1.5, 0.0], [3.0, 0.0]]),
            0.0,
            PathControlConfig(max_speed_mps=3.0, min_speed_mps=0.25, speed_kp=0.0),
            planned_speed_mps=0.0,
        )
        self.assertEqual(command.speed_mps, 0.0)

    def test_imu_stationary_updates_learn_bias_and_zero_velocity(self):
        estimator = ImuSpeedEstimator(
            ImuSpeedEstimatorConfig(bias_learning_rate=0.5)
        )
        first = estimator.update(0.2, 0, stationary=True)
        second = estimator.update(0.2, 10_000_000, stationary=True)
        self.assertTrue(first.stationary_update)
        self.assertAlmostEqual(first.bias_mps2, 0.2)
        self.assertAlmostEqual(second.bias_mps2, 0.2)
        self.assertEqual(second.speed_mps, 0.0)

    def test_imu_integrates_bias_corrected_acceleration_while_driving(self):
        estimator = ImuSpeedEstimator(
            ImuSpeedEstimatorConfig(
                initial_bias_mps2=0.2,
                velocity_leak_per_sec=0.0,
            )
        )
        estimator.update(0.2, 0, stationary=True)
        estimate = estimator.update(1.2, 100_000_000, stationary=False)
        self.assertAlmostEqual(estimate.corrected_accel_mps2, 1.0)
        self.assertAlmostEqual(estimate.speed_mps, 0.1)

    def test_imu_feedback_reduces_command_when_estimate_is_above_target(self):
        command = control_local_path(
            np.asarray([[0.5, 0.0], [1.5, 0.0], [3.0, 0.0]]),
            2.0,
            PathControlConfig(max_speed_mps=3.0, speed_kp=0.15),
            planned_speed_mps=1.5,
        )
        self.assertAlmostEqual(command.speed_mps, 1.425)

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

    def test_logging_gps_trail_uses_local_east_north_coordinates(self):
        east, north = local_gps_xy_m(0.0, 0.0, 0.000001, 0.000002)
        self.assertAlmostEqual(east, 0.222, places=2)
        self.assertAlmostEqual(north, 0.111, places=2)


class StuckRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.gate = StuckRecoveryGate(
            StuckRecoveryConfig(hold_sec=1.0, duration_sec=0.75)
        )

    def update(
        self,
        time_sec,
        *,
        path_valid=True,
        speed=0.4,
        steering_deg=18.0,
        estimated_speed=0.0,
        yaw_rate=0.0,
    ):
        return self.gate.update(
            int(time_sec * 1e9),
            path_valid=path_valid,
            commanded_speed_mps=speed,
            steering_rad=math.radians(steering_deg),
            estimated_speed_mps=estimated_speed,
            yaw_rate_rps=yaw_rate,
        )

    def test_stable_turn_with_no_motion_gets_one_bounded_attempt(self):
        self.assertEqual(self.update(0.0).state, "candidate")
        self.assertEqual(self.update(0.5).state, "candidate")
        active = self.update(1.0)
        self.assertTrue(active.active)
        self.assertEqual(active.state, "active")
        self.assertTrue(self.update(1.5).active)
        exhausted = self.update(1.8)
        self.assertFalse(exhausted.active)
        self.assertEqual(exhausted.state, "exhausted")
        self.assertEqual(self.update(3.0).state, "exhausted")

    def test_straight_path_never_triggers_recovery(self):
        self.assertEqual(self.update(0.0, steering_deg=0.0).state, "inactive")
        self.assertEqual(self.update(2.0, steering_deg=0.0).state, "inactive")

    def test_motion_evidence_rearms_after_an_attempt(self):
        self.update(0.0)
        self.assertTrue(self.update(1.0).active)
        moving = self.update(1.1, estimated_speed=0.2)
        self.assertEqual(moving.state, "moving")
        self.assertFalse(moving.active)
        self.assertEqual(self.update(1.2).state, "candidate")

    def test_invalid_or_blocked_path_cancels_candidate(self):
        self.update(0.0)
        stopped = self.update(1.0, path_valid=False)
        self.assertEqual(stopped.state, "inactive")
        self.assertFalse(stopped.active)


if __name__ == "__main__":
    unittest.main()
