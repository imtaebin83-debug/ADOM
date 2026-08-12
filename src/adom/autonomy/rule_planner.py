from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
import math

import numpy as np

from .costmap import CostmapConfig


@dataclass(frozen=True)
class PlannerConfig:
    wheelbase_m: float = 0.33
    max_steering_deg: float = 20.0
    lookahead_m: float = 3.0
    path_step_m: float = 0.10
    tree_depth: int = 3
    tree_branch_steering_deg: float = 10.0
    steering_change_penalty: float = 4.0
    corridor_half_width_m: float = 0.18
    unknown_cost: float = 70.0
    lethal_cost: int = 90
    stop_distance_m: float = 0.75
    max_speed_mps: float = 0.25
    min_speed_mps: float = 0.08
    steering_penalty: float = 8.0
    distance_decay_m: float = 1.25
    clearance_penalty: float = 35.0
    slow_distance_m: float = 2.0
    gap_enabled: bool = True
    gap_field_of_view_deg: float = 120.0
    gap_ray_count: int = 41
    gap_detection_half_angle_deg: float = 12.0
    gap_trigger_distance_m: float = 3.5
    gap_min_depth_m: float = 0.8
    gap_min_width_m: float = 0.45
    gap_switch_margin_m: float = 0.25
    gap_unknown_penalty_m: float = 0.50


@dataclass(frozen=True)
class GapAnalysis:
    obstacle_detected: bool = False
    obstacle_distance_m: float = math.inf
    selected_side: int = 0
    left_width_m: float = 0.0
    right_width_m: float = 0.0
    left_depth_m: float = 0.0
    right_depth_m: float = 0.0
    left_score: float = 0.0
    right_score: float = 0.0
    left_goal_angle_rad: float = 0.0
    right_goal_angle_rad: float = 0.0
    selected_goal_angle_rad: float = 0.0


@dataclass(frozen=True)
class RulePlan:
    speed_mps: float
    steering_rad: float
    score: float
    blocked: bool
    path_xy: np.ndarray
    clearance_m: float
    steering_sequence_rad: tuple[float, ...] = ()
    gap: GapAnalysis = field(default_factory=GapAnalysis)
    candidate_count: int = 0


def _steering_actions(config: PlannerConfig) -> tuple[float, ...]:
    if config.tree_depth < 1:
        raise ValueError("tree_depth must be at least one")
    if config.tree_branch_steering_deg <= 0.0:
        raise ValueError("tree_branch_steering_deg must be positive")
    actions = np.arange(
        -config.max_steering_deg,
        config.max_steering_deg + config.tree_branch_steering_deg * 0.5,
        config.tree_branch_steering_deg,
    )
    actions = np.append(actions, [-config.max_steering_deg, 0.0, config.max_steering_deg])
    unique = sorted({round(float(value), 9) for value in actions})
    return tuple(math.radians(value) for value in unique)


def _tree_path(sequence: tuple[float, ...], config: PlannerConfig) -> np.ndarray:
    """Integrate one root-to-leaf sequence with the Ackermann bicycle model."""
    segment_m = config.lookahead_m / config.tree_depth
    steps = max(1, int(round(segment_m / config.path_step_m)))
    step_m = segment_m / steps
    x = y = heading = 0.0
    points: list[tuple[float, float]] = []
    for steering in sequence:
        if abs(steering) < 1e-9:
            for _ in range(steps):
                x += step_m * math.cos(heading)
                y += step_m * math.sin(heading)
                points.append((x, y))
            continue
        radius = config.wheelbase_m / math.tan(steering)
        delta_heading = step_m / radius
        for _ in range(steps):
            next_heading = heading + delta_heading
            x += radius * (math.sin(next_heading) - math.sin(heading))
            y -= radius * (math.cos(next_heading) - math.cos(heading))
            heading = next_heading
            points.append((x, y))
    return np.asarray(points, dtype=np.float64)


def _sample_costs(
    grid: np.ndarray,
    path: np.ndarray,
    costmap: CostmapConfig,
    planner: PlannerConfig,
) -> np.ndarray:
    offsets = np.arange(
        -planner.corridor_half_width_m,
        planner.corridor_half_width_m + costmap.resolution_m,
        costmap.resolution_m,
    )
    x = np.repeat(path[:, 0], len(offsets))
    y = (path[:, 1, None] + offsets[None, :]).reshape(-1)
    columns = np.floor(x / costmap.resolution_m).astype(np.int64)
    rows = np.floor((y + costmap.width_m / 2.0) / costmap.resolution_m).astype(
        np.int64
    )
    valid = (
        (rows >= 0)
        & (rows < grid.shape[0])
        & (columns >= 0)
        & (columns < grid.shape[1])
    )
    values = np.full(len(rows), planner.unknown_cost, dtype=np.float64)
    observed = grid[rows[valid], columns[valid]].astype(np.float64)
    values[valid] = np.where(observed < 0, planner.unknown_cost, observed)
    return values.reshape(len(path), len(offsets))


def _widest_ray_gap(
    angles: np.ndarray,
    ranges: np.ndarray,
    observed_ratios: np.ndarray,
    required_range_m: float,
    obstacle_distance_m: float,
    unknown_penalty_m: float,
) -> tuple[float, float, float, float]:
    """Return effective width, mean reach, score, and center angle."""
    eligible = ranges >= required_range_m
    best = (0.0, 0.0, 0.0, 0.0)
    start: int | None = None
    angle_step = abs(float(angles[1] - angles[0])) if len(angles) > 1 else 0.0
    for index in range(len(angles) + 1):
        in_gap = index < len(angles) and bool(eligible[index])
        if in_gap and start is None:
            start = index
        if in_gap or start is None:
            continue
        stop = index
        angular_width = angle_step * (stop - start)
        physical_width = obstacle_distance_m * angular_width
        reach = float(np.mean(ranges[start:stop]))
        observed = float(np.mean(observed_ratios[start:stop]))
        effective_width = max(
            0.0, physical_width - unknown_penalty_m * (1.0 - observed)
        )
        score = effective_width + 0.25 * max(0.0, reach - required_range_m)
        center_angle = float(np.mean(angles[start:stop]))
        best = max(
            best,
            (effective_width, reach, score, center_angle),
            key=lambda value: value[2],
        )
        start = None
    return best


def analyze_directional_gaps(
    grid: np.ndarray,
    costmap: CostmapConfig,
    planner: PlannerConfig,
    preferred_side: int = 0,
) -> GapAnalysis:
    """Measure left/right free angular gaps around a central lethal obstacle.

    Positive lateral coordinates and steering are reported as ``left``. Unknown
    cells remain traversable, matching tree scoring, but reduce the gap width.
    """
    if not planner.gap_enabled:
        return GapAnalysis()
    if planner.gap_ray_count < 5:
        raise ValueError("gap_ray_count must be at least five")
    if planner.gap_field_of_view_deg <= 0.0:
        raise ValueError("gap_field_of_view_deg must be positive")

    half_fov = math.radians(planner.gap_field_of_view_deg / 2.0)
    angles = np.linspace(-half_fov, half_fov, planner.gap_ray_count)
    max_range = min(planner.lookahead_m, costmap.length_m)
    distances = np.arange(
        costmap.resolution_m, max_range + costmap.resolution_m * 0.5,
        costmap.resolution_m,
    )
    offsets = np.arange(
        -planner.corridor_half_width_m,
        planner.corridor_half_width_m + costmap.resolution_m * 0.5,
        costmap.resolution_m,
    )
    x = distances[None, :, None] * np.cos(angles[:, None, None])
    y = (
        distances[None, :, None] * np.sin(angles[:, None, None])
        + offsets[None, None, :]
    )
    columns = np.floor(x / costmap.resolution_m).astype(np.int64)
    rows = np.floor((y + costmap.width_m / 2.0) / costmap.resolution_m).astype(
        np.int64
    )
    columns = np.broadcast_to(columns, rows.shape)
    valid = (
        (rows >= 0)
        & (rows < grid.shape[0])
        & (columns >= 0)
        & (columns < grid.shape[1])
    )
    values = np.full(valid.shape, -1, dtype=np.int16)
    values[valid] = grid[rows[valid], columns[valid]].astype(np.int16)
    lethal_by_step = np.any(values >= planner.lethal_cost, axis=2)
    has_lethal = np.any(lethal_by_step, axis=1)
    first_indices = np.argmax(lethal_by_step, axis=1)
    ranges = np.where(has_lethal, distances[first_indices], max_range)
    observed_ratios = np.mean(np.any(values >= 0, axis=2), axis=1)

    central = np.abs(angles) <= math.radians(planner.gap_detection_half_angle_deg)
    central_ranges = ranges[central & has_lethal]
    if len(central_ranges) == 0:
        return GapAnalysis()
    obstacle_distance = float(np.min(central_ranges))
    if obstacle_distance > planner.gap_trigger_distance_m:
        return GapAnalysis()

    required_range = min(max_range, obstacle_distance + planner.gap_min_depth_m)
    left = angles > 0.0
    right = angles < 0.0
    left_width, left_depth, left_score, left_goal = _widest_ray_gap(
        angles[left], ranges[left], observed_ratios[left], required_range,
        obstacle_distance, planner.gap_unknown_penalty_m,
    )
    right_width, right_depth, right_score, right_goal = _widest_ray_gap(
        angles[right], ranges[right], observed_ratios[right], required_range,
        obstacle_distance, planner.gap_unknown_penalty_m,
    )
    left_feasible = left_width >= planner.gap_min_width_m
    right_feasible = right_width >= planner.gap_min_width_m
    selected_side = 0
    if left_feasible and right_feasible:
        if (
            preferred_side > 0
            and left_score + planner.gap_switch_margin_m >= right_score
        ):
            selected_side = 1
        elif (
            preferred_side < 0
            and right_score + planner.gap_switch_margin_m >= left_score
        ):
            selected_side = -1
        else:
            selected_side = 1 if left_score >= right_score else -1
    elif left_feasible:
        selected_side = 1
    elif right_feasible:
        selected_side = -1
    selected_goal = (
        left_goal if selected_side > 0 else right_goal if selected_side < 0 else 0.0
    )
    return GapAnalysis(
        obstacle_detected=True,
        obstacle_distance_m=obstacle_distance,
        selected_side=selected_side,
        left_width_m=left_width,
        right_width_m=right_width,
        left_depth_m=left_depth,
        right_depth_m=right_depth,
        left_score=left_score,
        right_score=right_score,
        left_goal_angle_rad=left_goal,
        right_goal_angle_rad=right_goal,
        selected_goal_angle_rad=selected_goal,
    )


def plan_corridor(
    grid: np.ndarray,
    costmap: CostmapConfig,
    planner: PlannerConfig,
    preferred_gap_side: int = 0,
) -> RulePlan:
    """Score Ackermann-feasible corridors and return a conservative command."""
    if grid.shape != (costmap.columns, costmap.rows):
        raise ValueError(
            f"grid shape {grid.shape} does not match {(costmap.columns, costmap.rows)}"
        )
    gap = analyze_directional_gaps(grid, costmap, planner, preferred_gap_side)
    if gap.obstacle_detected and gap.selected_side == 0:
        return RulePlan(
            0.0,
            0.0,
            math.inf,
            True,
            np.empty((0, 2), dtype=np.float64),
            gap.obstacle_distance_m,
            gap=gap,
            candidate_count=0,
        )
    actions = _steering_actions(planner)
    first_gap_action = None
    if gap.selected_side:
        side_actions = [
            action for action in actions if action * gap.selected_side > 1e-9
        ]
        first_gap_action = min(
            side_actions,
            key=lambda action: abs(action - gap.selected_goal_angle_rad),
        )
    candidates: list[
        tuple[
            float,
            tuple[float, ...],
            np.ndarray,
            np.ndarray,
            float,
            float,
            int | None,
        ]
    ] = []
    for sequence in product(actions, repeat=planner.tree_depth):
        if first_gap_action is not None and sequence[0] != first_gap_action:
            continue
        path = _tree_path(sequence, planner)
        costs = _sample_costs(grid, path, costmap, planner)
        step_costs = np.max(costs, axis=1)
        distances = np.linalg.norm(path, axis=1)
        weights = np.exp(-distances / max(planner.distance_decay_m, 1e-3))
        distance_weighted_risk = float(np.average(step_costs, weights=weights))
        lethal_steps = np.flatnonzero(step_costs >= planner.lethal_cost)
        lethal_index = int(lethal_steps[0]) if len(lethal_steps) else None
        clearance_m = (
            float(distances[lethal_index])
            if lethal_index is not None
            else planner.lookahead_m + planner.path_step_m
        )
        clearance_ratio = min(1.0, clearance_m / max(planner.lookahead_m, 1e-3))
        steering_magnitudes = [abs(math.degrees(value)) for value in sequence]
        steering_changes = [
            abs(math.degrees(current - previous))
            for previous, current in zip((0.0, *sequence[:-1]), sequence)
        ]
        score = (
            0.45 * float(np.max(step_costs))
            + 0.40 * distance_weighted_risk
            + planner.clearance_penalty * (1.0 - clearance_ratio)
            + planner.steering_penalty
            * float(np.mean(steering_magnitudes))
            / planner.max_steering_deg
            + planner.steering_change_penalty
            * float(np.mean(steering_changes))
            / planner.max_steering_deg
        )
        candidates.append(
            (
                score,
                sequence,
                path,
                step_costs,
                clearance_m,
                distance_weighted_risk,
                lethal_index,
            )
        )

    score, sequence, path, costs, clearance_m, weighted_risk, lethal_index = min(
        candidates, key=lambda item: item[0]
    )
    angle = sequence[0]
    blocked = clearance_m <= planner.stop_distance_m
    if blocked:
        speed = 0.0
    else:
        risk_scale = max(0.0, 1.0 - min(weighted_risk, 100.0) / 100.0)
        turn_scale = 1.0 - 0.55 * abs(angle) / math.radians(planner.max_steering_deg)
        clearance_scale = min(
            1.0,
            max(0.0, clearance_m - planner.stop_distance_m)
            / max(planner.slow_distance_m - planner.stop_distance_m, 1e-3),
        )
        speed = planner.max_speed_mps * risk_scale * turn_scale * clearance_scale
        speed = min(planner.max_speed_mps, max(planner.min_speed_mps, speed))
    if blocked:
        safe_path = np.empty((0, 2), dtype=np.float64)
    elif lethal_index is not None:
        safe_path = path[:lethal_index]
    else:
        safe_path = path
    return RulePlan(
        speed,
        angle,
        float(score),
        blocked,
        safe_path,
        clearance_m,
        tuple(sequence),
        gap,
        len(candidates),
    )
