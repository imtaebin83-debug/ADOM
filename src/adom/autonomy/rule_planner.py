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
    stop_distance_m: float = 0.30
    max_speed_mps: float = 1.0
    min_speed_mps: float = 0.10
    steering_penalty: float = 8.0
    distance_decay_m: float = 1.25
    clearance_penalty: float = 35.0
    slow_distance_m: float = 2.0
    side_cost_enabled: bool = True
    avoid_trigger_distance_m: float = 1.5


@dataclass(frozen=True)
class SideCostAnalysis:
    active: bool = False
    mode: str = "straight"
    selected_side: int = 0
    left_cost: float = 0.0
    right_cost: float = 0.0
    straight_obstacle_distance_m: float = math.inf


@dataclass(frozen=True)
class RulePlan:
    speed_mps: float
    steering_rad: float
    score: float
    blocked: bool
    path_xy: np.ndarray
    clearance_m: float
    steering_sequence_rad: tuple[float, ...] = ()
    side_cost: SideCostAnalysis = field(default_factory=SideCostAnalysis)
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


def analyze_side_costs(
    grid: np.ndarray,
    planner: PlannerConfig,
) -> SideCostAnalysis:
    """Choose the lower-cost half of the complete robot-frame costmap.

    Positive lateral rows are ``left``. Unknown cells use the same conservative
    cost as tree scoring. A tie deterministically selects left so AVOID mode
    always evaluates 25 trees. This helper never decides BLOCKED.
    """
    if not planner.side_cost_enabled:
        return SideCostAnalysis()
    costs = np.where(grid < 0, planner.unknown_cost, grid).astype(
        np.float64, copy=False
    )
    center = costs.shape[0] // 2
    right_cost = float(np.sum(costs[:center]))
    left_cost = float(np.sum(costs[-center:]))
    selected_side = 1 if left_cost <= right_cost else -1
    return SideCostAnalysis(
        active=True,
        selected_side=selected_side,
        left_cost=left_cost,
        right_cost=right_cost,
    )


def plan_corridor(
    grid: np.ndarray,
    costmap: CostmapConfig,
    planner: PlannerConfig,
) -> RulePlan:
    """Score Ackermann-feasible corridors and return a conservative command."""
    if grid.shape != (costmap.columns, costmap.rows):
        raise ValueError(
            f"grid shape {grid.shape} does not match {(costmap.columns, costmap.rows)}"
        )
    actions = _steering_actions(planner)
    straight_sequence = (0.0,) * planner.tree_depth
    straight_path = _tree_path(straight_sequence, planner)
    straight_costs = _sample_costs(grid, straight_path, costmap, planner)
    straight_step_costs = np.max(straight_costs, axis=1)
    straight_distances = np.linalg.norm(straight_path, axis=1)
    straight_lethal = np.flatnonzero(straight_step_costs >= planner.lethal_cost)
    straight_obstacle_distance = (
        float(straight_distances[int(straight_lethal[0])])
        if len(straight_lethal)
        else math.inf
    )
    if straight_obstacle_distance <= planner.stop_distance_m:
        return RulePlan(
            0.0,
            0.0,
            math.inf,
            True,
            np.empty((0, 2), dtype=np.float64),
            straight_obstacle_distance,
            side_cost=SideCostAnalysis(
                mode="blocked",
                straight_obstacle_distance_m=straight_obstacle_distance,
            ),
            candidate_count=0,
        )

    avoid_active = straight_obstacle_distance <= planner.avoid_trigger_distance_m
    side_cost = (
        analyze_side_costs(grid, planner) if avoid_active else SideCostAnalysis()
    )
    side_cost = SideCostAnalysis(
        active=avoid_active and side_cost.active,
        mode="avoid" if avoid_active else "straight",
        selected_side=side_cost.selected_side if avoid_active else 0,
        left_cost=side_cost.left_cost,
        right_cost=side_cost.right_cost,
        straight_obstacle_distance_m=straight_obstacle_distance,
    )
    first_side_action = None
    if side_cost.active:
        side_actions = [
            action for action in actions if action * side_cost.selected_side > 1e-9
        ]
        first_side_action = max(side_actions, key=abs)
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
        if not avoid_active and sequence != straight_sequence:
            continue
        if (
            avoid_active
            and first_side_action is not None
            and sequence[0] != first_side_action
        ):
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
        side_cost,
        len(candidates),
    )
