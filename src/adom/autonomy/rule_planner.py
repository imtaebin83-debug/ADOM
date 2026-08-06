from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .costmap import CostmapConfig


@dataclass(frozen=True)
class PlannerConfig:
    wheelbase_m: float = 0.33
    max_steering_deg: float = 20.0
    steering_step_deg: float = 4.0
    lookahead_m: float = 3.0
    path_step_m: float = 0.10
    corridor_half_width_m: float = 0.18
    unknown_cost: float = 70.0
    lethal_cost: int = 90
    stop_distance_m: float = 0.75
    max_speed_mps: float = 0.25
    min_speed_mps: float = 0.08
    steering_penalty: float = 8.0


@dataclass(frozen=True)
class RulePlan:
    speed_mps: float
    steering_rad: float
    score: float
    blocked: bool
    path_xy: np.ndarray


def _candidate_path(angle: float, config: PlannerConfig) -> np.ndarray:
    distance = np.arange(
        config.path_step_m, config.lookahead_m + config.path_step_m, config.path_step_m
    )
    if abs(angle) < 1e-6:
        return np.stack((distance, np.zeros_like(distance)), axis=1)
    radius = config.wheelbase_m / math.tan(angle)
    heading = distance / radius
    return np.stack(
        (radius * np.sin(heading), radius * (1.0 - np.cos(heading))), axis=1
    )


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
    return values


def plan_corridor(
    grid: np.ndarray, costmap: CostmapConfig, planner: PlannerConfig
) -> RulePlan:
    """Score Ackermann-feasible corridors and return a conservative command."""
    if grid.shape != (costmap.columns, costmap.rows):
        raise ValueError(
            f"grid shape {grid.shape} does not match {(costmap.columns, costmap.rows)}"
        )
    angles_deg = np.arange(
        -planner.max_steering_deg,
        planner.max_steering_deg + planner.steering_step_deg * 0.5,
        planner.steering_step_deg,
    )
    candidates: list[tuple[float, float, np.ndarray, np.ndarray]] = []
    for angle_deg in angles_deg:
        angle = math.radians(float(angle_deg))
        path = _candidate_path(angle, planner)
        costs = _sample_costs(grid, path, costmap, planner)
        score = (
            0.65 * float(np.max(costs))
            + 0.35 * float(np.mean(costs))
            + planner.steering_penalty * abs(angle_deg) / planner.max_steering_deg
        )
        candidates.append((score, angle, path, costs))

    score, angle, path, costs = min(candidates, key=lambda item: item[0])
    near_count = max(1, int(planner.stop_distance_m / planner.path_step_m))
    corridor_width = max(
        1, int(round(2.0 * planner.corridor_half_width_m / costmap.resolution_m)) + 1
    )
    blocked = bool(np.max(costs[: near_count * corridor_width]) >= planner.lethal_cost)
    if blocked:
        speed = 0.0
    else:
        risk_scale = max(0.0, 1.0 - min(float(np.max(costs)), 100.0) / 100.0)
        turn_scale = 1.0 - 0.55 * abs(angle) / math.radians(planner.max_steering_deg)
        speed = planner.max_speed_mps * risk_scale * turn_scale
        speed = min(planner.max_speed_mps, max(planner.min_speed_mps, speed))
    return RulePlan(speed, angle, float(score), blocked, path)
