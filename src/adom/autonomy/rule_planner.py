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
    distance_decay_m: float = 1.25
    clearance_penalty: float = 35.0
    slow_distance_m: float = 2.0


@dataclass(frozen=True)
class RulePlan:
    speed_mps: float
    steering_rad: float
    score: float
    blocked: bool
    path_xy: np.ndarray
    clearance_m: float


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
    return values.reshape(len(path), len(offsets))


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
    candidates: list[
        tuple[float, float, np.ndarray, np.ndarray, float, float, int | None]
    ] = []
    for angle_deg in angles_deg:
        angle = math.radians(float(angle_deg))
        path = _candidate_path(angle, planner)
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
        score = (
            0.45 * float(np.max(step_costs))
            + 0.40 * distance_weighted_risk
            + planner.clearance_penalty * (1.0 - clearance_ratio)
            + planner.steering_penalty * abs(angle_deg) / planner.max_steering_deg
        )
        candidates.append(
            (
                score,
                angle,
                path,
                step_costs,
                clearance_m,
                distance_weighted_risk,
                lethal_index,
            )
        )

    score, angle, path, costs, clearance_m, weighted_risk, lethal_index = min(
        candidates, key=lambda item: item[0]
    )
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
    return RulePlan(speed, angle, float(score), blocked, safe_path, clearance_m)
