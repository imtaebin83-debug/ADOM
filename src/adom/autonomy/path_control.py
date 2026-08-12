from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


EARTH_RADIUS_M = 6_371_008.8


def local_gps_xy_m(
    origin_latitude: float,
    origin_longitude: float,
    latitude: float,
    longitude: float,
) -> tuple[float, float]:
    """Approximate a short GPS trail in a local east/north metric frame."""
    origin_latitude_rad = math.radians(origin_latitude)
    east = (
        EARTH_RADIUS_M
        * math.cos(origin_latitude_rad)
        * math.radians(longitude - origin_longitude)
    )
    north = EARTH_RADIUS_M * math.radians(latitude - origin_latitude)
    return east, north


@dataclass(frozen=True)
class PathControlConfig:
    wheelbase_m: float = 0.33
    lookahead_m: float = 0.80
    max_steering_deg: float = 20.0
    max_speed_mps: float = 0.25
    min_speed_mps: float = 0.06
    curvature_speed_gain: float = 1.5
    speed_kp: float = 0.6
    path_stop_distance_m: float = 0.50
    path_slow_distance_m: float = 2.0


@dataclass(frozen=True)
class PathControlCommand:
    speed_mps: float
    steering_rad: float
    curvature: float
    target_x_m: float
    target_y_m: float
    available_path_m: float


def haversine_distance_m(
    latitude_a: float,
    longitude_a: float,
    latitude_b: float,
    longitude_b: float,
) -> float:
    latitude_a_rad = math.radians(latitude_a)
    latitude_b_rad = math.radians(latitude_b)
    delta_latitude = latitude_b_rad - latitude_a_rad
    delta_longitude = math.radians(longitude_b - longitude_a)
    value = (
        math.sin(delta_latitude / 2.0) ** 2
        + math.cos(latitude_a_rad)
        * math.cos(latitude_b_rad)
        * math.sin(delta_longitude / 2.0) ** 2
    )
    return 2.0 * EARTH_RADIUS_M * math.asin(min(1.0, math.sqrt(value)))


def gps_speed_mps(
    previous: tuple[float, float, int],
    current: tuple[float, float, int],
    *,
    minimum_dt_sec: float = 0.05,
    maximum_dt_sec: float = 2.0,
    maximum_speed_mps: float = 10.0,
) -> float | None:
    dt_sec = (current[2] - previous[2]) / 1e9
    if dt_sec < minimum_dt_sec or dt_sec > maximum_dt_sec:
        return None
    speed = haversine_distance_m(
        previous[0], previous[1], current[0], current[1]
    ) / dt_sec
    if not math.isfinite(speed) or speed > maximum_speed_mps:
        return None
    return speed


def select_lookahead_point(path_xy: np.ndarray, lookahead_m: float) -> np.ndarray:
    if path_xy.ndim != 2 or path_xy.shape[1] != 2 or len(path_xy) == 0:
        raise ValueError("path_xy must be a non-empty Nx2 array")
    if lookahead_m <= 0.0:
        raise ValueError("lookahead_m must be positive")
    distances = np.linalg.norm(path_xy, axis=1)
    indexes = np.flatnonzero(distances >= lookahead_m)
    return path_xy[int(indexes[0])] if len(indexes) else path_xy[-1]


def control_local_path(
    path_xy: np.ndarray,
    measured_speed_mps: float,
    config: PathControlConfig,
) -> PathControlCommand:
    target = select_lookahead_point(path_xy, config.lookahead_m)
    distance_squared = max(float(np.dot(target, target)), 1e-6)
    curvature = 2.0 * float(target[1]) / distance_squared
    steering = math.atan(config.wheelbase_m * curvature)
    steering_limit = math.radians(config.max_steering_deg)
    steering = max(-steering_limit, min(steering_limit, steering))

    target_speed = config.max_speed_mps / (
        1.0 + config.curvature_speed_gain * abs(curvature)
    )
    segments = np.diff(
        np.vstack((np.zeros((1, 2), dtype=np.float64), path_xy)), axis=0
    )
    available_path_m = float(np.sum(np.linalg.norm(segments, axis=1)))
    distance_speed_scale = min(
        1.0,
        max(0.0, available_path_m - config.path_stop_distance_m)
        / max(config.path_slow_distance_m - config.path_stop_distance_m, 1e-3),
    )
    target_speed *= distance_speed_scale
    if available_path_m <= config.path_stop_distance_m:
        target_speed = 0.0
    else:
        target_speed = max(
            config.min_speed_mps, min(config.max_speed_mps, target_speed)
        )
    speed_error = target_speed - max(0.0, measured_speed_mps)
    command_speed = target_speed + config.speed_kp * speed_error
    command_speed = max(0.0, min(config.max_speed_mps, command_speed))
    return PathControlCommand(
        speed_mps=command_speed,
        steering_rad=steering,
        curvature=curvature,
        target_x_m=float(target[0]),
        target_y_m=float(target[1]),
        available_path_m=available_path_m,
    )
