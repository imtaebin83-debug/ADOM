from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CostmapConfig:
    resolution_m: float = 0.10
    length_m: float = 8.0
    width_m: float = 6.0
    min_range_m: float = 0.30
    max_range_m: float = 8.0
    sample_stride: int = 4
    # base_link is ground-referenced. Retain a small tolerance for TF/depth
    # error, but reject physically impossible points well below the road.
    min_height_m: float = -0.05
    max_height_m: float = 1.50
    class_costs: tuple[int, ...] = (0, 15, 60, 100)
    geometric_obstacle_min_height_m: float = 0.10
    inflation_radius_m: float = 0.25
    inflation_seed_cost: int = 90
    inflation_min_cost: int = 60

    @property
    def rows(self) -> int:
        return max(1, int(round(self.length_m / self.resolution_m)))

    @property
    def columns(self) -> int:
        return max(1, int(round(self.width_m / self.resolution_m)))


@dataclass(frozen=True)
class ProjectionDiagnostics:
    sampled_pixels: int
    finite_depth_pixels: int
    in_range_depth_pixels: int
    semantic_label_pixels: int
    depth_label_pixels: int
    height_valid_points: int
    transformed_z_min_m: float | None
    transformed_z_max_m: float | None


def quaternion_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    norm = x * x + y * y + z * z + w * w
    if norm < 1e-12:
        return np.eye(3, dtype=np.float64)
    scale = 2.0 / norm
    return np.asarray(
        [
            [
                1.0 - scale * (y * y + z * z),
                scale * (x * y - z * w),
                scale * (x * z + y * w),
            ],
            [
                scale * (x * y + z * w),
                1.0 - scale * (x * x + z * z),
                scale * (y * z - x * w),
            ],
            [
                scale * (x * z - y * w),
                scale * (y * z + x * w),
                1.0 - scale * (x * x + y * y),
            ],
        ],
        dtype=np.float64,
    )


def project_mask_depth(
    mask: np.ndarray,
    depth_m: np.ndarray,
    intrinsics: tuple[float, float, float, float],
    rotation: np.ndarray,
    translation: np.ndarray,
    config: CostmapConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Return base-frame XYZ points and configured semantic IDs."""
    points, labels, _ = project_mask_depth_with_diagnostics(
        mask, depth_m, intrinsics, rotation, translation, config
    )
    return points, labels


def project_mask_depth_with_diagnostics(
    mask: np.ndarray,
    depth_m: np.ndarray,
    intrinsics: tuple[float, float, float, float],
    rotation: np.ndarray,
    translation: np.ndarray,
    config: CostmapConfig,
) -> tuple[np.ndarray, np.ndarray, ProjectionDiagnostics]:
    """Project mask/depth and report which filter removed observations."""
    if mask.ndim != 2 or depth_m.ndim != 2:
        raise ValueError("mask and depth must both be HxW")
    if mask.shape != depth_m.shape:
        raise ValueError(f"mask/depth shapes differ: {mask.shape} vs {depth_m.shape}")

    fx, fy, cx, cy = intrinsics
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError("camera focal lengths must be positive")
    stride = max(1, int(config.sample_stride))
    v, u = np.mgrid[0 : mask.shape[0] : stride, 0 : mask.shape[1] : stride]
    z = depth_m[::stride, ::stride].astype(np.float64, copy=False)
    labels = mask[::stride, ::stride]
    finite = np.isfinite(z)
    in_range = finite & (z >= config.min_range_m) & (z <= config.max_range_m)
    semantic = labels < len(config.class_costs)
    valid = in_range & semantic
    if not np.any(valid):
        diagnostics = ProjectionDiagnostics(
            sampled_pixels=int(z.size),
            finite_depth_pixels=int(np.count_nonzero(finite)),
            in_range_depth_pixels=int(np.count_nonzero(in_range)),
            semantic_label_pixels=int(np.count_nonzero(semantic)),
            depth_label_pixels=0,
            height_valid_points=0,
            transformed_z_min_m=None,
            transformed_z_max_m=None,
        )
        return (
            np.empty((0, 3), dtype=np.float64),
            np.empty(0, dtype=np.uint8),
            diagnostics,
        )

    z = z[valid]
    optical = np.stack(
        (((u[valid] - cx) * z / fx), ((v[valid] - cy) * z / fy), z), axis=1
    )
    points = optical @ rotation.T + translation.reshape(1, 3)
    height_valid = (
        (points[:, 2] >= config.min_height_m)
        & (points[:, 2] <= config.max_height_m)
    )
    diagnostics = ProjectionDiagnostics(
        sampled_pixels=int(finite.size),
        finite_depth_pixels=int(np.count_nonzero(finite)),
        in_range_depth_pixels=int(np.count_nonzero(in_range)),
        semantic_label_pixels=int(np.count_nonzero(semantic)),
        depth_label_pixels=int(np.count_nonzero(valid)),
        height_valid_points=int(np.count_nonzero(height_valid)),
        transformed_z_min_m=float(np.min(points[:, 2])),
        transformed_z_max_m=float(np.max(points[:, 2])),
    )
    return (
        points[height_valid],
        labels[valid][height_valid].astype(np.uint8),
        diagnostics,
    )


def _inflate(grid: np.ndarray, config: CostmapConfig) -> np.ndarray:
    radius = int(np.ceil(config.inflation_radius_m / config.resolution_m))
    if radius <= 0:
        return grid
    # Only cells that are already lethal seed inflation.  Lower semantic costs
    # remain useful for scoring and slowing without being promoted to a hard
    # stop merely because inflation is enabled.
    danger = grid >= config.inflation_seed_cost
    if not np.any(danger):
        return grid
    inflated = grid.copy()
    rows, columns = grid.shape
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            distance = float(np.hypot(dx, dy))
            if distance > radius:
                continue
            source_y0, source_y1 = max(0, -dy), min(rows, rows - dy)
            source_x0, source_x1 = max(0, -dx), min(columns, columns - dx)
            target_y0, target_y1 = source_y0 + dy, source_y1 + dy
            target_x0, target_x1 = source_x0 + dx, source_x1 + dx
            source = danger[source_y0:source_y1, source_x0:source_x1]
            decay = 1.0 - distance / max(radius + 1.0, 1.0)
            cost = max(config.inflation_min_cost, int(round(100.0 * decay)))
            target = inflated[target_y0:target_y1, target_x0:target_x1]
            target[source] = np.maximum(target[source], cost)
    return inflated


def build_costmap(
    points_base: np.ndarray, labels: np.ndarray, config: CostmapConfig
) -> np.ndarray:
    """Rasterize semantic 3D observations into a robot-centric OccupancyGrid."""
    grid = np.full((config.columns, config.rows), -1, dtype=np.int8)
    if points_base.size == 0:
        return grid
    if points_base.ndim != 2 or points_base.shape[1] != 3:
        raise ValueError(f"points must be Nx3, got {points_base.shape}")
    if len(points_base) != len(labels):
        raise ValueError("points and labels must have equal lengths")

    forward = np.floor(points_base[:, 0] / config.resolution_m).astype(np.int64)
    lateral = np.floor(
        (points_base[:, 1] + config.width_m / 2.0) / config.resolution_m
    ).astype(np.int64)
    valid = (
        (forward >= 0)
        & (forward < config.rows)
        & (lateral >= 0)
        & (lateral < config.columns)
        & (labels >= 0)
        & (labels < len(config.class_costs))
    )
    costs = np.asarray(config.class_costs, dtype=np.int16)[labels[valid]]
    costs = np.where(
        points_base[valid, 2] >= config.geometric_obstacle_min_height_m,
        100,
        costs,
    )
    # Multiple projected pixels can land in one cell. np.maximum.at preserves
    # the highest cost without a Python loop over tens of thousands of points.
    np.maximum.at(grid, (lateral[valid], forward[valid]), costs.astype(np.int8))
    return _inflate(grid, config)
