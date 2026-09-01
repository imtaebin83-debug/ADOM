#!/usr/bin/env python3
"""Compare the hybrid gap+tree planner with the original full-tree path."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adom.autonomy import CostmapConfig, PlannerConfig, plan_corridor  # noqa: E402


MAX_ALLOWED_P95_INCREASE_MS = 50.0


def _scenes(costmap: CostmapConfig) -> dict[str, np.ndarray]:
    clear = np.zeros((costmap.columns, costmap.rows), dtype=np.int8)
    center = costmap.columns // 2
    left_open = clear.copy()
    left_open[center - 3 : center + 4, 10:18] = 100
    left_open[: center - 5, 10:35] = 100
    right_open = clear.copy()
    right_open[center - 3 : center + 4, 10:18] = 100
    right_open[center + 5 :, 10:35] = 100
    closed = clear.copy()
    closed[center - 3 : center + 4, 10:18] = 100
    closed[: center - 3, 8:40] = 100
    closed[center + 4 :, 8:40] = 100
    return {
        "clear": clear,
        "left_open": left_open,
        "right_open": right_open,
        "closed": closed,
    }


def _measure(
    grid: np.ndarray,
    costmap: CostmapConfig,
    planner: PlannerConfig,
    samples: int = 60,
) -> tuple[np.ndarray, int]:
    for _ in range(5):
        plan_corridor(grid, costmap, planner)
    elapsed = []
    plan = None
    for _ in range(samples):
        started = time.perf_counter_ns()
        plan = plan_corridor(grid, costmap, planner)
        elapsed.append((time.perf_counter_ns() - started) / 1e6)
    assert plan is not None
    return np.asarray(elapsed), plan.candidate_count


def main() -> None:
    costmap = CostmapConfig(inflation_radius_m=0.0)
    hybrid = PlannerConfig(
        max_steering_deg=24.0,
        lookahead_m=4.0,
        path_step_m=0.10,
        tree_depth=3,
        tree_branch_steering_deg=12.0,
        stop_distance_m=0.30,
        max_speed_mps=1.0,
        min_speed_mps=0.10,
        distance_decay_m=2.0,
        slow_distance_m=3.0,
    )
    full_tree = replace(hybrid, side_cost_enabled=False)
    worst_increase = float("-inf")
    for name, grid in _scenes(costmap).items():
        baseline_ms, baseline_candidates = _measure(grid, costmap, full_tree)
        hybrid_ms, hybrid_candidates = _measure(grid, costmap, hybrid)
        increase = float(
            np.percentile(hybrid_ms, 95) - np.percentile(baseline_ms, 95)
        )
        worst_increase = max(worst_increase, increase)
        print(
            json.dumps(
                {
                    "scene": name,
                    "baseline_p95_ms": round(
                        float(np.percentile(baseline_ms, 95)), 3
                    ),
                    "hybrid_p95_ms": round(float(np.percentile(hybrid_ms, 95)), 3),
                    "p95_increase_ms": round(increase, 3),
                    "baseline_candidates": baseline_candidates,
                    "hybrid_candidates": hybrid_candidates,
                },
                sort_keys=True,
            )
        )
    if worst_increase >= MAX_ALLOWED_P95_INCREASE_MS:
        raise SystemExit(
            f"REJECT: worst p95 increase {worst_increase:.3f} ms is at least "
            f"{MAX_ALLOWED_P95_INCREASE_MS:.1f} ms"
        )
    print(
        f"ACCEPT: worst p95 increase {worst_increase:.3f} ms is below "
        f"{MAX_ALLOWED_P95_INCREASE_MS:.1f} ms"
    )


if __name__ == "__main__":
    main()
