from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class StuckRecoveryConfig:
    command_min_mps: float = 0.25
    max_estimated_speed_mps: float = 0.08
    max_abs_yaw_rate_rps: float = 0.08
    min_abs_steering_deg: float = 12.0
    steering_stability_deg: float = 5.0
    hold_sec: float = 1.50
    duration_sec: float = 0.75


@dataclass(frozen=True)
class StuckRecoveryDecision:
    state: str
    active: bool
    candidate_age_sec: float


class StuckRecoveryGate:
    """Allow one bounded high-throttle attempt on a stable turning path.

    The gate deliberately does not trigger while driving straight. With no wheel
    encoder, low longitudinal IMU speed alone cannot distinguish a stalled car
    from steady motion. A persistent turning request plus low speed and low yaw
    response is a more conservative stuck proxy.
    """

    def __init__(self, config: StuckRecoveryConfig) -> None:
        if config.command_min_mps <= 0.0:
            raise ValueError("stuck command_min_mps must be positive")
        if config.max_estimated_speed_mps < 0.0:
            raise ValueError("stuck max_estimated_speed_mps must be non-negative")
        if config.max_abs_yaw_rate_rps < 0.0:
            raise ValueError("stuck max_abs_yaw_rate_rps must be non-negative")
        if config.min_abs_steering_deg <= 0.0:
            raise ValueError("stuck min_abs_steering_deg must be positive")
        if config.steering_stability_deg < 0.0:
            raise ValueError("stuck steering_stability_deg must be non-negative")
        if config.hold_sec <= 0.0 or config.duration_sec <= 0.0:
            raise ValueError("stuck hold and duration must be positive")
        self.config = config
        self._candidate_start_ns: int | None = None
        self._candidate_steering_deg: float | None = None
        self._active_until_ns: int | None = None
        self._attempt_used = False

    def reset(self) -> None:
        self._candidate_start_ns = None
        self._candidate_steering_deg = None
        self._active_until_ns = None
        self._attempt_used = False

    def update(
        self,
        now_ns: int,
        *,
        path_valid: bool,
        commanded_speed_mps: float,
        steering_rad: float,
        estimated_speed_mps: float,
        yaw_rate_rps: float,
    ) -> StuckRecoveryDecision:
        values = (
            commanded_speed_mps,
            steering_rad,
            estimated_speed_mps,
            yaw_rate_rps,
        )
        if now_ns < 0 or not all(math.isfinite(value) for value in values):
            raise ValueError("stuck recovery inputs must be finite and time non-negative")

        steering_deg = math.degrees(steering_rad)
        progress = (
            estimated_speed_mps > self.config.max_estimated_speed_mps
            or abs(yaw_rate_rps) > self.config.max_abs_yaw_rate_rps
        )
        if progress:
            self.reset()
            return StuckRecoveryDecision("moving", False, 0.0)

        eligible = (
            path_valid
            and commanded_speed_mps >= self.config.command_min_mps
            and abs(steering_deg) >= self.config.min_abs_steering_deg
        )
        if not eligible:
            self._candidate_start_ns = None
            self._candidate_steering_deg = None
            self._active_until_ns = None
            return StuckRecoveryDecision(
                "exhausted" if self._attempt_used else "inactive", False, 0.0
            )

        if self._active_until_ns is not None:
            if now_ns < self._active_until_ns:
                return StuckRecoveryDecision("active", True, 0.0)
            self._active_until_ns = None
            return StuckRecoveryDecision("exhausted", False, 0.0)
        if self._attempt_used:
            return StuckRecoveryDecision("exhausted", False, 0.0)

        stable = (
            self._candidate_steering_deg is not None
            and self._candidate_steering_deg * steering_deg > 0.0
            and abs(steering_deg - self._candidate_steering_deg)
            <= self.config.steering_stability_deg
        )
        if self._candidate_start_ns is None or not stable:
            self._candidate_start_ns = now_ns
            self._candidate_steering_deg = steering_deg
            return StuckRecoveryDecision("candidate", False, 0.0)

        candidate_age_sec = (now_ns - self._candidate_start_ns) / 1e9
        if candidate_age_sec < self.config.hold_sec:
            return StuckRecoveryDecision("candidate", False, candidate_age_sec)

        self._attempt_used = True
        self._active_until_ns = now_ns + int(self.config.duration_sec * 1e9)
        return StuckRecoveryDecision("active", True, candidate_age_sec)
