from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ImuSpeedEstimatorConfig:
    initial_bias_mps2: float = 0.0
    bias_learning_rate: float = 0.02
    stationary_accel_threshold_mps2: float = 0.35
    max_abs_accel_mps2: float = 5.0
    max_integration_dt_sec: float = 0.10
    speed_limit_mps: float = 4.0
    velocity_leak_per_sec: float = 0.02


@dataclass(frozen=True)
class ImuSpeedEstimate:
    speed_mps: float
    bias_mps2: float
    corrected_accel_mps2: float
    stationary_update: bool


class ImuSpeedEstimator:
    """Bias-aware longitudinal IMU integrator with zero-velocity updates.

    IMU acceleration alone cannot observe absolute speed during steady motion.
    This estimator therefore learns gravity/mounting bias only while the control
    chain says the vehicle is stationary, resets velocity to zero there, and
    provides a bounded short-horizon speed estimate while driving.
    """

    def __init__(self, config: ImuSpeedEstimatorConfig) -> None:
        if not 0.0 <= config.bias_learning_rate <= 1.0:
            raise ValueError("bias_learning_rate must be in [0, 1]")
        if config.stationary_accel_threshold_mps2 < 0.0:
            raise ValueError("stationary_accel_threshold_mps2 must be non-negative")
        if config.max_abs_accel_mps2 <= 0.0 or config.speed_limit_mps <= 0.0:
            raise ValueError("acceleration and speed limits must be positive")
        if config.max_integration_dt_sec <= 0.0:
            raise ValueError("max_integration_dt_sec must be positive")
        self.config = config
        self.bias_mps2 = float(config.initial_bias_mps2)
        self.speed_mps = 0.0
        self.last_stamp_ns: int | None = None

    def update(
        self, raw_accel_mps2: float, stamp_ns: int, *, stationary: bool
    ) -> ImuSpeedEstimate:
        if not math.isfinite(raw_accel_mps2):
            raise ValueError("raw_accel_mps2 must be finite")
        stationary_update = False
        residual = raw_accel_mps2 - self.bias_mps2
        if stationary:
            if self.last_stamp_ns is None:
                self.bias_mps2 = raw_accel_mps2
                stationary_update = True
            elif abs(residual) <= self.config.stationary_accel_threshold_mps2:
                alpha = self.config.bias_learning_rate
                self.bias_mps2 += alpha * residual
                stationary_update = True
            self.speed_mps = 0.0
        elif self.last_stamp_ns is not None:
            dt = (stamp_ns - self.last_stamp_ns) / 1e9
            if 0.0 < dt <= self.config.max_integration_dt_sec:
                corrected = max(
                    -self.config.max_abs_accel_mps2,
                    min(self.config.max_abs_accel_mps2, residual),
                )
                leak = max(0.0, 1.0 - self.config.velocity_leak_per_sec * dt)
                self.speed_mps = max(
                    0.0,
                    min(
                        self.config.speed_limit_mps,
                        self.speed_mps * leak + corrected * dt,
                    ),
                )
        self.last_stamp_ns = stamp_ns
        corrected = max(
            -self.config.max_abs_accel_mps2,
            min(
                self.config.max_abs_accel_mps2,
                raw_accel_mps2 - self.bias_mps2,
            ),
        )
        return ImuSpeedEstimate(
            speed_mps=self.speed_mps,
            bias_mps2=self.bias_mps2,
            corrected_accel_mps2=corrected,
            stationary_update=stationary_update,
        )
