from __future__ import annotations

from dataclasses import dataclass
import math

from .path_control import haversine_distance_m


@dataclass(frozen=True)
class GnssSpeedEstimatorConfig:
    smoothing_alpha: float = 0.5
    minimum_dt_sec: float = 0.05
    maximum_dt_sec: float = 2.0
    maximum_speed_mps: float = 10.0


@dataclass(frozen=True)
class SpeedFeedback:
    speed_mps: float
    source: str


class GnssSpeedEstimator:
    """Ground speed from consecutive GNSS fixes, smoothed with an EMA.

    Position differencing is noisy at low speed unless RTK corrections are
    stable, so the estimate is only an optional feedback source. Samples that
    arrive too quickly are skipped without replacing the reference fix, long
    gaps restart the reference, and implausible jumps are rejected.
    """

    def __init__(self, config: GnssSpeedEstimatorConfig) -> None:
        if not 0.0 < config.smoothing_alpha <= 1.0:
            raise ValueError("smoothing_alpha must be in (0, 1]")
        if config.minimum_dt_sec <= 0.0:
            raise ValueError("minimum_dt_sec must be positive")
        if config.maximum_dt_sec <= config.minimum_dt_sec:
            raise ValueError("maximum_dt_sec must exceed minimum_dt_sec")
        if config.maximum_speed_mps <= 0.0:
            raise ValueError("maximum_speed_mps must be positive")
        self.config = config
        self._reference: tuple[float, float, int] | None = None
        self._speed_mps: float | None = None

    @property
    def speed_mps(self) -> float | None:
        return self._speed_mps

    def reset(self) -> None:
        self._reference = None
        self._speed_mps = None

    def update(self, latitude: float, longitude: float, stamp_ns: int) -> float | None:
        """Add one fix and return the smoothed speed when a new sample is valid."""
        if not (math.isfinite(latitude) and math.isfinite(longitude)):
            return None
        sample = (float(latitude), float(longitude), int(stamp_ns))
        if self._reference is None:
            self._reference = sample
            return None
        dt_sec = (sample[2] - self._reference[2]) / 1e9
        if dt_sec < self.config.minimum_dt_sec:
            return None
        if dt_sec > self.config.maximum_dt_sec:
            self._reference = sample
            return None
        raw_speed = (
            haversine_distance_m(
                self._reference[0], self._reference[1], sample[0], sample[1]
            )
            / dt_sec
        )
        self._reference = sample
        if not math.isfinite(raw_speed) or raw_speed > self.config.maximum_speed_mps:
            return None
        alpha = self.config.smoothing_alpha
        self._speed_mps = (
            raw_speed
            if self._speed_mps is None
            else alpha * raw_speed + (1.0 - alpha) * self._speed_mps
        )
        return self._speed_mps


def select_feedback_speed(
    imu_speed_mps: float,
    gnss_speed_mps: float | None,
    gnss_age_sec: float,
    *,
    gnss_timeout_sec: float,
    gnss_enabled: bool = True,
) -> SpeedFeedback:
    """Prefer a fresh GNSS speed; otherwise fall back to the IMU estimate.

    GNSS never gates safety: a missing or stale fix only changes the feedback
    source back to the IMU estimate, and the controller watchdogs are unchanged.
    """
    if (
        gnss_enabled
        and gnss_speed_mps is not None
        and math.isfinite(gnss_speed_mps)
        and math.isfinite(gnss_age_sec)
        and 0.0 <= gnss_age_sec <= gnss_timeout_sec
    ):
        return SpeedFeedback(max(0.0, float(gnss_speed_mps)), "gnss")
    return SpeedFeedback(float(imu_speed_mps), "imu")
