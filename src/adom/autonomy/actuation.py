from __future__ import annotations


def speed_to_pwm_us(
    speed_mps: float,
    *,
    max_forward_speed_mps: float,
    max_reverse_speed_mps: float,
    neutral_us: float,
    forward_max_us: float,
    reverse_max_us: float,
) -> float:
    """Linearly map a bounded nominal speed command to ESC pulse width."""
    if max_forward_speed_mps <= 0.0:
        raise ValueError("max_forward_speed_mps must be positive")
    if max_reverse_speed_mps < 0.0:
        raise ValueError("max_reverse_speed_mps must be non-negative")
    if speed_mps > 0.0:
        ratio = min(1.0, speed_mps / max_forward_speed_mps)
        forward_min_us = 1560.0
        return forward_min_us + ratio * (forward_max_us - forward_min_us)
    if speed_mps == 0.0:
        return neutral_us
    if max_reverse_speed_mps == 0.0:
        return neutral_us
    ratio = min(1.0, abs(speed_mps) / max_reverse_speed_mps)
    return neutral_us + ratio * (reverse_max_us - neutral_us)
