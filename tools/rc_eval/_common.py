from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Iterable


TRIAL_FIELDS = (
    "trial_id",
    "order",
    "operator",
    "model",
    "checkpoint_sha256",
    "scene_id",
    "hazard_type",
    "hazard_present",
    "start_position_marker",
    "obstacle_position_marker",
    "commanded_speed_mps",
    "battery_voltage",
    "lighting_weather_note",
    "rosbag_path",
    "video_path",
    "emergency_intervention",
    "human_final_outcome",
    "exclusion_reason",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_bool(value: Any, *, field: str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"{field} must be boolean, got {value!r}")


def parse_optional_bool(value: Any, *, field: str) -> bool | None:
    if value is None or str(value).strip().lower() in {"", "null", "none", "n/a"}:
        return None
    return parse_bool(value, field=field)


def parse_optional_float(value: Any, *, field: str) -> float | None:
    if value is None or str(value).strip().lower() in {"", "null", "none", "n/a"}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be numeric, got {value!r}") from error


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float | None, float | None]:
    if total <= 0:
        return None, None
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    margin = z * math.sqrt(
        proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
    ) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def validate_metadata(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required = {
        "trial_id",
        "operator",
        "model",
        "scene_id",
        "hazard_type",
        "hazard_present",
        "start_position_marker",
        "commanded_speed_mps",
    }
    for field in sorted(required):
        if field not in payload or payload[field] in {None, ""}:
            errors.append(f"missing required field: {field}")
    if payload.get("model") not in {None, "", "b0-e0", "eadom"}:
        errors.append("model must be b0-e0 or eadom")
    if payload.get("hazard_type") not in {None, "", "log", "rubble", "none"}:
        errors.append("hazard_type must be log, rubble, or none")
    try:
        if "hazard_present" in payload:
            present = parse_bool(payload["hazard_present"], field="hazard_present")
            if present and payload.get("hazard_type") == "none":
                errors.append("hazard_present=true conflicts with hazard_type=none")
            if not present and payload.get("hazard_type") not in {None, "", "none"}:
                errors.append("hazard_present=false requires hazard_type=none")
    except ValueError as error:
        errors.append(str(error))
    try:
        if "commanded_speed_mps" in payload:
            speed = parse_optional_float(payload["commanded_speed_mps"], field="commanded_speed_mps")
            if speed is not None and speed < 0:
                errors.append("commanded_speed_mps must be non-negative")
    except ValueError as error:
        errors.append(str(error))
    return errors
