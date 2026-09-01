from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any


B5_GO_SCHEMA = "adom-b5-capacity-domain-go-v1"
# B2 fresh-evaluation summaries embed the resolved B2 model in this contract.
# The B0 summaries therefore have a different contract SHA and must not be used
# as provenance for the B2 evidence that authorizes a B5 run.
FROZEN_EVALUATION_CONTRACT_SHA256 = (
    "4adfcb3ae550274ed3436c695c872e030c804bb8c16c09025958797312d8d592"
)
FROZEN_RELLIS_TEST_MANIFEST_SHA256 = (
    "2e078a3ac89d870b4dfb5838f8cc2772e788ecdd7cb011c309d59b4ca6a66918"
)
FROZEN_KOREAN_TEST_MANIFEST_SHA256 = (
    "1eb86ff65620fb5c0afc1d58c572c517cacc937468ebd865375aaa26d81eb782"
)
TRIGGERS = {
    "abs_capacity_only_common_miou_ge_10pp",
    "abs_b2_difference_in_differences_ge_10pp",
    "class_direction_discordance_spread_ge_10pp",
}


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"B5 go decision field {field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"B5 go decision field {field} must be finite")
    return result


def _sha256(value: Any, field: str) -> str:
    text = str(value).lower()
    if not re.fullmatch(r"[0-9a-f]{64}", text):
        raise RuntimeError(f"B5 go decision field {field} must be a SHA-256")
    return text


def _recorded_sha256(value: str, field: str) -> str:
    try:
        return _sha256(value, field)
    except RuntimeError as error:
        raise RuntimeError(
            f"B5 is blocked: the preserved B2 run record value for {field} has "
            f"{len(value)} hex characters, not a valid 64-character SHA-256. "
            "Re-audit the raw artifact; do not guess or truncate the identifier."
        ) from error


def validate_b5_go_decision(path: Path) -> dict[str, Any]:
    """Fail closed unless frozen B2 evidence meets one preregistered trigger."""
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"B5 go decision is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if payload.get("schema_version") != B5_GO_SCHEMA:
        raise RuntimeError("B5 go decision schema mismatch")
    if payload.get("decision") != "GO":
        raise RuntimeError("B5 training is locked unless decision is GO")
    trigger = payload.get("trigger")
    if trigger not in TRIGGERS:
        raise RuntimeError(f"B5 go decision has an unregistered trigger: {trigger}")
    if payload.get("primary_split") != "matched-legacy-4568":
        raise RuntimeError("B5 primary run is locked to matched-legacy-4568")
    if payload.get("korean_heldout_used_for_selection") is not False:
        raise RuntimeError("Korean held-out must remain test-only")

    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        raise RuntimeError("B5 go decision requires B2 provenance")
    for field in ("b2_e0_checkpoint_sha256", "b2_eadom_checkpoint_sha256"):
        _sha256(provenance.get(field), f"provenance.{field}")
    frozen = {
        "evaluation_contract_sha256": FROZEN_EVALUATION_CONTRACT_SHA256,
        "rellis_test_manifest_sha256": FROZEN_RELLIS_TEST_MANIFEST_SHA256,
        "korean_test_manifest_sha256": FROZEN_KOREAN_TEST_MANIFEST_SHA256,
    }
    for field, expected in frozen.items():
        _recorded_sha256(expected, f"recorded.{field}")
        actual = _sha256(provenance.get(field), f"provenance.{field}")
        if actual != expected:
            raise RuntimeError(
                f"B5 go decision changed frozen {field}: {actual} != {expected}"
            )

    metrics = payload.get("metrics_pp")
    if not isinstance(metrics, dict):
        raise RuntimeError("B5 go decision requires metrics_pp")
    capacity = _finite_number(metrics.get("capacity_only_common_miou"), "capacity")
    interaction = _finite_number(
        metrics.get("b2_difference_in_differences"), "interaction"
    )
    log_effect = _finite_number(metrics.get("log_capacity_effect"), "log_effect")
    rubble_effect = _finite_number(
        metrics.get("rubble_capacity_effect"), "rubble_effect"
    )
    trigger_passed = {
        "abs_capacity_only_common_miou_ge_10pp": abs(capacity) >= 10.0,
        "abs_b2_difference_in_differences_ge_10pp": abs(interaction) >= 10.0,
        "class_direction_discordance_spread_ge_10pp": (
            log_effect * rubble_effect < 0
            and abs(log_effect - rubble_effect) >= 10.0
        ),
    }[trigger]
    if not trigger_passed:
        raise RuntimeError(f"B5 go trigger {trigger} is not supported by metrics_pp")

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "schema_version": B5_GO_SCHEMA,
        "status": "PASS",
        "decision": "GO",
        "trigger": trigger,
        "path": str(path),
        "sha256": digest,
        "metrics_pp": {
            "capacity_only_common_miou": capacity,
            "b2_difference_in_differences": interaction,
            "log_capacity_effect": log_effect,
            "rubble_capacity_effect": rubble_effect,
        },
        "provenance": provenance,
    }
