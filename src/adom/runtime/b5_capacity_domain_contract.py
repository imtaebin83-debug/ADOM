from __future__ import annotations

import argparse
import hashlib
import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any

from adom.data.io import write_json
from adom.data.semantic20 import resource_path
from adom.runtime.b2_eadom_contract import (
    _diff,
    _load_config,
    _normalize,
    _remove_path,
    _sha256_json,
)
from adom.runtime.b5_gate import (
    FROZEN_EVALUATION_CONTRACT_SHA256,
    FROZEN_KOREAN_TEST_MANIFEST_SHA256,
    FROZEN_RELLIS_TEST_MANIFEST_SHA256,
)
from adom.runtime.doctor import GPU_PROFILES
from adom.runtime.semantic20_cycle import (
    CONFIG_DIR,
    _file_sha256,
    validate_semantic20_dataset,
)


B2_CHECKPOINT = (
    "https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/"
    "segformer/mit_b2_20220624-66e8bf70.pth"
)
B5_CHECKPOINT = (
    "https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/"
    "segformer/mit_b5_20220624-658746d9.pth"
)
B2_TO_B5_ARCHITECTURE_DIFFS: dict[str, tuple[Any, Any]] = {
    "checkpoint": (B2_CHECKPOINT, B5_CHECKPOINT),
    "model.backbone.init_cfg.checkpoint": (B2_CHECKPOINT, B5_CHECKPOINT),
    "model.backbone.num_layers": ([3, 4, 6, 3], [3, 6, 40, 3]),
}
FROZEN_PRIMARY_DATASET = {
    "split_counts": {"train": 4568, "val": 900, "test": 899},
    "verified_pairs": 14636,
    "split_contract_sha256": (
        "fab9c136c81081464d9db099656680dac3bf2921a4ae2bbd76055c383b309ab93"
    ),
    "manifest_sha256": (
        "183dda705e76b451dc383a81f517d36df3d6032f00002ab225421b9ae3316b9dd"
    ),
    "dataset_images_sha256": (
        "ce06265e6146bcd37692938786386cbd9b844e9742f831284ee55d26aedd15305"
    ),
    "dataset_masks_sha256": (
        "5ae15ab1eff69921168b15811683edab41472456a439b58aa63844c6d472c377e"
    ),
    "dataset_content_sha256": (
        "a70c6b9467b692a4797976659c6dcd501c80938626226000a6cc214efcdec5e42"
    ),
    "mapping_sha256": {
        "bridge_mapping.yaml": (
            "ecfa61662ddbf16c801bcac22db11b0e7ee2408d635e3018a21dd389933a6bc55"
        )
    },
}
FROZEN_CANONICAL_SOURCE = {
    "train_count": 4435,
    "val_count": 900,
    "test_count": 899,
    "combined_split_sha256": (
        "14758eb11fb087386a0c9f2f28d4bd9c740064ecb668f78dc98a9532696c6da9"
    ),
    "rellis_mapping_sha256": (
        "001e13dea20bbcba81efbbb52fbada742ec6cbaeb99c4e8f0eae6790d0af38ed"
    ),
}


def _canonical_source_contract() -> dict[str, Any]:
    split_root = resource_path("rellis", "splits")
    split_values = {
        split: [
            line.strip()
            for line in (split_root / f"{split}.txt")
            .read_text(encoding="utf-8-sig")
            .splitlines()
            if line.strip()
        ]
        for split in ("train", "val", "test")
    }
    payload = b"".join(
        ("\n".join(split_values[split]) + "\n").encode("utf-8")
        for split in ("train", "val", "test")
    )
    actual = {
        "train_count": len(split_values["train"]),
        "val_count": len(split_values["val"]),
        "test_count": len(split_values["test"]),
        "combined_split_sha256": hashlib.sha256(payload).hexdigest(),
        "rellis_mapping_sha256": _file_sha256(
            resource_path("rellis", "config", "class_mapping.yaml")
        ),
    }
    if actual != FROZEN_CANONICAL_SOURCE:
        raise RuntimeError(
            f"Canonical RELLIS split/mapping lock changed: {actual}"
        )
    return actual


def _frozen_primary_dataset(dataset_root: Path) -> dict[str, Any]:
    invalid_recorded_digests = {
        field: len(value)
        for field, value in FROZEN_PRIMARY_DATASET.items()
        if isinstance(value, str) and len(value) != 64
    }
    if invalid_recorded_digests:
        raise RuntimeError(
            "B5 is blocked: preserved B2 primary dataset SHA-256 values are not "
            f"64 hex characters: {invalid_recorded_digests}. Re-audit the raw "
            "dataset artifacts; do not guess or truncate identifiers."
        )
    actual = validate_semantic20_dataset(dataset_root.resolve(), "eadom")
    mismatches = {
        field: {"actual": _normalize(actual.get(field)), "expected": expected}
        for field, expected in FROZEN_PRIMARY_DATASET.items()
        if _normalize(actual.get(field)) != expected
    }
    if mismatches:
        raise RuntimeError(f"B5 primary dataset lock changed: {mismatches}")
    return actual


def _stage_condition_contract(condition: str, stage: str) -> dict[str, Any]:
    suffix = "e0_rellis" if condition == "e0" else "eadom"
    b2_path = CONFIG_DIR / f"segformer_b2_{stage}_{suffix}.py"
    b5_path = CONFIG_DIR / f"segformer_b5_{stage}_{suffix}.py"
    b2 = _load_config(b2_path)
    b5 = _load_config(b5_path)
    actual_diffs = _diff(b2, b5)
    if actual_diffs != B2_TO_B5_ARCHITECTURE_DIFFS:
        missing = sorted(set(B2_TO_B5_ARCHITECTURE_DIFFS) - set(actual_diffs))
        unexpected = sorted(set(actual_diffs) - set(B2_TO_B5_ARCHITECTURE_DIFFS))
        mismatched = sorted(
            path
            for path in set(actual_diffs) & set(B2_TO_B5_ARCHITECTURE_DIFFS)
            if actual_diffs[path] != B2_TO_B5_ARCHITECTURE_DIFFS[path]
        )
        raise RuntimeError(
            f"B5 {condition} {stage} is not architecture-only: "
            f"missing={missing}, unexpected={unexpected}, mismatched={mismatched}"
        )

    b2_common = deepcopy(b2)
    b5_common = deepcopy(b5)
    for path in B2_TO_B5_ARCHITECTURE_DIFFS:
        _remove_path(b2_common, path)
        _remove_path(b5_common, path)
    b2_common_sha = _sha256_json(b2_common)
    b5_common_sha = _sha256_json(b5_common)
    if b2_common_sha != b5_common_sha:
        raise RuntimeError("B2/B5 non-architecture fingerprints differ")

    hook_types = [item["type"] for item in b5["custom_hooks"]]
    expected_audit = (
        "FreezeBackboneHook" if stage == "stage1" else "BackboneAuditHook"
    )
    expected_updates = 4000 if stage == "stage1" else 40000
    accumulation = int(b5["optim_wrapper"].get("accumulative_counts", 1))
    dataset = b5["train_dataloader"]["dataset"]
    checks = {
        "train_split": dataset["split"],
        "train_manifest": dataset.get("manifest"),
        "val_split": b5["val_dataloader"]["dataset"]["split"],
        "test_split": b5["test_dataloader"]["dataset"]["split"],
        "num_classes": b5["model"]["decode_head"]["num_classes"],
        "ignore_index": b5["model"]["decode_head"]["ignore_index"],
        "initialization": b5["model"]["backbone"]["init_cfg"]["checkpoint"],
        "load_from": b5.get("load_from"),
        "loss": b5["model"]["decode_head"]["loss_decode"]["type"],
        "crop_size": b5["model"]["data_preprocessor"]["size"],
        "optimizer_updates": b5["train_cfg"]["max_iters"] // accumulation,
        "effective_batch": b5["train_dataloader"]["batch_size"] * accumulation,
        "seed": b5["randomness"]["seed"],
        "deterministic": b5["randomness"]["deterministic"],
        "audit_hook": expected_audit in hook_types,
        "canonical_test_lock": "CanonicalTestLockHook" in hook_types,
        "rellis_val_selection": (
            "ConstrainedCheckpointSelectionHook" in hook_types
            and b5["val_dataloader"]["dataset"]["split"] == "splits/val.txt"
        ),
    }
    expected = {
        "train_split": (
            "splits/train.txt" if condition == "e0" else "splits/ta1_train.txt"
        ),
        "train_manifest": None if condition == "e0" else "manifest.csv",
        "val_split": "splits/val.txt",
        "test_split": "splits/test.txt",
        "num_classes": 19,
        "ignore_index": 255,
        "initialization": B5_CHECKPOINT,
        "load_from": None,
        "loss": "CrossEntropyLoss",
        "crop_size": [512, 512],
        "optimizer_updates": expected_updates,
        "effective_batch": 16,
        "seed": 42,
        "deterministic": True,
        "audit_hook": True,
        "canonical_test_lock": True,
        "rellis_val_selection": True,
    }
    failures = {
        field: {"actual": checks[field], "expected": value}
        for field, value in expected.items()
        if checks[field] != value
    }
    if failures:
        raise RuntimeError(f"B5 {condition} {stage} contract failed: {failures}")
    return {
        "condition": condition,
        "stage": stage,
        "reference_config": b2_path.as_posix(),
        "b5_config": b5_path.as_posix(),
        "non_architecture_sha256": b5_common_sha,
        "architecture_differences": [
            {"path": path, "b2": values[0], "b5": values[1]}
            for path, values in sorted(actual_diffs.items())
        ],
        "checks": checks,
    }


def build_contract(
    dataset_root: Path,
    *,
    gpu_profile: str,
    micro_batch: int,
    accumulative_counts: int,
) -> dict[str, Any]:
    if gpu_profile not in GPU_PROFILES:
        raise RuntimeError(f"Unknown B5 GPU profile: {gpu_profile}")
    candidates = list(GPU_PROFILES[gpu_profile]["proposed_micro_batches"])
    if micro_batch not in candidates:
        raise RuntimeError(
            f"micro-batch {micro_batch} is outside {gpu_profile} proposal {candidates}"
        )
    if micro_batch * accumulative_counts != 16:
        raise RuntimeError("B5 effective batch must remain 16")

    previous = {
        name: os.environ.get(name)
        for name in (
            "ADOM_DATA_ROOT",
            "ADOM_MICRO_BATCH",
            "ADOM_ACCUMULATIVE_COUNTS",
            "ADOM_SEED",
            "ADOM_DETERMINISTIC",
        )
    }
    os.environ.update(
        {
            "ADOM_DATA_ROOT": dataset_root.resolve().as_posix(),
            "ADOM_MICRO_BATCH": str(micro_batch),
            "ADOM_ACCUMULATIVE_COUNTS": str(accumulative_counts),
            "ADOM_SEED": "42",
            "ADOM_DETERMINISTIC": "true",
        }
    )
    try:
        configs = [
            _stage_condition_contract(condition, stage)
            for condition in ("e0", "eadom")
            for stage in ("stage1", "stage2")
        ]
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    return {
        "schema_version": "adom-b5-capacity-domain-static-contract-v1",
        "status": "PASS",
        "execution_authorized": False,
        "execution_gate": "separate validated adom-b5-capacity-domain-go-v1 artifact",
        "initialization": "official MiT-B5 ImageNet; no B0/B2 experiment checkpoint",
        "gpu_profile": gpu_profile,
        "batch_plan": {
            "micro_batch": micro_batch,
            "accumulative_counts": accumulative_counts,
            "effective_batch": 16,
            "fallback_order": candidates,
            "status": "proposal-until-memory-probe",
        },
        "canonical_source": _canonical_source_contract(),
        "primary_dataset": _frozen_primary_dataset(dataset_root),
        "evaluation_lock": {
            "evaluation_contract_sha256": FROZEN_EVALUATION_CONTRACT_SHA256,
            "rellis_test_manifest_sha256": FROZEN_RELLIS_TEST_MANIFEST_SHA256,
            "korean_test_manifest_sha256": FROZEN_KOREAN_TEST_MANIFEST_SHA256,
            "korean_heldout_policy": (
                "test-only after RELLIS-val checkpoint freeze; never recipe, "
                "threshold, early-stopping, or checkpoint input"
            ),
        },
        "configs": configs,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Fail-closed B5 E0/E-ADOM preregistration static contract"
    )
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--gpu-profile", required=True, choices=tuple(GPU_PROFILES))
    parser.add_argument(
        "--micro-batch", required=True, type=int, choices=(16, 8, 4, 2, 1)
    )
    parser.add_argument(
        "--accumulative-counts", required=True, type=int, choices=(1, 2, 4, 8, 16)
    )
    args = parser.parse_args(argv)
    report = build_contract(
        args.dataset,
        gpu_profile=args.gpu_profile,
        micro_batch=args.micro_batch,
        accumulative_counts=args.accumulative_counts,
    )
    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
