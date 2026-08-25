from __future__ import annotations

import argparse
import hashlib
import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any

from adom.data.io import write_json
from adom.runtime.semantic20_cycle import CONFIG_DIR, validate_semantic20_dataset


ARCHITECTURE_DIFFS: dict[str, tuple[Any, Any]] = {
    "checkpoint": (
        "https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/"
        "segformer/mit_b0_20220624-7e0fe6dd.pth",
        "https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/"
        "segformer/mit_b2_20220624-66e8bf70.pth",
    ),
    "model.backbone.embed_dims": (32, 64),
    "model.backbone.init_cfg.checkpoint": (
        "https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/"
        "segformer/mit_b0_20220624-7e0fe6dd.pth",
        "https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/"
        "segformer/mit_b2_20220624-66e8bf70.pth",
    ),
    "model.backbone.num_layers": ([2, 2, 2, 2], [3, 4, 6, 3]),
    "model.decode_head.channels": (256, 768),
    "model.decode_head.in_channels": (
        [32, 64, 160, 256],
        [64, 128, 320, 512],
    ),
}


def _normalize(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    if isinstance(value, dict):
        return {str(key): _normalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    return value


def _diff(left: Any, right: Any, prefix: str = "") -> dict[str, tuple[Any, Any]]:
    if isinstance(left, dict) and isinstance(right, dict):
        result: dict[str, tuple[Any, Any]] = {}
        for key in sorted(set(left) | set(right)):
            path = f"{prefix}.{key}" if prefix else key
            if key not in left:
                result[path] = (None, right[key])
            elif key not in right:
                result[path] = (left[key], None)
            else:
                result.update(_diff(left[key], right[key], path))
        return result
    if left != right:
        return {prefix: (left, right)}
    return {}


def _remove_path(value: dict[str, Any], path: str) -> None:
    parts = path.split(".")
    parent: Any = value
    for part in parts[:-1]:
        if not isinstance(parent, dict) or part not in parent:
            return
        parent = parent[part]
    if isinstance(parent, dict):
        parent.pop(parts[-1], None)


def _sha256_json(value: Any) -> str:
    payload = json.dumps(
        _normalize(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_config(path: Path) -> dict[str, Any]:
    try:
        from mmengine.config import Config
    except ImportError as error:
        raise RuntimeError(
            "MMEngine is required to resolve the B2 E-ADOM config contract"
        ) from error
    config = Config.fromfile(path, import_custom_modules=False)
    return _normalize(config)


def _stage_contract(stage: str) -> dict[str, Any]:
    b0_path = CONFIG_DIR / f"segformer_b0_{stage}_eadom.py"
    b2_path = CONFIG_DIR / f"segformer_b2_{stage}_eadom.py"
    b0 = _load_config(b0_path)
    b2 = _load_config(b2_path)
    actual = _diff(b0, b2)
    if actual != ARCHITECTURE_DIFFS:
        missing = sorted(set(ARCHITECTURE_DIFFS) - set(actual))
        unexpected = sorted(set(actual) - set(ARCHITECTURE_DIFFS))
        mismatched = sorted(
            path
            for path in set(actual) & set(ARCHITECTURE_DIFFS)
            if actual[path] != ARCHITECTURE_DIFFS[path]
        )
        raise RuntimeError(
            "B2 E-ADOM is not architecture-only: "
            f"missing={missing}, unexpected={unexpected}, mismatched={mismatched}"
        )

    b0_common = deepcopy(b0)
    b2_common = deepcopy(b2)
    for path in ARCHITECTURE_DIFFS:
        _remove_path(b0_common, path)
        _remove_path(b2_common, path)
    b0_common_sha = _sha256_json(b0_common)
    b2_common_sha = _sha256_json(b2_common)
    if b0_common_sha != b2_common_sha:
        raise RuntimeError("Non-architecture resolved config digests do not match")

    config = b2
    expected_updates = 4000 if stage == "stage1" else 40000
    accumulative = int(config["optim_wrapper"].get("accumulative_counts", 1))
    checks = {
        "train_split": config["train_dataloader"]["dataset"]["split"],
        "val_split": config["val_dataloader"]["dataset"]["split"],
        "test_split": config["test_dataloader"]["dataset"]["split"],
        "num_classes": config["model"]["decode_head"]["num_classes"],
        "ignore_index": config["model"]["decode_head"]["ignore_index"],
        "loss_type": config["model"]["decode_head"]["loss_decode"]["type"],
        "avg_non_ignore": config["model"]["decode_head"]["loss_decode"][
            "avg_non_ignore"
        ],
        "crop_size": config["model"]["data_preprocessor"]["size"],
        "micro_batch": config["train_dataloader"]["batch_size"],
        "accumulative_counts": accumulative,
        "runner_iterations": config["train_cfg"]["max_iters"],
        "optimizer_updates": config["train_cfg"]["max_iters"] // accumulative,
        "deterministic": config["randomness"]["deterministic"],
        "val_evaluators": [item["type"] for item in config["val_evaluator"]],
    }
    expected = {
        "train_split": "splits/ta1_train.txt",
        "val_split": "splits/val.txt",
        "test_split": "splits/test.txt",
        "num_classes": 19,
        "ignore_index": 255,
        "loss_type": "CrossEntropyLoss",
        "avg_non_ignore": True,
        "crop_size": [512, 512],
        "optimizer_updates": expected_updates,
        "deterministic": True,
        "val_evaluators": ["AdomSemantic20Metric"],
    }
    failures = {
        key: {"actual": checks[key], "expected": value}
        for key, value in expected.items()
        if checks[key] != value
    }
    if failures:
        raise RuntimeError(f"B2 E-ADOM resolved contract failed: {failures}")
    if checks["micro_batch"] * checks["accumulative_counts"] != 16:
        raise RuntimeError("B2 E-ADOM effective batch must remain 16")

    return {
        "stage": stage,
        "b0_config": b0_path.as_posix(),
        "b2_config": b2_path.as_posix(),
        "b0_resolved_sha256": _sha256_json(b0),
        "b2_resolved_sha256": _sha256_json(b2),
        "non_architecture_sha256": b0_common_sha,
        "architecture_differences": [
            {"path": path, "b0": pair[0], "b2": pair[1]}
            for path, pair in sorted(actual.items())
        ],
        "checks": checks,
        "effective_batch": checks["micro_batch"] * checks["accumulative_counts"],
    }


def build_contract(
    dataset_root: Path,
    *,
    micro_batch: int,
    accumulative_counts: int,
) -> dict[str, Any]:
    if micro_batch * accumulative_counts != 16:
        raise RuntimeError("micro-batch × accumulative-counts must equal 16")
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
        stages = [_stage_contract("stage1"), _stage_contract("stage2")]
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    dataset = validate_semantic20_dataset(dataset_root.resolve(), "eadom")
    return {
        "schema_version": "adom-b2-eadom-static-contract-v1",
        "status": "PASS",
        "condition": "B2-E-ADOM primary matched-legacy seed42",
        "architecture_only": True,
        "korean_heldout_policy": "test-only after RELLIS validation checkpoint freeze",
        "stages": stages,
        "dataset_contract": dataset,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Fail-closed B2 E-ADOM config, split, mapping, and manifest gate"
    )
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--micro-batch", required=True, type=int, choices=(16, 8, 4))
    parser.add_argument("--accumulative-counts", required=True, type=int, choices=(1, 2, 4))
    args = parser.parse_args(argv)
    report = build_contract(
        args.dataset,
        micro_batch=args.micro_batch,
        accumulative_counts=args.accumulative_counts,
    )
    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
