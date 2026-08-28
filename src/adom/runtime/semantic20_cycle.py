from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from adom.data.io import write_json
from adom.data.semantic20 import resource_path
from adom.evaluation_semantic20 import (
    SEMANTIC20_CLASSES,
    TEST_SUPPORTED11,
    VAL_SUPPORTED13,
)
from adom.runtime.checkpoints import resolve_single_best_checkpoint
from adom.runtime.b5_gate import validate_b5_go_decision
from adom.runtime.cycle import (
    CONFIG_ROOT,
    EFFECTIVE_BATCH,
    REPO_ROOT,
    CycleState,
    _now,
    _resumable_checkpoint,
    _run_phase,
    _tool_path,
    _tracking_env,
)
from adom.runtime.doctor import GPU_PROFILES


CONFIG_DIR = CONFIG_ROOT / "adom" / "phase1_semantic20"
REFERENCE_SPLITS = resource_path("rellis", "splits")
GATE_UPDATES = {"smoke": 50, "mini": 500}
EXPECTED_SPLIT_COUNTS = {
    "e0": {"train": 4435, "val": 900, "test": 899},
    "e1": {"train": 9868, "val": 900, "test": 899},
    "eadom": {"train": 4568, "val": 900, "test": 899},
    "ta0": {"train": 4435, "val": 900, "test": 899},
    "ta1": {"train": 4568, "val": 900, "test": 899},
    "ta2": {"train": 10001, "val": 900, "test": 899},
}
PRODUCTION_CANONICAL_EVAL_COUNTS = {"val": 900, "test": 899}
CANONICAL_EVAL_COUNTS = PRODUCTION_CANONICAL_EVAL_COUNTS.copy()
TA_EXPERIMENTS = {"ta0", "ta1", "ta2"}
EADOM_EXPERIMENT = "eadom"
COMBINED_EXPERIMENTS = {"e1", "e2", EADOM_EXPERIMENT} | TA_EXPERIMENTS
EXPECTED_E1_MANIFEST_COUNT = 14421
EXPECTED_E1_MAIN_SOURCE_COUNTS = Counter(
    {"rellis3d": 6234, "rugd": 4779, "ycor": 654}
)
EXPECTED_E1_MANIFEST_SOURCE_COUNTS = Counter(
    {"rellis3d": 6234, "rugd": 7436, "ycor": 751}
)
EXPECTED_TA_MANIFEST_SOURCE_COUNTS = Counter(
    {"rellis3d": 6234, "rugd": 7436, "ycor": 751, "adom_zed2i": 215}
)
EXPECTED_TA_MAIN_SOURCE_COUNTS = {
    "ta0": Counter({"rellis3d": 6234}),
    "ta1": Counter({"rellis3d": 6234, "adom_zed2i": 133}),
    "ta2": Counter(
        {"rellis3d": 6234, "rugd": 4779, "ycor": 654, "adom_zed2i": 133}
    ),
}
EXPECTED_TA_TRAIN_SOURCES = {
    "ta0": {"rellis3d"},
    "ta1": {"rellis3d", "adom_zed2i"},
    "ta2": {"rellis3d", "rugd", "ycor", "adom_zed2i"},
}
EXPECTED_EADOM_TRAIN_SOURCES = {"rellis3d", "adom_zed2i"}
ALLOWED_TARGET_IDS = set(range(19)) | {255}


def _read_split(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Semantic20 split is missing: {path}")
    values = [
        line.strip()
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    if not values or len(values) != len(set(values)):
        raise RuntimeError(f"Semantic20 split is empty or has duplicates: {path}")
    return values


def _rellis_key(value: str) -> str:
    return value.removeprefix("rellis3d/")


def _validate_semantic20_pair(
    image: Path,
    mask: Path,
    key: str,
) -> dict[str, Any]:
    if not image.is_file() or not mask.is_file():
        raise FileNotFoundError(f"Missing dataset pair: {image}, {mask}")
    with Image.open(image) as image_file:
        image_file.load()
        image_size = image_file.size
        image_digest = hashlib.sha256()
        image_digest.update(image_file.mode.encode("ascii"))
        image_digest.update(str(image_size).encode("ascii"))
        image_digest.update(image_file.tobytes())
    with Image.open(mask) as mask_file:
        mask_file.load()
        mask_mode = mask_file.mode
        mask_array = np.asarray(mask_file)
    if mask_array.ndim != 2 or mask_mode not in {"L", "P"}:
        raise RuntimeError(
            f"Mask must be single-channel for {key}: mode={mask_mode}, "
            f"shape={mask_array.shape}"
        )
    if mask_array.dtype != np.uint8:
        raise RuntimeError(f"Mask must be uint8 for {key}: {mask_array.dtype}")
    if image_size != (mask_array.shape[1], mask_array.shape[0]):
        raise RuntimeError(
            f"Image/mask size mismatch for {key}: "
            f"image={image_size}, mask={mask_array.shape}"
        )
    invalid_ids = {int(value) for value in np.unique(mask_array)} - ALLOWED_TARGET_IDS
    if invalid_ids:
        raise RuntimeError(f"Invalid Semantic20 target IDs for {key}: {sorted(invalid_ids)}")
    valid = mask_array != 255
    valid_values = mask_array[valid]
    pixel_counts = np.bincount(valid_values, minlength=19).astype(np.int64)
    digest = hashlib.sha256()
    digest.update(mask_array.tobytes())
    return {
        "total_pixels": int(mask_array.size),
        "non_ignore_pixels": int(valid_values.size),
        "pixel_counts": pixel_counts.tolist(),
        "image_presence": (pixel_counts > 0).astype(np.int64).tolist(),
        "image_sha256": image_digest.hexdigest(),
        "mask_sha256": digest.hexdigest(),
    }


def _support_payload(
    splits: dict[str, list[str]],
    pair_audits: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    def aggregate(keys: list[str]) -> dict[str, Any]:
        pixels = np.zeros(19, dtype=np.int64)
        images = np.zeros(19, dtype=np.int64)
        total_pixels = 0
        non_ignore_pixels = 0
        for key in keys:
            audit = pair_audits[key]
            pixels += np.asarray(audit["pixel_counts"], dtype=np.int64)
            images += np.asarray(audit["image_presence"], dtype=np.int64)
            total_pixels += int(audit["total_pixels"])
            non_ignore_pixels += int(audit["non_ignore_pixels"])
        return {
            "sample_count": len(keys),
            "total_pixels": total_pixels,
            "non_ignore_pixels": non_ignore_pixels,
            "classes": [
                {
                    "id": index,
                    "name": name,
                    "pixels": int(pixels[index]),
                    "images": int(images[index]),
                    "pixel_share_non_ignore": (
                        float(pixels[index] / non_ignore_pixels)
                        if non_ignore_pixels
                        else 0.0
                    ),
                    "image_share": (
                        float(images[index] / len(keys)) if keys else 0.0
                    ),
                }
                for index, name in enumerate(SEMANTIC20_CLASSES)
            ],
        }

    by_split = {split: aggregate(keys) for split, keys in splits.items()}
    by_source_split: dict[str, dict[str, Any]] = {}
    for split, keys in splits.items():
        source_keys: dict[str, list[str]] = {}
        for key in keys:
            source = key.split("/", 1)[0] if "/" in key else "rellis3d"
            source_keys.setdefault(source, []).append(key)
        for source, values in source_keys.items():
            by_source_split[f"{source}/{split}"] = aggregate(values)
    return {
        "schema_version": "semantic20-support-v1",
        "by_split": by_split,
        "by_source_split": by_source_split,
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_ta_initial_checkpoint(
    checkpoint_path: Path,
    expected_sha256: str,
) -> dict[str, Any]:
    checkpoint_path = checkpoint_path.expanduser().resolve()
    expected = expected_sha256.strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError("Expected initial checkpoint SHA-256 must be 64 hex characters")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Initial B0-E0 checkpoint is missing: {checkpoint_path}")
    actual = _file_sha256(checkpoint_path)
    if actual != expected:
        raise RuntimeError(
            f"Initial checkpoint SHA-256 mismatch: actual={actual}, expected={expected}"
        )

    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict") if isinstance(checkpoint, dict) else None
    if not isinstance(state_dict, dict) or not state_dict:
        raise RuntimeError("Initial checkpoint has no non-empty state_dict")
    head_keys = [
        key for key in state_dict if key.endswith("decode_head.conv_seg.weight")
    ]
    if len(head_keys) != 1:
        raise RuntimeError(
            "Initial checkpoint must contain exactly one Semantic20 decode head weight"
        )
    head_shape = tuple(state_dict[head_keys[0]].shape)
    if len(head_shape) != 4 or head_shape[0] != 19 or head_shape[1] != 256:
        raise RuntimeError(
            f"Initial checkpoint is not the 19-class SegFormer head contract: {head_shape}"
        )

    blocks: dict[int, set[int]] = {stage: set() for stage in range(4)}
    pattern = re.compile(r"backbone\.layers\.(\d+)\.1\.(\d+)\.")
    for key in state_dict:
        match = pattern.search(key)
        if match:
            stage, block = (int(value) for value in match.groups())
            if stage in blocks:
                blocks[stage].add(block)
    block_counts = [max(blocks[stage]) + 1 if blocks[stage] else 0 for stage in range(4)]
    if block_counts != [2, 2, 2, 2]:
        raise RuntimeError(
            f"Initial checkpoint is not a SegFormer-B0 encoder: blocks={block_counts}"
        )
    return {
        "path": str(checkpoint_path),
        "sha256": actual,
        "architecture": "segformer_b0",
        "num_classes": 19,
        "decode_head_weight_shape": list(head_shape),
        "backbone_blocks": block_counts,
    }


def _mapping_digests(experiment: str) -> dict[str, str]:
    paths = [
        resource_path("rellis", "config", "class_mapping.yaml")
        if experiment == "e0"
        else resource_path("semantic_20", "config", "bridge_mapping.yaml")
    ]
    if experiment == "e2":
        paths.append(
            resource_path("semantic_20", "config", "goose_direct_mapping.yaml")
        )
    return {path.name: _file_sha256(path) for path in paths}


def validate_semantic20_dataset(dataset_root: Path, experiment: str) -> dict[str, Any]:
    success_marker = dataset_root / "_SUCCESS"
    if not success_marker.is_file():
        raise FileNotFoundError(f"Dataset success marker is missing: {success_marker}")
    expected: dict[str, list[str]] = {
        split: _read_split(REFERENCE_SPLITS / f"{split}.txt")
        for split in ("train", "val", "test")
    }
    train_filename = (
        "ta1_train.txt"
        if experiment == EADOM_EXPERIMENT
        else f"{experiment}_train.txt"
        if experiment in TA_EXPERIMENTS
        else "train.txt"
    )
    actual = {
        "train": _read_split(dataset_root / "splits" / train_filename),
        "val": _read_split(dataset_root / "splits" / "val.txt"),
        "test": _read_split(dataset_root / "splits" / "test.txt"),
    }
    actual_counts = {key: len(value) for key, value in actual.items()}
    if experiment in EXPECTED_SPLIT_COUNTS and (
        actual_counts != EXPECTED_SPLIT_COUNTS[experiment]
    ):
        raise RuntimeError(
            f"{experiment} split counts differ from the Semantic20 contract: "
            f"actual={actual_counts}, expected={EXPECTED_SPLIT_COUNTS[experiment]}"
        )
    if experiment == "e2" and (
        actual_counts["val"] != CANONICAL_EVAL_COUNTS["val"]
        or actual_counts["test"] != CANONICAL_EVAL_COUNTS["test"]
        or actual_counts["train"] <= EXPECTED_SPLIT_COUNTS["e1"]["train"]
    ):
        raise RuntimeError(
            "E2 must keep canonical RELLIS val/test and add GOOSE train samples: "
            f"{actual_counts}"
        )
    all_keys = [key for values in actual.values() for key in values]
    if len(all_keys) != len(set(all_keys)):
        raise RuntimeError(f"{experiment} sample occurs in more than one main split")
    for split in ("val", "test"):
        normalized = [_rellis_key(value) for value in actual[split]]
        if normalized != expected[split]:
            raise RuntimeError(
                f"{experiment} {split} is not the canonical RELLIS {split} split"
            )
        if any(not value.startswith("rellis3d/") for value in actual[split]) and (
            experiment in COMBINED_EXPERIMENTS
        ):
            raise RuntimeError(
                f"{experiment.upper()} {split} must contain RELLIS samples only"
            )
    if experiment == "e0":
        if [_rellis_key(value) for value in actual["train"]] != expected["train"]:
            raise RuntimeError("E0 train is not the canonical RELLIS train split")
    else:
        rellis_train = [
            _rellis_key(value)
            for value in actual["train"]
            if value.startswith("rellis3d/")
        ]
        if rellis_train != expected["train"]:
            raise RuntimeError(
                f"{experiment.upper()} does not contain the canonical RELLIS train split"
            )
        sources = {value.split("/", 1)[0] for value in actual["train"]}
        required_sources = (
            EXPECTED_TA_TRAIN_SOURCES[experiment]
            if experiment in TA_EXPERIMENTS
            else EXPECTED_EADOM_TRAIN_SOURCES
            if experiment == EADOM_EXPERIMENT
            else {"rellis3d", "rugd", "ycor"}
        )
        if experiment == "e2":
            required_sources = required_sources | {"goose"}
        if sources != required_sources:
            raise RuntimeError(
                f"{experiment.upper()} train sources must be "
                f"{sorted(required_sources)}, got {sorted(sources)}"
            )

    manifest_rows: dict[str, dict[str, str]] = {}
    manifest_path: Path | None = None
    if experiment in COMBINED_EXPERIMENTS:
        final_check_path = dataset_root / "results" / "final_check.json"
        if not final_check_path.is_file():
            raise FileNotFoundError(
                f"{experiment.upper()} final check is missing: {final_check_path}"
            )
        final_check = json.loads(final_check_path.read_text(encoding="utf-8-sig"))
        if str(final_check.get("status", "")).upper() != "PASS":
            raise RuntimeError(
                f"{experiment.upper()} final check is not PASS: {final_check_path}"
            )
        manifest_path = dataset_root / "manifest.csv"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"{experiment.upper()} manifest is missing: {manifest_path}"
            )
        with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"sample_key", "image_path", "mask_path"}
            missing = required - set(reader.fieldnames or ())
            if missing:
                raise RuntimeError(
                    f"E1 manifest is missing fields: {sorted(missing)}"
                )
            for row in reader:
                key = row["sample_key"]
                if key in manifest_rows:
                    raise RuntimeError(
                        f"Duplicate {experiment.upper()} manifest sample: {key}"
                    )
                manifest_rows[key] = {
                    "image_path": row["image_path"],
                    "mask_path": row["mask_path"],
                }
        if experiment == "e1" and len(manifest_rows) != EXPECTED_E1_MANIFEST_COUNT:
            raise RuntimeError(
                "E1 manifest count differs from contract: "
                f"{len(manifest_rows)} != {EXPECTED_E1_MANIFEST_COUNT}"
            )
        manifest_sources = Counter(key.split("/", 1)[0] for key in manifest_rows)
        if experiment in TA_EXPERIMENTS | {EADOM_EXPERIMENT} and (
            manifest_sources != EXPECTED_TA_MANIFEST_SOURCE_COUNTS
        ):
            raise RuntimeError(
                "TA manifest source counts differ from contract: "
                f"{dict(manifest_sources)}"
            )
        if experiment == "e2":
            required_manifest_sources = {"rellis3d", "rugd", "ycor", "goose"}
            if set(manifest_sources) != required_manifest_sources:
                raise RuntimeError(
                    "E2 manifest must contain exactly RELLIS, RUGD, YCOR, and GOOSE: "
                    f"{dict(manifest_sources)}"
                )
            for source, expected_count in EXPECTED_E1_MANIFEST_SOURCE_COUNTS.items():
                if manifest_sources[source] != expected_count:
                    raise RuntimeError(
                        f"E2 changed the existing {source} manifest count: "
                        f"{manifest_sources[source]} != {expected_count}"
                    )
            if manifest_sources["goose"] <= 0:
                raise RuntimeError("E2 manifest contains no GOOSE samples")

    # Check every pair before paying for GPU time. E1 must use manifest paths
    # because RUGD images are PNG while RELLIS and YCOR images are JPEG.
    main_source_counts: Counter[str] = Counter()
    pair_audits: dict[str, dict[str, Any]] = {}
    verified_pairs = 0
    if experiment in COMBINED_EXPERIMENTS:
        manifest_source_counts: Counter[str] = Counter()
        for key, row in manifest_rows.items():
            image = (dataset_root / row["image_path"]).resolve()
            mask = (dataset_root / row["mask_path"]).resolve()
            if dataset_root.resolve() not in image.parents or dataset_root.resolve() not in mask.parents:
                raise RuntimeError(
                    f"{experiment.upper()} manifest path escapes dataset root: {key}"
                )
            pair_audits[key] = _validate_semantic20_pair(image, mask, key)
            manifest_source_counts[key.split("/", 1)[0]] += 1
            verified_pairs += 1
        if experiment == "e1" and (
            manifest_source_counts != EXPECTED_E1_MANIFEST_SOURCE_COUNTS
        ):
            raise RuntimeError(
                "E1 manifest source counts differ from contract: "
                f"{dict(manifest_source_counts)}"
            )

    for split, keys in actual.items():
        for key in keys:
            if experiment in COMBINED_EXPERIMENTS:
                if key not in manifest_rows:
                    raise RuntimeError(
                        f"{experiment.upper()} split sample is absent from manifest: {key}"
                    )
                main_source_counts[key.split("/", 1)[0]] += 1
            else:
                image = dataset_root / "images" / f"{key}.jpg"
                mask = dataset_root / "masks" / f"{key}.png"
                pair_audits[key] = _validate_semantic20_pair(image, mask, key)
                verified_pairs += 1

    if experiment == "e1" and main_source_counts != EXPECTED_E1_MAIN_SOURCE_COUNTS:
        # Main val/test are RELLIS-only, so 4,435+900+899 RELLIS pairs are checked.
        raise RuntimeError(
            f"E1 main split source counts differ from contract: {dict(main_source_counts)}"
        )
    if experiment == "e2":
        for source, expected_count in EXPECTED_E1_MAIN_SOURCE_COUNTS.items():
            if main_source_counts[source] != expected_count:
                raise RuntimeError(
                    f"E2 changed the existing {source} main split count: "
                    f"{main_source_counts[source]} != {expected_count}"
                )
        if main_source_counts["goose"] <= 0:
            raise RuntimeError("E2 main train contains no GOOSE samples")
    if experiment in TA_EXPERIMENTS and (
        main_source_counts != EXPECTED_TA_MAIN_SOURCE_COUNTS[experiment]
    ):
        raise RuntimeError(
            f"{experiment.upper()} main split source counts differ from contract: "
            f"{dict(main_source_counts)}"
        )
    if experiment == EADOM_EXPERIMENT and (
        main_source_counts != EXPECTED_TA_MAIN_SOURCE_COUNTS["ta1"]
    ):
        raise RuntimeError(
            "EADOM main split source counts differ from the TA1 data-only contract: "
            f"{dict(main_source_counts)}"
        )

    digest = hashlib.sha256()
    for split in ("train", "val", "test"):
        digest.update(("\n".join(actual[split]) + "\n").encode("utf-8"))
    image_dataset_digest = hashlib.sha256()
    mask_dataset_digest = hashlib.sha256()
    content_dataset_digest = hashlib.sha256()
    for key in sorted(pair_audits):
        key_bytes = key.encode("utf-8")
        image_sha = pair_audits[key]["image_sha256"].encode("ascii")
        mask_sha = pair_audits[key]["mask_sha256"].encode("ascii")
        image_dataset_digest.update(key_bytes)
        image_dataset_digest.update(image_sha)
        mask_dataset_digest.update(key_bytes)
        mask_dataset_digest.update(mask_sha)
        content_dataset_digest.update(key_bytes)
        content_dataset_digest.update(image_sha)
        content_dataset_digest.update(mask_sha)
    support = _support_payload(actual, pair_audits)
    production_reference = all(
        len(expected[split]) == PRODUCTION_CANONICAL_EVAL_COUNTS[split]
        for split in ("val", "test")
    )
    if production_reference:
        for split, expected_names in (
            ("val", VAL_SUPPORTED13),
            ("test", TEST_SUPPORTED11),
        ):
            actual_names = {
                row["name"]
                for row in support["by_split"][split]["classes"]
                if row["pixels"] > 0
            }
            if actual_names != set(expected_names):
                raise RuntimeError(
                    f"Canonical {split} class support changed: "
                    f"missing={sorted(set(expected_names) - actual_names)}, "
                    f"unexpected={sorted(actual_names - set(expected_names))}"
                )
    mapping_digests = _mapping_digests(experiment)
    return {
        "experiment": experiment,
        "num_classes": 19,
        "ignore_index": 255,
        "split_counts": actual_counts,
        "verified_pairs": verified_pairs,
        "split_contract_sha256": digest.hexdigest(),
        "manifest_sha256": _file_sha256(manifest_path) if manifest_path else None,
        "dataset_images_sha256": image_dataset_digest.hexdigest(),
        "dataset_masks_sha256": mask_dataset_digest.hexdigest(),
        "dataset_content_sha256": content_dataset_digest.hexdigest(),
        "mapping_sha256": mapping_digests,
        "class_support": support,
        "validation_test_policy": "canonical RELLIS-only",
    }


def _config(model: str, stage: str, experiment: str) -> Path:
    suffix = {
        "e0": "e0_rellis",
        "e1": "e1_combined",
        "e2": "e2_combined_goose",
        "eadom": "eadom",
        "ta0": "ta0",
        "ta1": "ta1",
        "ta2": "ta2",
    }[experiment]
    return CONFIG_DIR / f"segformer_{model}_{stage}_{suffix}.py"


def _batch_candidates(model: str, gpu_profile: str | None = None) -> list[int]:
    if model == "b0":
        return [16]
    if model == "b2":
        return [16, 8, 4]
    if model == "b5":
        if gpu_profile not in GPU_PROFILES:
            raise RuntimeError("B5 requires an exact --gpu-profile")
        return list(GPU_PROFILES[gpu_profile]["proposed_micro_batches"])
    raise RuntimeError(f"Unsupported Semantic20 model: {model}")


def _requested_models(value: str, experiment: str) -> list[str]:
    models = [item.strip().lower() for item in value.split(",") if item.strip()]
    if not models or any(item not in {"b0", "b2", "b5"} for item in models):
        raise RuntimeError("--models accepts b0, b2, and/or b5")
    if len(models) != len(set(models)):
        raise RuntimeError("--models contains duplicates")
    if experiment in TA_EXPERIMENTS and models != ["b0"]:
        raise RuntimeError("TA0/TA1/TA2 are locked to --models b0")
    if experiment == EADOM_EXPERIMENT and len(models) != 1:
        raise RuntimeError("E-ADOM runs exactly one architecture at a time")
    if "b5" in models and experiment not in {"e0", EADOM_EXPERIMENT}:
        raise RuntimeError("B5 is preregistered only for E0 and E-ADOM")
    return models


def _probe_batch(
    *,
    model: str,
    config: Path,
    output_root: Path,
    env: dict[str, str],
    train_tool: Path,
    resume: bool,
    load_from: Path | None = None,
    gpu_profile: str | None = None,
) -> tuple[int, int]:
    plan_path = output_root / model / "batch_plan.json"
    if resume and plan_path.is_file():
        value = json.loads(plan_path.read_text(encoding="utf-8"))
        return int(value["micro_batch"]), int(value["accumulative_counts"])
    for micro_batch in _batch_candidates(model, gpu_profile):
        accumulative = EFFECTIVE_BATCH // micro_batch
        probe_dir = output_root / model / "probes" / f"micro_batch_{micro_batch}"
        probe_dir.mkdir(parents=True, exist_ok=True)
        log_path = probe_dir / "probe.log"
        probe_env = dict(env)
        probe_env.update(
            {
                "ADOM_MICRO_BATCH": str(micro_batch),
                "ADOM_ACCUMULATIVE_COUNTS": "1",
                "ADOM_TA_CONFIG_PROBE": "true",
                "WANDB_MODE": "disabled",
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            }
        )
        command = [
            sys.executable,
            str(train_tool),
            str(config),
            "--work-dir",
            str(probe_dir),
            "--cfg-options",
            "train_cfg.max_iters=2",
            "train_cfg.val_interval=3",
            "default_hooks.checkpoint.interval=3",
            "val_cfg=None",
            "val_dataloader=None",
            "val_evaluator=None",
        ]
        if load_from is not None:
            command.append(f"load_from={load_from}")
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(
                command,
                cwd=REPO_ROOT,
                env=probe_env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        if result.returncode == 0:
            write_json(
                plan_path,
                {
                    "model": model,
                    "probe_runner_iterations": 2,
                    "micro_batch": micro_batch,
                    "accumulative_counts": accumulative,
                    "effective_batch": micro_batch * accumulative,
                    "fallback_order": _batch_candidates(model, gpu_profile),
                    "gpu_profile": gpu_profile,
                    "probe_log": str(log_path.resolve()),
                },
            )
            return micro_batch, accumulative
        log_text = log_path.read_text(encoding="utf-8", errors="replace").lower()
        if "out of memory" not in log_text:
            raise RuntimeError(f"Non-OOM batch probe failure; see {log_path}")
    raise RuntimeError(f"No supported effective-batch-16 plan fits {model}")


def _stage_env(
    env: dict[str, str],
    *,
    gate: str,
    stage: str,
    work_dir: Path,
) -> dict[str, str]:
    value = dict(env)
    value["ADOM_METRIC_OUTPUT_DIR"] = work_dir.as_posix()
    if gate in GATE_UPDATES:
        updates = GATE_UPDATES[gate]
        value["ADOM_MAX_OPTIMIZER_UPDATES"] = str(updates)
        value["ADOM_TA_TOTAL_OPTIMIZER_UPDATES"] = str(updates)
        lp_head_updates = int(value.get("ADOM_TA_LP_HEAD_OPTIMIZER_UPDATES", "1000"))
        if stage == "stage1":
            phase_updates = max(1, round(updates * lp_head_updates / 6000))
        elif stage == "stage2":
            phase_updates = updates - max(1, round(updates * lp_head_updates / 6000))
        else:
            phase_updates = updates
        value["ADOM_VAL_INTERVAL_OPTIMIZER_UPDATES"] = str(
            phase_updates if gate == "mini" else phase_updates + 1
        )
    else:
        value.pop("ADOM_MAX_OPTIMIZER_UPDATES", None)
        value["ADOM_VAL_INTERVAL_OPTIMIZER_UPDATES"] = (
            "500" if stage == "stage1" else "1000"
        )
    if gate == "smoke":
        value["WANDB_MODE"] = "disabled"
    else:
        value["WANDB_MODE"] = "online"
    return value


def _train_stage(
    *,
    state: CycleState,
    model: str,
    stage: str,
    experiment: str,
    gate: str,
    output_root: Path,
    env: dict[str, str],
    train_tool: Path,
    resume: bool,
    load_from: Path | None = None,
) -> Path | None:
    work_dir = output_root / model / stage
    work_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("dataset_contract.json", "class_support.json"):
        source = output_root / filename
        destination = work_dir / filename
        if source.is_file() and not destination.is_file():
            shutil.copy2(source, destination)
    audit_name = (
        "backbone_freeze_check.json"
        if stage == "stage1"
        else "backbone_update_check.json"
    )
    command = [
        sys.executable,
        str(train_tool),
        str(_config(model, stage, experiment)),
        "--work-dir",
        str(work_dir),
    ]
    checkpoint = _resumable_checkpoint(work_dir) if resume else None
    if checkpoint is not None:
        command.extend(["--resume", "--cfg-options", "load_from=None"])
    elif load_from is not None:
        command.extend(["--cfg-options", f"load_from={load_from}"])
    stage_env = _stage_env(env, gate=gate, stage=stage, work_dir=work_dir)
    if checkpoint is not None:
        import torch

        checkpoint_payload = torch.load(checkpoint, map_location="cpu")
        checkpoint_meta = checkpoint_payload.get("meta", {})
        runner_iterations = int(checkpoint_meta.get("iter", 0))
        micro_batch = int(stage_env["ADOM_MICRO_BATCH"])
        stage_env["ADOM_SAMPLER_START_INDEX"] = str(
            runner_iterations * micro_batch
        )
    else:
        stage_env["ADOM_SAMPLER_START_INDEX"] = "0"
    stage_env["ADOM_EXPERIMENT_TAG"] = f"experiment:{experiment}"
    stage_env["ADOM_MODEL_TAG"] = f"model:{model}"
    stage_env["ADOM_PHASE_TAG"] = f"phase:{stage}"
    stage_env = _tracking_env(
        stage_env,
        output_root=output_root,
        model=model,
        phase=f"{experiment}-{stage}-{gate}",
        job_type="training",
    )
    _run_phase(
        state,
        name=f"{model}_{stage}",
        command=command,
        artifacts=(
            [work_dir / audit_name]
            if gate == "smoke"
            else [work_dir / audit_name, work_dir / "checkpoint_selection.json"]
        ),
        env=stage_env,
        resume=resume,
    )
    if gate == "smoke":
        checkpoint = _resumable_checkpoint(work_dir)
        if checkpoint is None:
            raise RuntimeError(f"Smoke phase did not create a checkpoint: {work_dir}")
        return checkpoint
    best = resolve_single_best_checkpoint(work_dir)
    state.value["phases"][f"{model}_{stage}"]["best_checkpoint"] = str(best)
    state.save()
    return best


def validate_stage2_handoff(
    output_root: Path,
    model: str,
    stage1_checkpoint: Path,
) -> dict[str, Any]:
    """Prove Stage 2 consumes the RELLIS-val-selected Stage 1 checkpoint."""
    stage1_root = (output_root / model / "stage1").resolve()
    checkpoint = stage1_checkpoint.resolve()
    if stage1_root not in checkpoint.parents:
        raise RuntimeError(
            f"Stage 2 handoff escaped {stage1_root}: {checkpoint}"
        )
    selection_path = stage1_root / "checkpoint_selection.json"
    if not selection_path.is_file():
        raise RuntimeError("Stage 2 handoff requires Stage 1 checkpoint selection")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("schema_version") != "semantic20-clean-v1":
        raise RuntimeError("Stage 1 selection schema mismatch")
    selected = selection.get("selected", {})
    selected_checkpoint = Path(str(selected.get("checkpoint", ""))).resolve()
    if selected_checkpoint != checkpoint:
        raise RuntimeError(
            "Stage 2 checkpoint differs from Stage 1 RELLIS-val selection: "
            f"selected={selected_checkpoint}, handoff={checkpoint}"
        )
    report = {
        "schema_version": "semantic20-stage2-handoff-v1",
        "status": "PASS",
        "model": model,
        "selection_schema": selection["schema_version"],
        "selection_rule": selection.get("rule"),
        "selected_iteration": selected.get("iteration"),
        "stage1_checkpoint": str(checkpoint),
        "stage1_checkpoint_sha256": _file_sha256(checkpoint),
        "stage2_load_from": str(checkpoint),
    }
    write_json(output_root / model / "stage2_handoff.json", report)
    return report


def _run_resume_gate(
    *,
    state: CycleState,
    model: str,
    experiment: str,
    output_root: Path,
    env: dict[str, str],
    train_tool: Path,
    initial_checkpoint: Path | None = None,
) -> None:
    work_dir = output_root / model / "resume_check"
    base_env = dict(env)
    base_env.update(
        {
            "WANDB_MODE": "disabled",
            "ADOM_VAL_INTERVAL_OPTIMIZER_UPDATES": "5",
            "ADOM_CHECKPOINT_INTERVAL_OPTIMIZER_UPDATES": "1",
            "ADOM_METRIC_OUTPUT_DIR": work_dir.as_posix(),
        }
    )
    config = _config(model, "stage1", experiment)
    first_env = dict(base_env)
    first_env["ADOM_MAX_OPTIMIZER_UPDATES"] = "2"
    lp_head_updates = int(first_env.get("ADOM_TA_LP_HEAD_OPTIMIZER_UPDATES", "1000"))
    first_env["ADOM_TA_TOTAL_OPTIMIZER_UPDATES"] = str(
        round(2 * 6000 / lp_head_updates)
    )
    _run_phase(
        state,
        name=f"{model}_resume_seed",
        command=[
            sys.executable,
            str(train_tool),
            str(config),
            "--work-dir",
            str(work_dir),
        ]
        + (
            ["--cfg-options", f"load_from={initial_checkpoint}"]
            if initial_checkpoint is not None
            else []
        ),
        artifacts=[work_dir / "last_checkpoint"],
        env=first_env,
        resume=False,
    )
    first_checkpoint = _resumable_checkpoint(work_dir)
    if first_checkpoint is None:
        raise RuntimeError("Resume gate did not create its seed checkpoint")

    second_env = dict(base_env)
    second_env["ADOM_MAX_OPTIMIZER_UPDATES"] = "4"
    second_env["ADOM_TA_TOTAL_OPTIMIZER_UPDATES"] = str(
        round(4 * 6000 / lp_head_updates)
    )
    _run_phase(
        state,
        name=f"{model}_resume_restore",
        command=[
            sys.executable,
            str(train_tool),
            str(config),
            "--work-dir",
            str(work_dir),
            "--resume",
            "--cfg-options",
            "load_from=None",
        ],
        artifacts=[work_dir / "last_checkpoint"],
        env=second_env,
        resume=False,
    )
    second_checkpoint = _resumable_checkpoint(work_dir)
    if second_checkpoint is None or second_checkpoint == first_checkpoint:
        raise RuntimeError("Resume gate did not advance to a new checkpoint")

    import torch

    checkpoint = torch.load(second_checkpoint, map_location="cpu")
    required = {"state_dict", "optimizer", "param_schedulers"}
    missing = required - set(checkpoint)
    if missing:
        raise RuntimeError(
            f"Resumed checkpoint is missing optimizer/scheduler state: {sorted(missing)}"
        )
    write_json(
        work_dir / "resume_check.json",
        {
            "status": "PASS",
            "seed_checkpoint": str(first_checkpoint),
            "resumed_checkpoint": str(second_checkpoint),
            "optimizer_state": True,
            "scheduler_state": True,
        },
    )


def run_cycle(args: argparse.Namespace) -> None:
    dataset_root = args.dataset.resolve()
    output_root = args.output.resolve()
    b5_go_contract: dict[str, Any] | None = None
    models = _requested_models(args.models, args.experiment)
    if "b5" in models:
        if args.b5_go_decision is None:
            raise RuntimeError("B5 requires --b5-go-decision")
        if args.gpu_profile is None:
            raise RuntimeError("B5 requires an exact --gpu-profile")
        b5_go_contract = validate_b5_go_decision(args.b5_go_decision)
    elif args.b5_go_decision is not None:
        raise RuntimeError("--b5-go-decision is reserved for B5")
    if output_root.exists() and not args.resume:
        raise RuntimeError(f"Output exists: {output_root}; use --resume explicitly")
    output_root.mkdir(parents=True, exist_ok=True)
    state = CycleState(output_root / "status.json", args.resume)
    dataset_contract = validate_semantic20_dataset(dataset_root, args.experiment)
    initial_checkpoint: Path | None = None
    initial_checkpoint_contract: dict[str, Any] | None = None
    if args.experiment in TA_EXPERIMENTS:
        if args.initial_checkpoint is None or not args.expected_initial_checkpoint_sha256:
            raise RuntimeError(
                "TA experiments require --initial-checkpoint and "
                "--expected-initial-checkpoint-sha256"
            )
        initial_checkpoint_contract = validate_ta_initial_checkpoint(
            args.initial_checkpoint,
            args.expected_initial_checkpoint_sha256,
        )
        initial_checkpoint = Path(initial_checkpoint_contract["path"])
    elif args.initial_checkpoint is not None or args.expected_initial_checkpoint_sha256:
        raise RuntimeError("Initial checkpoint arguments are reserved for TA experiments")

    resume_contract = {
        "experiment": args.experiment,
        "seed": args.seed,
        "dataset_content_sha256": dataset_contract["dataset_content_sha256"],
        "initial_checkpoint_sha256": (
            initial_checkpoint_contract["sha256"] if initial_checkpoint_contract else None
        ),
        "b5_go_decision_sha256": (
            b5_go_contract["sha256"] if b5_go_contract else None
        ),
    }
    if args.resume and "resume_contract" in state.value:
        if state.value["resume_contract"] != resume_contract:
            raise RuntimeError(
                "Unsafe resume contract mismatch: "
                f"stored={state.value['resume_contract']}, requested={resume_contract}"
            )
    write_json(output_root / "dataset_contract.json", dataset_contract)
    write_json(output_root / "class_support.json", dataset_contract["class_support"])

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "src") + os.pathsep + env.get(
        "PYTHONPATH", ""
    )
    env["ADOM_DATA_ROOT"] = dataset_root.as_posix()
    env["ADOM_SEED"] = str(args.seed)
    env["ADOM_DETERMINISTIC"] = "true"
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    if initial_checkpoint_contract is not None:
        env["ADOM_EXPECTED_INITIAL_CHECKPOINT_SHA256"] = initial_checkpoint_contract[
            "sha256"
        ]
        env["ADOM_INITIAL_CHECKPOINT"] = initial_checkpoint_contract["path"]
    tags = [item for item in env.get("WANDB_TAGS", "").split(",") if item]
    tags.extend(
        (
            "phase1",
            "semantic20",
            "clean-v1",
            f"experiment:{args.experiment}",
            f"seed:{args.seed}",
        )
    )
    env["WANDB_TAGS"] = ",".join(dict.fromkeys(tags))

    doctor_path = output_root / "doctor.json"
    doctor_command = [
        sys.executable,
        "-m",
        "adom.runtime.doctor",
        "--require-gpu",
        "--require-gpu-name",
        args.require_gpu_name,
        "--minimum-gpu-memory-gib",
        str(args.minimum_gpu_memory_gib),
        "--skip-deployment",
        "--output",
        str(doctor_path),
    ]
    if args.gpu_profile:
        doctor_command.extend(["--gpu-profile", args.gpu_profile])
    else:
        doctor_command.extend(
            [
                "--require-gpu-name",
                args.require_gpu_name,
                "--minimum-gpu-memory-gib",
                str(args.minimum_gpu_memory_gib),
            ]
        )
    if args.expected_image_sha:
        doctor_command.extend(["--expected-image-sha", args.expected_image_sha])
    _run_phase(
        state,
        name="runtime_doctor",
        command=doctor_command,
        artifacts=[doctor_path],
        env=env,
        resume=args.resume,
    )
    train_tool = _tool_path("mmseg", ".mim/tools/train.py")
    test_tool = _tool_path("mmseg", ".mim/tools/test.py")

    state.value.update(
        {
            "experiment": args.experiment,
            "seed": args.seed,
            "gate": args.gate,
            "dataset_root": str(dataset_root),
            "initial_checkpoint": initial_checkpoint_contract,
            "b5_go_decision": b5_go_contract,
            "gpu_profile": args.gpu_profile,
            "resume_contract": resume_contract,
            "optimizer_update_domain": True,
            "export": "independent/not part of Phase 1 training cycle",
            "test_policy": {
                "locked_by_default": True,
                "run_test": bool(args.run_test),
                "final_test_model": args.final_test_model,
            },
            "wandb": {
                key: env.get(key)
                for key in ("WANDB_PROJECT", "WANDB_ENTITY", "WANDB_RUN_GROUP", "WANDB_TAGS")
                if env.get(key)
            },
        }
    )
    state.save()

    for model in models:
        if args.micro_batch is not None:
            if args.micro_batch not in _batch_candidates(model, args.gpu_profile):
                raise RuntimeError(
                    f"Unsupported {model} micro-batch {args.micro_batch}; "
                    f"choose from {_batch_candidates(model, args.gpu_profile)}"
                )
            micro_batch = args.micro_batch
            accumulative = EFFECTIVE_BATCH // micro_batch
        elif args.skip_batch_probe:
            micro_batch = 16
            accumulative = 1
        else:
            micro_batch, accumulative = _probe_batch(
                model=model,
                config=_config(model, "stage1", args.experiment),
                output_root=output_root,
                env=env,
                train_tool=train_tool,
                resume=args.resume,
                load_from=initial_checkpoint,
                gpu_profile=args.gpu_profile,
            )
        if args.gate == "probe":
            continue
        model_env = dict(env)
        model_env["ADOM_MICRO_BATCH"] = str(micro_batch)
        model_env["ADOM_ACCUMULATIVE_COUNTS"] = str(accumulative)
        if args.gate == "resume":
            _run_resume_gate(
                state=state,
                model=model,
                experiment=args.experiment,
                output_root=output_root,
                env=model_env,
                train_tool=train_tool,
                initial_checkpoint=initial_checkpoint,
            )
            continue
        stage1_best = _train_stage(
            state=state,
            model=model,
            stage="stage1",
            experiment=args.experiment,
            gate=args.gate,
            output_root=output_root,
            env=model_env,
            train_tool=train_tool,
            resume=args.resume,
            load_from=initial_checkpoint,
        )
        if args.gate != "full" and args.experiment not in TA_EXPERIMENTS:
            continue
        if stage1_best is None:
            raise RuntimeError("TA/full Stage 2 requires a Stage 1 checkpoint")
        if args.gate != "smoke":
            handoff = validate_stage2_handoff(output_root, model, stage1_best)
            state.value["phases"][f"{model}_stage1"]["stage2_handoff"] = handoff
            state.save()
        stage2_best = _train_stage(
            state=state,
            model=model,
            stage="stage2",
            experiment=args.experiment,
            gate=args.gate,
            output_root=output_root,
            env=model_env,
            train_tool=train_tool,
            resume=args.resume,
            load_from=stage1_best,
        )
        if stage2_best is None:
            raise RuntimeError("Full training requires a Stage 2 selected checkpoint")
        if args.gate != "full":
            continue
        if not args.run_test or model != args.final_test_model:
            continue
        test_dir = output_root / model / "test"
        test_env = dict(model_env)
        test_env["ADOM_METRIC_OUTPUT_DIR"] = test_dir.as_posix()
        test_env["ADOM_CANONICAL_TEST_UNLOCK"] = "final-model-confirmed"
        test_env["ADOM_EXPERIMENT_TAG"] = f"experiment:{args.experiment}"
        test_env["ADOM_MODEL_TAG"] = f"model:{model}"
        test_env["ADOM_PHASE_TAG"] = "phase:test"
        test_env = _tracking_env(
            test_env,
            output_root=output_root,
            model=model,
            phase=f"{args.experiment}-test",
            job_type="evaluation",
        )
        _run_phase(
            state,
            name=f"{model}_test",
            command=[
                sys.executable,
                str(test_tool),
                str(_config(model, "stage2", args.experiment)),
                str(stage2_best),
                "--work-dir",
                str(test_dir),
            ],
            artifacts=[test_dir / "test_metrics.json", test_dir / "confusion_matrix.json"],
            env=test_env,
            resume=args.resume,
        )

    summary_models: list[dict[str, Any]] = []
    for model in models:
        selection_path = output_root / model / "stage2" / "checkpoint_selection.json"
        model_summary: dict[str, Any] = {"model": model}
        if selection_path.is_file():
            selection = json.loads(selection_path.read_text(encoding="utf-8"))
            model_summary["validation_selection"] = selection.get("selected", {})
        metric_path = output_root / model / "test" / "test_metrics.json"
        if metric_path.is_file():
            metric_value = json.loads(metric_path.read_text(encoding="utf-8"))
            model_summary["test_metrics"] = metric_value.get("metrics", {})
        if len(model_summary) > 1:
            summary_models.append(model_summary)
    write_json(
        output_root / "summary.json",
        {
            "experiment": args.experiment,
            "seed": args.seed,
            "gate": args.gate,
            "optimizer_update_domain": True,
            "dataset_contract": dataset_contract,
            "models": summary_models,
        },
    )
    state.value["status"] = "completed"
    state.value["finished_at"] = _now()
    state.save()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run Clean v1 E0/E1/E2/E-ADOM/TA0/TA1/TA2 Semantic20 SegFormer gates"
        )
    )
    parser.add_argument("--dataset", type=Path)
    parser.add_argument(
        "--experiment",
        choices=("e0", "e1", "e2", "eadom", "ta0", "ta1", "ta2"),
        required=True,
    )
    parser.add_argument("--models", default="b0,b2")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--gate",
        choices=("probe", "smoke", "mini", "resume", "full"),
        default="full",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-batch-probe", action="store_true")
    parser.add_argument("--micro-batch", type=int)
    parser.add_argument(
        "--require-gpu-name",
        default="A100",
        help="Substring required in the runtime GPU name; defaults to the A100 contract.",
    )
    parser.add_argument(
        "--minimum-gpu-memory-gib",
        type=float,
        default=75.0,
        help="Minimum physical GPU memory; defaults to the A100 80GB contract.",
    )
    parser.add_argument(
        "--gpu-profile",
        choices=tuple(GPU_PROFILES),
        help="Exact model/VRAM profile required by the preregistered B5 study.",
    )
    parser.add_argument("--seed", type=int, choices=(42, 43, 44), default=42)
    parser.add_argument(
        "--run-test",
        action="store_true",
        help="Unlock canonical test for the one final model named by --final-test-model.",
    )
    parser.add_argument("--final-test-model", choices=("b0", "b2", "b5"))
    parser.add_argument(
        "--expected-image-sha",
        help="Require ADOM_GIT_SHA inside the immutable Docker image to match.",
    )
    parser.add_argument("--initial-checkpoint", type=Path)
    parser.add_argument("--expected-initial-checkpoint-sha256")
    parser.add_argument(
        "--b5-go-decision",
        type=Path,
        help="Frozen B2 evidence decision artifact required before any B5 gate.",
    )
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="Compatibility flag; export is already independent from this training cycle",
    )
    args = parser.parse_args(argv)
    if not args.require_gpu_name.strip():
        parser.error("--require-gpu-name must not be empty")
    if args.minimum_gpu_memory_gib <= 0:
        parser.error("--minimum-gpu-memory-gib must be positive")
    if args.run_test != bool(args.final_test_model):
        parser.error("--run-test and --final-test-model must be provided together")
    if args.run_test and args.gate != "full":
        parser.error("canonical test can only run with --gate full")
    requested_models = {
        item.strip().lower() for item in args.models.split(",") if item.strip()
    }
    if args.final_test_model and args.final_test_model not in requested_models:
        parser.error("--final-test-model must be included in --models")
    if args.dataset is None:
        value = os.getenv("ADOM_DATA_ROOT")
        if not value:
            parser.error("provide --dataset or ADOM_DATA_ROOT")
        args.dataset = Path(value)
    try:
        run_cycle(args)
    except BaseException as error:
        status_path = args.output.resolve() / "status.json"
        if status_path.is_file():
            try:
                status = json.loads(status_path.read_text(encoding="utf-8"))
                status.update(
                    {
                        "status": "failed",
                        "finished_at": _now(),
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
                write_json(status_path, status)
            except Exception:
                pass
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
