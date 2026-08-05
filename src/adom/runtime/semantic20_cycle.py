from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from adom.data.io import write_json
from adom.data.semantic20 import resource_path
from adom.runtime.checkpoints import resolve_single_best_checkpoint
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


CONFIG_DIR = CONFIG_ROOT / "adom" / "phase1_semantic20"
REFERENCE_SPLITS = resource_path("rellis", "splits")
GATE_UPDATES = {"smoke": 50, "mini": 500}
EXPECTED_SPLIT_COUNTS = {
    "e0": {"train": 4435, "val": 900, "test": 899},
    "e1": {"train": 9868, "val": 900, "test": 899},
}
EXPECTED_E1_MANIFEST_COUNT = 14421
EXPECTED_E1_MAIN_SOURCE_COUNTS = Counter(
    {"rellis3d": 6234, "rugd": 4779, "ycor": 654}
)
EXPECTED_E1_MANIFEST_SOURCE_COUNTS = Counter(
    {"rellis3d": 6234, "rugd": 7436, "ycor": 751}
)
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


def _validate_semantic20_pair(image: Path, mask: Path, key: str) -> None:
    if not image.is_file() or not mask.is_file():
        raise FileNotFoundError(f"Missing dataset pair: {image}, {mask}")
    with Image.open(image) as image_file:
        image_file.load()
        image_size = image_file.size
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


def validate_semantic20_dataset(dataset_root: Path, experiment: str) -> dict[str, Any]:
    success_marker = dataset_root / "_SUCCESS"
    if not success_marker.is_file():
        raise FileNotFoundError(f"Dataset success marker is missing: {success_marker}")
    expected: dict[str, list[str]] = {
        split: _read_split(REFERENCE_SPLITS / f"{split}.txt")
        for split in ("train", "val", "test")
    }
    actual = {
        split: _read_split(dataset_root / "splits" / f"{split}.txt")
        for split in ("train", "val", "test")
    }
    actual_counts = {key: len(value) for key, value in actual.items()}
    if actual_counts != EXPECTED_SPLIT_COUNTS[experiment]:
        raise RuntimeError(
            f"{experiment} split counts differ from the Semantic20 contract: "
            f"actual={actual_counts}, expected={EXPECTED_SPLIT_COUNTS[experiment]}"
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
            experiment == "e1"
        ):
            raise RuntimeError(f"E1 {split} must contain RELLIS samples only")
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
            raise RuntimeError("E1 does not contain the canonical RELLIS train split")
        sources = {value.split("/", 1)[0] for value in actual["train"]}
        if not {"rellis3d", "rugd", "ycor"}.issubset(sources):
            raise RuntimeError("E1 train must contain RELLIS, RUGD, and YCOR")

    manifest_rows: dict[str, tuple[str, str]] = {}
    if experiment == "e1":
        final_check_path = dataset_root / "results" / "final_check.json"
        if not final_check_path.is_file():
            raise FileNotFoundError(f"E1 final check is missing: {final_check_path}")
        final_check = json.loads(final_check_path.read_text(encoding="utf-8-sig"))
        if str(final_check.get("status", "")).upper() != "PASS":
            raise RuntimeError(f"E1 final check is not PASS: {final_check_path}")
        manifest_path = dataset_root / "manifest.csv"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"E1 manifest is missing: {manifest_path}")
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
                    raise RuntimeError(f"Duplicate E1 manifest sample: {key}")
                manifest_rows[key] = (row["image_path"], row["mask_path"])
        if len(manifest_rows) != EXPECTED_E1_MANIFEST_COUNT:
            raise RuntimeError(
                "E1 manifest count differs from contract: "
                f"{len(manifest_rows)} != {EXPECTED_E1_MANIFEST_COUNT}"
            )

    # Check every pair before paying for GPU time. E1 must use manifest paths
    # because RUGD images are PNG while RELLIS and YCOR images are JPEG.
    main_source_counts: Counter[str] = Counter()
    verified_pairs = 0
    if experiment == "e1":
        manifest_source_counts: Counter[str] = Counter()
        for key, (image_relpath, mask_relpath) in manifest_rows.items():
            image = (dataset_root / image_relpath).resolve()
            mask = (dataset_root / mask_relpath).resolve()
            if dataset_root.resolve() not in image.parents or dataset_root.resolve() not in mask.parents:
                raise RuntimeError(f"E1 manifest path escapes dataset root: {key}")
            _validate_semantic20_pair(image, mask, key)
            manifest_source_counts[key.split("/", 1)[0]] += 1
            verified_pairs += 1
        if manifest_source_counts != EXPECTED_E1_MANIFEST_SOURCE_COUNTS:
            raise RuntimeError(
                "E1 manifest source counts differ from contract: "
                f"{dict(manifest_source_counts)}"
            )

    for split, keys in actual.items():
        for key in keys:
            if experiment == "e1":
                if key not in manifest_rows:
                    raise RuntimeError(f"E1 split sample is absent from manifest: {key}")
                main_source_counts[key.split("/", 1)[0]] += 1
            else:
                image = dataset_root / "images" / f"{key}.jpg"
                mask = dataset_root / "masks" / f"{key}.png"
                _validate_semantic20_pair(image, mask, key)
                verified_pairs += 1

    if experiment == "e1" and main_source_counts != EXPECTED_E1_MAIN_SOURCE_COUNTS:
        # Main val/test are RELLIS-only, so 4,435+900+899 RELLIS pairs are checked.
        raise RuntimeError(
            f"E1 main split source counts differ from contract: {dict(main_source_counts)}"
        )

    digest = hashlib.sha256()
    for split in ("train", "val", "test"):
        digest.update(("\n".join(actual[split]) + "\n").encode("utf-8"))
    return {
        "experiment": experiment,
        "num_classes": 19,
        "ignore_index": 255,
        "split_counts": actual_counts,
        "verified_pairs": verified_pairs,
        "split_contract_sha256": digest.hexdigest(),
        "validation_test_policy": "canonical RELLIS-only",
    }


def _config(model: str, stage: str, experiment: str) -> Path:
    suffix = "e0_rellis" if experiment == "e0" else "e1_combined"
    return CONFIG_DIR / f"segformer_{model}_{stage}_{suffix}.py"


def _batch_candidates(model: str) -> list[int]:
    return [16] if model == "b0" else [16, 8, 4]


def _probe_batch(
    *,
    model: str,
    config: Path,
    output_root: Path,
    env: dict[str, str],
    train_tool: Path,
    resume: bool,
) -> tuple[int, int]:
    plan_path = output_root / model / "batch_plan.json"
    if resume and plan_path.is_file():
        value = json.loads(plan_path.read_text(encoding="utf-8"))
        return int(value["micro_batch"]), int(value["accumulative_counts"])
    for micro_batch in _batch_candidates(model):
        accumulative = EFFECTIVE_BATCH // micro_batch
        probe_dir = output_root / model / "probes" / f"micro_batch_{micro_batch}"
        probe_dir.mkdir(parents=True, exist_ok=True)
        log_path = probe_dir / "probe.log"
        probe_env = dict(env)
        probe_env.update(
            {
                "ADOM_MICRO_BATCH": str(micro_batch),
                "ADOM_ACCUMULATIVE_COUNTS": "1",
                "WANDB_MODE": "disabled",
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
        ]
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
                    "fallback_order": _batch_candidates(model),
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
        value["ADOM_VAL_INTERVAL_OPTIMIZER_UPDATES"] = str(
            updates if gate == "mini" else updates + 1
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
    stage_env["WANDB_EXTRA_TAG"] = "extra:" + stage_env.get(
        "WANDB_TAGS", "runpod"
    ).replace(",", "+")
    _run_phase(
        state,
        name=f"{model}_{stage}",
        command=command,
        artifacts=[work_dir / audit_name],
        env=stage_env,
        resume=resume,
    )
    if gate == "smoke":
        return None
    best = resolve_single_best_checkpoint(work_dir)
    state.value["phases"][f"{model}_{stage}"]["best_checkpoint"] = str(best)
    state.save()
    return best


def _run_resume_gate(
    *,
    state: CycleState,
    model: str,
    experiment: str,
    output_root: Path,
    env: dict[str, str],
    train_tool: Path,
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
    _run_phase(
        state,
        name=f"{model}_resume_seed",
        command=[
            sys.executable,
            str(train_tool),
            str(config),
            "--work-dir",
            str(work_dir),
        ],
        artifacts=[work_dir / "last_checkpoint"],
        env=first_env,
        resume=False,
    )
    first_checkpoint = _resumable_checkpoint(work_dir)
    if first_checkpoint is None:
        raise RuntimeError("Resume gate did not create its seed checkpoint")

    second_env = dict(base_env)
    second_env["ADOM_MAX_OPTIMIZER_UPDATES"] = "4"
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
    if output_root.exists() and not args.resume:
        raise RuntimeError(f"Output exists: {output_root}; use --resume explicitly")
    output_root.mkdir(parents=True, exist_ok=True)
    state = CycleState(output_root / "status.json", args.resume)
    dataset_contract = validate_semantic20_dataset(dataset_root, args.experiment)
    write_json(output_root / "dataset_contract.json", dataset_contract)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "src") + os.pathsep + env.get(
        "PYTHONPATH", ""
    )
    env["ADOM_DATA_ROOT"] = dataset_root.as_posix()
    tags = [item for item in env.get("WANDB_TAGS", "").split(",") if item]
    tags.extend(("phase1", "semantic20", f"experiment:{args.experiment}", "seed:42"))
    env["WANDB_TAGS"] = ",".join(dict.fromkeys(tags))

    models = [item.strip().lower() for item in args.models.split(",") if item.strip()]
    if not models or any(item not in {"b0", "b2"} for item in models):
        raise RuntimeError("--models accepts b0 and/or b2")
    if len(models) != len(set(models)):
        raise RuntimeError("--models contains duplicates")

    doctor_path = output_root / "doctor.json"
    doctor_command = [
        sys.executable,
        "-m",
        "adom.runtime.doctor",
        "--require-gpu",
        "--require-gpu-name",
        "A100",
        "--minimum-gpu-memory-gib",
        "75",
        "--skip-deployment",
        "--output",
        str(doctor_path),
    ]
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
            "gate": args.gate,
            "dataset_root": str(dataset_root),
            "optimizer_update_domain": True,
            "export": "independent/not part of Phase 1 training cycle",
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
            if args.micro_batch not in _batch_candidates(model):
                raise RuntimeError(
                    f"Unsupported {model} micro-batch {args.micro_batch}; "
                    f"choose from {_batch_candidates(model)}"
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
        )
        if args.gate != "full":
            continue
        if stage1_best is None:
            raise RuntimeError("Full Stage 2 requires a Stage 1 best checkpoint")
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
            raise RuntimeError("Full test requires a Stage 2 best checkpoint")
        test_dir = output_root / model / "test"
        test_env = dict(model_env)
        test_env["ADOM_METRIC_OUTPUT_DIR"] = test_dir.as_posix()
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
        test_env["WANDB_EXTRA_TAG"] = "extra:" + test_env.get(
            "WANDB_TAGS", "runpod"
        ).replace(",", "+")
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
        metric_path = output_root / model / "test" / "test_metrics.json"
        if metric_path.is_file():
            metric_value = json.loads(metric_path.read_text(encoding="utf-8"))
            summary_models.append(
                {"model": model, "metrics": metric_value.get("metrics", {})}
            )
    write_json(
        output_root / "summary.json",
        {
            "experiment": args.experiment,
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
        description="Run Phase 1 Semantic20 SegFormer optimizer-update gates"
    )
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--experiment", choices=("e0", "e1"), required=True)
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
        "--expected-image-sha",
        help="Require ADOM_GIT_SHA inside the immutable Docker image to match.",
    )
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="Compatibility flag; export is already independent from this training cycle",
    )
    args = parser.parse_args(argv)
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
