from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from adom.data.io import sha256_file, write_json
from adom.data.schema import LabelSchema
from adom.data.validation import validate_package
from adom.runtime.checkpoints import resolve_single_best_checkpoint


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_ROOT = REPO_ROOT / "configs"
EFFECTIVE_BATCH = 16
PROFILES = {
    "640x384": (640, 384),
    "384x384": (384, 384),
}


def _bounded_wandb_id(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_-]+", "-", value).strip("-") or "adom-run"
    if len(normalized) <= 64:
        return normalized
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:8]
    return f"{normalized[:55]}-{digest}"


def _tracking_env(
    base_env: dict[str, str],
    *,
    output_root: Path,
    model: str,
    phase: str,
    job_type: str,
) -> dict[str, str]:
    env = dict(base_env)
    logical_run = env.get("ADOM_RUN_ID", output_root.name)
    run_prefix = env.get("WANDB_RUN_ID", logical_run)
    name_prefix = env.get("WANDB_NAME", logical_run)
    env.setdefault("WANDB_PROJECT", "adom")
    env.setdefault("WANDB_RUN_GROUP", logical_run)
    env.setdefault("WANDB_DIR", str(output_root / "wandb"))
    env["WANDB_RUN_ID"] = _bounded_wandb_id(f"{run_prefix}-{model}-{phase}")
    env["WANDB_NAME"] = f"{name_prefix}-{model}-{phase}"
    env["WANDB_JOB_TYPE"] = job_type
    env.setdefault("WANDB_RESUME", "allow")
    tags = [item.strip() for item in env.get("WANDB_TAGS", "").split(",")]
    tags.extend((f"model:{model}", f"phase:{phase}"))
    env["WANDB_TAGS"] = ",".join(dict.fromkeys(item for item in tags if item))
    return env


def _resumable_checkpoint(work_dir: Path) -> Path | None:
    marker = work_dir / "last_checkpoint"
    if not marker.is_file():
        return None
    value = marker.read_text(encoding="utf-8").strip()
    if not value:
        raise RuntimeError(f"Checkpoint marker is empty: {marker}")
    checkpoint = Path(value)
    if not checkpoint.is_absolute():
        checkpoint = work_dir / checkpoint
    if not checkpoint.is_file():
        raise RuntimeError(
            f"Checkpoint marker points to a missing file: {marker} -> {checkpoint}"
        )
    return checkpoint.resolve()


class CycleState:
    def __init__(self, path: Path, resume: bool) -> None:
        self.path = path
        if resume and path.is_file():
            self.value = json.loads(path.read_text(encoding="utf-8"))
        else:
            self.value = {
                "format_version": 1,
                "status": "running",
                "started_at": _now(),
                "phases": {},
            }
            self.save()

    def save(self) -> None:
        write_json(self.path, self.value)

    def completed(self, name: str, artifacts: list[Path]) -> bool:
        phase = self.value["phases"].get(name, {})
        return phase.get("status") == "completed" and all(
            path.exists() for path in artifacts
        )

    def start(self, name: str, command: list[str]) -> None:
        self.value["phases"][name] = {
            "status": "running",
            "started_at": _now(),
            "command": command,
        }
        self.save()

    def finish(self, name: str, artifacts: list[Path]) -> None:
        phase = self.value["phases"][name]
        phase["status"] = "completed"
        phase["finished_at"] = _now()
        phase["artifacts"] = [str(path.resolve()) for path in artifacts]
        self.save()

    def fail(self, name: str, error: BaseException) -> None:
        phase = self.value["phases"].setdefault(name, {})
        phase["status"] = "failed"
        phase["finished_at"] = _now()
        phase["error"] = f"{type(error).__name__}: {error}"
        self.value["status"] = "failed"
        self.save()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _tool_path(package: str, relative: str) -> Path:
    module = importlib.import_module(package)
    path = Path(module.__file__).resolve().parent / relative
    if not path.is_file():
        raise RuntimeError(f"{package} tool not found: {path}")
    return path


def _run_phase(
    state: CycleState,
    *,
    name: str,
    command: list[str],
    artifacts: list[Path],
    env: dict[str, str],
    resume: bool,
) -> None:
    if resume and state.completed(name, artifacts):
        print(f"[resume] {name}: already completed")
        return
    state.start(name, command)
    print(f"[run] {name}")
    try:
        subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)
        missing = [str(path) for path in artifacts if not path.exists()]
        if missing:
            raise RuntimeError(f"phase did not create required artifacts: {missing}")
        state.finish(name, artifacts)
    except BaseException as error:
        state.fail(name, error)
        raise


def _first_test_image(dataset_root: Path) -> Path:
    manifest = dataset_root / "metadata" / "manifest_test.csv"
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        row = next(csv.DictReader(handle), None)
    if not row:
        raise RuntimeError(f"No test sample in {manifest}")
    path = dataset_root / row["image_relpath"]
    if not path.is_file():
        raise RuntimeError(f"Test image is missing: {path}")
    return path


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

    candidates = [16, 8, 4, 2, 1] if model == "b0" else [8, 4, 2, 1]
    for micro_batch in candidates:
        probe_dir = output_root / model / "probes" / f"micro_batch_{micro_batch}"
        probe_dir.mkdir(parents=True, exist_ok=True)
        log_path = probe_dir / "probe.log"
        probe_env = dict(env)
        probe_env["ADOM_MICRO_BATCH"] = str(micro_batch)
        probe_env["ADOM_ACCUMULATIVE_COUNTS"] = "1"
        probe_env["WANDB_MODE"] = "disabled"
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
        print(f"[probe] {model}: trying micro-batch {micro_batch}")
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(
                command,
                cwd=REPO_ROOT,
                env=probe_env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        if result.returncode == 0:
            accumulative = math.ceil(EFFECTIVE_BATCH / micro_batch)
            write_json(
                plan_path,
                {
                    "model": model,
                    "micro_batch": micro_batch,
                    "accumulative_counts": accumulative,
                    "effective_batch": micro_batch * accumulative,
                    "probe_log": str(log_path.resolve()),
                },
            )
            return micro_batch, accumulative
        log_text = log_path.read_text(encoding="utf-8", errors="replace").lower()
        if "out of memory" not in log_text:
            raise RuntimeError(
                f"{model} batch probe failed for a non-OOM reason; see {log_path}"
            )
    raise RuntimeError(f"No viable micro-batch found for {model}")


def _write_summary(output_root: Path, state: CycleState) -> None:
    rows: list[dict[str, Any]] = []
    for model in ("b0", "b2"):
        metric_path = output_root / model / "test" / "test_metrics.json"
        if not metric_path.is_file():
            continue
        value = json.loads(metric_path.read_text(encoding="utf-8"))
        row: dict[str, Any] = {"model": model}
        row.update(value.get("metrics", {}))
        rows.append(row)
    summary = {
        "status": "completed",
        "finished_at": _now(),
        "models": rows,
        "state_file": str(state.path.resolve()),
    }
    write_json(output_root / "summary.json", summary)
    fieldnames = sorted({key for row in rows for key in row}) or ["model"]
    with (output_root / "summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_artifact_manifest(output_root: Path) -> None:
    forbidden = [
        path
        for path in output_root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".engine", ".plan", ".trt"}
    ]
    if forbidden:
        raise RuntimeError(
            "RunPod output must not contain TensorRT engines: "
            + ", ".join(str(path) for path in forbidden)
        )
    selected: set[Path] = set()
    for pattern in (
        "doctor.json",
        "status.json",
        "summary.json",
        "summary.csv",
        "**/batch_plan.json",
        "**/backbone_*_check.json",
        "**/test_metrics.json",
        "**/best_mIoU_iter_*.pth",
        "**/end2end.onnx",
        "**/deploy.json",
        "**/parity.json",
        "**/metadata.json",
    ):
        selected.update(
            path for path in output_root.glob(pattern) if path.is_file()
        )
    write_json(
        output_root / "artifact_manifest.json",
        {
            "format_version": 1,
            "created_at": _now(),
            "artifacts": [
                {
                    "path": path.relative_to(output_root).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for path in sorted(selected)
            ],
            "tensorrt_engine_included": False,
        },
    )


def run_cycle(args: argparse.Namespace) -> None:
    dataset_root = args.dataset.resolve()
    output_root = args.output.resolve()
    if output_root.exists() and not args.resume:
        raise RuntimeError(
            f"Output already exists: {output_root}. Use --resume explicitly."
        )
    output_root.mkdir(parents=True, exist_ok=True)
    state = CycleState(output_root / "status.json", args.resume)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "src") + os.pathsep + env.get(
        "PYTHONPATH", ""
    )
    env["ADOM_DATA_ROOT"] = str(dataset_root)

    models = [item.strip().lower() for item in args.models.split(",") if item.strip()]
    if not models or models[0] != "b0" or any(item not in {"b0", "b2"} for item in models):
        raise RuntimeError("--models must start with b0 and contain only b0,b2")
    if len(models) != len(set(models)):
        raise RuntimeError("--models contains duplicates")

    doctor_path = output_root / "doctor.json"
    doctor_command = [
        sys.executable,
        "-m",
        "adom.runtime.doctor",
        "--dataset-root",
        str(dataset_root),
        "--require-gpu",
        "--output",
        str(doctor_path),
    ]
    _run_phase(
        state,
        name="doctor_and_dataset_qc",
        command=doctor_command,
        artifacts=[doctor_path],
        env=env,
        resume=args.resume,
    )

    report = validate_package(dataset_root, verify_checksums=True)
    report.require_success()
    for split in ("train", "val", "test"):
        package_split = dataset_root / "splits" / f"{split}.txt"
        committed_split = (
            REPO_ROOT / "data" / "splits" / "rellis3d" / "official" / f"{split}.txt"
        )
        if sha256_file(package_split) != sha256_file(committed_split):
            raise RuntimeError(
                f"Dataset {split} split differs from committed official split"
            )
    committed_mapping = CONFIG_ROOT / "datasets" / "rellis3d" / "label_mapping.yaml"
    metadata = json.loads(
        (dataset_root / "metadata" / "dataset.json").read_text(encoding="utf-8")
    )
    if metadata.get("mapping_sha256") != sha256_file(committed_mapping):
        raise RuntimeError("Dataset mapping does not match the committed Cost4 mapping")
    if (
        LabelSchema.from_path(dataset_root / "config" / "label_mapping.yaml").snapshot()
        != LabelSchema.from_path(committed_mapping).snapshot()
    ):
        raise RuntimeError("Packaged mapping semantics differ from the committed mapping")
    state.value["dataset_checksum_manifest_sha256"] = sha256_file(
        dataset_root / "SHA256SUMS.txt"
    )
    state.save()

    train_tool = _tool_path("mmseg", ".mim/tools/train.py")
    test_tool = _tool_path("mmseg", ".mim/tools/test.py")
    deploy_tool = _tool_path("mmdeploy", ".mim/tools/deploy.py")
    image = _first_test_image(dataset_root)
    mapping = dataset_root / "config" / "label_mapping.yaml"

    for model in models:
        stage1_config = CONFIG_ROOT / "adom" / f"segformer_{model}_stage1_rellis3d.py"
        stage2_config = CONFIG_ROOT / "adom" / f"segformer_{model}_stage2_rellis3d.py"
        if args.skip_batch_probe:
            micro_batch = 4 if model == "b0" else 2
            accumulative = math.ceil(EFFECTIVE_BATCH / micro_batch)
        else:
            micro_batch, accumulative = _probe_batch(
                model=model,
                config=stage1_config,
                output_root=output_root,
                env=env,
                train_tool=train_tool,
                resume=args.resume,
            )
        model_env = dict(env)
        model_env["ADOM_MICRO_BATCH"] = str(micro_batch)
        model_env["ADOM_ACCUMULATIVE_COUNTS"] = str(accumulative)

        stage1_dir = output_root / model / "stage1"
        stage1_audit = stage1_dir / "backbone_freeze_check.json"
        stage1_command = [
            sys.executable,
            str(train_tool),
            str(stage1_config),
            "--work-dir",
            str(stage1_dir),
        ]
        stage1_checkpoint = _resumable_checkpoint(stage1_dir) if args.resume else None
        if stage1_checkpoint is not None:
            stage1_command.extend(["--resume", "--cfg-options", "load_from=None"])
        stage1_env = _tracking_env(
            model_env,
            output_root=output_root,
            model=model,
            phase="stage1",
            job_type="training",
        )
        _run_phase(
            state,
            name=f"{model}_stage1",
            command=stage1_command,
            artifacts=[stage1_audit],
            env=stage1_env,
            resume=args.resume,
        )
        stage1_best = resolve_single_best_checkpoint(stage1_dir)
        state.value["phases"][f"{model}_stage1"]["best_checkpoint"] = str(stage1_best)
        state.save()

        stage2_dir = output_root / model / "stage2"
        stage2_audit = stage2_dir / "backbone_update_check.json"
        stage2_command = [
            sys.executable,
            str(train_tool),
            str(stage2_config),
            "--work-dir",
            str(stage2_dir),
        ]
        stage2_checkpoint = _resumable_checkpoint(stage2_dir) if args.resume else None
        if stage2_checkpoint is not None:
            stage2_command.extend(["--resume", "--cfg-options", "load_from=None"])
        else:
            stage2_command.extend(["--cfg-options", f"load_from={stage1_best}"])
        stage2_env = _tracking_env(
            model_env,
            output_root=output_root,
            model=model,
            phase="stage2",
            job_type="training",
        )
        _run_phase(
            state,
            name=f"{model}_stage2",
            command=stage2_command,
            artifacts=[stage2_audit],
            env=stage2_env,
            resume=args.resume,
        )
        stage2_best = resolve_single_best_checkpoint(stage2_dir)
        state.value["phases"][f"{model}_stage2"]["best_checkpoint"] = str(stage2_best)
        state.save()

        test_dir = output_root / model / "test"
        test_metrics = test_dir / "test_metrics.json"
        test_command = [
            sys.executable,
            str(test_tool),
            str(stage2_config),
            str(stage2_best),
            "--work-dir",
            str(test_dir),
        ]
        test_env = _tracking_env(
            model_env,
            output_root=output_root,
            model=model,
            phase="test",
            job_type="evaluation",
        )
        _run_phase(
            state,
            name=f"{model}_test",
            command=test_command,
            artifacts=[test_metrics],
            env=test_env,
            resume=args.resume,
        )

        for profile, (width, height) in PROFILES.items():
            export_model_config = (
                CONFIG_ROOT
                / "adom"
                / "export"
                / f"segformer_{model}_{profile}_rellis3d.py"
            )
            deploy_config = (
                CONFIG_ROOT / "deployment" / f"mmseg_onnxruntime_{profile}.py"
            )
            export_dir = output_root / model / "onnx" / profile
            onnx_path = export_dir / "end2end.onnx"
            deploy_info = export_dir / "deploy.json"
            parity_path = export_dir / "parity.json"
            metadata_path = export_dir / "metadata.json"
            export_command = [
                sys.executable,
                str(deploy_tool),
                str(deploy_config),
                str(export_model_config),
                str(stage2_best),
                str(image),
                "--work-dir",
                str(export_dir),
                "--device",
                args.device,
                "--dump-info",
            ]
            artifact_env = dict(model_env)
            artifact_env["WANDB_MODE"] = "disabled"
            _run_phase(
                state,
                name=f"{model}_onnx_{profile}",
                command=export_command,
                artifacts=[onnx_path, deploy_info],
                env=artifact_env,
                resume=args.resume,
            )
            parity_command = [
                sys.executable,
                "-m",
                "adom.runtime.onnx_parity",
                "--deploy-config",
                str(deploy_config),
                "--model-config",
                str(export_model_config),
                "--checkpoint",
                str(stage2_best),
                "--onnx",
                str(onnx_path),
                "--image",
                str(image),
                "--device",
                args.device,
                "--output",
                str(parity_path),
            ]
            _run_phase(
                state,
                name=f"{model}_parity_{profile}",
                command=parity_command,
                artifacts=[parity_path],
                env=artifact_env,
                resume=args.resume,
            )
            metadata_command = [
                sys.executable,
                "-m",
                "adom.runtime.artifacts",
                "--repo-root",
                str(REPO_ROOT),
                "--dataset-root",
                str(dataset_root),
                "--model-config",
                str(export_model_config),
                "--checkpoint",
                str(stage2_best),
                "--onnx",
                str(onnx_path),
                "--deploy-info",
                str(deploy_info),
                "--mapping",
                str(mapping),
                "--profile",
                profile,
                "--width",
                str(width),
                "--height",
                str(height),
                "--output",
                str(metadata_path),
            ]
            _run_phase(
                state,
                name=f"{model}_metadata_{profile}",
                command=metadata_command,
                artifacts=[metadata_path],
                env=artifact_env,
                resume=args.resume,
            )

    _write_summary(output_root, state)
    state.value["status"] = "completed"
    state.value["finished_at"] = _now()
    state.save()
    _write_artifact_manifest(output_root)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run ADOM Cost4 B0/B2 training, evaluation, and ONNX export"
    )
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--models", default="b0,b2")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-batch-probe", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    try:
        run_cycle(args)
    except BaseException as error:
        status_path = args.output.resolve() / "status.json"
        if status_path.is_file():
            try:
                status = json.loads(status_path.read_text(encoding="utf-8"))
                status["status"] = "failed"
                status["finished_at"] = _now()
                status["error"] = f"{type(error).__name__}: {error}"
                write_json(status_path, status)
            except Exception:
                pass
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
