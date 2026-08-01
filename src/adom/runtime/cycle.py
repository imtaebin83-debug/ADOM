from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from adom.data.io import sha256_file, write_json
from adom.data.schema import LabelSchema
from adom.data.validation import validate_manual_approval, validate_package
from adom.runtime.checkpoints import resolve_single_best_checkpoint


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_ROOT = REPO_ROOT / "configs"
EFFECTIVE_BATCH = 16
PROFILES = {
    "640x384": (640, 384),
    "384x384": (384, 384),
}


class CycleState:
    def __init__(self, path: Path, resume: bool) -> None:
        self.path = path
        if resume and path.is_file():
            self.value = json.loads(path.read_text(encoding="utf-8"))
        else:
            self.value = {
                "format_version": 2,
                "status": "running",
                "started_at": _now(),
                "phases": {},
            }
            self.save()

    def save(self) -> None:
        write_json(self.path, self.value)

    def completed(self, name: str, artifacts: list[Path]) -> bool:
        phase = self.value["phases"].get(name, {})
        expected_hashes = phase.get("artifact_sha256")
        if phase.get("status") != "completed" or not isinstance(
            expected_hashes, dict
        ):
            return False
        for path in artifacts:
            resolved = str(path.resolve())
            if (
                not path.is_file()
                or expected_hashes.get(resolved) != sha256_file(path)
            ):
                return False
        return True

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
        phase["artifact_sha256"] = {
            str(path.resolve()): sha256_file(path) for path in artifacts
        }
        self.save()

    def fail(self, name: str, error: BaseException) -> None:
        phase = self.value["phases"].setdefault(name, {})
        phase["status"] = "failed"
        phase["finished_at"] = _now()
        phase["error"] = f"{type(error).__name__}: {error}"
        self.value["status"] = "failed"
        self.save()

    def bind_run_context(self, context: dict[str, Any], resume: bool) -> None:
        existing = self.value.get("run_context")
        if resume and existing is None and self.value.get("phases"):
            raise RuntimeError(
                "Existing run has no reproducibility fingerprint; start a new "
                "output directory instead of resuming it"
            )
        if existing is not None and existing != context:
            keys = sorted(set(existing) | set(context))
            changed = [
                key for key in keys if existing.get(key) != context.get(key)
            ]
            raise RuntimeError(
                "Run context changed; refusing unsafe resume. "
                f"Different fields: {changed}"
            )
        self.value["run_context"] = context
        self.save()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _tool_path(package: str, relative: str) -> Path:
    module = importlib.import_module(package)
    path = Path(module.__file__).resolve().parent / relative
    if not path.is_file():
        raise RuntimeError(f"{package} tool not found: {path}")
    return path


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _source_tree_sha256() -> str:
    digest = hashlib.sha256()
    roots = [
        REPO_ROOT / "src" / "adom",
        REPO_ROOT / "configs" / "adom",
        REPO_ROOT / "configs" / "deployment",
    ]
    files = [
        path
        for root in roots
        for path in root.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    ]
    files.extend(
        [
            REPO_ROOT / "scripts" / "run_training_cycle.sh",
            REPO_ROOT / "requirements" / "openmmlab.txt",
        ]
    )
    for path in sorted(files):
        relative = path.relative_to(REPO_ROOT).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


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


def _select_parity_images(dataset_root: Path, limit: int) -> list[Path]:
    if limit < 1:
        raise RuntimeError("--parity-samples must be at least 1")
    manifest = dataset_root / "metadata" / "manifest_test.csv"
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"No test samples in {manifest}")

    ranked_by_class: dict[int, list[tuple[int, str, Path]]] = {
        class_id: [] for class_id in (0, 1, 2, 3, 255)
    }
    first_by_sequence: dict[str, tuple[str, Path]] = {}
    ordered: list[tuple[str, Path]] = []
    for row in rows:
        sample_id = row["sample_id"]
        image_path = dataset_root / row["image_relpath"]
        mask_path = dataset_root / row["mask_relpath"]
        if not image_path.is_file() or not mask_path.is_file():
            raise RuntimeError(f"Parity sample pair is missing for {sample_id}")
        with Image.open(mask_path) as mask_image:
            mask = np.asarray(mask_image)
        ids, counts = np.unique(mask, return_counts=True)
        count_by_id = {
            int(class_id): int(count) for class_id, count in zip(ids, counts)
        }
        for class_id in ranked_by_class:
            ranked_by_class[class_id].append(
                (count_by_id.get(class_id, 0), sample_id, image_path)
            )
        sequence = row.get("sequence", "")
        first_by_sequence.setdefault(sequence, (sample_id, image_path))
        ordered.append((sample_id, image_path))

    selected: dict[str, Path] = {}
    for class_id in (0, 1, 2, 3, 255):
        ranked = sorted(
            ranked_by_class[class_id],
            key=lambda item: (-item[0], item[1]),
        )
        if ranked and ranked[0][0] > 0:
            selected.setdefault(ranked[0][1], ranked[0][2])
            if len(selected) >= limit:
                return list(selected.values())
    for sample_id, path in first_by_sequence.values():
        selected.setdefault(sample_id, path)
        if len(selected) >= limit:
            return list(selected.values())
    if limit > 1:
        for index in np.linspace(0, len(ordered) - 1, num=limit, dtype=int):
            sample_id, path = ordered[int(index)]
            selected.setdefault(sample_id, path)
            if len(selected) >= limit:
                break
    return list(selected.values())


def _probe_batch(
    *,
    model: str,
    stage: str,
    config: Path,
    output_root: Path,
    env: dict[str, str],
    train_tool: Path,
    resume: bool,
) -> tuple[int, int]:
    plan_path = output_root / model / "batch_plan.json"
    if resume and plan_path.is_file():
        value = json.loads(plan_path.read_text(encoding="utf-8"))
        stage_plan = value.get("stages", {}).get(stage)
        if stage_plan:
            return (
                int(stage_plan["micro_batch"]),
                int(stage_plan["accumulative_counts"]),
            )

    candidates = [16, 8, 4, 2, 1] if model == "b0" else [8, 4, 2, 1]
    for micro_batch in candidates:
        probe_dir = (
            output_root
            / model
            / "probes"
            / stage
            / f"micro_batch_{micro_batch}"
        )
        probe_dir.mkdir(parents=True, exist_ok=True)
        log_path = probe_dir / "probe.log"
        probe_env = dict(env)
        probe_env["ADOM_MICRO_BATCH"] = str(micro_batch)
        probe_env["ADOM_ACCUMULATIVE_COUNTS"] = "1"
        command = [
            sys.executable,
            str(train_tool),
            str(config),
            "--work-dir",
            str(probe_dir),
            "--cfg-options",
            "train_cfg.max_iters=2",
            "val_cfg=None",
            "val_dataloader=None",
            "val_evaluator=None",
            "default_hooks.checkpoint.interval=3",
        ]
        print(f"[probe] {model} {stage}: trying micro-batch {micro_batch}")
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
            plan: dict[str, Any] = {
                "format_version": 2,
                "model": model,
                "stages": {},
            }
            if plan_path.is_file():
                existing = json.loads(plan_path.read_text(encoding="utf-8"))
                if (
                    existing.get("format_version") == 2
                    and existing.get("model") == model
                    and isinstance(existing.get("stages"), dict)
                ):
                    plan = existing
            plan["stages"][stage] = {
                "config": config.relative_to(REPO_ROOT).as_posix(),
                "micro_batch": micro_batch,
                "accumulative_counts": accumulative,
                "effective_batch": micro_batch * accumulative,
                "probe_log": str(log_path.resolve()),
            }
            write_json(
                plan_path,
                plan,
            )
            return micro_batch, accumulative
        log_text = log_path.read_text(encoding="utf-8", errors="replace").lower()
        if "out of memory" not in log_text:
            raise RuntimeError(
                f"{model} batch probe failed for a non-OOM reason; see {log_path}"
            )
    raise RuntimeError(f"No viable micro-batch found for {model} {stage}")


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
        "parity_inputs.json",
        "**/batch_plan.json",
        "**/backbone_*_check.json",
        "**/test_metrics.json",
        "**/best_mIoU_iter_*.pth",
        "**/end2end.onnx",
        "**/deploy.json",
        "**/detail.json",
        "**/pipeline.json",
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
        # Runtime and dataset health are intentionally rechecked on every
        # invocation, including resume.
        resume=False,
    )

    report = validate_package(dataset_root, verify_checksums=True)
    report.require_success()
    approval_errors = validate_manual_approval(dataset_root)
    if approval_errors:
        raise RuntimeError(
            "Dataset has no valid manual preview approval: "
            + "; ".join(approval_errors)
        )
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
    doctor = json.loads(doctor_path.read_text(encoding="utf-8"))
    gpu = doctor.get("gpu", {})
    state.bind_run_context(
        {
            "git_sha": _git_sha(),
            "source_tree_sha256": _source_tree_sha256(),
            "dataset_checksum_manifest_sha256": sha256_file(
                dataset_root / "SHA256SUMS.txt"
            ),
            "mapping_sha256": sha256_file(committed_mapping),
            "models": models,
            "device": args.device,
            "parity_samples": args.parity_samples,
            "gpu": {
                "name": gpu.get("name"),
                "memory_bytes": gpu.get("memory_bytes"),
                "torch": gpu.get("torch"),
            },
            "versions": doctor.get("versions", {}),
        },
        resume=args.resume,
    )
    state.save()

    train_tool = _tool_path("mmseg", ".mim/tools/train.py")
    test_tool = _tool_path("mmseg", ".mim/tools/test.py")
    deploy_tool = _tool_path("mmdeploy", ".mim/tools/deploy.py")
    image = _first_test_image(dataset_root)
    parity_images = _select_parity_images(dataset_root, args.parity_samples)
    write_json(
        output_root / "parity_inputs.json",
        {
            "selection": "highest pixel count per Cost4/ignore class, then "
            "sequence coverage and deterministic test-set spacing",
            "images": [
                path.relative_to(dataset_root).as_posix()
                for path in parity_images
            ],
        },
    )
    mapping = dataset_root / "config" / "label_mapping.yaml"

    for model in models:
        stage1_config = CONFIG_ROOT / "adom" / f"segformer_{model}_stage1_rellis3d.py"
        stage2_config = CONFIG_ROOT / "adom" / f"segformer_{model}_stage2_rellis3d.py"
        if args.skip_batch_probe:
            stage1_micro_batch = 4 if model == "b0" else 2
            stage2_micro_batch = 2 if model == "b0" else 1
            stage1_accumulative = math.ceil(EFFECTIVE_BATCH / stage1_micro_batch)
            stage2_accumulative = math.ceil(EFFECTIVE_BATCH / stage2_micro_batch)
        else:
            stage1_micro_batch, stage1_accumulative = _probe_batch(
                model=model,
                stage="stage1",
                config=stage1_config,
                output_root=output_root,
                env=env,
                train_tool=train_tool,
                resume=args.resume,
            )
            stage2_micro_batch, stage2_accumulative = _probe_batch(
                model=model,
                stage="stage2",
                config=stage2_config,
                output_root=output_root,
                env=env,
                train_tool=train_tool,
                resume=args.resume,
            )
        stage1_env = dict(env)
        stage1_env["ADOM_MICRO_BATCH"] = str(stage1_micro_batch)
        stage1_env["ADOM_ACCUMULATIVE_COUNTS"] = str(stage1_accumulative)
        stage2_env = dict(env)
        stage2_env["ADOM_MICRO_BATCH"] = str(stage2_micro_batch)
        stage2_env["ADOM_ACCUMULATIVE_COUNTS"] = str(stage2_accumulative)

        stage1_dir = output_root / model / "stage1"
        stage1_audit = stage1_dir / "backbone_freeze_check.json"
        stage1_command = [
            sys.executable,
            str(train_tool),
            str(stage1_config),
            "--work-dir",
            str(stage1_dir),
        ]
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
            "--cfg-options",
            f"load_from={stage1_best}",
        ]
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
        _run_phase(
            state,
            name=f"{model}_test",
            command=test_command,
            artifacts=[test_metrics],
            env=stage2_env,
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
            detail_info = export_dir / "detail.json"
            pipeline_info = export_dir / "pipeline.json"
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
            _run_phase(
                state,
                name=f"{model}_onnx_{profile}",
                command=export_command,
                artifacts=[onnx_path, deploy_info, detail_info, pipeline_info],
                env=stage2_env,
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
                "--device",
                args.device,
                "--output",
                str(parity_path),
            ]
            for parity_image in parity_images:
                parity_command.extend(["--image", str(parity_image)])
            _run_phase(
                state,
                name=f"{model}_parity_{profile}",
                command=parity_command,
                artifacts=[parity_path],
                env=stage2_env,
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
                "--detail-info",
                str(detail_info),
                "--pipeline-info",
                str(pipeline_info),
                "--test-metrics",
                str(test_metrics),
                "--mapping",
                str(mapping),
                "--model-variant",
                model,
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
                env=stage2_env,
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
    parser.add_argument("--parity-samples", type=int, default=16)
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
