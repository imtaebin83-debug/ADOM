from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib
import inspect
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tarfile
from typing import Any

from _common import (
    IGNORE_INDEX,
    SEMANTIC20_CLASSES,
    canonical_json_sha256,
    read_json,
    sha256_file,
    write_json,
)


B0_CHECKPOINT_SHA256 = "d76229ff623eb382fd48011decf54c342d88a113bcbe650fb58cc20e42cabe73"
EADOM_CHECKPOINT_SHA256 = "f4cc41fd91e9df8e7aa3f726498e80636b736dfadf0e1baf338fe7c82a83399c"
EADOM_ARCHIVE_SHA256 = "8468bca1840c89b19145e743d877ffbcf6e5b4f50013de3bcb3d76b6ed45f77b"
B0_CANONICAL_PATH = Path(
    "/workspace/adom/runs/semantic20/e0/"
    "20260805T122006Z-5c50bfdf2900-b0-full/b0/stage2/"
    "best_mIoU_iter_6000.pth"
)
EADOM_ARTIFACT_DIR = Path("/workspace/adom/artifacts/eadom-b0-seed42-iter26000")
EADOM_ARCHIVE = Path("/workspace/adom/exports/canonical-compare-20260814T013811Z.tar.gz")


def _command(command: list[str], cwd: Path) -> dict[str, Any]:
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            check=False,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"command": command, "status": "ERROR", "error": str(error)}
    return {
        "command": command,
        "status": "PASS" if result.returncode == 0 else "ERROR",
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def _module_version(name: str) -> dict[str, Any]:
    try:
        module = importlib.import_module(name)
    except Exception as error:
        return {"status": "MISSING", "error": str(error)}
    return {
        "status": "PASS",
        "version": getattr(module, "__version__", "unknown"),
        "path": inspect.getfile(module),
    }


def environment(repo: Path) -> dict[str, Any]:
    git_head = _command(["git", "rev-parse", "HEAD"], repo)
    git_branch = _command(["git", "branch", "--show-current"], repo)
    git_status = _command(["git", "status", "--short"], repo)
    gpu = _command(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ],
        repo,
    )
    modules = {
        name: _module_version(name)
        for name in ("torch", "mmengine", "mmcv", "mmseg")
    }
    torch_info: dict[str, Any] = {}
    if modules["torch"]["status"] == "PASS":
        import torch

        torch_info = {
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "cudnn_version": torch.backends.cudnn.version(),
            "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
        }
    image_git_sha_path = Path(
        "/workspace/adom/artifacts/eadom-b0-seed42-iter26000/image_git_sha.txt"
    )
    return {
        "schema_version": "adom-paper-eval-environment-v1",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository": str(repo),
        "git": {
            "head": git_head,
            "branch": git_branch,
            "status_short": git_status,
        },
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "packages": modules,
        "torch_runtime": torch_info,
        "gpu": gpu,
        "immutable_image_git_sha": (
            image_git_sha_path.read_text(encoding="utf-8").strip()
            if image_git_sha_path.is_file()
            else None
        ),
    }


def _iteration(path: Path, metadata: dict[str, Any] | None = None) -> int | None:
    for pattern in (r"(?:iter|iteration)[_-]?(\d+)", r"iter_(\d+)"):
        match = re.search(pattern, path.name, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    if metadata:
        for key in ("iter", "iteration"):
            if key in metadata:
                try:
                    return int(metadata[key])
                except (TypeError, ValueError):
                    pass
    return None


def _linked_configs(path: Path) -> list[str]:
    candidates: list[Path] = []
    for parent in [path.parent, *list(path.parents)[:3]]:
        for pattern in ("resolved*config*.py", "*config*.py", "*.py"):
            candidates.extend(parent.glob(pattern))
    return sorted({str(value.resolve()) for value in candidates if value.is_file()})


def _candidate(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "filename": path.name,
        "size_bytes": stat.st_size,
        "modification_time_utc": datetime.fromtimestamp(
            stat.st_mtime, tz=timezone.utc
        ).isoformat(),
        "iteration": _iteration(path),
        "sha256": sha256_file(path),
        "linked_configs": _linked_configs(path),
    }


def _find_candidates(paths: list[Path]) -> list[dict[str, Any]]:
    found: set[Path] = set()
    for value in paths:
        if value.is_file() and value.suffix == ".pth":
            found.add(value.resolve())
        elif value.is_dir():
            found.update(path.resolve() for path in value.rglob("*.pth") if path.is_file())
    return [_candidate(path) for path in sorted(found)]


def _resolve_selected(
    *,
    explicit: Path | None,
    candidates: list[dict[str, Any]],
    expected_sha256: str,
    name: str,
) -> tuple[Path | None, list[str]]:
    blockers: list[str] = []
    if explicit is not None:
        if not explicit.is_file():
            return None, [f"{name} checkpoint is missing: {explicit}"]
        actual = sha256_file(explicit)
        if actual != expected_sha256:
            blockers.append(
                f"{name} checkpoint SHA-256 mismatch: {actual} != {expected_sha256}"
            )
        return explicit.resolve(), blockers
    matches = [item for item in candidates if item["sha256"] == expected_sha256]
    if len(matches) != 1:
        blockers.append(
            f"{name} expected SHA-256 matched {len(matches)} candidates; exactly one is required"
        )
        return None, blockers
    return Path(matches[0]["path"]), blockers


def _checkpoint_metadata(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"status": "BLOCKED", "reason": "checkpoint not selected"}
    try:
        import torch
    except Exception as error:
        return {"status": "BLOCKED", "reason": f"PyTorch unavailable: {error}"}
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch 2.1 does not expose weights_only on all builds.
        checkpoint = torch.load(path, map_location="cpu")
    except Exception as error:
        return {"status": "BLOCKED", "reason": f"checkpoint load failed: {error}"}
    if not isinstance(checkpoint, dict):
        return {"status": "BLOCKED", "reason": "checkpoint root is not a mapping"}
    metadata = checkpoint.get("meta", {})
    if not isinstance(metadata, dict):
        metadata = {"raw_type": type(metadata).__name__}
    state_dict = checkpoint.get("state_dict", checkpoint)
    shapes: dict[str, list[int]] = {}
    if isinstance(state_dict, dict):
        for key, value in state_dict.items():
            if "decode_head" in key and (
                key.endswith("conv_seg.weight") or key.endswith("conv_seg.bias")
            ):
                shapes[key] = list(getattr(value, "shape", ()))
    config_value = metadata.get("cfg", metadata.get("config"))
    return {
        "status": "PASS",
        "iteration": _iteration(path, metadata),
        "epoch": metadata.get("epoch"),
        "meta_keys": sorted(str(key) for key in metadata),
        "classes": metadata.get("CLASSES", metadata.get("classes")),
        "palette": metadata.get("PALETTE", metadata.get("palette")),
        "config_present": config_value is not None,
        "config_sha256": (
            canonical_json_sha256(config_value)
            if config_value is not None
            else None
        ),
        "decode_head_shapes": shapes,
    }


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _config_contract(path: Path) -> dict[str, Any]:
    try:
        from mmengine.config import Config
    except Exception as error:
        return {"status": "BLOCKED", "reason": f"MMEngine unavailable: {error}"}
    try:
        config = Config.fromfile(str(path))
    except Exception as error:
        return {"status": "BLOCKED", "reason": f"config load failed: {error}"}
    pipeline = config.get("test_pipeline")
    if pipeline is None:
        pipeline = config.get("test_dataloader", {}).get("dataset", {}).get("pipeline")
    model = _plain(config.get("model", {}))
    contract = {
        "model_type": model.get("type"),
        "backbone": model.get("backbone"),
        "decode_head": model.get("decode_head"),
        "data_preprocessor": model.get("data_preprocessor"),
        "test_cfg": model.get("test_cfg"),
        "test_pipeline": _plain(pipeline),
        "class_order": list(SEMANTIC20_CLASSES),
        "ignore_index": IGNORE_INDEX,
    }
    return {
        "status": "PASS",
        "path": str(path.resolve()),
        "file_sha256": sha256_file(path),
        "contract": contract,
        "contract_sha256": canonical_json_sha256(contract),
    }


def _decode_class_count(metadata: dict[str, Any], config: dict[str, Any]) -> int | None:
    shapes = metadata.get("decode_head_shapes", {})
    counts = {shape[0] for shape in shapes.values() if shape}
    if len(counts) == 1:
        return int(next(iter(counts)))
    try:
        return int(config["contract"]["decode_head"]["num_classes"])
    except (KeyError, TypeError, ValueError):
        return None


def _report(
    environment_data: dict[str, Any],
    checkpoints: dict[str, Any],
    datasets: dict[str, Any],
    blockers: list[str],
) -> str:
    status = "PASS" if not blockers else "BLOCKED"
    git = environment_data["git"]
    rows = [
        "# ADOM paper evaluation audit",
        "",
        f"**Status: {status}**",
        "",
        "## Audit plan",
        "",
        "1. Freeze Git/environment identity and verify both checkpoint bytes.",
        "2. Compare model, normalization, test pipeline, ontology and inference contracts.",
        "3. Freeze ordered RELLIS/Korean manifests and audit pair, hash and sequence leakage.",
        "4. Start inference only when every audit gate passes.",
        "",
        "## Environment",
        "",
        f"- Git HEAD: `{git['head'].get('stdout', 'unavailable')}`",
        f"- Branch: `{git['branch'].get('stdout', 'unavailable')}`",
        f"- Python: `{environment_data['python']['version'].splitlines()[0]}`",
        f"- GPU: `{environment_data['gpu'].get('stdout', 'unavailable')}`",
        "",
        "## Checkpoints",
        "",
    ]
    for name in ("b0_e0", "eadom"):
        item = checkpoints[name]
        rows.extend(
            [
                f"### {name}",
                "",
                f"- Selected path: `{item.get('selected_path')}`",
                f"- Expected SHA-256: `{item['expected_sha256']}`",
                f"- Actual SHA-256: `{item.get('actual_sha256')}`",
                f"- Iteration: `{item.get('metadata', {}).get('iteration')}`",
                f"- Decode classes: `{item.get('decode_head_num_classes')}`",
                "",
            ]
        )
    rows.extend(
        [
            "## Dataset manifests",
            "",
            f"- Status: `{datasets.get('status', 'MISSING')}`",
            f"- Common supported classes: `{', '.join(datasets.get('common_supported_classes', []))}`",
            "",
            "## Blockers",
            "",
        ]
    )
    rows.extend(f"- {value}" for value in blockers)
    if not blockers:
        rows.append("- None. The four inference runs may start.")
    rows.append("")
    return "\n".join(rows)


def audit(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repo.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    environment_data = environment(repo)
    write_json(output_dir / "environment.json", environment_data)
    blockers: list[str] = []
    for package in ("torch", "mmengine", "mmcv", "mmseg"):
        if environment_data["packages"][package]["status"] != "PASS":
            blockers.append(f"required package unavailable: {package}")
    if environment_data["gpu"]["status"] != "PASS":
        blockers.append("nvidia-smi GPU audit failed")

    b0_paths = [args.b0_search_root]
    if args.b0_checkpoint is not None:
        b0_paths.append(args.b0_checkpoint)
    eadom_paths = [args.eadom_artifact_dir]
    if args.eadom_checkpoint is not None:
        eadom_paths.append(args.eadom_checkpoint)
    b0_candidates = _find_candidates(b0_paths)
    eadom_candidates = _find_candidates(eadom_paths)
    b0_selected, b0_blockers = _resolve_selected(
        explicit=args.b0_checkpoint,
        candidates=b0_candidates,
        expected_sha256=args.expected_b0_sha256,
        name="B0-E0",
    )
    eadom_selected, eadom_blockers = _resolve_selected(
        explicit=args.eadom_checkpoint,
        candidates=eadom_candidates,
        expected_sha256=args.expected_eadom_sha256,
        name="E-ADOM",
    )
    blockers.extend(b0_blockers + eadom_blockers)

    b0_metadata = _checkpoint_metadata(b0_selected)
    eadom_metadata = _checkpoint_metadata(eadom_selected)
    b0_config = _config_contract(args.b0_config)
    eadom_config = _config_contract(args.eadom_config)
    for name, value in (
        ("B0-E0 checkpoint metadata", b0_metadata),
        ("E-ADOM checkpoint metadata", eadom_metadata),
        ("B0-E0 config", b0_config),
        ("E-ADOM config", eadom_config),
    ):
        if value["status"] != "PASS":
            blockers.append(f"{name} audit failed: {value.get('reason')}")
    if b0_config.get("contract_sha256") != eadom_config.get("contract_sha256"):
        blockers.append("B0-E0 and E-ADOM model/test contracts differ")
    b0_classes = _decode_class_count(b0_metadata, b0_config)
    eadom_classes = _decode_class_count(eadom_metadata, eadom_config)
    if b0_classes != 19 or eadom_classes != 19:
        blockers.append(
            f"decode head class count is not Semantic20/19: B0-E0={b0_classes}, E-ADOM={eadom_classes}"
        )

    archive: dict[str, Any] = {
        "path": str(args.eadom_archive.resolve()),
        "expected_sha256": args.expected_archive_sha256,
        "exists": args.eadom_archive.is_file(),
    }
    if args.eadom_archive.is_file():
        archive["actual_sha256"] = sha256_file(args.eadom_archive)
        with tarfile.open(args.eadom_archive, "r:gz") as stream:
            archive["members"] = sorted(stream.getnames())
        archive["contains_manifest_evidence"] = any(
            "manifest" in name.lower() for name in archive["members"]
        )
        if archive["actual_sha256"] != args.expected_archive_sha256:
            blockers.append("canonical comparison archive SHA-256 mismatch")
    else:
        blockers.append(f"canonical comparison archive is missing: {args.eadom_archive}")

    dataset_summary_path = output_dir / "dataset_manifest_summary.json"
    if dataset_summary_path.is_file():
        datasets = read_json(dataset_summary_path)
        if datasets.get("status") != "PASS":
            blockers.extend(
                f"dataset audit: {value}" for value in datasets.get("blockers", [])
            )
    else:
        datasets = {"status": "MISSING"}
        blockers.append(f"dataset manifest audit is missing: {dataset_summary_path}")

    checkpoints = {
        "schema_version": "adom-paper-eval-checkpoints-v1",
        "b0_e0": {
            "selected_path": str(b0_selected) if b0_selected else None,
            "expected_sha256": args.expected_b0_sha256,
            "actual_sha256": sha256_file(b0_selected) if b0_selected else None,
            "selection_basis": (
                "legacy canonical/deployment checkpoint: maximum raw validation MMSeg mIoU, iter 6000"
            ),
            "candidates": b0_candidates,
            "metadata": b0_metadata,
            "config": b0_config,
            "decode_head_num_classes": b0_classes,
        },
        "eadom": {
            "selected_path": str(eadom_selected) if eadom_selected else None,
            "expected_sha256": args.expected_eadom_sha256,
            "actual_sha256": sha256_file(eadom_selected) if eadom_selected else None,
            "selection_basis": "exact expected SHA-256 within the frozen E-ADOM artifact directory",
            "candidates": eadom_candidates,
            "metadata": eadom_metadata,
            "config": eadom_config,
            "decode_head_num_classes": eadom_classes,
        },
        "canonical_comparison_archive": archive,
        "shared_contract_sha256": (
            b0_config.get("contract_sha256")
            if b0_config.get("contract_sha256") == eadom_config.get("contract_sha256")
            else None
        ),
        "status": "PASS" if not blockers else "BLOCKED",
    }
    write_json(output_dir / "checkpoint_manifest.json", checkpoints)
    (output_dir / "audit_report.md").write_text(
        _report(environment_data, checkpoints, datasets, blockers), encoding="utf-8"
    )
    return {"status": "PASS" if not blockers else "BLOCKED", "blockers": blockers}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit environment, checkpoints and frozen dataset manifests before paper evaluation"
    )
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--b0-checkpoint", type=Path, default=B0_CANONICAL_PATH)
    parser.add_argument("--b0-search-root", type=Path, default=B0_CANONICAL_PATH.parent)
    parser.add_argument("--eadom-checkpoint", type=Path)
    parser.add_argument("--eadom-artifact-dir", type=Path, default=EADOM_ARTIFACT_DIR)
    parser.add_argument("--eadom-archive", type=Path, default=EADOM_ARCHIVE)
    parser.add_argument(
        "--b0-config",
        type=Path,
        default=Path("configs/adom/phase1_semantic20/segformer_b0_stage2_e0_rellis.py"),
    )
    parser.add_argument(
        "--eadom-config",
        type=Path,
        default=Path("configs/adom/phase1_semantic20/segformer_b0_stage2_eadom.py"),
    )
    parser.add_argument("--expected-b0-sha256", default=B0_CHECKPOINT_SHA256)
    parser.add_argument("--expected-eadom-sha256", default=EADOM_CHECKPOINT_SHA256)
    parser.add_argument("--expected-archive-sha256", default=EADOM_ARCHIVE_SHA256)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    result = audit(parse_args(argv))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if result["status"] != "PASS":
        sys.exit(2)


if __name__ == "__main__":
    main()
