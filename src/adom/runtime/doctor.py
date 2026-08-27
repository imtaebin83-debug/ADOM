from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import platform
import re
import sys
from pathlib import Path
from typing import Any

from adom.data.io import sha256_file
from adom.data.validation import validate_package


EXPECTED_VERSIONS = {
    "numpy": "1.24.4",
    "setuptools": "69.5.1",
    "opencv-python": "4.8.0.76",
    "opencv-python-headless": "4.8.0.76",
    "albumentations": "1.3.1",
    "mmcv": "2.1.0",
    "mmengine": "0.10.7",
    "mmsegmentation": "1.2.2",
    "mmdeploy": "1.3.1",
    "wandb": "0.22.3",
    "ftfy": "6.1.1",
    "regex": "2023.10.3",
    "prettytable": "3.9.0",
}


# Marketing capacities are decimal GB while torch reports bytes. The ranges
# below identify the requested full-device VRAM class without assuming an
# exact GiB rendering from a particular driver. MIG/vGPU slices fail closed.
GPU_PROFILES: dict[str, dict[str, Any]] = {
    "a100-40gb": {
        "display_name": "NVIDIA A100 40GB",
        "name_pattern": r"\bA100(?:[- ](?:SXM\d*|PCIE))?[- ]40GB\b",
        "memory_gib_range": (37.0, 45.0),
        "proposed_micro_batches": [8, 4, 2, 1],
        "status": "proposal-until-memory-probe",
    },
    "a100-80gb": {
        "display_name": "NVIDIA A100 80GB",
        "name_pattern": r"\bA100(?:[- ](?:SXM\d*|PCIE))?[- ]80GB\b",
        "memory_gib_range": (74.0, 85.0),
        "proposed_micro_batches": [16, 8, 4, 2, 1],
        "status": "proposal-until-memory-probe",
    },
    "rtx-a6000-48gb": {
        "display_name": "NVIDIA RTX A6000 48GB",
        "name_pattern": r"\bRTX A6000\b",
        "memory_gib_range": (44.0, 52.0),
        "proposed_micro_batches": [8, 4, 2, 1],
        "status": "proposal-until-memory-probe",
    },
    "rtx-pro-6000-blackwell-96gb": {
        "display_name": "NVIDIA RTX PRO 6000 Blackwell 96GB",
        "name_pattern": r"\bRTX PRO 6000 BLACKWELL\b",
        "memory_gib_range": (89.0, 101.0),
        "proposed_micro_batches": [16, 8, 4, 2, 1],
        "status": "proposal-until-memory-probe",
    },
}


def validate_gpu_profile(
    profile_name: str,
    gpu_name: str,
    memory_bytes: int,
) -> dict[str, Any]:
    """Validate an exact B5 study GPU family and full-device VRAM class."""
    if profile_name not in GPU_PROFILES:
        raise ValueError(f"Unknown GPU profile: {profile_name}")
    profile = GPU_PROFILES[profile_name]
    actual_gib = float(memory_bytes) / (1024**3)
    lower, upper = profile["memory_gib_range"]
    name_matches = bool(
        re.search(str(profile["name_pattern"]), gpu_name, flags=re.IGNORECASE)
    )
    memory_matches = float(lower) <= actual_gib < float(upper)
    errors: list[str] = []
    if not name_matches:
        errors.append(
            f"GPU profile {profile_name} requires {profile['display_name']!r}; "
            f"runtime reported {gpu_name!r}"
        )
    if not memory_matches:
        errors.append(
            f"GPU profile {profile_name} requires full-device memory in "
            f"[{lower:.1f}, {upper:.1f}) GiB; runtime reported {actual_gib:.2f} GiB"
        )
    return {
        "profile": profile_name,
        "display_name": profile["display_name"],
        "observed_name": gpu_name,
        "observed_memory_gib": actual_gib,
        "required_memory_gib_range": [lower, upper],
        "proposed_micro_batches": list(profile["proposed_micro_batches"]),
        "proposal_status": profile["status"],
        "name_matches": name_matches,
        "memory_matches": memory_matches,
        "errors": errors,
    }


def _distribution_version(name: str) -> str:
    return importlib.metadata.version(name)


def run_doctor(
    dataset_root: Path | None,
    require_gpu: bool,
    require_deployment: bool = True,
    require_gpu_name: str | None = None,
    minimum_gpu_memory_gib: float | None = None,
    expected_image_sha: str | None = None,
    gpu_profile: str | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    versions: dict[str, str] = {}
    for distribution, expected in EXPECTED_VERSIONS.items():
        if not require_deployment and distribution == "mmdeploy":
            continue
        try:
            actual = _distribution_version(distribution)
        except importlib.metadata.PackageNotFoundError:
            errors.append(f"missing package: {distribution}=={expected}")
            continue
        versions[distribution] = actual
        if actual != expected:
            errors.append(
                f"version mismatch: {distribution}={actual}, expected={expected}"
            )

    imports: dict[str, str] = {}
    runtime_modules = [
        "cv2",
        "numpy",
        "mmcv",
        "mmcv.ops",
        "mmengine",
        "mmseg",
        "albumentations",
        "wandb",
        "ftfy",
        "regex",
        "prettytable",
        "torch",
        "adom.mmseg",
    ]
    if require_deployment:
        runtime_modules.extend(("mmdeploy", "onnx", "onnxruntime"))
    for module_name in runtime_modules:
        try:
            importlib.import_module(module_name)
            imports[module_name] = "ok"
        except Exception as error:  # pragma: no cover - exercised in container
            imports[module_name] = f"{type(error).__name__}: {error}"
            errors.append(f"import failed: {module_name}: {error}")

    gpu: dict[str, Any] = {"available": False}
    try:
        import torch

        gpu["torch"] = torch.__version__
        gpu["available"] = bool(torch.cuda.is_available())
        if gpu["available"]:
            gpu["device_count"] = torch.cuda.device_count()
            gpu["name"] = torch.cuda.get_device_name(0)
            gpu["memory_bytes"] = torch.cuda.get_device_properties(0).total_memory
    except Exception as error:  # pragma: no cover - exercised in container
        errors.append(f"torch GPU check failed: {error}")
    if require_gpu and not gpu["available"]:
        errors.append("CUDA GPU is required but torch.cuda.is_available() is false")
    if require_gpu_name and require_gpu_name.casefold() not in str(gpu.get("name", "")).casefold():
        errors.append(
            f"GPU name must contain {require_gpu_name!r}, got {gpu.get('name')!r}"
        )
    if minimum_gpu_memory_gib is not None:
        actual_gib = float(gpu.get("memory_bytes", 0)) / (1024**3)
        if actual_gib < minimum_gpu_memory_gib:
            errors.append(
                f"GPU memory is {actual_gib:.2f} GiB, expected at least "
                f"{minimum_gpu_memory_gib:.2f} GiB"
            )
    gpu_profile_report: dict[str, Any] | None = None
    if gpu_profile is not None:
        gpu_profile_report = validate_gpu_profile(
            gpu_profile,
            str(gpu.get("name", "")),
            int(gpu.get("memory_bytes", 0)),
        )
        errors.extend(gpu_profile_report["errors"])

    image_sha = os.getenv("ADOM_GIT_SHA", "unknown")
    if expected_image_sha and image_sha != expected_image_sha:
        errors.append(
            f"image Git SHA mismatch: ADOM_GIT_SHA={image_sha}, expected={expected_image_sha}"
        )

    dataset: dict[str, Any] | None = None
    if dataset_root is not None:
        report = validate_package(dataset_root, verify_checksums=True)
        dataset = report.to_dict()
        if not report.passed:
            errors.append(
                f"dataset strict validation failed with {len(report.errors)} errors"
            )
        checksum_path = dataset_root / "SHA256SUMS.txt"
        if checksum_path.is_file():
            dataset["checksum_manifest_sha256"] = sha256_file(checksum_path)

    return {
        "status": "PASS" if not errors else "FAIL",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "versions": versions,
        "imports": imports,
        "gpu": gpu,
        "gpu_profile": gpu_profile_report,
        "dataset": dataset,
        "deployment_required": require_deployment,
        "image_git_sha": image_sha,
        "errors": errors,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Validate the ADOM training runtime")
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--skip-deployment", action="store_true")
    parser.add_argument("--require-gpu-name")
    parser.add_argument("--minimum-gpu-memory-gib", type=float)
    parser.add_argument("--gpu-profile", choices=tuple(GPU_PROFILES))
    parser.add_argument("--expected-image-sha")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = run_doctor(
        args.dataset_root,
        args.require_gpu,
        require_deployment=not args.skip_deployment,
        require_gpu_name=args.require_gpu_name,
        minimum_gpu_memory_gib=args.minimum_gpu_memory_gib,
        expected_image_sha=args.expected_image_sha,
        gpu_profile=args.gpu_profile,
    )
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    if report["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
