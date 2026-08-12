from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path, PurePosixPath


REPO_ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_SUFFIXES = {
    ".bag",
    ".calib",
    ".ckpt",
    ".db3",
    ".engine",
    ".onnx",
    ".plan",
    ".pt",
    ".pth",
    ".safetensors",
    ".trt",
    ".uff",
}
FORBIDDEN_PATH_PREFIXES = tuple(
    PurePosixPath(value)
    for value in (
        "datasets",
        "data/raw",
        "data/interim",
        "data/processed",
        "data/external",
        "data/captures",
        "data/autonomy_bags",
        "models/checkpoints",
        "models/exports",
        "logs",
        "mlruns",
        "outputs",
        "runs",
        "tensorboard",
        "wandb",
    )
)
ALLOWED_DATA_PREFIX = PurePosixPath("data/splits")
LEGACY_PATH_PREFIX = PurePosixPath("study/gahyung/Datasets_Repo")
PERSONAL_PATH_ALLOWLIST = {
    PurePosixPath("docs/dataset-preprocessing-migration-plan.md"),
}
PERSONAL_PATH_PATTERNS = (
    re.compile(rb"[A-Za-z]:[\\/]+Users[\\/]+", re.IGNORECASE),
    re.compile(rb"/Users/[A-Za-z0-9._-]+/"),
    re.compile(rb"/home/[A-Za-z0-9._-]+/"),
)


def tracked_files() -> list[Path]:
    output = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=REPO_ROOT,
    )
    return [
        REPO_ROOT / item.decode("utf-8")
        for item in output.split(b"\0")
        if item
    ]


def main() -> None:
    errors: list[str] = []
    for path in tracked_files():
        relative = PurePosixPath(path.relative_to(REPO_ROOT).as_posix())
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            errors.append(f"forbidden binary artifact: {relative}")
        if (
            any(
                relative == prefix or relative.is_relative_to(prefix)
                for prefix in FORBIDDEN_PATH_PREFIXES
            )
            and not relative.is_relative_to(ALLOWED_DATA_PREFIX)
            and relative.parts[0] != "study"
        ):
            errors.append(f"forbidden artifact directory: {relative}")
        if not path.is_file():
            continue
        if path.stat().st_size > 10 * 1024 * 1024:
            errors.append(f"tracked file exceeds 10 MiB: {relative}")
            continue
        if (
            relative not in PERSONAL_PATH_ALLOWLIST
            and not relative.is_relative_to(LEGACY_PATH_PREFIX)
        ):
            content = path.read_bytes()
            for pattern in PERSONAL_PATH_PATTERNS:
                if pattern.search(content):
                    errors.append(f"personal absolute path found: {relative}")
                    break
    if errors:
        print("\n".join(f"ERROR: {error}" for error in errors), file=sys.stderr)
        raise SystemExit(2)
    print("Git artifact guard: PASS")


if __name__ == "__main__":
    main()
