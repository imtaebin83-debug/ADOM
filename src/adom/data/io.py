from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from .models import DatasetError


def load_structured_text(path: Path) -> dict[str, Any]:
    """Load JSON-compatible YAML, falling back to PyYAML for legacy files."""
    text = path.read_text(encoding="utf-8")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise DatasetError(
                f"{path} is not JSON-compatible YAML and PyYAML is unavailable"
            ) from exc
        value = yaml.safe_load(text)
    if not isinstance(value, dict):
        raise DatasetError(f"Expected a mapping at the root of {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_checksum_manifest(root: Path) -> Path:
    output = root / "SHA256SUMS.txt"
    entries: list[str] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path == output:
            continue
        relative = path.relative_to(root).as_posix()
        entries.append(f"{sha256_file(path)}  {relative}")
    output.write_text("\n".join(entries) + "\n", encoding="utf-8")
    return output


def verify_checksum_manifest(root: Path) -> list[str]:
    manifest = root / "SHA256SUMS.txt"
    if not manifest.is_file():
        return ["SHA256SUMS.txt is missing"]
    errors: list[str] = []
    seen: set[str] = set()
    for number, line in enumerate(
        manifest.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            expected, relative = line.split("  ", 1)
        except ValueError:
            errors.append(f"malformed checksum line {number}")
            continue
        if relative in seen:
            errors.append(f"duplicate checksum entry: {relative}")
            continue
        seen.add(relative)
        if Path(relative).is_absolute() or ".." in PurePosixPath(relative).parts:
            errors.append(f"unsafe checksum path: {relative}")
            continue
        target = root / Path(relative)
        if not target.is_file():
            errors.append(f"checksum target missing: {relative}")
        elif sha256_file(target) != expected:
            errors.append(f"checksum mismatch: {relative}")
    if not seen:
        errors.append("checksum manifest has no entries")
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != manifest
    }
    for relative in sorted(actual - seen):
        errors.append(f"file missing from checksum manifest: {relative}")
    return errors


def is_portable_relative(path_text: str) -> bool:
    if not path_text or Path(path_text).is_absolute():
        return False
    if len(path_text) >= 3 and path_text[1:3] in {":\\", ":/"}:
        return False
    return ".." not in PurePosixPath(path_text.replace("\\", "/")).parts


def atomic_replace_directory(staging: Path, destination: Path, overwrite: bool) -> None:
    backup = destination.with_name(destination.name + ".previous")
    if destination.exists() and not overwrite:
        raise DatasetError(
            f"Output already exists: {destination}. Pass --overwrite explicitly."
        )
    if backup.exists():
        raise DatasetError(f"Refusing to replace existing backup: {backup}")
    if destination.exists():
        destination.rename(backup)
    try:
        staging.rename(destination)
    except Exception:
        if backup.exists() and not destination.exists():
            backup.rename(destination)
        raise
    if backup.exists():
        import shutil

        shutil.rmtree(backup)


def ensure_no_symlink_escape(root: Path, path: Path) -> None:
    resolved_root = root.resolve()
    resolved = path.resolve()
    if os.path.commonpath([resolved_root, resolved]) != str(resolved_root):
        raise DatasetError(f"Path escapes dataset root: {path}")
