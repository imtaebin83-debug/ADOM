from __future__ import annotations

import tarfile
from pathlib import Path

from .io import sha256_file
from .models import DatasetError
from .validation import validate_package


def create_deterministic_tar(dataset_root: Path, archive: Path) -> tuple[Path, Path]:
    report = validate_package(dataset_root, verify_checksums=True)
    report.require_success()
    if archive.exists():
        raise DatasetError(f"Archive already exists: {archive}")
    archive.parent.mkdir(parents=True, exist_ok=True)

    def normalize(info: tarfile.TarInfo) -> tarfile.TarInfo:
        info.uid = 0
        info.gid = 0
        info.uname = ""
        info.gname = ""
        info.mtime = 0
        info.mode = 0o755 if info.isdir() else 0o644
        return info

    with tarfile.open(archive, "w", format=tarfile.PAX_FORMAT) as handle:
        for path in sorted(dataset_root.rglob("*")):
            arcname = path.relative_to(dataset_root).as_posix()
            handle.add(path, arcname=arcname, recursive=False, filter=normalize)
    checksum = archive.with_suffix(archive.suffix + ".sha256")
    checksum.write_text(f"{sha256_file(archive)}  {archive.name}\n", encoding="utf-8")
    return archive, checksum


def verify_archive_checksum(archive: Path, checksum: Path) -> None:
    line = checksum.read_text(encoding="utf-8").strip()
    expected, filename = line.split("  ", 1)
    if filename != archive.name:
        raise DatasetError(
            f"Checksum filename {filename!r} does not match {archive.name!r}"
        )
    actual = sha256_file(archive)
    if actual != expected:
        raise DatasetError(f"Archive checksum mismatch: {archive}")
