from __future__ import annotations

import tarfile
from datetime import datetime, timezone
from pathlib import Path

from .io import sha256_file, write_checksum_manifest, write_json
from .models import DatasetError
from .validation import validate_manual_approval, validate_package


def approve_dataset(
    dataset_root: Path,
    approver: str,
    notes: str = "",
    replace: bool = False,
) -> Path:
    approver = approver.strip()
    if not approver:
        raise DatasetError("Approver must not be empty")
    report = validate_package(dataset_root, verify_checksums=True)
    report.require_success()
    approval_path = dataset_root / "reports" / "approval.json"
    if approval_path.exists() and not replace:
        raise DatasetError(
            f"Approval already exists: {approval_path}. Pass --replace explicitly."
        )
    previews = sorted(
        path
        for path in (dataset_root / "reports" / "previews").glob("*")
        if path.is_file() and path.name != "legend.png"
    )
    if not previews:
        raise DatasetError("No generated previews are available for manual review")
    mapping_path = dataset_root / "config" / "label_mapping.yaml"
    payload = {
        "format_version": 1,
        "status": "APPROVED",
        "approver": approver,
        "approved_at": datetime.now(timezone.utc).isoformat(),
        "notes": notes,
        "mapping_sha256": sha256_file(mapping_path),
        "split_sha256": {
            split: sha256_file(dataset_root / "splits" / f"{split}.txt")
            for split in ("train", "val", "test")
        },
        "reviewed_previews": [
            {
                "path": path.relative_to(dataset_root).as_posix(),
                "sha256": sha256_file(path),
            }
            for path in previews
        ],
    }
    write_json(approval_path, payload)
    write_checksum_manifest(dataset_root)
    errors = validate_manual_approval(dataset_root)
    if errors:
        raise DatasetError("Approval validation failed: " + "; ".join(errors))
    validate_package(dataset_root, verify_checksums=True).require_success()
    return approval_path


def create_deterministic_tar(dataset_root: Path, archive: Path) -> tuple[Path, Path]:
    report = validate_package(dataset_root, verify_checksums=True)
    report.require_success()
    approval_errors = validate_manual_approval(dataset_root)
    if approval_errors:
        raise DatasetError(
            "Dataset requires a valid manual preview approval before packaging: "
            + "; ".join(approval_errors)
        )
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
