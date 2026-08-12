#!/usr/bin/env python3
"""Build a verified server-upload package from date-level normalized data."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dates", nargs="*", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def collect_pngs(root: Path) -> dict[Path, Path]:
    return {
        path.relative_to(root): path
        for path in sorted(root.rglob("*.png"), key=lambda item: item.as_posix())
        if path.is_file()
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_dates(source_root: Path, requested: list[str] | None) -> list[str]:
    if requested:
        dates = requested
    else:
        dates = sorted(
            path.name
            for path in source_root.iterdir()
            if path.is_dir() and (path / "normalized").is_dir()
        )
    if not dates:
        raise ValueError("No date folders with normalized data were found")
    if len(dates) != len(set(dates)):
        raise ValueError("Duplicate date arguments are not allowed")
    return dates


def validate_date(source_root: Path, date: str) -> tuple[dict[Path, Path], dict[Path, Path]]:
    date_root = (source_root / date).resolve()
    try:
        date_root.relative_to(source_root)
    except ValueError as exc:
        raise ValueError(f"Date escapes source root: {date}") from exc
    raw = collect_pngs(date_root / "normalized" / "raw")
    masks = collect_pngs(date_root / "normalized" / "masks")
    if not raw or set(raw) != set(masks):
        raise ValueError(f"Raw/mask relative paths are not 1:1 for {date}")
    for relative_path in sorted(raw, key=lambda item: item.as_posix()):
        with Image.open(raw[relative_path]) as raw_image, Image.open(
            masks[relative_path]
        ) as mask_image:
            if raw_image.size != mask_image.size:
                raise ValueError(
                    f"Image size mismatch: {date}/{relative_path.as_posix()}"
                )
    return raw, masks


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    source_root = args.source_root.expanduser().resolve()
    output_root = args.output.expanduser().resolve()
    try:
        if not source_root.is_dir():
            raise ValueError(f"Source root is not a directory: {source_root}")
        if output_root == source_root:
            raise ValueError("Output must differ from source root")
        if not args.dry_run and output_root.exists() and any(output_root.iterdir()):
            raise ValueError("Output must be new or empty")

        dates = discover_dates(source_root, args.dates)
        validated: dict[str, tuple[dict[Path, Path], dict[Path, Path]]] = {}
        for date in dates:
            validated[date] = validate_date(source_root, date)
            print(f"{date}: {len(validated[date][0])} pairs")

        if args.dry_run:
            print(f"DRY RUN: would package {sum(len(item[0]) for item in validated.values())} pairs")
            return 0

        manifest: dict[str, object] = {"format_version": 1, "dates": {}}
        for date, (raw, masks) in validated.items():
            date_output = output_root / date
            records: list[dict[str, object]] = []
            for relative_path in sorted(raw, key=lambda item: item.as_posix()):
                raw_destination = date_output / "raw" / relative_path
                mask_destination = date_output / "masks" / relative_path
                raw_destination.parent.mkdir(parents=True, exist_ok=True)
                mask_destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(raw[relative_path], raw_destination)
                shutil.copy2(masks[relative_path], mask_destination)
                records.append(
                    {
                        "relative_path": relative_path.as_posix(),
                        "raw_sha256": sha256(raw_destination),
                        "mask_sha256": sha256(mask_destination),
                    }
                )
            labelmap = source_root / date / "labelmap.txt"
            if labelmap.is_file():
                shutil.copy2(labelmap, date_output / "labelmap.txt")
            manifest["dates"][date] = {"pairs": len(records), "files": records}

        output_root.mkdir(parents=True, exist_ok=True)
        (output_root / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Packaged pairs: {sum(len(item[0]) for item in validated.values())}")
        print("✅ UPLOAD PACKAGE VERIFIED")
        return 0
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
