#!/usr/bin/env python3
"""Normalize legacy 260810 filenames into session/frame relative paths."""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

from PIL import Image


PREFIXED_NAME = re.compile(
    r"^\d+_[0-9]+(?:\.[0-9]+)?_"
    r"(?P<session>\d{8}_\d{6}_\+\d{4})_"
    r"(?P<frame>frame_\d+\.png)$",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--masks", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def pngs_by_name(root: Path) -> dict[str, Path]:
    indexed: dict[str, Path] = {}
    for path in sorted(root.rglob("*.png"), key=lambda item: item.as_posix()):
        if path.name in indexed:
            raise ValueError(f"Duplicate filename prevents safe matching: {path.name}")
        indexed[path.name] = path
    return indexed


def normalized_path(name: str) -> Path:
    match = PREFIXED_NAME.fullmatch(name)
    if match is None:
        raise ValueError(f"Unrecognized 260810 filename: {name}")
    return Path(match.group("session")) / match.group("frame").lower()


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    raw_root = args.raw.expanduser().resolve()
    mask_root = args.masks.expanduser().resolve()
    output_root = args.output.expanduser().resolve()

    try:
        if not raw_root.is_dir() or not mask_root.is_dir():
            raise ValueError("Both source roots must be directories")
        if output_root == raw_root or raw_root in output_root.parents:
            raise ValueError("Output must not be inside the raw root")
        if output_root == mask_root or mask_root in output_root.parents:
            raise ValueError("Output must not be inside the mask root")
        if not args.dry_run and output_root.exists() and any(output_root.iterdir()):
            raise ValueError("Output must be new or empty")

        raw = pngs_by_name(raw_root)
        masks = pngs_by_name(mask_root)
        matched = sorted(set(raw) & set(masks))
        raw_only = sorted(set(raw) - set(masks))
        mask_only = sorted(set(masks) - set(raw))
        valid: dict[Path, tuple[Path, Path]] = {}
        mismatches: list[str] = []

        for name in matched:
            destination = normalized_path(name)
            if destination in valid:
                raise ValueError(f"Normalized path collision: {destination.as_posix()}")
            with Image.open(raw[name]) as raw_image, Image.open(masks[name]) as mask_image:
                if raw_image.size != mask_image.size:
                    mismatches.append(name)
                    continue
            valid[destination] = (raw[name], masks[name])

        print(f"Total raw PNG: {len(raw)}")
        print(f"Total mask PNG: {len(masks)}")
        print(f"Matched pairs: {len(matched)}")
        print(f"Raw without mask: {len(raw_only)}")
        print(f"Mask without raw: {len(mask_only)}")
        print(f"Size mismatches: {len(mismatches)}")

        if not args.dry_run:
            for relative_path, (raw_path, mask_path) in valid.items():
                raw_destination = output_root / "raw" / relative_path
                mask_destination = output_root / "masks" / relative_path
                raw_destination.parent.mkdir(parents=True, exist_ok=True)
                mask_destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(raw_path, raw_destination)
                shutil.copy2(mask_path, mask_destination)

        print(f"Final raw: {len(valid)}")
        print(f"Final masks: {len(valid)}")
        print("✅ RAW / MASK PERFECT 1:1 SYNC")
        return 0
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
