#!/usr/bin/env python3
"""Copy size-checked raw/mask PNG pairs using root-relative paths."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from PIL import Image, UnidentifiedImageError


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy raw/mask PNG pairs with identical paths relative to their "
            "roots into OUTPUT_ROOT/{raw,masks}."
        )
    )
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--masks", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def configure_utf8_output() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def collect_pngs(root: Path) -> dict[Path, Path]:
    """Index PNG files by full relative path, never by stem alone."""
    indexed: dict[Path, Path] = {}
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if not path.is_file() or path.suffix.lower() != ".png":
            continue
        relative_path = path.relative_to(root)
        if relative_path in indexed:
            raise ValueError(f"Duplicate relative path: {relative_path.as_posix()}")
        indexed[relative_path] = path
    return indexed


def image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def print_paths(title: str, paths: set[Path]) -> None:
    if not paths:
        return
    print(f"\n{title}")
    for path in sorted(paths, key=lambda item: item.as_posix()):
        print(f"  {path.as_posix()}")


def validate_roots(raw_root: Path, mask_root: Path, output_root: Path) -> None:
    for label, root in (("RAW_ROOT", raw_root), ("MASK_ROOT", mask_root)):
        if not root.is_dir():
            raise ValueError(f"{label} is not a directory: {root}")
    if raw_root == mask_root:
        raise ValueError("RAW_ROOT and MASK_ROOT must differ")
    if output_root == raw_root or raw_root in output_root.parents:
        raise ValueError("OUTPUT_ROOT must not be inside RAW_ROOT")
    if output_root == mask_root or mask_root in output_root.parents:
        raise ValueError("OUTPUT_ROOT must not be inside MASK_ROOT")


def ensure_empty_output(output_root: Path) -> None:
    if output_root.exists() and not output_root.is_dir():
        raise ValueError(f"OUTPUT_ROOT is not a directory: {output_root}")
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError(
            "OUTPUT_ROOT must be new or empty to avoid overwrites: "
            f"{output_root}"
        )


def run(
    raw_root: Path,
    mask_root: Path,
    output_root: Path,
    dry_run: bool,
) -> int:
    raw_root = raw_root.expanduser().resolve()
    mask_root = mask_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    validate_roots(raw_root, mask_root, output_root)
    if not dry_run:
        ensure_empty_output(output_root)

    raw_files = collect_pngs(raw_root)
    mask_files = collect_pngs(mask_root)
    raw_paths = set(raw_files)
    mask_paths = set(mask_files)
    matched_paths = raw_paths & mask_paths
    raw_without_mask = raw_paths - mask_paths
    mask_without_raw = mask_paths - raw_paths
    valid_pairs: set[Path] = set()
    size_mismatches: list[tuple[Path, tuple[int, int], tuple[int, int]]] = []
    read_errors: list[tuple[Path, str]] = []

    for relative_path in sorted(matched_paths, key=lambda item: item.as_posix()):
        try:
            raw_size = image_size(raw_files[relative_path])
            mask_size = image_size(mask_files[relative_path])
        except (OSError, UnidentifiedImageError) as exc:
            read_errors.append((relative_path, str(exc)))
            continue
        if raw_size != mask_size:
            size_mismatches.append((relative_path, raw_size, mask_size))
            continue
        valid_pairs.add(relative_path)

    print(f"Total raw PNG: {len(raw_files)}")
    print(f"Total mask PNG: {len(mask_files)}")
    print(f"Matched pairs: {len(matched_paths)}")
    print(f"Raw without mask: {len(raw_without_mask)}")
    print(f"Mask without raw: {len(mask_without_raw)}")
    print(f"Size mismatches: {len(size_mismatches)}")
    print(f"Image read errors: {len(read_errors)}")
    print_paths("Excluded raw without mask:", raw_without_mask)
    print_paths("Excluded masks without raw:", mask_without_raw)

    if size_mismatches:
        print("\nExcluded pairs with different image sizes:")
        for path, raw_size, mask_size in size_mismatches:
            print(f"  {path.as_posix()}: raw={raw_size}, mask={mask_size}")
    if read_errors:
        print("\nExcluded unreadable image pairs:")
        for path, error in read_errors:
            print(f"  {path.as_posix()}: {error}")

    if dry_run:
        final_raw = set(valid_pairs)
        final_masks = set(valid_pairs)
        print("\nDRY RUN: no files were copied or deleted")
    else:
        final_raw_root = output_root / "raw"
        final_mask_root = output_root / "masks"
        for relative_path in sorted(valid_pairs, key=lambda item: item.as_posix()):
            raw_destination = final_raw_root / relative_path
            mask_destination = final_mask_root / relative_path
            raw_destination.parent.mkdir(parents=True, exist_ok=True)
            mask_destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(raw_files[relative_path], raw_destination)
            shutil.copy2(mask_files[relative_path], mask_destination)
        final_raw = set(collect_pngs(final_raw_root))
        final_masks = set(collect_pngs(final_mask_root))

    print(f"Final raw: {len(final_raw)}")
    print(f"Final masks: {len(final_masks)}")
    if final_raw == final_masks:
        print("✅ RAW / MASK PERFECT 1:1 SYNC")
        return 0
    print("❌ SYNC ERROR")
    print_paths("Files only in final raw:", final_raw - final_masks)
    print_paths("Files only in final masks:", final_masks - final_raw)
    return 1


def main() -> int:
    configure_utf8_output()
    args = parse_args()
    try:
        return run(args.raw, args.masks, args.output, args.dry_run)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
