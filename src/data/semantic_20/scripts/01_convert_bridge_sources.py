from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image


ALLOWED_TARGET_IDS = set(range(19)) | {255}
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_BRIDGE_MAPPING = SCRIPT_DIR.parent / "config" / "bridge_mapping.yaml"
DEFAULT_RUGD_SOURCE_MAPPING = DATA_ROOT / "rugd" / "config" / "label_mapping.json"
DEFAULT_RUGD_SPLITS = DATA_ROOT / "rugd" / "splits"


class SampleConversionError(RuntimeError):
    def __init__(
        self,
        stage: str,
        sample_id: str,
        input_path: Path,
        cause: BaseException,
    ) -> None:
        self.stage = stage
        self.sample_id = sample_id
        self.input_path = str(input_path)
        super().__init__(
            f"{stage} failed for {sample_id} at {input_path}: "
            f"{type(cause).__name__}: {cause}"
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert RUGD and YCOR source masks into the "
            "RELLIS-3D Semantic20 target ID space."
        )
    )

    parser.add_argument(
        "--mapping",
        type=Path,
        default=DEFAULT_BRIDGE_MAPPING,
    )
    parser.add_argument(
        "--rugd-image-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--rugd-mask-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--rugd-split-root",
        type=Path,
        default=DEFAULT_RUGD_SPLITS,
    )
    parser.add_argument(
        "--rugd-source-map",
        type=Path,
        default=DEFAULT_RUGD_SOURCE_MAPPING,
        help="Official RUGD RGB palette mapping shipped with ADOM.",
    )
    parser.add_argument(
        "--rugd-mask-mode",
        choices=("auto", "rgb", "indexed"),
        default="auto",
        help="RUGD annotation encoding; auto selects RGB for RGB/RGBA/P images.",
    )
    parser.add_argument(
        "--ycor-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--ycor-source-map",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--min-non-ignore-ratio",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a validated <output-root>.partial staging directory.",
    )

    return parser.parse_args(argv)


def require_path(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"{description} not found: {path}"
        )


def load_bridge_config(path: Path) -> dict[str, Any]:
    require_path(path, "Bridge mapping")

    with path.open("r", encoding="utf-8-sig") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            f"Invalid YAML root: {path}"
        )

    if int(config.get("num_classes", -1)) != 19:
        raise ValueError(
            "num_classes must be 19."
        )

    if int(config.get("ignore_index", -1)) != 255:
        raise ValueError(
            "ignore_index must be 255."
        )

    return config


def load_source_to_target(
    config: dict[str, Any],
    source_name: str,
) -> dict[int, int]:
    source_config = config.get(source_name)

    if not isinstance(source_config, dict):
        raise KeyError(
            f"Missing source section: {source_name}"
        )

    entries = source_config.get("source_to_target")

    if not isinstance(entries, dict):
        raise KeyError(
            f"Missing source_to_target: {source_name}"
        )

    mapping: dict[int, int] = {}

    for source_key, entry in entries.items():
        if not isinstance(entry, dict):
            raise ValueError(
                f"Invalid mapping entry: {source_name}/{source_key}"
            )

        source_id = int(source_key)
        target_id = int(entry["target_id"])
        use = bool(entry.get("use", target_id != 255))

        if target_id not in ALLOWED_TARGET_IDS:
            raise ValueError(
                f"Invalid target ID {target_id} "
                f"for {source_name} source ID {source_id}"
            )

        if not use and target_id != 255:
            raise ValueError(
                f"use=false must map to 255: "
                f"{source_name} source ID {source_id}"
            )

        if source_id in mapping:
            raise ValueError(
                f"Duplicate source ID: "
                f"{source_name}/{source_id}"
            )

        mapping[source_id] = target_id

    return mapping


def prepare_output_root(
    output_root: Path,
    overwrite: bool,
    resume: bool,
) -> None:
    if output_root.exists() and any(output_root.iterdir()):
        if not resume and not overwrite:
            raise FileExistsError(
                f"Output root is not empty: {output_root}\n"
                "Use --overwrite to replace it."
            )
        if overwrite:
            shutil.rmtree(output_root)

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    for relative_path in (
        "images/rugd",
        "images/ycor",
        "masks/rugd",
        "masks/ycor",
        "splits",
        "results",
    ):
        (output_root / relative_path).mkdir(
            parents=True,
            exist_ok=True,
        )


def read_split_file(path: Path) -> list[str]:
    require_path(path, "Split file")

    entries = [
        line.strip()
        for line in path.read_text(
            encoding="utf-8-sig"
        ).splitlines()
        if line.strip()
    ]

    if len(entries) != len(set(entries)):
        raise ValueError(
            f"Duplicate entries in split: {path}"
        )

    return entries


def collect_sample_map(root: Path, suffixes: set[str], description: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        sample_id = path.stem
        if sample_id in result:
            raise ValueError(
                f"Duplicate {description} sample ID {sample_id}: "
                f"{result[sample_id]}, {path}"
            )
        result[sample_id] = path
    if not result:
        raise ValueError(f"No {description} files found under {root}")
    return result


def validate_rugd_inputs(
    *,
    image_root: Path,
    mask_root: Path,
    split_root: Path,
    mask_mode: str,
    rgb_to_target: dict[int, int],
    indexed_mapping: dict[int, int],
) -> tuple[dict[str, Path], dict[str, Path], dict[str, list[str]]]:
    """Perform a deterministic, full RUGD pair/palette/split audit."""
    splits = {
        name: read_split_file(split_root / f"{name}.txt")
        for name in ("train", "val", "test")
    }
    owners: dict[str, str] = {}
    for split, sample_ids in splits.items():
        for sample_id in sample_ids:
            if sample_id in owners:
                raise ValueError(
                    f"RUGD sample occurs in more than one split: "
                    f"{sample_id} ({owners[sample_id]}, {split})"
                )
            owners[sample_id] = split

    images = collect_sample_map(image_root, {".png", ".jpg", ".jpeg"}, "RUGD image")
    masks = collect_sample_map(mask_root, {".png"}, "RUGD mask")
    expected = set(owners)
    problems = {
        "missing_images": sorted(expected - set(images)),
        "missing_masks": sorted(expected - set(masks)),
        "unexpected_images": sorted(set(images) - expected),
        "unexpected_masks": sorted(set(masks) - expected),
    }
    if any(problems.values()):
        summary = "; ".join(
            f"{name}={len(values)} examples={values[:10]}"
            for name, values in problems.items()
            if values
        )
        raise ValueError(f"RUGD split/file inventory mismatch: {summary}")

    for sample_id in sorted(expected):
        with Image.open(images[sample_id]) as image:
            image.load()
            image_size = image.size
        target = load_rugd_target_mask(
            masks[sample_id],
            mask_mode=mask_mode,
            rgb_to_target=rgb_to_target,
            indexed_mapping=indexed_mapping,
        )
        mask_size = (target.shape[1], target.shape[0])
        if image_size != mask_size:
            raise ValueError(
                f"RUGD image/mask size mismatch for {sample_id}: "
                f"image={image_size}, mask={mask_size}"
            )
    return images, masks, splits


def load_index_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        image.load()
        array = np.asarray(image)

    if array.ndim == 2:
        return array.astype(
            np.int32,
            copy=False,
        )

    if array.ndim == 3 and array.shape[2] >= 3:
        rgb = array[:, :, :3]

        if (
            np.array_equal(rgb[:, :, 0], rgb[:, :, 1])
            and np.array_equal(rgb[:, :, 1], rgb[:, :, 2])
        ):
            return rgb[:, :, 0].astype(
                np.int32,
                copy=False,
            )

    raise ValueError(
        f"Expected indexed mask, got shape "
        f"{array.shape}: {path}"
    )


def _packed_rgb(rgb: np.ndarray) -> np.ndarray:
    return (
        (rgb[:, :, 0].astype(np.uint32) << 16)
        | (rgb[:, :, 1].astype(np.uint32) << 8)
        | rgb[:, :, 2].astype(np.uint32)
    )


def load_rugd_rgb_to_target(
    mapping_path: Path,
    bridge_config: dict[str, Any],
) -> dict[int, int]:
    """Compose the official RUGD RGB palette with the Semantic20 bridge."""
    require_path(mapping_path, "RUGD source mapping")
    with mapping_path.open("r", encoding="utf-8-sig") as file:
        config = json.load(file)
    entries = config.get("source_palette_rgb")
    if not isinstance(entries, list) or not entries:
        raise KeyError("source_palette_rgb not found in RUGD source mapping")

    bridge_entries = bridge_config.get("rugd", {}).get("source_to_target", {})
    target_by_name = {
        str(entry["source_class"]): int(entry["target_id"])
        for entry in bridge_entries.values()
    }
    palette_names: set[str] = set()
    rgb_to_target: dict[int, int] = {}
    for entry in entries:
        class_name = str(entry["source_class"])
        palette_names.add(class_name)
        rgb = tuple(int(value) for value in entry["rgb"])
        if len(rgb) != 3 or any(value < 0 or value > 255 for value in rgb):
            raise ValueError(f"Invalid RUGD RGB value: {rgb}")
        key = (rgb[0] << 16) | (rgb[1] << 8) | rgb[2]
        if key in rgb_to_target:
            raise ValueError(f"Duplicate RUGD RGB value: {rgb}")
        # A known official class absent from the conservative bridge is explicit ignore.
        rgb_to_target[key] = target_by_name.get(class_name, 255)

    missing_classes = set(target_by_name) - palette_names
    if missing_classes:
        raise ValueError(
            "RUGD bridge classes are absent from the official palette: "
            f"{sorted(missing_classes)}"
        )
    return rgb_to_target


def load_rugd_target_mask(
    path: Path,
    *,
    mask_mode: str,
    rgb_to_target: dict[int, int],
    indexed_mapping: dict[int, int],
) -> np.ndarray:
    with Image.open(path) as image:
        image.load()
        selected_mode = mask_mode
        if selected_mode == "auto":
            selected_mode = "rgb" if image.mode in {"RGB", "RGBA", "P"} else "indexed"
        if selected_mode == "indexed":
            source = np.asarray(image)
            if source.ndim == 3 and source.shape[2] >= 3:
                rgb = source[:, :, :3]
                if not (
                    np.array_equal(rgb[:, :, 0], rgb[:, :, 1])
                    and np.array_equal(rgb[:, :, 1], rgb[:, :, 2])
                ):
                    raise ValueError(f"Expected indexed RUGD mask, got {source.shape}: {path}")
                source = rgb[:, :, 0]
            if source.ndim != 2:
                raise ValueError(f"Expected indexed RUGD mask, got {source.shape}: {path}")
            return remap_mask(
                source.astype(np.int32, copy=False), indexed_mapping, "RUGD", path
            )
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)

    packed = _packed_rgb(rgb)
    unique_keys, inverse = np.unique(packed, return_inverse=True)
    unknown_keys = [int(key) for key in unique_keys if int(key) not in rgb_to_target]
    if unknown_keys:
        unknown_rgb = [
            ((key >> 16) & 255, (key >> 8) & 255, key & 255)
            for key in unknown_keys[:20]
        ]
        raise ValueError(f"Unknown RUGD RGB labels in {path}: {unknown_rgb}")
    target_values = np.asarray(
        [rgb_to_target[int(key)] for key in unique_keys], dtype=np.uint8
    )
    target = target_values[inverse].reshape(packed.shape)
    invalid = {int(value) for value in np.unique(target)} - ALLOWED_TARGET_IDS
    if invalid:
        raise ValueError(f"Invalid target IDs after RUGD RGB remapping: {sorted(invalid)}")
    return target


def load_ycor_rgb_to_source(
    mapping_path: Path,
) -> dict[int, int]:
    require_path(
        mapping_path,
        "YCOR source mapping",
    )

    with mapping_path.open(
        "r",
        encoding="utf-8-sig",
    ) as file:
        config = json.load(file)

    entries = config.get("source_palette_rgb")

    if not isinstance(entries, list):
        raise KeyError(
            "source_palette_rgb not found "
            "in YCOR source mapping."
        )

    rgb_to_source: dict[int, int] = {}

    for entry in entries:
        source_id = int(
            entry["source_index_if_indexed"]
        )
        rgb = tuple(
            int(value)
            for value in entry["rgb"]
        )

        if len(rgb) != 3:
            raise ValueError(
                f"Invalid RGB value: {rgb}"
            )

        key = (
            (rgb[0] << 16)
            | (rgb[1] << 8)
            | rgb[2]
        )

        if key in rgb_to_source:
            raise ValueError(
                f"Duplicate YCOR RGB value: {rgb}"
            )

        rgb_to_source[key] = source_id

    return rgb_to_source


def load_ycor_source_mask(
    path: Path,
    rgb_to_source: dict[int, int],
) -> np.ndarray:
    with Image.open(path) as image:
        image.load()
        mode = image.mode

        if mode in ("RGB", "RGBA", "P"):
            rgb = np.asarray(
                image.convert("RGB"),
                dtype=np.uint8,
            )

            packed = (
                (
                    rgb[:, :, 0].astype(np.uint32)
                    << 16
                )
                | (
                    rgb[:, :, 1].astype(np.uint32)
                    << 8
                )
                | rgb[:, :, 2].astype(np.uint32)
            )

            unique_keys, inverse = np.unique(
                packed,
                return_inverse=True,
            )

            unknown_keys = [
                int(key)
                for key in unique_keys
                if int(key) not in rgb_to_source
            ]

            if unknown_keys:
                unknown_rgb = [
                    (
                        (key >> 16) & 255,
                        (key >> 8) & 255,
                        key & 255,
                    )
                    for key in unknown_keys[:20]
                ]

                raise ValueError(
                    f"Unknown YCOR RGB labels in "
                    f"{path}: {unknown_rgb}"
                )

            source_values = np.asarray(
                [
                    rgb_to_source[int(key)]
                    for key in unique_keys
                ],
                dtype=np.int32,
            )

            return source_values[inverse].reshape(
                packed.shape
            )

        array = np.asarray(image)

    if array.ndim == 2:
        return array.astype(
            np.int32,
            copy=False,
        )

    if array.ndim == 3 and array.shape[2] >= 3:
        rgb = array[:, :, :3]

        if (
            np.array_equal(rgb[:, :, 0], rgb[:, :, 1])
            and np.array_equal(rgb[:, :, 1], rgb[:, :, 2])
        ):
            return rgb[:, :, 0].astype(
                np.int32,
                copy=False,
            )

    raise ValueError(
        f"Unsupported YCOR mask shape: "
        f"{array.shape} {path}"
    )


def remap_mask(
    source_mask: np.ndarray,
    mapping: dict[int, int],
    source_name: str,
    path: Path,
) -> np.ndarray:
    source_ids = {
        int(value)
        for value in np.unique(source_mask)
    }

    unknown_ids = sorted(
        source_ids - set(mapping)
    )

    if unknown_ids:
        raise ValueError(
            f"Unknown {source_name} source IDs "
            f"in {path}: {unknown_ids}"
        )

    target_mask = np.full(
        source_mask.shape,
        255,
        dtype=np.uint8,
    )

    for source_id, target_id in mapping.items():
        target_mask[source_mask == source_id] = target_id

    target_ids = {
        int(value)
        for value in np.unique(target_mask)
    }

    invalid_ids = sorted(
        target_ids - ALLOWED_TARGET_IDS
    )

    if invalid_ids:
        raise ValueError(
            f"Invalid target IDs after remapping "
            f"{path}: {invalid_ids}"
        )

    return target_mask


def link_or_copy(
    source: Path,
    destination: Path,
) -> str:
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def save_mask(
    mask: np.ndarray,
    destination: Path,
) -> None:
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = destination.with_name(destination.name + ".tmp")
    Image.fromarray(
        mask.astype(np.uint8)
    ).save(
        temporary,
        format="PNG",
    )
    os.replace(temporary, destination)


def materialize_pair(
    *,
    source_image: Path,
    target_mask: np.ndarray,
    destination_image: Path,
    destination_mask: Path,
    resume: bool,
) -> str:
    if destination_image.exists() or destination_mask.exists():
        if not resume:
            raise FileExistsError(
                f"Incomplete or unexpected staging pair: {destination_image}, {destination_mask}"
            )
        if not destination_image.is_file() or not destination_mask.is_file():
            for path in (destination_image, destination_mask):
                if path.is_file():
                    path.unlink()
        else:
            with Image.open(destination_mask) as existing_image:
                existing_image.load()
                existing = np.asarray(existing_image)
            if (
                existing.ndim != 2
                or existing.dtype != np.uint8
                or not np.array_equal(existing, target_mask)
            ):
                raise ValueError(f"Staged mask failed resume validation: {destination_mask}")
            return "resumed"
    copy_mode = link_or_copy(source_image, destination_image)
    save_mask(target_mask, destination_mask)
    return copy_mode


def relative_posix(
    path: Path,
    root: Path,
) -> str:
    return path.relative_to(root).as_posix()


def add_pixel_counts(
    mask: np.ndarray,
    pixel_counts: Counter[int],
) -> None:
    values, counts = np.unique(
        mask,
        return_counts=True,
    )

    for value, count in zip(values, counts):
        pixel_counts[int(value)] += int(count)


def run(args: argparse.Namespace) -> None:
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")
    published_root = args.output_root.resolve()
    if published_root.exists():
        raise FileExistsError(
            f"Published output already exists: {published_root}; use a new versioned path"
        )
    if not 0.0 <= args.min_non_ignore_ratio <= 1.0:
        raise ValueError(
            "--min-non-ignore-ratio must be "
            "between 0 and 1."
        )

    require_path(
        args.rugd_image_root,
        "RUGD image root",
    )
    require_path(
        args.rugd_mask_root,
        "RUGD mask root",
    )
    require_path(
        args.rugd_split_root,
        "RUGD split root",
    )
    require_path(
        args.ycor_root,
        "YCOR root",
    )

    config = load_bridge_config(args.mapping)

    rugd_mapping = load_source_to_target(
        config,
        "rugd",
    )
    rugd_rgb_to_target = load_rugd_rgb_to_target(
        args.rugd_source_map,
        config,
    )
    ycor_mapping = load_source_to_target(
        config,
        "ycor",
    )

    ycor_rgb_to_source = load_ycor_rgb_to_source(
        args.ycor_source_map
    )

    rugd_image_map, rugd_mask_map, rugd_splits = validate_rugd_inputs(
        image_root=args.rugd_image_root,
        mask_root=args.rugd_mask_root,
        split_root=args.rugd_split_root,
        mask_mode=args.rugd_mask_mode,
        rgb_to_target=rugd_rgb_to_target,
        indexed_mapping=rugd_mapping,
    )

    output_root = published_root.with_name(published_root.name + ".partial")

    prepare_output_root(
        output_root,
        args.overwrite,
        args.resume,
    )

    manifest_rows: list[dict[str, Any]] = []

    output_splits: dict[str, list[str]] = {
        "train": [],
        "val": [],
        "test": [],
    }

    pixel_counts: Counter[int] = Counter()
    image_copy_counts: Counter[str] = Counter()

    source_summary: dict[str, dict[str, int]] = {
        "rugd": {
            "input": 0,
            "kept": 0,
            "skipped_low_non_ignore": 0,
        },
        "ycor": {
            "input": 0,
            "kept": 0,
            "skipped_low_non_ignore": 0,
        },
    }

    seen_rugd_samples: set[str] = set()

    for split_name in ("train", "val", "test"):
        sample_ids = rugd_splits[split_name]

        for sample_id in sample_ids:
            if sample_id in seen_rugd_samples:
                raise ValueError(
                    f"RUGD sample occurs in more "
                    f"than one split: {sample_id}"
                )

            seen_rugd_samples.add(sample_id)

            source_summary["rugd"]["input"] += 1

            image_path = rugd_image_map[sample_id]
            mask_path = rugd_mask_map[sample_id]

            require_path(
                image_path,
                "RUGD image",
            )
            require_path(
                mask_path,
                "RUGD mask",
            )

            try:
                target_mask = load_rugd_target_mask(
                    mask_path,
                    mask_mode=args.rugd_mask_mode,
                    rgb_to_target=rugd_rgb_to_target,
                    indexed_mapping=rugd_mapping,
                )
                with Image.open(image_path) as image:
                    image.load()
                    image_size = image.size
                if image_size != (target_mask.shape[1], target_mask.shape[0]):
                    raise ValueError(
                        f"image={image_size}, "
                        f"mask={(target_mask.shape[1], target_mask.shape[0])}"
                    )
            except Exception as error:
                raise SampleConversionError(
                    "rugd_conversion", sample_id, mask_path, error
                ) from error

            non_ignore_ratio = float(
                np.count_nonzero(
                    target_mask != 255
                )
                / target_mask.size
            )

            if (
                non_ignore_ratio
                < args.min_non_ignore_ratio
            ):
                source_summary["rugd"][
                    "skipped_low_non_ignore"
                ] += 1
                continue

            destination_image = (
                output_root
                / "images"
                / "rugd"
                / split_name
                / image_path.name
            )

            destination_mask = (
                output_root
                / "masks"
                / "rugd"
                / split_name
                / f"{sample_id}.png"
            )

            copy_mode = materialize_pair(
                source_image=image_path,
                target_mask=target_mask,
                destination_image=destination_image,
                destination_mask=destination_mask,
                resume=args.resume,
            )
            image_copy_counts[copy_mode] += 1

            sample_key = (
                f"rugd/{split_name}/{sample_id}"
            )

            output_splits[split_name].append(
                sample_key
            )

            add_pixel_counts(
                target_mask,
                pixel_counts,
            )

            manifest_rows.append(
                {
                    "sample_key": sample_key,
                    "source": "rugd",
                    "source_split": split_name,
                    "output_split": split_name,
                    "sample_id": sample_id,
                    "image_path": relative_posix(
                        destination_image,
                        output_root,
                    ),
                    "mask_path": relative_posix(
                        destination_mask,
                        output_root,
                    ),
                    "non_ignore_ratio": (
                        f"{non_ignore_ratio:.8f}"
                    ),
                }
            )

            source_summary["rugd"]["kept"] += 1

    ycor_split_mapping = {
        "train": "train",
        "valid": "val",
    }

    for source_split, output_split in (
        ycor_split_mapping.items()
    ):
        split_root = (
            args.ycor_root
            / source_split
        )

        require_path(
            split_root,
            f"YCOR {source_split} split",
        )

        sample_dirs = sorted(
            path
            for path in split_root.iterdir()
            if path.is_dir()
        )

        for sample_dir in sample_dirs:
            source_summary["ycor"]["input"] += 1

            image_path = sample_dir / "rgb.jpg"
            mask_path = sample_dir / "labels.png"

            require_path(
                image_path,
                "YCOR image",
            )
            require_path(
                mask_path,
                "YCOR mask",
            )

            try:
                source_mask = load_ycor_source_mask(
                    mask_path,
                    ycor_rgb_to_source,
                )
                target_mask = remap_mask(
                    source_mask,
                    ycor_mapping,
                    "YCOR",
                    mask_path,
                )
                with Image.open(image_path) as image:
                    image.load()
                    image_size = image.size
                if image_size != (target_mask.shape[1], target_mask.shape[0]):
                    raise ValueError(
                        f"image={image_size}, "
                        f"mask={(target_mask.shape[1], target_mask.shape[0])}"
                    )
            except Exception as error:
                raise SampleConversionError(
                    "ycor_conversion", sample_dir.name, mask_path, error
                ) from error

            non_ignore_ratio = float(
                np.count_nonzero(
                    target_mask != 255
                )
                / target_mask.size
            )

            puddle_pixel_count = int(
                np.count_nonzero(
                    target_mask == 16
                )
            )

            keep_sample = (
                puddle_pixel_count > 0
                or non_ignore_ratio
                >= args.min_non_ignore_ratio
            )

            if not keep_sample:
                source_summary["ycor"][
                    "skipped_low_non_ignore"
                ] += 1
                continue

            sample_id = sample_dir.name

            destination_image = (
                output_root
                / "images"
                / "ycor"
                / output_split
                / f"{sample_id}.jpg"
            )

            destination_mask = (
                output_root
                / "masks"
                / "ycor"
                / output_split
                / f"{sample_id}.png"
            )

            copy_mode = materialize_pair(
                source_image=image_path,
                target_mask=target_mask,
                destination_image=destination_image,
                destination_mask=destination_mask,
                resume=args.resume,
            )
            image_copy_counts[copy_mode] += 1

            sample_key = (
                f"ycor/{output_split}/{sample_id}"
            )

            output_splits[output_split].append(
                sample_key
            )

            add_pixel_counts(
                target_mask,
                pixel_counts,
            )

            manifest_rows.append(
                {
                    "sample_key": sample_key,
                    "source": "ycor",
                    "source_split": source_split,
                    "output_split": output_split,
                    "sample_id": sample_id,
                    "image_path": relative_posix(
                        destination_image,
                        output_root,
                    ),
                    "mask_path": relative_posix(
                        destination_mask,
                        output_root,
                    ),
                    "non_ignore_ratio": (
                        f"{non_ignore_ratio:.8f}"
                    ),
                }
            )

            source_summary["ycor"]["kept"] += 1

    sample_keys = [
        row["sample_key"]
        for row in manifest_rows
    ]

    if len(sample_keys) != len(set(sample_keys)):
        raise ValueError(
            "Duplicate sample_key values "
            "in output manifest."
        )

    manifest_path = (
        output_root
        / "manifest.csv"
    )

    manifest_fields = [
        "sample_key",
        "source",
        "source_split",
        "output_split",
        "sample_id",
        "image_path",
        "mask_path",
        "non_ignore_ratio",
    ]

    with manifest_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=manifest_fields,
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    for split_name, entries in output_splits.items():
        split_path = (
            output_root
            / "splits"
            / f"{split_name}.txt"
        )

        split_path.write_text(
            (
                "\n".join(entries) + "\n"
                if entries
                else ""
            ),
            encoding="utf-8",
        )

    target_classes = {
        int(key): str(value)
        for key, value in config[
            "target_classes"
        ].items()
    }

    total_pixels = sum(
        pixel_counts.values()
    )
    trainable_pixels = (
        total_pixels
        - pixel_counts.get(255, 0)
    )

    statistics_path = (
        output_root
        / "results"
        / "target_class_statistics.csv"
    )

    statistics_fields = [
        "target_id",
        "target_class",
        "pixel_count",
        "pixel_percent_all",
        "pixel_percent_trainable",
    ]

    with statistics_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=statistics_fields,
        )
        writer.writeheader()

        for target_id in sorted(
            pixel_counts,
            key=lambda value: (
                value == 255,
                value,
            ),
        ):
            count = pixel_counts[target_id]

            writer.writerow(
                {
                    "target_id": target_id,
                    "target_class": (
                        target_classes.get(
                            target_id,
                            "unknown",
                        )
                    ),
                    "pixel_count": count,
                    "pixel_percent_all": (
                        100.0 * count / total_pixels
                        if total_pixels
                        else 0.0
                    ),
                    "pixel_percent_trainable": (
                        ""
                        if target_id == 255
                        else (
                            100.0
                            * count
                            / trainable_pixels
                            if trainable_pixels
                            else 0.0
                        )
                    ),
                }
            )

    summary = {
        "dataset_name": config["dataset_name"],
        "mapping_version": config["mapping_version"],
        "target_dataset": config["target_dataset"],
        "num_classes": 19,
        "ignore_index": 255,
        "min_non_ignore_ratio": (
            args.min_non_ignore_ratio
        ),
        "source_summary": source_summary,
        "output_split_counts": {
            split: len(entries)
            for split, entries
            in output_splits.items()
        },
        "output_sample_count": len(
            manifest_rows
        ),
        "image_storage_mode_counts": dict(
            image_copy_counts
        ),
        "target_pixel_counts": {
            str(target_id): count
            for target_id, count
            in sorted(pixel_counts.items())
        },
        "total_pixels": total_pixels,
        "trainable_pixels": trainable_pixels,
        "ignore_pixels": pixel_counts.get(
            255,
            0,
        ),
    }

    summary_path = (
        output_root
        / "results"
        / "conversion_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("Bridge conversion completed.")
    (output_root / "results" / "status.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "output_sample_count": len(manifest_rows),
                "mapping_version": config["mapping_version"],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_root / "_SUCCESS").write_text("PASS\n", encoding="utf-8")
    os.replace(output_root, published_root)
    print(f"Output root: {published_root}")
    print(
        f"RUGD: input="
        f"{source_summary['rugd']['input']}, "
        f"kept={source_summary['rugd']['kept']}, "
        f"skipped="
        f"{source_summary['rugd']['skipped_low_non_ignore']}"
    )
    print(
        f"YCOR: input="
        f"{source_summary['ycor']['input']}, "
        f"kept={source_summary['ycor']['kept']}, "
        f"skipped="
        f"{source_summary['ycor']['skipped_low_non_ignore']}"
    )
    print(
        "Output splits: "
        f"{summary['output_split_counts']}"
    )
    print(
        f"Manifest: {published_root / manifest_path.relative_to(output_root)}"
    )
    print(
        f"Summary: {published_root / summary_path.relative_to(output_root)}"
    )
    print(
        f"Statistics: {published_root / statistics_path.relative_to(output_root)}"
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    staging = args.output_root.resolve().with_name(args.output_root.resolve().name + ".partial")
    try:
        run(args)
    except BaseException as error:
        if args.output_root.resolve().exists():
            raise
        status_path = staging / "results" / "status.json"
        status_path.parent.mkdir(parents=True, exist_ok=True)
        status_path.write_text(
            json.dumps(
                {
                    "status": "FAILED",
                    "stage": getattr(error, "stage", "preflight_or_publish"),
                    "sample_id": getattr(error, "sample_id", None),
                    "input_path": getattr(error, "input_path", None),
                    "error": f"{type(error).__name__}: {error}",
                    "staging_root": str(staging),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        raise


if __name__ == "__main__":
    main()
