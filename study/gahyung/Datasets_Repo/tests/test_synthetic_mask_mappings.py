from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType

import numpy as np
import yaml
from PIL import Image


DATASET_REPOSITORY_ROOT = (
    Path(__file__).resolve().parents[1]
)

VALID_ADOM_IDS = {
    0,
    1,
    2,
    3,
    255,
}

ORDERED_ADOM_IDS = (
    0,
    1,
    2,
    3,
    255,
)

RELLIS_SCRIPT_PATH = (
    DATASET_REPOSITORY_ROOT
    / "RELLIS-3D"
    / "rellis3d_cost4_v1"
    / "scripts"
    / "02_convert_masks.py"
)

RELLIS_MAPPING_PATH = (
    DATASET_REPOSITORY_ROOT
    / "RELLIS-3D"
    / "rellis3d_cost4_v1"
    / "config"
    / "class_mapping.yaml"
)

RUGD_SCRIPT_PATH = (
    DATASET_REPOSITORY_ROOT
    / "RUGD"
    / "scripts"
    / "04_remap_rugd.py"
)

RUGD_MAPPING_PATH = (
    DATASET_REPOSITORY_ROOT
    / "RUGD"
    / "config"
    / "label_mapping.json"
)

YCOR_COMMON_PATH = (
    DATASET_REPOSITORY_ROOT
    / "YCOR"
    / "scripts"
    / "common.py"
)

YCOR_MAPPING_PATH = (
    DATASET_REPOSITORY_ROOT
    / "YCOR"
    / "config"
    / "label_mapping.json"
)


def load_python_module(
    module_name: str,
    module_path: Path,
) -> ModuleType:
    if not module_path.is_file():
        raise FileNotFoundError(
            f"Python module does not exist: {module_path}"
        )

    specification = (
        importlib.util.spec_from_file_location(
            module_name,
            module_path,
        )
    )

    if (
        specification is None
        or specification.loader is None
    ):
        raise ImportError(
            f"Could not create module specification: "
            f"{module_path}"
        )

    module = importlib.util.module_from_spec(
        specification
    )

    sys.modules[module_name] = module

    specification.loader.exec_module(
        module
    )

    return module


def read_uint8_mask(
    path: Path,
) -> np.ndarray:
    with Image.open(path) as image:
        image.load()

        if image.mode != "L":
            image = image.convert("L")

        return np.asarray(
            image,
            dtype=np.uint8,
        )


class SyntheticMaskMappingTests(
    unittest.TestCase,
):
    @classmethod
    def setUpClass(
        cls,
    ) -> None:
        cls.rugd_module = load_python_module(
            "adom_rugd_remap",
            RUGD_SCRIPT_PATH,
        )

        cls.ycor_module = load_python_module(
            "adom_ycor_common",
            YCOR_COMMON_PATH,
        )

    def test_rellis_synthetic_mask_conversion(
        self,
    ) -> None:
        with RELLIS_MAPPING_PATH.open(
            "r",
            encoding="utf-8-sig",
        ) as file:
            config = yaml.safe_load(file)

        source_to_target = {
            int(source_id): int(target_id)
            for source_id, target_id
            in config[
                "rellis_to_target"
            ].items()
        }

        source_for_target: dict[
            int,
            int,
        ] = {}

        for source_id, target_id in (
            source_to_target.items()
        ):
            source_for_target.setdefault(
                target_id,
                source_id,
            )

        missing_targets = (
            VALID_ADOM_IDS
            - set(source_for_target)
        )

        self.assertFalse(
            missing_targets,
            msg=(
                "RELLIS mapping does not cover "
                f"ADOM IDs: {sorted(missing_targets)}"
            ),
        )

        source_row = np.asarray(
            [
                source_for_target[target_id]
                for target_id
                in ORDERED_ADOM_IDS
            ],
            dtype=np.int32,
        )

        expected_row = np.asarray(
            ORDERED_ADOM_IDS,
            dtype=np.uint8,
        )

        source_mask = np.vstack(
            (
                source_row,
                source_row[::-1],
            )
        )

        expected_mask = np.vstack(
            (
                expected_row,
                expected_row[::-1],
            )
        )

        with tempfile.TemporaryDirectory(
            prefix="adom_rellis_mask_test_",
        ) as temporary_directory:
            test_root = Path(
                temporary_directory
            )

            raw_root = (
                test_root
                / "raw"
            )

            output_root = (
                test_root
                / "output"
            )

            rgb_directory = (
                raw_root
                / "00000"
                / "pylon_camera_node"
            )

            source_mask_directory = (
                raw_root
                / "00000"
                / "pylon_camera_node_label_id"
            )

            rgb_directory.mkdir(
                parents=True,
                exist_ok=True,
            )

            source_mask_directory.mkdir(
                parents=True,
                exist_ok=True,
            )

            rgb_path = (
                rgb_directory
                / "frame_000001.png"
            )

            source_mask_path = (
                source_mask_directory
                / "frame_000001.png"
            )

            rgb = np.zeros(
                (
                    source_mask.shape[0],
                    source_mask.shape[1],
                    3,
                ),
                dtype=np.uint8,
            )

            rgb[:, :, 0] = 120
            rgb[:, :, 1] = 80
            rgb[:, :, 2] = 40

            Image.fromarray(
                rgb,
                mode="RGB",
            ).save(rgb_path)

            source_dtype = (
                np.uint8
                if int(source_mask.max()) <= 255
                else np.uint16
            )

            Image.fromarray(
                source_mask.astype(
                    source_dtype
                )
            ).save(source_mask_path)

            completed = subprocess.run(
                [
                    sys.executable,
                    str(RELLIS_SCRIPT_PATH),
                    "--input-root",
                    str(raw_root),
                    "--output-root",
                    str(output_root),
                    "--mapping",
                    str(RELLIS_MAPPING_PATH),
                    "--limit",
                    "1",
                    "--overwrite",
                ],
                cwd=DATASET_REPOSITORY_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )

            self.assertEqual(
                completed.returncode,
                0,
                msg=(
                    "RELLIS synthetic conversion failed.\n"
                    f"stdout:\n{completed.stdout}\n"
                    f"stderr:\n{completed.stderr}"
                ),
            )

            converted_paths = sorted(
                (
                    output_root
                    / "masks"
                ).rglob("*.png")
            )

            self.assertEqual(
                len(converted_paths),
                1,
                msg=(
                    "Expected exactly one converted "
                    f"RELLIS mask: {converted_paths}"
                ),
            )

            actual_mask = read_uint8_mask(
                converted_paths[0]
            )

            self.assertEqual(
                actual_mask.dtype,
                np.dtype(np.uint8),
            )

            np.testing.assert_array_equal(
                actual_mask,
                expected_mask,
            )

    def test_rellis_missing_mask_pair_fails(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(
            prefix="adom_rellis_pair_test_",
        ) as temporary_directory:
            test_root = Path(
                temporary_directory
            )

            raw_root = (
                test_root
                / "raw"
            )

            output_root = (
                test_root
                / "output"
            )

            rgb_directory = (
                raw_root
                / "00000"
                / "pylon_camera_node"
            )

            source_mask_directory = (
                raw_root
                / "00000"
                / "pylon_camera_node_label_id"
            )

            rgb_directory.mkdir(
                parents=True,
                exist_ok=True,
            )

            source_mask_directory.mkdir(
                parents=True,
                exist_ok=True,
            )

            rgb = np.zeros(
                (4, 6, 3),
                dtype=np.uint8,
            )

            Image.fromarray(
                rgb,
                mode="RGB",
            ).save(
                rgb_directory
                / "frame_without_mask.png"
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    str(RELLIS_SCRIPT_PATH),
                    "--input-root",
                    str(raw_root),
                    "--output-root",
                    str(output_root),
                    "--mapping",
                    str(RELLIS_MAPPING_PATH),
                    "--overwrite",
                ],
                cwd=DATASET_REPOSITORY_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )

            self.assertNotEqual(
                completed.returncode,
                0,
                msg=(
                    "RELLIS conversion incorrectly passed "
                    "with an RGB image that has no mask."
                ),
            )

    def test_rellis_official_raw_ids_29_30_32(
        self,
    ) -> None:
        with RELLIS_MAPPING_PATH.open(
            "r",
            encoding="utf-8-sig",
        ) as file:
            config = yaml.safe_load(file)

        source_to_target = {
            int(source_id): int(target_id)
            for source_id, target_id
            in config["rellis_to_target"].items()
        }

        self.assertEqual(source_to_target[29], 1)
        self.assertEqual(source_to_target[30], 1)
        self.assertEqual(source_to_target[32], 3)

    def test_rugd_synthetic_rgb_mapping(
        self,
    ) -> None:
        # 04_remap_rugd.load_mapping() returns one dictionary:
        # {(R, G, B): ADOM target ID}.
        rgb_to_adom = (
            self.rugd_module.load_mapping(
                RUGD_MAPPING_PATH
            )
        )

        self.assertIsInstance(
            rgb_to_adom,
            dict,
        )

        self.assertTrue(
            rgb_to_adom,
            msg=(
                "RUGD load_mapping() returned "
                "an empty mapping."
            ),
        )

        with RUGD_MAPPING_PATH.open(
            "r",
            encoding="utf-8-sig",
        ) as file:
            mapping_config = json.load(file)

        rgb_to_name = {
            self.rugd_module.parse_rgb_key(
                key
            ): str(value)
            for key, value
            in mapping_config[
                "RGB_TO_NAME"
            ].items()
            if key != "_comment"
        }

        rugd_to_adom = {
            str(key): int(value)
            for key, value
            in mapping_config[
                "RUGD_TO_ADOM"
            ].items()
            if key != "_comment"
        }

        declared_rgb_to_adom = {
            self.rugd_module.parse_rgb_key(
                key
            ): int(value)
            for key, value
            in mapping_config[
                "RGB_TO_ADOM"
            ].items()
            if key != "_comment"
        }

        self.assertEqual(
            set(rgb_to_name),
            set(declared_rgb_to_adom),
            msg=(
                "RGB_TO_NAME and RGB_TO_ADOM "
                "use different RGB keys."
            ),
        )

        composed_rgb_to_adom = {}

        for rgb, class_name in (
            rgb_to_name.items()
        ):
            self.assertIn(
                class_name,
                rugd_to_adom,
                msg=(
                    "RUGD_TO_ADOM is missing "
                    f"class: {class_name}"
                ),
            )

            composed_rgb_to_adom[rgb] = (
                rugd_to_adom[class_name]
            )

        self.assertEqual(
            composed_rgb_to_adom,
            declared_rgb_to_adom,
            msg=(
                "RGB_TO_ADOM differs from "
                "RGB_TO_NAME + RUGD_TO_ADOM."
            ),
        )

        # Verify that the actual mapping loader returns
        # the same mapping declared in label_mapping.json.
        self.assertEqual(
            rgb_to_adom,
            declared_rgb_to_adom,
            msg=(
                "04_remap_rugd.load_mapping() "
                "returned an unexpected mapping."
            ),
        )

        actual_target_ids = {
            int(target_id)
            for target_id
            in rgb_to_adom.values()
        }

        self.assertTrue(
            actual_target_ids.issubset(
                VALID_ADOM_IDS
            ),
            msg=(
                "RUGD mapping contains invalid "
                f"ADOM IDs: "
                f"{sorted(actual_target_ids)}"
            ),
        )

        source_rgb_for_target: dict[
            int,
            tuple[int, int, int],
        ] = {}

        for rgb, target_id in (
            rgb_to_adom.items()
        ):
            source_rgb_for_target.setdefault(
                int(target_id),
                rgb,
            )

        missing_targets = (
            VALID_ADOM_IDS
            - set(source_rgb_for_target)
        )

        self.assertFalse(
            missing_targets,
            msg=(
                "RUGD mapping does not cover "
                f"ADOM IDs: "
                f"{sorted(missing_targets)}"
            ),
        )

        source_mask = np.asarray(
            [
                [
                    source_rgb_for_target[
                        target_id
                    ]
                    for target_id
                    in ORDERED_ADOM_IDS
                ],
                [
                    source_rgb_for_target[
                        target_id
                    ]
                    for target_id
                    in reversed(
                        ORDERED_ADOM_IDS
                    )
                ],
            ],
            dtype=np.uint8,
        )

        expected_mask = np.asarray(
            [
                ORDERED_ADOM_IDS,
                tuple(
                    reversed(
                        ORDERED_ADOM_IDS
                    )
                ),
            ],
            dtype=np.uint8,
        )

        actual_mask = np.full(
            source_mask.shape[:2],
            fill_value=255,
            dtype=np.uint8,
        )

        for rgb, target_id in (
            rgb_to_adom.items()
        ):
            rgb_array = np.asarray(
                rgb,
                dtype=np.uint8,
            )

            matching_pixels = np.all(
                source_mask == rgb_array,
                axis=2,
            )

            actual_mask[
                matching_pixels
            ] = int(target_id)

        self.assertEqual(
            actual_mask.dtype,
            np.dtype(np.uint8),
        )

        np.testing.assert_array_equal(
            actual_mask,
            expected_mask,
        )

    def test_ycor_synthetic_rgb_mask(
        self,
    ) -> None:
        with YCOR_MAPPING_PATH.open(
            "r",
            encoding="utf-8-sig",
        ) as file:
            config = json.load(file)

        mapping_entries = config[
            "source_palette_rgb"
        ]

        source_mask = np.asarray(
            [
                [
                    entry["rgb"]
                    for entry
                    in mapping_entries
                ]
            ],
            dtype=np.uint8,
        )

        expected_mask = np.asarray(
            [
                [
                    int(entry["target_id"])
                    for entry
                    in mapping_entries
                ]
            ],
            dtype=np.uint8,
        )

        original_mapping_file = (
            self.ycor_module.MAPPING_FILE
        )

        try:
            self.ycor_module.MAPPING_FILE = (
                YCOR_MAPPING_PATH
            )

            with tempfile.TemporaryDirectory(
                prefix="adom_ycor_rgb_test_",
            ) as temporary_directory:
                mask_path = (
                    Path(temporary_directory)
                    / "synthetic_rgb_mask.png"
                )

                Image.fromarray(
                    source_mask,
                    mode="RGB",
                ).save(mask_path)

                (
                    actual_mask,
                    source_encoding,
                ) = self.ycor_module.remap_mask(
                    mask_path
                )

            self.assertEqual(
                source_encoding,
                "rgb",
            )

            self.assertEqual(
                actual_mask.dtype,
                np.dtype(np.uint8),
            )

            np.testing.assert_array_equal(
                actual_mask,
                expected_mask,
            )

        finally:
            self.ycor_module.MAPPING_FILE = (
                original_mapping_file
            )

    def test_ycor_synthetic_index_mask(
        self,
    ) -> None:
        with YCOR_MAPPING_PATH.open(
            "r",
            encoding="utf-8-sig",
        ) as file:
            config = json.load(file)

        mapping_entries = config[
            "source_palette_rgb"
        ]

        source_mask = np.asarray(
            [
                [
                    int(
                        entry[
                            "source_index_if_indexed"
                        ]
                    )
                    for entry
                    in mapping_entries
                ]
            ],
            dtype=np.uint8,
        )

        expected_mask = np.asarray(
            [
                [
                    int(entry["target_id"])
                    for entry
                    in mapping_entries
                ]
            ],
            dtype=np.uint8,
        )

        original_mapping_file = (
            self.ycor_module.MAPPING_FILE
        )

        try:
            self.ycor_module.MAPPING_FILE = (
                YCOR_MAPPING_PATH
            )

            with tempfile.TemporaryDirectory(
                prefix="adom_ycor_index_test_",
            ) as temporary_directory:
                mask_path = (
                    Path(temporary_directory)
                    / "synthetic_index_mask.png"
                )

                Image.fromarray(
                    source_mask,
                    mode="L",
                ).save(mask_path)

                (
                    actual_mask,
                    source_encoding,
                ) = self.ycor_module.remap_mask(
                    mask_path
                )

            self.assertEqual(
                source_encoding,
                "indexed",
            )

            self.assertEqual(
                actual_mask.dtype,
                np.dtype(np.uint8),
            )

            np.testing.assert_array_equal(
                actual_mask,
                expected_mask,
            )

        finally:
            self.ycor_module.MAPPING_FILE = (
                original_mapping_file
            )


if __name__ == "__main__":
    unittest.main(
        verbosity=2,
    )
