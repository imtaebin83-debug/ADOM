from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]

CONVERTER_PATH = (
    PACKAGE_ROOT
    / "scripts"
    / "01_convert_bridge_sources.py"
)

MAPPING_PATH = (
    PACKAGE_ROOT
    / "config"
    / "bridge_mapping.yaml"
)


def load_converter():
    spec = importlib.util.spec_from_file_location(
        "adom_bridge_converter",
        CONVERTER_PATH,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Failed to load: {CONVERTER_PATH}"
        )

    module = importlib.util.module_from_spec(
        spec
    )
    spec.loader.exec_module(module)

    return module


converter = load_converter()


EXPECTED_RUGD_MAPPING = {
    0: 255,
    3: 1,
    4: 2,
    6: 255,
    7: 9,
    10: 8,
    11: 4,
    12: 5,
    14: 255,
    17: 13,
    19: 11,
    20: 18,
}

EXPECTED_YCOR_MAPPING = {
    0: 255,
    1: 255,
    2: 1,
    3: 255,
    4: 255,
    5: 255,
    6: 255,
    7: 16,
    8: 255,
}


def test_target_metadata() -> None:
    config = converter.load_bridge_config(
        MAPPING_PATH
    )

    assert config["num_classes"] == 19
    assert config["ignore_index"] == 255
    assert config["reduce_zero_label"] is False

    target_ids = {
        int(target_id)
        for target_id
        in config["target_classes"]
    }

    assert target_ids == (
        set(range(19)) | {255}
    )


def test_rugd_synthetic_mapping() -> None:
    config = converter.load_bridge_config(
        MAPPING_PATH
    )

    mapping = converter.load_source_to_target(
        config,
        "rugd",
    )

    assert mapping == EXPECTED_RUGD_MAPPING

    source_mask = np.array(
        [
            [0, 3, 4, 6, 7, 10],
            [11, 12, 14, 17, 19, 20],
        ],
        dtype=np.uint8,
    )

    expected_mask = np.array(
        [
            [255, 1, 2, 255, 9, 8],
            [4, 5, 255, 13, 11, 18],
        ],
        dtype=np.uint8,
    )

    result = converter.remap_mask(
        source_mask,
        mapping,
        "RUGD",
        Path("synthetic_rugd.png"),
    )

    assert np.array_equal(
        result,
        expected_mask,
    )


def test_ycor_synthetic_mapping() -> None:
    config = converter.load_bridge_config(
        MAPPING_PATH
    )

    mapping = converter.load_source_to_target(
        config,
        "ycor",
    )

    assert mapping == EXPECTED_YCOR_MAPPING

    source_mask = np.array(
        [
            [0, 1, 2, 3, 4],
            [5, 6, 7, 8, 2],
        ],
        dtype=np.uint8,
    )

    expected_mask = np.array(
        [
            [255, 255, 1, 255, 255],
            [255, 255, 16, 255, 1],
        ],
        dtype=np.uint8,
    )

    result = converter.remap_mask(
        source_mask,
        mapping,
        "YCOR",
        Path("synthetic_ycor.png"),
    )

    assert np.array_equal(
        result,
        expected_mask,
    )


def test_unknown_source_id_is_rejected() -> None:
    config = converter.load_bridge_config(
        MAPPING_PATH
    )

    mapping = converter.load_source_to_target(
        config,
        "rugd",
    )

    source_mask = np.array(
        [[3, 4, 99]],
        dtype=np.uint8,
    )

    with pytest.raises(
        ValueError,
        match="Unknown RUGD source IDs",
    ):
        converter.remap_mask(
            source_mask,
            mapping,
            "RUGD",
            Path("synthetic_unknown.png"),
        )
