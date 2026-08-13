from __future__ import annotations

import os
from pathlib import Path

from mmengine.config import Config


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = REPO_ROOT / "configs" / "adom" / "phase1_semantic20"
PAIRED_RECIPES = {
    "c0": ("segformer_b0_ta0_c0_stage1.py", "segformer_b0_ta0_c0_stage2.py"),
    "i0": ("segformer_b0_ta0_i0_stage1.py", "segformer_b0_ta0_i0_stage2.py"),
    "i1": ("segformer_b0_ta0_i1_stage1.py", "segformer_b0_ta0_i1_stage2.py"),
    "i2": ("segformer_b0_ta0_i2_stage1.py", "segformer_b0_ta0_i2_stage2.py"),
    "o1": (
        "segformer_b0_ta0_o1_lp_ft_stage1.py",
        "segformer_b0_ta0_o1_lp_ft_stage2.py",
    ),
    "b0": (
        "segformer_b0_ta0_b0_uniform_stage1.py",
        "segformer_b0_ta0_b0_uniform_stage2.py",
    ),
    "b1": (
        "segformer_b0_ta0_b1_rcs_stage1.py",
        "segformer_b0_ta0_b1_rcs_stage2.py",
    ),
    "l0": (
        "segformer_b0_ta0_l0_ce_stage1.py",
        "segformer_b0_ta0_l0_ce_stage2.py",
    ),
    "l1": (
        "segformer_b0_ta0_l1_ce_lovasz_stage1.py",
        "segformer_b0_ta0_l1_ce_lovasz_stage2.py",
    ),
}
SINGLE_RECIPES = {
    "o0": "segformer_b0_ta0_o0_direct_ft.py",
    "o2": "segformer_b0_ta0_o2_discriminative_lr.py",
}


def _load(filename: str) -> Config:
    return Config.fromfile(CONFIG_ROOT / filename, import_custom_modules=False)


def _optimizer_updates(config: Config) -> int:
    accumulation = int(config.optim_wrapper.get("accumulative_counts", 1))
    iterations = int(config.train_cfg.max_iters)
    if iterations % accumulation:
        raise RuntimeError(f"Runner iterations are not divisible by accumulation: {iterations}")
    return iterations // accumulation


def _assert_common(config: Config) -> None:
    assert config.train_dataloader.dataset.split == "splits/ta0_train.txt"
    assert dict(config.train_dataloader.sampler.source_weights) == {"rellis3d": 1.0}
    assert int(config.randomness.seed) == 42
    assert int(config.ta0_contract.effective_batch) == 16
    assert (
        int(config.train_dataloader.batch_size)
        * int(config.optim_wrapper.get("accumulative_counts", 1))
        == 16
    )
    assert int(config.ta0_contract.total_optimizer_updates) == 6000
    hook_types = {hook["type"] for hook in config.custom_hooks}
    assert "CanonicalTestLockHook" in hook_types
    assert "TA0AblationContractHook" in hook_types
    assert "SourceExposureAuditHook" in hook_types


def main() -> None:
    os.environ.setdefault("ADOM_DATA_ROOT", "/tmp/adom-ta0-config-import")
    os.environ.setdefault("ADOM_SEED", "42")
    os.environ.setdefault("ADOM_ACCUMULATIVE_COUNTS", "2")
    os.environ.setdefault("ADOM_MICRO_BATCH", "8")
    os.environ.setdefault("ADOM_TA_TOTAL_OPTIMIZER_UPDATES", "6000")
    os.environ.setdefault("WANDB_MODE", "disabled")

    for recipe, filenames in PAIRED_RECIPES.items():
        configs = [_load(filename) for filename in filenames]
        for config in configs:
            _assert_common(config)
        actual = sum(_optimizer_updates(config) for config in configs)
        assert actual == 6000, (recipe, actual)

    for recipe, filename in SINGLE_RECIPES.items():
        config = _load(filename)
        _assert_common(config)
        actual = _optimizer_updates(config)
        assert actual == 6000, (recipe, actual)

    for head_updates in (500, 1000):
        os.environ["ADOM_TA_LP_HEAD_OPTIMIZER_UPDATES"] = str(head_updates)
        lp_configs = [_load(filename) for filename in PAIRED_RECIPES["o1"]]
        assert [_optimizer_updates(config) for config in lp_configs] == [
            head_updates,
            6000 - head_updates,
        ]
    os.environ["ADOM_TA_LP_HEAD_OPTIMIZER_UPDATES"] = "1000"

    rcs = _load("segformer_b0_ta0_b1_rcs_stage1.py")
    assert rcs.train_dataloader.sampler.type == "SourceRareClassInfiniteSampler"
    assert list(rcs.train_dataloader.sampler.rare_class_ids) == [3, 10, 15, 18]
    ce = _load("segformer_b0_ta0_l0_ce_stage1.py")
    assert ce.model.decode_head.loss_decode.type == "CrossEntropyLoss"
    lovasz = _load("segformer_b0_ta0_l1_ce_lovasz_stage1.py")
    assert [loss.type for loss in lovasz.model.decode_head.loss_decode] == [
        "CrossEntropyLoss",
        "LovaszLoss",
    ]

    combined_path = CONFIG_ROOT / "segformer_b0_ta0_r_combined.py"
    os.environ.pop("ADOM_TA0_COMBINED_CONFIG_IMPORT_ONLY", None)
    try:
        Config.fromfile(combined_path, import_custom_modules=False)
    except ValueError as error:
        assert "combined config is locked" in str(error)
    else:
        raise AssertionError("TA0-R combined config imported without approval")
    os.environ["ADOM_TA0_COMBINED_CONFIG_IMPORT_ONLY"] = "true"
    combined = Config.fromfile(combined_path, import_custom_modules=False)
    _assert_common(combined)
    assert bool(combined.ta0_contract.provisional)
    assert _optimizer_updates(combined) == 6000
    print("TA0 config import and comparison contracts: PASS")


if __name__ == "__main__":
    main()
