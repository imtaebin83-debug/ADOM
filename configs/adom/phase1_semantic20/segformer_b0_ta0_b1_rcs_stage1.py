_base_ = [
    "./_base_/models/segformer_b0_ta.py",
    "./_base_/datasets/ta0_i0_crop512_rcs.py",
    "./_base_/semantic_default_runtime.py",
    "./_base_/schedules/target_adapt_stage1_head.py",
]
custom_imports = dict(imports=["adom.mmseg"], allow_failed_imports=False)
semantic20_experiment = "TA0_B1_RCS"
ta0_contract = dict(
    config_id="ta0-b1-rcs-stage1",
    ablation_axis="imbalance",
    input_profile="i0_crop512",
    optimization="lp_ft",
    sampling="source_quota_rcs",
    loss="ce_only",
    seed=int("{{$ADOM_SEED:42}}"),
    effective_batch=16,
    total_optimizer_updates=6000,
    phase_optimizer_updates=int("{{$ADOM_TA_LP_HEAD_OPTIMIZER_UPDATES:1000}}"),
)
