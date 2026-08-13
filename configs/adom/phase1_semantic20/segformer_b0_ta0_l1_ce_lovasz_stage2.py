_base_ = [
    "./_base_/models/segformer_b0_ta_ce_lovasz.py",
    "./_base_/datasets/ta0_i0_crop512.py",
    "./_base_/semantic_default_runtime.py",
    "./_base_/schedules/target_adapt_stage2_full.py",
]
custom_imports = dict(imports=["adom.mmseg"], allow_failed_imports=False)
semantic20_experiment = "TA0_L1_CE_LOVASZ"
ta0_contract = dict(
    config_id="ta0-l1-ce-lovasz-stage2",
    ablation_axis="loss",
    input_profile="i0_crop512",
    optimization="lp_ft",
    sampling="source_uniform",
    loss="ce_plus_lovasz",
    seed=int("{{$ADOM_SEED:42}}"),
    effective_batch=16,
    total_optimizer_updates=6000,
    phase_optimizer_updates=6000
    - int("{{$ADOM_TA_LP_HEAD_OPTIMIZER_UPDATES:1000}}"),
)
