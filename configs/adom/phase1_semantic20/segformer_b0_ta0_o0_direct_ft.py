_base_ = [
    "./_base_/models/segformer_b0_ta.py",
    "./_base_/datasets/ta0_i0_crop512.py",
    "./_base_/semantic_default_runtime.py",
    "./_base_/schedules/target_adapt_direct_ft.py",
]
custom_imports = dict(imports=["adom.mmseg"], allow_failed_imports=False)
semantic20_experiment = "TA0_O0_DIRECT_FT"
ta0_contract = dict(
    config_id="ta0-o0-direct-ft",
    ablation_axis="optimization",
    input_profile="i0_crop512",
    optimization="direct_ft",
    sampling="source_uniform",
    loss="ce_only",
    seed=int("{{$ADOM_SEED:42}}"),
    effective_batch=16,
    total_optimizer_updates=6000,
    phase_optimizer_updates=6000,
)
