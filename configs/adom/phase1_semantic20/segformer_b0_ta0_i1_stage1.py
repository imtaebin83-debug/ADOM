_base_ = [
    "./_base_/models/segformer_b0_ta.py",
    "./_base_/datasets/ta0_i1_nocrop_640x384.py",
    "./_base_/semantic_default_runtime.py",
    "./_base_/schedules/target_adapt_stage1_head.py",
]
custom_imports = dict(imports=["adom.mmseg"], allow_failed_imports=False)
model = dict(data_preprocessor=dict(size=(384, 640)))
semantic20_experiment = "TA0_I1_NOCROP_640X384"
ta0_contract = dict(
    config_id="ta0-i1-stage1",
    ablation_axis="input",
    input_profile="i1_nocrop_640x384",
    optimization="lp_ft",
    sampling="source_uniform",
    loss="ce_only",
    seed=int("{{$ADOM_SEED:42}}"),
    effective_batch=16,
    total_optimizer_updates=6000,
    phase_optimizer_updates=int("{{$ADOM_TA_LP_HEAD_OPTIMIZER_UPDATES:1000}}"),
)
