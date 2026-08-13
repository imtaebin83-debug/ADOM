_base_ = [
    "./_base_/models/segformer_b0_ta.py",
    "./_base_/datasets/ta0_i2_nocrop_640x480.py",
    "./_base_/semantic_default_runtime.py",
    "./_base_/schedules/target_adapt_stage1_head.py",
]
custom_imports = dict(imports=["adom.mmseg"], allow_failed_imports=False)
model = dict(data_preprocessor=dict(size=(480, 640)))
semantic20_experiment = "TA0_I2_NOCROP_640X480"
ta0_contract = dict(
    config_id="ta0-i2-stage1",
    ablation_axis="input",
    input_profile="i2_nocrop_640x480",
    optimization="lp_ft",
    sampling="source_uniform",
    loss="ce_only",
    seed=int("{{$ADOM_SEED:42}}"),
    effective_batch=16,
    total_optimizer_updates=6000,
    phase_optimizer_updates=int("{{$ADOM_TA_LP_HEAD_OPTIMIZER_UPDATES:1000}}"),
)
