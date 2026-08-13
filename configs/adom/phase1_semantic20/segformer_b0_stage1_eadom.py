_base_ = [
    "./_base_/models/segformer_b0.py",
    "./_base_/datasets/eadom_rellis_adom.py",
    "./_base_/semantic_default_runtime.py",
    "./_base_/schedules/stage1_head_4k_updates.py",
]
custom_imports = dict(imports=["adom.mmseg"], allow_failed_imports=False)
semantic20_experiment = "EADOM_RELLIS_PLUS_ADOM_E0_RECIPE"
