_base_ = [
    "./_base_/models/segformer_b0_cost4.py",
    "./_base_/datasets/rellis3d_cost4.py",
    "./_base_/default_runtime.py",
    "./_base_/schedules/stage1_head_4k.py",
]
custom_imports = dict(imports=["adom.mmseg"], allow_failed_imports=False)
