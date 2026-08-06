_base_ = [
    "./_base_/models/segformer_b2.py",
    "./_base_/datasets/e2_combined_goose.py",
    "./_base_/semantic_default_runtime.py",
    "./_base_/schedules/stage1_head_4k_updates.py",
]
custom_imports = dict(imports=["adom.mmseg"], allow_failed_imports=False)
