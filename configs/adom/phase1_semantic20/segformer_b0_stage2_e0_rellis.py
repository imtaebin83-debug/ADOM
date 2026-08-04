_base_ = [
    "./_base_/models/segformer_b0.py",
    "./_base_/datasets/e0_rellis.py",
    "./_base_/semantic_default_runtime.py",
    "./_base_/schedules/stage2_full_40k_updates.py",
]
custom_imports = dict(imports=["adom.mmseg"], allow_failed_imports=False)
semantic20_experiment = "E0_RELLIS_ONLY"
