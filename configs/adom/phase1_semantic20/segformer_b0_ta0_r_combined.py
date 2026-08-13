_base_ = [
    "./_base_/models/segformer_b0_ta_ce_lovasz.py",
    "./_base_/datasets/ta0_i1_nocrop_640x384_rcs.py",
    "./_base_/semantic_default_runtime.py",
    "./_base_/schedules/target_adapt_discriminative_lr.py",
]

# This file is deliberately fail-closed. It is a final interaction-check
# candidate, not a claim that I1/O2/B1/L1 have won their independent ablations.
import_only = "{{$ADOM_TA0_COMBINED_CONFIG_IMPORT_ONLY:false}}".strip().lower()
if import_only not in {"true", "1"}:
    raise ValueError(
        "TA0-R combined config is locked until the independent input, "
        "optimization, imbalance, and loss ablations have been reviewed"
    )
custom_imports = dict(imports=["adom.mmseg"], allow_failed_imports=False)
model = dict(data_preprocessor=dict(size=(384, 640)))
semantic20_experiment = "TA0_R_COMBINED_INTERACTION_CHECK"
ta0_contract = dict(
    config_id="ta0-r-combined-interaction-check",
    ablation_axis="combined_interaction_check",
    input_profile="i1_nocrop_640x384",
    optimization="discriminative_lr",
    sampling="source_quota_rcs",
    loss="ce_plus_lovasz",
    seed=int("{{$ADOM_SEED:42}}"),
    effective_batch=16,
    total_optimizer_updates=6000,
    phase_optimizer_updates=6000,
    provisional=True,
)
