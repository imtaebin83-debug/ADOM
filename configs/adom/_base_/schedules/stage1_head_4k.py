import os

accumulative_counts = int(os.getenv("ADOM_ACCUMULATIVE_COUNTS", "4"))
optim_wrapper = dict(
    type="AmpOptimWrapper",
    loss_scale="dynamic",
    accumulative_counts=accumulative_counts,
    optimizer=dict(
        type="AdamW",
        lr=0.0006,
        betas=(0.9, 0.999),
        weight_decay=0.01,
    ),
    paramwise_cfg=dict(
        custom_keys={
            "norm": dict(decay_mult=0.0),
            "pos_block": dict(decay_mult=0.0),
        }
    ),
)
param_scheduler = [
    dict(
        type="LinearLR",
        start_factor=1e-3,
        by_epoch=False,
        begin=0,
        end=200,
    ),
    dict(
        type="PolyLR",
        eta_min=0.0,
        power=1.0,
        begin=200,
        end=4000,
        by_epoch=False,
    ),
]
train_cfg = dict(type="IterBasedTrainLoop", max_iters=4000, val_interval=500)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")
custom_hooks = [
    dict(type="FreezeBackboneHook"),
    dict(type="FiniteLossHook"),
    dict(type="MetricArtifactHook"),
]
