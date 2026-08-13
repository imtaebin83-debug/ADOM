accumulative_counts = int("{{$ADOM_ACCUMULATIVE_COUNTS:1}}")
if accumulative_counts < 1:
    raise ValueError("ADOM_ACCUMULATIVE_COUNTS must be at least 1")

optimizer_updates = int("{{$ADOM_MAX_OPTIMIZER_UPDATES:5000}}")
warmup_updates = min(int("{{$ADOM_WARMUP_OPTIMIZER_UPDATES:100}}"), optimizer_updates)
val_updates = int("{{$ADOM_VAL_INTERVAL_OPTIMIZER_UPDATES:500}}")


def runner_iters(update_count):
    return update_count * accumulative_counts


optim_wrapper = dict(
    type="AmpOptimWrapper",
    loss_scale="dynamic",
    accumulative_counts=accumulative_counts,
    optimizer=dict(type="AdamW", lr=6e-5, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            "decode_head": dict(lr_mult=10.0),
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
        end=runner_iters(warmup_updates),
    )
]
if warmup_updates < optimizer_updates:
    param_scheduler.append(
        dict(
            type="PolyLR",
            eta_min=0.0,
            power=1.0,
            begin=runner_iters(warmup_updates),
            end=runner_iters(optimizer_updates),
            by_epoch=False,
        )
    )
train_cfg = dict(
    type="IterBasedTrainLoop",
    max_iters=runner_iters(optimizer_updates),
    val_interval=runner_iters(val_updates),
)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")
custom_hooks = [
    dict(type="CanonicalTestLockHook"),
    dict(type="BackboneAuditHook"),
    dict(type="FiniteLossHook"),
    dict(type="MetricArtifactHook"),
    dict(type="SourceExposureAuditHook", interval=100),
    dict(type="ConstrainedCheckpointSelectionHook", tolerance_pp=1.0),
]
