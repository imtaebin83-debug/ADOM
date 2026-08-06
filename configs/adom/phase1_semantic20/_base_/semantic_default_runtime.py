default_scope = "mmseg"
runtime_seed = int("{{$ADOM_SEED:42}}")
runtime_deterministic = "{{$ADOM_DETERMINISTIC:true}}".strip().lower()
if runtime_deterministic not in {"true", "false", "1", "0"}:
    raise ValueError("ADOM_DETERMINISTIC must be true/false or 1/0")
runtime_deterministic = runtime_deterministic in {"true", "1"}
env_cfg = dict(
    cudnn_benchmark=not runtime_deterministic,
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
    dist_cfg=dict(backend="nccl"),
)

# Environment substitution is native MMEngine config syntax. Avoiding Python
# imports here keeps the complete inheritance chain in non-lazy config mode.
wandb_init_kwargs = dict(
    project="{{$WANDB_PROJECT:adom}}",
    group="{{$WANDB_RUN_GROUP:semantic20}}",
    name="{{$WANDB_NAME:semantic20-run}}",
    id="{{$WANDB_RUN_ID:semantic20-run}}",
    resume="allow",
    job_type="{{$WANDB_JOB_TYPE:training}}",
    tags=[
        "phase1",
        "semantic20",
        "{{$ADOM_EXPERIMENT_TAG:experiment-unset}}",
        "{{$ADOM_MODEL_TAG:model-unset}}",
        "{{$ADOM_PHASE_TAG:phase-unset}}",
        "{{$WANDB_EXTRA_TAG:runpod}}",
    ],
)
wandb_mode = "{{$WANDB_MODE:online}}".strip().lower()
vis_backends = [
    dict(type="TensorboardVisBackend"),
    dict(type="LocalVisBackend"),
]
if wandb_mode != "disabled":
    vis_backends.insert(
        0, dict(type="WandbVisBackend", init_kwargs=wandb_init_kwargs)
    )
visualizer = dict(
    type="SegLocalVisualizer",
    vis_backends=vis_backends,
    name="visualizer",
)
log_processor = dict(by_epoch=False)
log_level = "INFO"
load_from = None
resume = False
randomness = dict(seed=runtime_seed, deterministic=runtime_deterministic)

runtime_accumulative_counts = int("{{$ADOM_ACCUMULATIVE_COUNTS:1}}")
runtime_checkpoint_updates = int(
    "{{$ADOM_CHECKPOINT_INTERVAL_OPTIMIZER_UPDATES:500}}"
)
default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(
        type="CheckpointHook",
        by_epoch=False,
        interval=runtime_checkpoint_updates * runtime_accumulative_counts,
        max_keep_ckpts=3,
        save_last=True,
        save_optimizer=True,
        save_param_scheduler=True,
        # Clean v1 selection is handled by ConstrainedCheckpointSelectionHook.
        save_best=None,
    ),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    visualization=dict(type="SegVisualizationHook"),
)
