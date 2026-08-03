import os


default_scope = "mmseg"
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
    dist_cfg=dict(backend="nccl"),
)
wandb_init_kwargs = dict(
    project=os.getenv("WANDB_PROJECT", "adom"),
    job_type=os.getenv("WANDB_JOB_TYPE", "training"),
)
for key, environment_name in (
    ("entity", "WANDB_ENTITY"),
    ("group", "WANDB_RUN_GROUP"),
    ("name", "WANDB_NAME"),
):
    value = os.getenv(environment_name)
    if value:
        wandb_init_kwargs[key] = value

wandb_run_id = os.getenv("WANDB_RUN_ID")
if wandb_run_id:
    wandb_init_kwargs["id"] = wandb_run_id
    wandb_init_kwargs["resume"] = os.getenv("WANDB_RESUME", "allow")

wandb_tags = [
    item.strip() for item in os.getenv("WANDB_TAGS", "").split(",") if item.strip()
]
if wandb_tags:
    wandb_init_kwargs["tags"] = wandb_tags

vis_backends = [
    dict(type="WandbVisBackend", init_kwargs=wandb_init_kwargs),
    dict(type="TensorboardVisBackend"),
    dict(type="LocalVisBackend"),
]
visualizer = dict(
    type="SegLocalVisualizer",
    vis_backends=vis_backends,
    name="visualizer",
)
log_processor = dict(by_epoch=False)
log_level = "INFO"
load_from = None
resume = False
randomness = dict(seed=42, deterministic=False)

default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(
        type="CheckpointHook",
        by_epoch=False,
        interval=int(os.getenv("ADOM_CHECKPOINT_INTERVAL", "500")),
        max_keep_ckpts=3,
        save_last=True,
        save_optimizer=True,
        save_param_scheduler=True,
        save_best="mIoU",
        rule="greater",
    ),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    visualization=dict(type="SegVisualizationHook"),
)
