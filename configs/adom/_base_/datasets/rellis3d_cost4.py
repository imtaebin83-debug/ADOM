import os

dataset_type = "AdomCost4Dataset"
data_root = os.getenv("ADOM_DATA_ROOT", "/workspace/adom/datasets/rellis3d")
crop_size = (512, 512)

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", reduce_zero_label=False),
    dict(
        type="RandomResize",
        scale=(1024, 512),
        ratio_range=(0.5, 2.0),
        keep_ratio=True,
    ),
    dict(type="RandomCrop", crop_size=crop_size, cat_max_ratio=0.75),
    dict(type="RandomFlip", prob=0.5, direction="horizontal"),
    dict(type="PhotoMetricDistortion"),
    dict(type="PackSegInputs"),
]

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=(1024, 512), keep_ratio=True),
    dict(type="LoadAnnotations", reduce_zero_label=False),
    dict(type="PackSegInputs"),
]

train_dataset = dict(
    type=dataset_type,
    data_root=data_root,
    manifest="metadata/manifest_train.csv",
    pipeline=train_pipeline,
)
val_dataset = dict(
    type=dataset_type,
    data_root=data_root,
    manifest="metadata/manifest_val.csv",
    pipeline=test_pipeline,
    test_mode=True,
)
test_dataset = dict(
    type=dataset_type,
    data_root=data_root,
    manifest="metadata/manifest_test.csv",
    pipeline=test_pipeline,
    test_mode=True,
)

micro_batch = int(os.getenv("ADOM_MICRO_BATCH", "4"))
num_workers = int(os.getenv("ADOM_NUM_WORKERS", "4"))

train_dataloader = dict(
    batch_size=micro_batch,
    num_workers=num_workers,
    persistent_workers=num_workers > 0,
    sampler=dict(type="InfiniteSampler", shuffle=True),
    dataset=train_dataset,
)
val_dataloader = dict(
    batch_size=1,
    num_workers=num_workers,
    persistent_workers=num_workers > 0,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=val_dataset,
)
test_dataloader = dict(
    batch_size=1,
    num_workers=num_workers,
    persistent_workers=num_workers > 0,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=test_dataset,
)

val_evaluator = [
    dict(type="IoUMetric", iou_metrics=["mIoU"]),
    dict(type="AdomSafetyMetric", ignore_index=255),
]
test_evaluator = val_evaluator
