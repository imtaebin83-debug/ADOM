# E0: RELLIS-only train, with the canonical RELLIS validation and test splits.
dataset_type = "AdomSemantic20Dataset"
data_root = "{{$ADOM_DATA_ROOT:ADOM_DATA_ROOT_NOT_SET}}"
crop_size = (512, 512)
pack_meta_keys = (
    "img_path",
    "seg_map_path",
    "ori_shape",
    "img_shape",
    "pad_shape",
    "scale_factor",
    "flip",
    "flip_direction",
    "reduce_zero_label",
    "sample_id",
)

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
    dict(type="PackSegInputs", meta_keys=pack_meta_keys),
]

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=(1024, 512), keep_ratio=True),
    dict(type="LoadAnnotations", reduce_zero_label=False),
    dict(type="PackSegInputs", meta_keys=pack_meta_keys),
]

train_dataset = dict(
    type=dataset_type,
    data_root=data_root,
    split="splits/train.txt",
    pipeline=train_pipeline,
)
val_dataset = dict(
    type=dataset_type,
    data_root=data_root,
    split="splits/val.txt",
    pipeline=test_pipeline,
    test_mode=True,
)
test_dataset = dict(
    type=dataset_type,
    data_root=data_root,
    split="splits/test.txt",
    pipeline=test_pipeline,
    test_mode=True,
)

micro_batch = int("{{$ADOM_MICRO_BATCH:16}}")
num_workers = int("{{$ADOM_NUM_WORKERS:4}}")
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

# IoUMetric is the official MMSeg 1.2.2 metric (mIoU, mAcc, aAcc and its
# per-class result table).
# The ADOM metric additionally persists the 19x19 confusion matrix and recall.
metric_output_dir = "{{$ADOM_METRIC_OUTPUT_DIR:work_dirs/semantic20_metrics}}"
val_evaluator = [
    dict(type="IoUMetric", iou_metrics=["mIoU"]),
    dict(
        type="AdomSemantic20Metric",
        ignore_index=255,
        output_dir=metric_output_dir,
        evaluation_split="val",
    ),
]
test_evaluator = [
    dict(type="IoUMetric", iou_metrics=["mIoU"]),
    dict(
        type="AdomSemantic20Metric",
        ignore_index=255,
        output_dir=metric_output_dir,
        evaluation_split="test",
    ),
]
