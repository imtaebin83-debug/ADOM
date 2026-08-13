_base_ = ["./ta_target_adaptation.py"]

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
    dict(type="Resize", scale=(640, 384), keep_ratio=True),
    dict(type="RandomFlip", prob=0.5, direction="horizontal"),
    dict(type="PhotoMetricDistortion"),
    dict(type="PackSegInputs", meta_keys=pack_meta_keys),
]
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=(640, 384), keep_ratio=True),
    dict(type="LoadAnnotations", reduce_zero_label=False),
    dict(type="PackSegInputs", meta_keys=pack_meta_keys),
]
train_dataloader = dict(
    dataset=dict(
        split="splits/ta0_train.txt",
        manifest="manifest.csv",
        pipeline=train_pipeline,
    )
)
val_dataloader = dict(
    dataset=dict(split="splits/val.txt", manifest="manifest.csv", pipeline=test_pipeline)
)
test_dataloader = dict(
    dataset=dict(split="splits/test.txt", manifest="manifest.csv", pipeline=test_pipeline)
)

ta0_input_profile = dict(
    id="i1_nocrop_640x384",
    train_shape_hw=(384, 640),
    crop=False,
    keep_ratio=True,
    pad_direction="right_bottom",
    seg_pad_value=255,
    mask_interpolation="nearest",
    image_interpolation="bilinear",
)
