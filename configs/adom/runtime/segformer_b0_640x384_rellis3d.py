_base_ = ["../phase1_semantic20/segformer_b0_stage2_e0_rellis.py"]

# ROS/PyTorch inference contract for a 640x360 ZED frame. Keep-ratio resize
# leaves the image at 640x360; SegDataPreProcessor then pads the bottom 24 rows
# and records img_padding_size so MMSeg can crop them before restoring ori_shape.
runtime_size = (384, 640)
model = dict(
    data_preprocessor=dict(
        size=runtime_size,
        test_cfg=dict(size=runtime_size),
    )
)
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=(640, 384), keep_ratio=True),
    dict(type="PackSegInputs"),
]
test_dataloader = dict(dataset=dict(pipeline=test_pipeline))
