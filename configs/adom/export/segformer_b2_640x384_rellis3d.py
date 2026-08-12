_base_ = ["../phase1_semantic20/segformer_b2_stage2_e0_rellis.py"]

# Semantic20 fallback export; kept separate from the Cost4 reference model.
model_input_size_hw = (384, 640)
pipeline_size_wh = (640, 384)
model = dict(data_preprocessor=dict(size=model_input_size_hw))
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=pipeline_size_wh, keep_ratio=True),
    dict(type="Pad", size=pipeline_size_wh, pad_val=dict(img=0, seg=255)),
    dict(type="PackSegInputs"),
]
test_dataloader = dict(dataset=dict(pipeline=test_pipeline))
