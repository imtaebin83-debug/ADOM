_base_ = ["../phase1_semantic20/segformer_b2_stage2_e0_rellis.py"]

# Semantic20 fallback export; kept separate from the Cost4 reference model.
export_size = (384, 384)
model = dict(data_preprocessor=dict(size=export_size))
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=(384, 384), keep_ratio=True),
    dict(type="Pad", size=export_size, pad_val=dict(img=0, seg=255)),
    dict(type="PackSegInputs"),
]
test_dataloader = dict(dataset=dict(pipeline=test_pipeline))
