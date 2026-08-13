_base_ = ["./segformer_b0.py"]

# TA always loads the verified B0-E0 checkpoint through the runtime. Avoid a
# second implicit ImageNet download or initialization path.
model = dict(backbone=dict(init_cfg=None))
