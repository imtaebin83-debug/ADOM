_base_ = ["./segformer_b2.py"]

# Official MMSegmentation v1.2.2 MiT-B5 ImageNet initialization. B5-E0 and
# B5-E-ADOM start here; they must never warm-start from a B0/B2 experiment.
checkpoint = (
    "https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/"
    "segformer/mit_b5_20220624-658746d9.pth"
)
model = dict(
    backbone=dict(
        num_layers=[3, 6, 40, 3],
        init_cfg=dict(type="Pretrained", checkpoint=checkpoint),
    ),
)
