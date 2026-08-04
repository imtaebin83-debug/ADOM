_base_ = ["./segformer_b0.py"]

checkpoint = (
    "https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/"
    "segformer/mit_b2_20220624-66e8bf70.pth"
)
model = dict(
    backbone=dict(
        embed_dims=64,
        num_layers=[3, 4, 6, 3],
        init_cfg=dict(type="Pretrained", checkpoint=checkpoint),
    ),
    decode_head=dict(
        in_channels=[64, 128, 320, 512],
        channels=768,
    ),
)
