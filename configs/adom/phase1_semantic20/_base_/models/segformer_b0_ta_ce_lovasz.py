_base_ = ["./segformer_b0_ta.py"]

# This is an independent loss ablation. Do not pair it with RCS in the L1 run.
model = dict(
    decode_head=dict(
        loss_decode=[
            dict(
                type="CrossEntropyLoss",
                use_sigmoid=False,
                avg_non_ignore=True,
                loss_weight=1.0,
            ),
            dict(
                type="LovaszLoss",
                loss_type="multi_class",
                per_image=False,
                reduction="none",
                loss_weight=1.0,
            ),
        ]
    )
)
