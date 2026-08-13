_base_ = ["./target_adapt_direct_ft.py"]

# Longest matching custom key wins in MMEngine. Early MiT stages receive the
# smallest LR; the late encoder and decoder retain adaptation capacity.
optim_wrapper = dict(
    paramwise_cfg=dict(
        _delete_=True,
        custom_keys={
            "backbone.layers.0": dict(lr_mult=0.1),
            "backbone.layers.1": dict(lr_mult=0.25),
            "backbone.layers.2": dict(lr_mult=0.5),
            "backbone.layers.3": dict(lr_mult=1.0),
            "decode_head": dict(lr_mult=10.0),
            "norm": dict(decay_mult=0.0),
            "pos_block": dict(decay_mult=0.0),
        },
    )
)
