_base_ = ["./ta0_i0_crop512.py"]

# Source quota remains RELLIS=1.0. RCS changes only the image chosen inside
# that already-selected source slot.
train_dataloader = dict(
    sampler=dict(
        _delete_=True,
        type="SourceRareClassInfiniteSampler",
        source_weights=dict(rellis3d=1.0),
        rare_class_ids=[3, 10, 15, 18],
        rare_probability=0.5,
        temperature=0.01,
        minimum_pixels=1,
        ignore_index=255,
        start_index=int("{{$ADOM_SAMPLER_START_INDEX:0}}"),
    )
)
