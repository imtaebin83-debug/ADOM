_base_ = ["./ta_target_adaptation.py"]

# I0 preserves the E0 RandomResize + 512 crop contract.
train_dataloader = dict(
    dataset=dict(split="splits/ta0_train.txt", manifest="manifest.csv")
)
val_dataloader = dict(dataset=dict(split="splits/val.txt", manifest="manifest.csv"))
test_dataloader = dict(dataset=dict(split="splits/test.txt", manifest="manifest.csv"))

ta0_input_profile = dict(
    id="i0_crop512",
    train_shape_hw=(512, 512),
    crop=True,
    keep_ratio=True,
    mask_interpolation="nearest",
    image_interpolation="bilinear",
)
