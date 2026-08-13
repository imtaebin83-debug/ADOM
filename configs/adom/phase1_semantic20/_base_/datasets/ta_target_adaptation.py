_base_ = ["./e1_combined.py"]

# Condition configs must override the train split and exact source weights.
# Canonical validation/test remain the RELLIS-only files in the shared package.
train_dataloader = dict(
    sampler=dict(
        _delete_=True,
        type="SourceWeightedInfiniteSampler",
        source_weights=dict(rellis3d=1.0),
        start_index=int("{{$ADOM_SAMPLER_START_INDEX:0}}"),
    ),
    dataset=dict(split="splits/TA_CONDITION_NOT_SET.txt", manifest="manifest.csv"),
)
val_dataloader = dict(dataset=dict(split="splits/val.txt", manifest="manifest.csv"))
test_dataloader = dict(dataset=dict(split="splits/test.txt", manifest="manifest.csv"))
