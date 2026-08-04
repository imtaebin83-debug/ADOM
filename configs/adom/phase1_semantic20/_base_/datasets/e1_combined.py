# E1 uses the combined RELLIS+RUGD+YCOR package. Its preprocessing contract
# keeps val/test RELLIS-only, so E0 and E1 are evaluated on identical samples.
_base_ = ["./e0_rellis.py"]

semantic20_experiment = "E1_RELLIS_RUGD_YCOR"

train_dataloader = dict(dataset=dict(manifest="manifest.csv"))
val_dataloader = dict(dataset=dict(manifest="manifest.csv"))
test_dataloader = dict(dataset=dict(manifest="manifest.csv"))
