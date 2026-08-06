# E2 extends the frozen E1 package with GOOSE Semantic20 direct-only samples.
# Main validation/test remain canonical RELLIS-only. Source validation splits
# are diagnostic artifacts and never participate in checkpoint selection.
_base_ = ["./e1_combined.py"]

semantic20_experiment = "E2_RELLIS_RUGD_YCOR_GOOSE_DIRECT"
