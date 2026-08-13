_base_ = ["./segformer_b0_ta0_c0_stage1.py"]
semantic20_experiment = "TA0_O1_LP_FT"
ta0_contract = dict(
    config_id="ta0-o1-lp-ft-stage1",
    ablation_axis="optimization",
)
