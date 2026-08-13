_base_ = ["./segformer_b0_ta0_c0_stage1.py"]
semantic20_experiment = "TA0_B0_SOURCE_UNIFORM"
ta0_contract = dict(config_id="ta0-b0-source-uniform-stage1", ablation_axis="imbalance")
