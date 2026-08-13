_base_ = ["./segformer_b0_ta0_c0_stage2.py"]
semantic20_experiment = "TA0_L0_CE_ONLY"
ta0_contract = dict(config_id="ta0-l0-ce-only-stage2", ablation_axis="loss")
