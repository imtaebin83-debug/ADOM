_base_ = ["./segformer_b0_ta0_c0_stage1.py"]
semantic20_experiment = "TA0_I0_CROP512"
ta0_contract = dict(config_id="ta0-i0-stage1", ablation_axis="input")
