_base_ = ["./segformer_b0_640x384_rellis3d.py"]

# E-ADOM uses the same SegFormer-B0 architecture, Semantic20 ontology and
# 640x384 inference preprocessing as B0-E0. This distinct config preserves the
# deployed model identity while the checkpoint supplies the adapted weights.
semantic20_deployment_profile = "eadom-b0-seed42-iter26000"
