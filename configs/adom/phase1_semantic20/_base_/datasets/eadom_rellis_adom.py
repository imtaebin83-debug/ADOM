_base_ = ["./e1_combined.py"]

# Emergency E-ADOM control: keep the E0 input, augmentation, normalization,
# uniform InfiniteSampler, canonical RELLIS validation and optimizer recipe.
# Only the train split changes, adding the labeled ADOM standalone train set.
train_dataloader = dict(dataset=dict(split="splits/ta1_train.txt"))

# The fixed Semantic20 metric computes its confusion matrix on CPU. The stock
# MMSeg IoUMetric uses CUDA histc, which PyTorch 2.1 rejects under strict
# deterministic algorithms. These metrics contain the authoritative mIoU,
# recall, precision, absent-class FP and safety panels used for selection.
metric_output_dir = "{{$ADOM_METRIC_OUTPUT_DIR:work_dirs/semantic20_metrics}}"
val_evaluator = [
    dict(
        type="AdomSemantic20Metric",
        ignore_index=255,
        output_dir=metric_output_dir,
        evaluation_split="val",
    )
]
test_evaluator = [
    dict(
        type="AdomSemantic20Metric",
        ignore_index=255,
        output_dir=metric_output_dir,
        evaluation_split="test",
    )
]
