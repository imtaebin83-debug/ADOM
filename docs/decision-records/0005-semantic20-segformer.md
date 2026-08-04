# 0005: Phase 1 Semantic20 SegFormer two-stage baseline

- Status: accepted for RunPod gates
- Date: 2026-08-04

## Scope

Phase 1 trains the original 20-label RELLIS ontology with `void` excluded:
19 trainable IDs (`0..18`), `ignore_index=255`, and `reduce_zero_label=False`.
The existing Cost4 configs and runtime remain as Phase 2/reference material under
their existing paths. Semantic20 has a separate config tree and execution entry
point.

E0 trains on RELLIS only. E1 trains on RELLIS+RUGD+YCOR. Both configs use the
canonical RELLIS-only validation and test splits. The runtime normalizes the E1
`rellis3d/` prefix and rejects a run unless val/test exactly match the committed
RELLIS split snapshots.
E1 resolves each image/mask through the combined package `manifest.csv`; this
is required because RUGD images are PNG while the RELLIS/YCOR images are JPEG.

## Architecture and optimization

SegFormer combines a hierarchical MiT encoder with a lightweight all-MLP
decoder and does not require positional encoding. B0 and B2 definitions and
ImageNet MiT pretrained checkpoints follow the official MMSegmentation configs.
The decoder has 19 randomly initialized outputs. Stage 1 freezes the encoder,
keeps it in eval mode, and trains the head. Stage 2 loads the best Stage 1
checkpoint and fine-tunes the full model with a 10x decoder-head LR.

- AdamW, betas `(0.9, 0.999)`, weight decay `0.01`
- norm and positional-block decay multiplier `0`
- Stage 1 head LR `6e-4`, 200-update warmup, 4,000 updates
- Stage 2 backbone LR `6e-5`, head LR `6e-4`, 500-update warmup, 40,000 updates
- linear warmup followed by PolyLR (`power=1.0`, `eta_min=0`)
- CrossEntropy only; no auxiliary loss, OHEM, class balancing, or layer decay
- seed 42 and dynamic-loss-scale `AmpOptimWrapper`

Primary references:

- [SegFormer paper](https://arxiv.org/abs/2105.15203)
- [MMSegmentation SegFormer configs](https://github.com/open-mmlab/mmsegmentation/tree/main/configs/segformer)
- [MMEngine OptimWrapper and AMP accumulation](https://mmengine.readthedocs.io/en/stable/tutorials/optim_wrapper.html)

The official BEiT layer-decay pattern is recorded only as a later experiment
candidate; it is deliberately absent from the first baseline:
[MMSegmentation BEiT configs](https://github.com/open-mmlab/mmsegmentation/tree/main/configs/beit).

## Optimizer-update clock

MMEngine's iter-based loop and parameter schedulers advance per runner/data
iteration. `AmpOptimWrapper(accumulative_counts=N)` performs one optimizer step
per N runner iterations. Consequently all update-domain targets are multiplied
by N:

| Target | Stage 1 runner iterations | Stage 2 runner iterations |
|---|---:|---:|
| maximum | `4,000 × N` | `40,000 × N` |
| warmup | `200 × N` | `500 × N` |
| validation | `500 × N` | `1,000 × N` |
| checkpoint | `500 × N` | `500 × N` |
| PolyLR end | `4,000 × N` | `40,000 × N` |

This prevents B2 fallback plans (`8/2` or `4/4`) from receiving only one-half
or one-quarter of the intended optimizer updates. Unit tests cover N=1, 2, 4.

## Metrics and artifacts

Official MMSeg 1.2.2 `IoUMetric` supplies mIoU, mAcc, aAcc and prints its
per-class IoU/accuracy table. `AdomSemantic20Metric` additionally logs a scalar
IoU and recall for each class and writes a 19×19 JSON confusion matrix (ground
truth rows, prediction columns). W&B is primary; TensorBoard and local MMEngine
visualization remain enabled. Stable W&B run IDs, group/tags, checkpoint
markers, best/latest checkpoints, commands, dataset split digest, and resume
state are recorded under the run output.

ONNX is intentionally outside the Semantic20 training cycle. The compatibility
flag `--skip-export` is accepted, while the Cost4 reference cycle now also has a
real `--skip-export` switch. Export cannot block the first Phase 1 training run.
