# ADOM paper evaluation

This directory rebuilds the B0-E0 versus E-ADOM comparison from direct
checkpoint inference. It never copies a stored metric into a paper table and
never overwrites a checkpoint, dataset, annotation, manifest, or prior output.

The evaluator is fail closed. `evaluate_checkpoint.py` will not start unless
`audit_report.md`, `environment.json`, `checkpoint_manifest.json`, and
`dataset_manifest_summary.json` all exist and both machine-readable audit
statuses are `PASS`.

## Frozen identities

- Ontology: Semantic20 IDs `0..18`, ignore `255`, class order from
  `tools/paper_eval/_common.py` and `src/adom/evaluation_semantic20.py`.
- B0-E0 canonical/deployment checkpoint: Stage 2
  `best_mIoU_iter_6000.pth`, expected SHA-256
  `d76229ff623eb382fd48011decf54c342d88a113bcbe650fb58cc20e42cabe73`.
- E-ADOM artifact directory:
  `/workspace/adom/artifacts/eadom-b0-seed42-iter26000`, expected checkpoint
  SHA-256
  `f4cc41fd91e9df8e7aa3f726498e80636b736dfadf0e1baf338fe7c82a83399c`.
- Frozen comparison archive:
  `/workspace/adom/exports/canonical-compare-20260814T013811Z.tar.gz`, expected
  SHA-256
  `8468bca1840c89b19145e743d877ffbcf6e5b4f50013de3bcb3d76b6ed45f77b`.
- Canonical paper inference config for both checkpoints:
  `configs/adom/phase1_semantic20/segformer_b0_stage2_e0_rellis.py`.
  The tool removes only `LoadAnnotations` from the test pipeline used by the
  standalone inference API. It preserves the keep-ratio `(1024, 512)` resize,
  model data preprocessor, ImageNet mean/std, `whole` inference, interpolation,
  postprocessing, and 19-way argmax for both models.

The repository documents the B0-E0 path as:

```text
/workspace/adom/runs/semantic20/e0/20260805T122006Z-5c50bfdf2900-b0-full/b0/stage2/best_mIoU_iter_6000.pth
```

`audit_experiment.py` lists every `.pth` candidate under each supplied search
root with path, iteration parsed from its name/metadata, modification time,
SHA-256, size, and nearby config candidates. E-ADOM is selected only by an
exact unique SHA match. Do not replace either with `latest.pth`, a final
iteration, or another best checkpoint.

## Environment

Run in the existing immutable RunPod image/environment used for the
checkpoints. Do not rebuild the training image or install a different OpenMMLab
stack during evaluation. The expected project stack is Python 3.10, PyTorch
2.1, MMSegmentation 1.2.2 and its pinned compatible MMEngine/MMCV versions; the
audit records the actual versions and blocks if a required package is absent.

From the RunPod repository checkout:

```bash
cd /workspace/adom
export PYTHONPATH=/workspace/adom/src
python3 -c 'import torch, mmengine, mmcv, mmseg; print(torch.__version__, mmengine.__version__, mmcv.__version__, mmseg.__version__)'
```

If the immutable container provides an activation script, activate that exact
environment before these commands. Do not create a replacement environment
just to bypass a failed version audit.

Set the two real dataset package roots. `RELLIS_ROOT` must contain canonical
`splits/test.txt` plus `images/` and `masks/` (or its `manifest.csv`).
`KOREAN_ROOT` must be the validated target-adaptation superset containing
`manifest.csv`, `splits/ta1_train.txt`,
`splits/adom_val_diagnostic.txt`, and
`splits/adom_test_diagnostic.txt`. Do not guess these paths; locate their
`_SUCCESS`, package summary, and SHA evidence first.

```bash
export RELLIS_ROOT=<ABSOLUTE_VALIDATED_RELLIS_SEMANTIC20_ROOT>
export KOREAN_ROOT=<ABSOLUTE_VALIDATED_TARGET_ADAPTATION_ROOT>
export PAPER_EVAL_ROOT=/workspace/adom/paper_eval_outputs/$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$PAPER_EVAL_ROOT"
```

## 1. Build and audit manifests

This command hashes every image and annotation, reads every GT mask, reports
class support, and checks basename/hash/sequence/adjacent-frame leakage. The
frozen production counts are RELLIS test 899 and Korean train/val/test
133/21/61. A mismatch is a blocker unless it is first resolved as an authorized
new dataset decision.

```bash
python3 tools/paper_eval/build_manifests.py \
  --rellis-root "$RELLIS_ROOT" \
  --korean-root "$KOREAN_ROOT" \
  --output-dir "$PAPER_EVAL_ROOT"
```

The four ordered CSVs and their content hashes are now frozen under
`$PAPER_EVAL_ROOT/manifests/`. The common class set is computed as the
Semantic20 classes with GT pixels in both test manifests; it is never entered
manually.

Run the environment/checkpoint/config audit only after manifest construction:

```bash
python3 tools/paper_eval/audit_experiment.py \
  --repo /workspace/adom \
  --output-dir "$PAPER_EVAL_ROOT"
```

Inspect the audit before inference:

```bash
sed -n '1,240p' "$PAPER_EVAL_ROOT/audit_report.md"
jq '.status, .blockers, .common_supported_classes' "$PAPER_EVAL_ROOT/dataset_manifest_summary.json"
jq '.status, .b0_e0.selected_path, .eadom.selected_path, .shared_contract_sha256' "$PAPER_EVAL_ROOT/checkpoint_manifest.json"
```

Any `BLOCKED` status means stop. Do not add `--allow-count-mismatch`, change a
split, or choose a different checkpoint merely to make the audit pass.

## 2. Run the four direct evaluations

Resolve the exact audited paths rather than using a glob:

```bash
export B0_CKPT=$(jq -r '.b0_e0.selected_path' "$PAPER_EVAL_ROOT/checkpoint_manifest.json")
export EADOM_CKPT=$(jq -r '.eadom.selected_path' "$PAPER_EVAL_ROOT/checkpoint_manifest.json")
export PAPER_CONFIG=/workspace/adom/configs/adom/phase1_semantic20/segformer_b0_stage2_e0_rellis.py
```

```bash
python3 tools/paper_eval/evaluate_checkpoint.py \
  --audit-dir "$PAPER_EVAL_ROOT" --output-dir "$PAPER_EVAL_ROOT" \
  --dataset rellis --model b0_e0 \
  --manifest "$PAPER_EVAL_ROOT/manifests/rellis_test_manifest.csv" \
  --config "$PAPER_CONFIG" --checkpoint "$B0_CKPT"

python3 tools/paper_eval/evaluate_checkpoint.py \
  --audit-dir "$PAPER_EVAL_ROOT" --output-dir "$PAPER_EVAL_ROOT" \
  --dataset rellis --model eadom \
  --manifest "$PAPER_EVAL_ROOT/manifests/rellis_test_manifest.csv" \
  --config "$PAPER_CONFIG" --checkpoint "$EADOM_CKPT"

python3 tools/paper_eval/evaluate_checkpoint.py \
  --audit-dir "$PAPER_EVAL_ROOT" --output-dir "$PAPER_EVAL_ROOT" \
  --dataset korean --model b0_e0 \
  --manifest "$PAPER_EVAL_ROOT/manifests/korean_test_manifest.csv" \
  --config "$PAPER_CONFIG" --checkpoint "$B0_CKPT"

python3 tools/paper_eval/evaluate_checkpoint.py \
  --audit-dir "$PAPER_EVAL_ROOT" --output-dir "$PAPER_EVAL_ROOT" \
  --dataset korean --model eadom \
  --manifest "$PAPER_EVAL_ROOT/manifests/korean_test_manifest.csv" \
  --config "$PAPER_CONFIG" --checkpoint "$EADOM_CKPT"
```

Each process sets seed 42, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, deterministic
cuDNN behavior, `torch.use_deterministic_algorithms(True)`, `model.eval()`, and
`torch.inference_mode()`. TTA and random transforms are rejected. Predictions
on GT ignore pixels are excluded before the confusion matrix, so partially
labeled Korean masks do not create false positives outside labeled regions.

Runtime and memory are hardware measurements, not predeclared estimates. Each
summary records total wall time, mean/p95 inference time, and peak allocated
CUDA bytes under `runtime`. Record `nvidia-smi` sampling separately if total
process VRAM rather than PyTorch allocated VRAM is required.

## 3. Paired sequence-aware bootstrap

The evaluator stores a 19×19 confusion matrix for every ordered image. The
bootstrap groups and sums those matrices by sequence when every image has a
sequence ID. If any sequence is unknown, it falls back to image resampling and
records that limitation. Class intervals require at least two independent
positive units. All intervals use E-ADOM minus B0-E0 and at least 10,000 paired
resamples.

```bash
python3 tools/paper_eval/bootstrap_ci.py \
  --dataset rellis --output-dir "$PAPER_EVAL_ROOT" \
  --baseline "$PAPER_EVAL_ROOT/metrics/rellis__b0_e0__per_image_confusions.npz" \
  --eadom "$PAPER_EVAL_ROOT/metrics/rellis__eadom__per_image_confusions.npz" \
  --samples 10000 --seed 20260814

python3 tools/paper_eval/bootstrap_ci.py \
  --dataset korean --output-dir "$PAPER_EVAL_ROOT" \
  --baseline "$PAPER_EVAL_ROOT/metrics/korean__b0_e0__per_image_confusions.npz" \
  --eadom "$PAPER_EVAL_ROOT/metrics/korean__eadom__per_image_confusions.npz" \
  --samples 10000 --seed 20260814
```

## 4. Tables, report, and qualitative grids

```bash
python3 tools/paper_eval/compare_results.py --output-dir "$PAPER_EVAL_ROOT"
python3 tools/paper_eval/generate_qualitative_grid.py --output-dir "$PAPER_EVAL_ROOT"
```

`compare_results.py` refuses different ordered manifest hashes, GT support,
common class sets, or evaluation-contract hashes. The paper table values trace
to summary/per-class artifacts, while every CI traces to a paired bootstrap
JSON. `comparison_manifest.json` records hashes for those sources and outputs.

Expected output structure:

```text
paper_eval_outputs/<UTC timestamp>/
├── audit_report.md
├── environment.json
├── checkpoint_manifest.json
├── dataset_manifest_summary.json
├── manifests/
├── metrics/
├── predictions/
├── qualitative/{log,rubble,regression_cases,negative_cases}/
├── paper_table.{csv,md}
├── paired_deltas.{csv,md}
├── comparison_manifest.json
└── report.md
```

The output root is intentionally ignored by Git. Preserve it as a versioned
external artifact/archive with its own checksum; do not commit predictions,
confusion arrays, checkpoints, or dataset paths to the repository.

## Supplemental B0-E0 evaluation on all self-collected annotations

Because B0-E0 was trained without the Korean collection, the Korean train,
validation, and held-out test splits can all be used to diagnose its external
domain behavior. This supplemental analysis must not be presented as an
E-ADOM generalization test because E-ADOM was exposed to Korean training data.

Evaluate the audited train and validation manifests with the same checkpoint
and inference contract. The default `--manifest-split test` keeps the original
paper-evaluation behavior and artifact names unchanged.

```bash
python3 tools/paper_eval/evaluate_checkpoint.py \
  --audit-dir "$PAPER_EVAL_ROOT" --output-dir "$PAPER_EVAL_ROOT/supplemental/b0_self_collected" \
  --dataset korean --manifest-split train --model b0_e0 \
  --manifest "$PAPER_EVAL_ROOT/manifests/korean_train_manifest.csv" \
  --config "$PAPER_CONFIG" --checkpoint "$B0_CKPT"

python3 tools/paper_eval/evaluate_checkpoint.py \
  --audit-dir "$PAPER_EVAL_ROOT" --output-dir "$PAPER_EVAL_ROOT/supplemental/b0_self_collected" \
  --dataset korean --manifest-split val --model b0_e0 \
  --manifest "$PAPER_EVAL_ROOT/manifests/korean_val_manifest.csv" \
  --config "$PAPER_CONFIG" --checkpoint "$B0_CKPT"

python3 tools/paper_eval/summarize_self_collected.py \
  --paper-eval-root "$PAPER_EVAL_ROOT" \
  --supplemental-dir "$PAPER_EVAL_ROOT/supplemental/b0_self_collected"
```

The source package has 215 rows but 203 unique RGB images. Twelve train/val
duplicate RGBs contain conflicting `rubble` versus `log` labels on the same
pixels. The summarizer reports the duplicate-weighted and label-preference
sensitivity estimates, but uses the 191-image conflict-free union as its
primary whole-collection result. It also writes per-sequence metrics and the
full GT-to-prediction distribution.

Generate deterministic visual evidence for the source/target shift, the
partially successful `person` class, and the duplicate-label conflict:

```bash
python3 tools/paper_eval/generate_domain_shift_figure.py \
  --paper-eval-root "$PAPER_EVAL_ROOT" \
  --supplemental-dir "$PAPER_EVAL_ROOT/supplemental/b0_self_collected" \
  --output-dir "$PAPER_EVAL_ROOT/supplemental/b0_self_collected/figures/domain_shift_v1"
```

RELLIS `log`/`rubble` panels use the median per-image class IoU among
GT-positive images. Korean `log`/`rubble` panels use the held-out image with
the largest class GT area, and the `person` panel uses the median positive
training image. The largest conflicting duplicate group is selected without
looking at model quality. `selection_manifest.json` records the exact sample
IDs and selection rules so the figures are not hand-picked after inspection.
