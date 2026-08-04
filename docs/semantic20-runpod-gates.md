# Semantic20 RunPod gates

Use the processed Network Volume directory as `--dataset`, or omit that option
and set `ADOM_DATA_ROOT`. Never point the command at the Git-tracked preprocessing
recipe directories because they intentionally do not contain images and masks.

```bash
export WANDB_PROJECT=adom
export WANDB_RUN_GROUP=semantic20-e0-baseline
export WANDB_TAGS=runpod,a100,baseline

# B0 2-runner-iteration automatic micro-batch probe (16/1)
scripts/run_semantic20_cycle.sh --experiment e0 --models b0 --gate probe \
  --dataset /workspace/adom/datasets/processed/rellis3d_semantic20_v1 \
  --output /workspace/adom/runs/e0-b0-probe

# B0 50 optimizer-update smoke; W&B is forced disabled, validation is skipped
scripts/run_semantic20_cycle.sh --experiment e0 --models b0 --gate smoke \
  --dataset /workspace/adom/datasets/processed/rellis3d_semantic20_v1 \
  --output /workspace/adom/runs/e0-b0-smoke

# B0 500 optimizer-update mini-run; W&B stays online and validation runs at 500
scripts/run_semantic20_cycle.sh --experiment e0 --models b0 --gate mini \
  --dataset /workspace/adom/datasets/processed/rellis3d_semantic20_v1 \
  --output /workspace/adom/runs/e0-b0-mini

# Full B0: Stage 1 4k, Stage 2 40k, then canonical RELLIS test; no export
scripts/run_semantic20_cycle.sh --experiment e0 --models b0 --gate full \
  --dataset /workspace/adom/datasets/processed/rellis3d_semantic20_v1 \
  --output /workspace/adom/runs/e0-b0-full --skip-export
```

Repeat the same gates with `--models b2`. Its automatic probe tries only the
effective-batch-16 plans `16/1`, `8/2`, `4/4`, in that order. Use `--resume` with
the same output directory after interruption. To bypass a completed probe,
provide `--micro-batch 16`, `8`, or `4` (valid choices depend on the model).

For E1, use `--experiment e1` and the processed combined package root. The
runtime refuses E1 unless its train set includes all three sources and its
validation/test lists exactly equal canonical RELLIS validation/test.

Expected per-model full-run outputs include:

- `batch_plan.json` and `status.json` with commands/resume metadata
- Stage 1/2 TensorBoard event files and W&B local backup directory
- `last_checkpoint`, periodic `iter_*.pth`, and `best_mIoU_*.pth`
- backbone freeze/update audit JSON
- validation/test metric JSON and `confusion_matrix.json`
