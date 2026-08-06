# ADOM Agent Guide

## Authority and decisions

- Read `ADOM_CONTEXT.md` before planning work. It is the current source of truth; record unverified values as proposals, not facts.
- Preserve decision records. Do not delete or rewrite their historical decisions. When an authorized change alters scope, an interface, ownership, or success criteria, update the root context and add a decision record with the date, decision, and reason.

## Current D-5 scope

- The five-day PoC uses the Semantic20 policy: RELLIS + RUGD + YCOR + newly collected, labeled data. Keep Semantic20 IDs `0..18` and `255` ignore unless an explicit decision changes the ontology.
- Keep Semantic20 and Cost4/Cost5 mappings, splits, configs, and artifacts separate. Do not silently substitute a mapping, split, checkpoint, or dataset version.
- The live PoC is a low-speed Go/Stop safety demonstration, not autonomous navigation. Preserve the watchdog/neutral-on-timeout behavior, require manual reset after STOP, and do not invent unverified ROS topic, QoS, or hardware values.
- ROS integration targets ROS 2 Jazzy. For target hardware work, build TensorRT engines on the target Jetson, not on RunPod.

## Repository boundaries

- Put one-off notes and checks in `study/`, reproducible experiments in `experiments/`, reusable logic in `src/`, and ROS adapters/launch/config in `ros2_ws/`.
- Do not commit datasets, generated labels, checkpoints, ONNX/TensorRT artifacts, ROS bags, run outputs, logs, secrets, or personal absolute paths. Keep source mappings canonical rather than duplicated.
- Preserve existing user changes and untracked files. Make the smallest scoped change; do not stage, revert, delete, commit, or push unrelated work.

## Verification

- For Python/config/data changes, run the relevant tests and `python scripts/check_git_artifacts.py`. The CI baseline is `python -m unittest discover -s tests -v`.
- Report the commands run, their results, and any checks not run. For Docker or training-image changes, keep the immutable Git-SHA image and `/opt/adom` code versus `/workspace` data/output boundary intact.

## Pull requests

- Write every PR body bilingually in Korean and English. For each major section,
  provide the Korean explanation first and the corresponding English explanation
  immediately after it; do not maintain two PRs or two conflicting descriptions.
- When the requested work is complete and required checks pass, open or convert the
  PR to Ready for review so it can be merged. Use Draft only when the user explicitly
  requests it or when required work/checks remain incomplete, and state the reason.
- Keep the PR body current when follow-up commits change validation results, known
  limitations, or merge readiness.
