# 0008. Defer D-5 target selection until baseline field visualization

- Status: Accepted
- Date: 2026-08-07
- Owners: 태빈, 가형, 명섭
- Supersedes: the `log` default-target portion of 0006

## Context

0006 selected `log` as the default D-5 target from canonical RELLIS metrics and
collection practicality. The team has not yet visualized the deployed E0 B0
Semantic20 output on the actual candidate scenes. Choosing a fine-tuning target
before that evidence could optimize the wrong failure mode or produce a weak
military/off-road demonstration scenario.

## Decision

The target class is not yet frozen. Baseline ONNX and TensorRT inference must
preserve all 19 Semantic20 logits and the full argmax mask. The team will inspect
color masks, overlays, and per-class valid-image/ROI area ratios on actual candidate
scenes before selecting one target for annotation, fine-tuning, and Go/Stop.

Until the target, ROI, and threshold are frozen, inference and ROI processing run
in observation mode only and must not trigger autonomous STOP.

## Rationale and evidence

- Semantic20 E0 has materially different support and failure modes by class.
- Canonical validation/test lacks GT support for several classes, so a table alone
  cannot decide field suitability.
- The selected class must combine baseline failure, military/off-road relevance,
  safe reproducibility, resize survival, and controllable false stops.
- Preserving all logits avoids re-exporting the model merely to change a downstream
  target class.

## Alternatives considered

- Keep `log` frozen: rejected until field visualization confirms it is the best
  demonstration and fine-tuning target.
- Export only one target channel: rejected because it discards Semantic20 evidence
  and couples model export to an unsettled downstream policy.
- Change ontology: rejected; train IDs `0..18` and ignore `255` remain unchanged.

## Consequences

- Baseline parity reports every class by default and excludes padding from class-area
  and ROI calculations.
- Hand-off includes 1–3 color masks/overlays for qualitative target selection.
- Production annotation begins only after a separate target-freeze update records
  the class ID, scenario, ROI, threshold, and validation evidence.
- Go/Stop safety scope, manual reset, watchdog, RGB-only input, and Jetson TensorRT
  build policy remain unchanged.

## Validation and rollback

Validate the decision with representative field frames and record the chosen target
in `ADOM_CONTEXT.md` plus a follow-up decision record. If no candidate supports a
credible live stop scenario, keep the full Semantic20 visualization as evidence and
use the existing recorded-input fallback without silently selecting a class.
