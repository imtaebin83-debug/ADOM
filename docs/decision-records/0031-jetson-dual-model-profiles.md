# 0031. Jetson dual Semantic20 model profiles

- Status: Accepted
- Date: 2026-08-14
- Owners: ADOM team
- Supersedes: none

## Context

The Jetson `t4` entrypoint previously implied one B0-E0 checkpoint. The emergency
E-ADOM B0 run completed its locked canonical test and PyTorch-to-ONNX export checks.
Its validation/log and test/rubble metrics improved, but canonical-test overall,
RareRisk4, barrier and terrain results did not surpass B0-E0. Selecting a checkpoint
or config by directory glob without a profile identity could silently mix these two
models.

The current ROS perception implementation uses the MMSeg/PyTorch CUDA backend. The
export package is ready for target-side engine work, but connecting a TensorRT engine
to the ROS node is a separate runtime-interface change and is not required to hand
off the two audited checkpoints now.

## Decision

`t4` requires one explicit profile: `b0-e0` or `eadom`. Each profile has a distinct
runtime config, checkpoint directory and expected checkpoint SHA256. Missing,
ambiguous or hash-mismatched checkpoints fail before ROS launch. B0-E0 remains the
default/fallback deployment candidate; E-ADOM remains a field A/B candidate and does
not replace B0-E0 based on the canonical test.

The named profile owns its runtime config; an inherited `ADOM_MODEL_CONFIG` cannot
replace the padding-safe ROS config. After a checkpoint matches one of the two
reviewed canonical hashes, the launcher enables PyTorch 2.6+ compatibility for the
full MMEngine checkpoint metadata. A custom SHA override does not inherit that
automatic legacy-load permission.

Both profiles continue to use the current MMSeg/PyTorch CUDA ROS backend. Target
Jetson TensorRT engine generation, standalone ONNX-to-TensorRT parity and the native
TensorRT ROS backend connection are tracked separately in `docs/TODO.md`.

## Rationale and evidence

- Exact checkpoint identities prevent B0-E0/E-ADOM or ontology mix-ups.
- Config ownership prevents a stale static-export config from bypassing the ROS
  padding metadata fix.
- Hash-gating the PyTorch compatibility override limits unrestricted checkpoint
  metadata loading to the two team-reviewed artifacts.
- An explicit CLI profile makes the active model visible in terminal logs and allows
  repeatable A/B trials without editing source files.
- E-ADOM canonical TestSupported11 mIoU was 58.04 versus B0-E0 59.11. Log improved
  only 0.24 pp, while rubble improved 9.78 pp and barrier fell 14.71 pp.
- E-ADOM export image, metadata and PyTorch-to-ONNX parity passed; maximum absolute
  error was `4.1961669921875e-05` and pixel argmax agreement was
  `0.9999996609157986`.

## Alternatives considered

- Replace B0-E0 with E-ADOM: rejected because the locked test does not show broad or
  robust improvement.
- Put both checkpoints in one directory and select by filename: rejected because it
  makes an accidental model swap possible.
- Block hand-off until TensorRT ROS integration: rejected because the audited
  PyTorch profiles can already support controlled Jetson A/B testing.

## Consequences

Existing Jetson shortcuts must change from `t4` to `t4 b0-e0` or `t4 eadom`. Model
artifacts remain outside Git. A new intended checkpoint requires an explicit expected
SHA override until its profile contract is reviewed, and it must use PyTorch's default
loader or receive a separate explicit trust decision.

## Validation and rollback

Repository tests assert both config identities, directories and immutable SHA256
values, config override rejection and ordering of the hash-gated PyTorch compatibility
setting. On 2026-08-14, the target Jetson loaded the frozen E-ADOM checkpoint and
padding-safe runtime config, then returned a live perception status from one ZED
publisher and one perception subscriber. The observed sample was 10.45 average FPS
and 873.35 ms capture-to-output latency; it is not a controlled benchmark, and the
source/mask dimension record remains required. Run the same fixed recorded input and
compare saved evidence before model selection. Rollback is `t4 b0-e0`; do not remove
the hash check.
