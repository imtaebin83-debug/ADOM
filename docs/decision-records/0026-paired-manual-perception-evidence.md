# 0026. Paired manual perception evidence

- Status: Accepted
- Date: 2026-08-13
- Owners: ADOM team
- Supersedes: none

## Context

`t2 mask` preserves deployed Semantic20 output but has no source RGB, so a bag alone cannot show
whether a predicted boundary matches the scene or support later manual annotation. Full-rate camera
recording on the Jetson would add substantial DDS serialization and disk I/O while the B0-E0 model is
already processing only about 9 FPS in the observed field sessions. The intended experiment uses
`t0`, `t1`, `t2`, `t3`, `t4` without t5 and drives the vehicle manually.

## Decision

Add a `t2 evidence` profile. When a recorder subscribes, the perception node publishes the BGR frame
actually passed to inference on `/adom/perception/image_evidence`, sharing the evidence mask's header
and 2 Hz schedule. The profile records both that image and
`/adom/perception/semantic20_mask_evidence` in addition to the base status/control/GPS topics.

The existing profiles remain unchanged:

- `t2`: no raster evidence
- `t2 mask`: sampled Semantic20 mask only
- `t2 evidence`: paired sampled BGR image and Semantic20 mask

Full-rate camera, full-rate mask, confidence and overlay remain excluded. `record_evidence=true`
requires `record_mask=true` and fails parameter validation otherwise.

## Rationale

Pairing the exact inference input and output by timestamp enables RGB/mask overlays, ROI and component
analysis, manual GT annotation and E0 versus tuned-model fixed-scene comparison. Sampling bounds the
load, and omitting t5 removes costmap/planner work from this manual perception trial. Final `/drive`,
PWM and control mode still capture how the operator moved the vehicle.

## Consequences

Without t5 there will be no costmap/planner messages even though their names remain allowed by the
recorder regex. Manual driving does not measure autonomous decision quality. At 640x360 and configured
2 Hz, paired BGR+mask raw payload is approximately 1.84 MB/s before DDS/MCAP overhead or compression;
the actual rate, latency and thermal effect remain unverified on the target Jetson.

## Validation

Run a wheels-off smoke test with `t0`, `t1`, `t2 evidence`, `t3`, `t4`. Confirm that `ros2 bag info`
contains status, paired image/mask, mode, `/drive` and PWM. For every sampled pair verify equal header
stamp and dimensions, valid Semantic20 IDs, and mask bincount equality with the matching perception
status. Compare 60-second mask/evidence A/B trials after a 10-second warm-up, including processing
p50/p95/p99, overwritten fraction, actual sample rate, bag bandwidth, temperature and power.
