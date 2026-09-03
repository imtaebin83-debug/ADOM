# Jetson onboard runtime, B0-E0 and E-ADOM

- Status: **Measured for B0 family. B2 family not measured.**
- Date: 2026-08-30
- Owner: 명섭
- Repo commit: `907df66`
- Method: offline recovery from recorded ROS 2 bags, no live vehicle
- Tool: [`tools/runtime_eval/measure_camera_latency.py`](../../tools/runtime_eval/measure_camera_latency.py)

## 1. What was measured and how

The deployed perception node publishes a JSON status object per frame on
`/adom/perception/status` carrying its own latency budget. That topic is
recorded in every `autonomy_*` bag, so the whole runtime section is recoverable
from the bags without touching the Jetson.

```bash
python tools/runtime_eval/measure_camera_latency.py \
  --bags-root <bag root> \
  --output-dir <output> \
  --include "autonomy_20260814_*"
```

Field meanings, as published by the node:

| Field | Meaning |
| --- | --- |
| `capture_to_receive_ms` | ZED capture stamp to the frame arriving in the node |
| `queue_wait_ms` | waiting in the latest-frame queue before inference started |
| `inference_ms` | TensorRT engine execution |
| `processing_ms` | inference plus pre/post processing |
| `capture_to_perception_output_ms` | ZED capture stamp to the perception output being published |

## 2. Data selection

56 bag directories were downloaded; 53 were readable. Four bags are corrupt in
transfer and were excluded: `20260812_205016`, `20260812_211419`,
`20260814_152451`, `20260814_154944` (`RecordLengthLimitExceeded` from the MCAP
reader).

**The bags are not one system.** Per-bag inference latency separates by
recording date:

| Date | Bags | `inference_ms` p50 | `average_fps` p50 |
| --- | ---: | ---: | ---: |
| 2026-08-12 | 5 | 87 - 94 | 9.0 - 10.1 |
| 2026-08-13 | 18 | 67 - 107 | 1.6 - 10.7 |
| 2026-08-14 | 16 | 47 - 57 | 12.5 - 16.7 |

Inference roughly halved on 08-14. The cause is not recorded in the bags; the
likely explanation is an engine or power-mode change, and it must be confirmed
with whoever operated the platform before this is described in a paper. **Only
2026-08-14 bags are used below.** Mixing the dates produces an average that
describes no configuration that ever ran, which is what the earlier
`11.34 FPS` / `203.32 ms` figures appear to be.

## 3. Model attribution (evidence, not provenance)

`session.json` does not record which checkpoint was loaded, so bags were
grouped by the class ratios the model itself predicted
(`per_bag_class_ratios.csv`). B0-E0 produces zero `log` and `rubble`
true positives on Korean scenes and pushes those pixels into `puddle`/`mud`;
E-ADOM recovers them.

| Group | Bags (2026-08-14) | mean `log` | mean `rubble` | mean `puddle` | mean `mud` |
| --- | --- | ---: | ---: | ---: | ---: |
| B0-E0 candidates | 132611, 132618, 132643, 132715, 133157, 133238, 133413, 133515 | 0.00% | 0.00 - 0.03% | 2.4 - 23.5% | 9.4 - 34.5% |
| E-ADOM candidates | 133747, 133802, 134130, 134357, 134547, 151012, 160233 | 2.8 - 47.9% | 1.3 - 8.4% | 0.0 - 0.9% | 0.1 - 1.9% |

The split is clean and the changeover falls between 13:35 and 13:37 on 08-14.
`133551` sits on the boundary (`log` 2.89% but `puddle` 11.77%) and is
**excluded from both groups** rather than assigned.

This is evidence, not provenance. A human who was there must confirm the
assignment before it is stated as fact in the paper.

## 4. Results

### 4.1 Latency budget, 2026-08-14, milliseconds

| Stage | B0-E0 mean | p50 | p95 | E-ADOM mean | p50 | p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| capture to receive | 48.86 | 50.68 | 54.92 | 49.36 | 51.52 | 55.88 |
| queue wait | 10.69 | 6.68 | 37.90 | 10.49 | 7.07 | 39.96 |
| inference (TensorRT) | 48.30 | 47.85 | 54.56 | 50.13 | 48.94 | 61.31 |
| processing (infer + pre/post) | 55.56 | 53.99 | 71.49 | 56.31 | 54.99 | 68.16 |
| **capture to perception output** | **115.06** | **117.48** | **143.37** | **116.10** | **118.60** | **146.35** |

- p99 of capture-to-output: B0-E0 155.25 ms, E-ADOM 161.78 ms.
- Samples: B0-E0 n = 6,759 over 8 bags; E-ADOM n = 16,641 over 7 bags.
- The budget closes: capture-to-receive + queue wait + processing equals
  capture-to-output within 1 ms for both groups.

### 4.2 Throughput and frame handling

| Metric | B0-E0 | E-ADOM |
| --- | ---: | ---: |
| `average_fps` mean / p50 | 16.51 / 16.64 | 14.67 / 15.12 |
| latest-frame drop rate | 21.8% | 19.7% |
| `target_fps` | 30.0 | 30.0 |
| `evidence_mask_fps` (recording throttle) | 2.0 | 2.0 |

The drop rate is the fraction of arriving frames the node overwrote without
running inference (`overwritten_frames` / `received_frames`). It is a property
of the latest-frame policy, which deliberately discards stale frames rather
than queueing them.

### 4.3 Independent cross-check

`/adom/perception/semantic20_mask_evidence` carries the camera header, so
`publish_time - header.stamp` measures the same quantity without the node's
cooperation. Across all 53 bags, 3,853 samples: mean 121.30 ms, p50 119.54 ms,
p95 163.89 ms — agreeing with the node's own report to within about 6 ms.

`/adom/navigation/semantic_costmap` and `/drive` re-stamp their output with the
current time, so their header-derived values are near zero and are **not**
camera-to-perception latency. The tool flags this automatically.

## 5. The runtime table for the paper

| Model | Engine (MB) | Inference p50 / p95 (ms) | Camera to perception p50 / p95 (ms) | FPS |
| --- | ---: | ---: | ---: | ---: |
| B0-E0 | 9.24 (unverified, see below) | 47.85 / 54.56 | 117.48 / 143.37 | 16.6 |
| B0-E-ADOM | 9.24 (unverified) | 48.94 / 61.31 | 118.60 / 146.35 | 15.1 |
| B2-E0 | not measured | not measured | not measured | not measured |
| B2-E-ADOM | not measured | not measured | not measured | not measured |

Measurement conditions: NVIDIA Jetson Orin Nano Super (8GB), TensorRT FP16,
static `1x3x384x640`, ZED 2i RGB, recorded 2026-08-14 on the deployed vehicle.

**Still to fill in, and none of it can be recovered from the bags:**

- JetPack / L4T / TensorRT version, `nvpmodel` power mode, active cooling.
  The board is confirmed as a Jetson Orin Nano Super 8GB, so the 08-14 speedup
  may be a switch into the Super (MAXN) power profile. `nvpmodel -q --verbose`
  settles it; do not state it as the cause until then.
- Engine size and SHA-256. The 9.24 MB figure is carried over from an earlier
  note and has no artifact behind it in this repo.

## 6. Claim this supports

Inference latency differs between B0-E0 and E-ADOM by 1.09 ms at p50 (2.3%),
and camera-to-perception by 1.12 ms, on 23,400 measured frames. The two models
share an architecture, input shape and precision, so this is the expected
result and it is now measured rather than assumed:

> Targeted domain adaptation changed the model's predictions without changing
> its onboard cost.

## 7. Limitations

- Single hardware unit, single session date, no repeat across power modes.
- Model attribution is inferred from predicted class ratios, not from recorded
  checkpoint provenance. See section 3.
- `average_fps` as published is a running average since node start, so short
  bags report a value that is still converging.
- The 08-12 and 08-13 recordings are excluded rather than explained. Whatever
  changed on 08-14 is undocumented.
- Frame drop rate depends on scene and on the ZED publish rate, not only on the
  model.
- No power draw, SoC temperature or thermal-throttling measurement. The bags do
  not carry them and the protocol in
  [`docs/metrics/benchmark-protocol.md`](../../docs/metrics/benchmark-protocol.md)
  requires power for a complete edge benchmark.

## 8. To complete the B2 rows

Runtime is set by the graph, not by the weight values, which section 6
demonstrates empirically for the B0 pair. So one B2 engine fills both B2 rows.

1. From 태빈: a hand-off package built from the **existing B2-E0 checkpoint**.
   B2-E-ADOM training does not need to finish first.
2. On the Jetson, by whoever is at the platform:

   ```bash
   ADOM_TRT_WORKSPACE_MIB=4096 scripts/build_semantic20_tensorrt.sh \
     <pkg>/model_static_1x3x384x640_fp32.onnx \
     <pkg>/model_static_1x3x384x640_fp16.engine

   scripts/validate_semantic20_tensorrt.sh <pkg> <pkg>/jetson-validation
   ```

   Send back the report JSON, the engine SHA-256 the build script prints, and
   `nvpmodel -q --verbose`.
3. Then, on any machine:

   ```bash
   python tools/runtime_eval/build_runtime_table.py \
     --report "B0-E0=<b0 report>" \
     --report "B0-E-ADOM=<b0-eadom report>" \
     --report "B2-E0=<b2 report>" \
     --alias  "B2-E-ADOM=B2-E0" \
     --output-dir results/runtime
   ```

If the B2 engine never arrives, report the two measured rows and say the B2
family was not deployed to the target platform before submission. A build
failure or a rate below the control-loop floor is itself a result worth one
sentence.
