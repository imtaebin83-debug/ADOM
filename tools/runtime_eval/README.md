# ADOM runtime evaluation tools

Two read-only tools that turn existing artifacts into the paper's runtime
numbers. Neither one launches a node, publishes a topic, or writes into a bag.

| Tool | Input | Answers |
| --- | --- | --- |
| `build_runtime_table.py` | Jetson TensorRT validation reports | engine size, inference latency, FPS, per model |
| `measure_camera_latency.py` | recorded `.mcap` bags | camera-to-perception latency budget, frame drop |

## build_runtime_table.py

Every row is read out of a `semantic20-onnx-tensorrt-parity-v1` report written
by `scripts/validate_semantic20_tensorrt.sh`, so all rows share one definition
of "inference time" and "FPS" instead of being retyped from different runs.

```bash
python tools/runtime_eval/build_runtime_table.py \
  --report "B0-E0=/path/to/b0-e0/jetson-validation" \
  --report "B0-E-ADOM=/path/to/b0-eadom/jetson-validation" \
  --report "B2-E0=/path/to/b2-e0/jetson-validation" \
  --alias  "B2-E-ADOM=B2-E0" \
  --output-dir results/runtime
```

`--report NAME=PATH` takes either the report JSON or the directory containing
it. `--alias NAME=SOURCE` declares a model whose engine was not built as
sharing SOURCE's architecture: TensorRT latency is set by the graph, not by the
weight values, so the row is legitimate as long as the table says so. Aliased
rows are marked `(*)` and the footnote is generated for you.

Outputs `runtime_table.md`, `runtime_table.csv`, `runtime_rows.json`.

The tool refuses to produce a silently wrong comparison. It warns when rows
were measured at different input shapes, different TensorRT versions, different
platforms or different iteration counts, when a parity status is not `PASS`, and
when two rows point at the same engine SHA-256.

The markdown ends with a measurement-conditions block. The fields the report
cannot know are left as **FILL IN**: the physical Jetson model, JetPack/L4T,
`nvpmodel` power mode, and cooling. Fill them before the table goes in a paper.

## measure_camera_latency.py

```bash
pip install mcap

python tools/runtime_eval/measure_camera_latency.py \
  --bags-root /path/to/downloaded/bags \
  --output-dir results/camera_latency \
  --dump-class-ratios
```

`--bags-root` is scanned recursively, so point it at the folder holding the
`autonomy_*` directories. A `<bag>/rosbag/*.mcap` layout is grouped under the
bag name, and `<bag>/session.json` is picked up when present.

Two independent sources are read, and they cross-check each other.

1. **The perception node's status topic.** The deployed Jetson node publishes a
   JSON object per frame with the whole budget: `capture_to_receive_ms`,
   `queue_wait_ms`, `capture_to_inference_start_ms`, `inference_ms`,
   `processing_ms`, `capture_to_perception_output_ms`, plus `average_fps` and
   the `received_frames` / `overwritten_frames` counters that give the
   latest-frame drop rate. Every numeric field is summarized, so this keeps
   working if the node gains or renames fields.

2. **Header propagation.** For any topic whose message starts with a
   `std_msgs/Header`, `publish_time - header.stamp` is the age of the camera
   frame when that message was published. This needs no cooperation from the
   node, so it is an independent check on source 1.

If a publisher re-stamps its output with the current time instead of
propagating the camera header, the header-based value collapses to about zero.
The summary flags that explicitly rather than reporting a meaningless number —
`/adom/navigation/semantic_costmap` and `/drive` do exactly this, which is why
only the mask topic is a valid cross-check.

Outputs `summary.md`, `summary.json`, `per_bag_status.csv`,
`per_bag_header_latency.csv`, and `per_bag_class_ratios.csv`. With
`--dump-class-ratios` it also writes `class_ratios.csv`, the full per-class
pixel ratio time series.

`per_bag_class_ratios.csv` is the mean predicted pixel ratio per class for each
bag. It helps narrow down which checkpoint a bag was recorded with when the
session metadata does not record it: a model that never predicts a class leaves
that column at zero for the whole bag. Treat it as evidence for a human to
confirm, not as provenance.

By default `summary.md` shows only the latency and rate fields. `--all-fields`
renders every numeric status field; the CSV and JSON always contain all of them.

## Reporting rule

Record whatever these tools produce the way
[docs/metrics/benchmark-protocol.md](../../docs/metrics/benchmark-protocol.md)
requires: the run summary in the relevant `experiments/` note, and the row in
[results/benchmark-results.csv](../../results/benchmark-results.csv). Keep the
engine SHA-256 with the numbers so a row can be traced back to one engine.
