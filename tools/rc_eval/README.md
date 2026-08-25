# ADOM RC Go/Stop evaluation logger

These tools are subscribe/read-only. They do not launch or modify the autonomy
or control nodes and never publish motor, servo, Go/Stop, or emergency-stop
commands. Every command that can start `ros2 bag record` is dry-run unless
`--execute-read-only --acknowledge-no-publish` is supplied.

The immutable `/opt/adom` training image does not contain the Jetson ROS graph,
launch files, deployment profiles, or a verified `ros2` installation. Therefore
`configs/rc_eval.yaml` intentionally leaves every topic and deployment command
unset. A human must run graph inspection on the actual Jetson and fill the
mapping from observed topic names and types. Do not infer them from this file.

## Prepare a balanced 40-trial plan

```bash
cd /opt/adom
python tools/rc_eval/create_trial_plan.py \
  --output rc_trials/<experiment_id>/trial_plan.csv \
  --repetitions 10 --seed 20260825
```

The four conditions are B0-E0/E-ADOM × hazard-present/hazard-absent. Their
order is randomized. Before use, set operator, exact scene/start/boundary
markers, `log` or `rubble`, checkpoint SHA, speed, and external-video path.

## Read-only graph discovery

Default dry-run:

```bash
python tools/rc_eval/inspect_ros_graph.py \
  --output-dir rc_trials/<experiment_id>/graph_audit
```

On the stationary Jetson with propulsion disabled, execute list commands only:

```bash
python tools/rc_eval/inspect_ros_graph.py \
  --output-dir rc_trials/<experiment_id>/graph_audit \
  --execute-read-only
```

Confirm topic types with `ros2 topic info --verbose` manually before editing
`configs/rc_eval.yaml`. The logger fails closed while any required topic is
unset.

## Record one trial

Create `metadata.json` from the schema, then preview:

```bash
python tools/rc_eval/start_trial_logger.py \
  --trial-dir rc_trials/<experiment_id>/trials/<trial_id> \
  --metadata /path/to/metadata.json \
  --config tools/rc_eval/configs/rc_eval.yaml
```

Only after the graph, emergency stop, wheels-off test, start marker, safety
boundary, and external video have been verified by people:

```bash
python tools/rc_eval/start_trial_logger.py \
  --trial-dir rc_trials/<experiment_id>/trials/<trial_id> \
  --metadata /path/to/metadata.json \
  --config tools/rc_eval/configs/rc_eval.yaml \
  --execute-read-only --acknowledge-no-publish
```

This starts only `ros2 bag record`. Start/stop of the car and existing control
nodes remains a separate human procedure. Stop the bag with:

```bash
python tools/rc_eval/stop_trial_logger.py \
  --trial-dir rc_trials/<experiment_id>/trials/<trial_id> \
  --execute-read-only
```

## Human annotation and analysis

```bash
python tools/rc_eval/annotate_trial.py \
  --trial-dir rc_trials/<experiment_id>/trials/<trial_id> \
  --hazard-detection-observed true \
  --stop-decision-observed true \
  --physical-stop-before-boundary true \
  --trial-completed true --emergency-intervention false

python tools/rc_eval/validate_trial.py \
  --experiment-root rc_trials/<experiment_id>

python tools/rc_eval/analyze_trials.py \
  --experiment-root rc_trials/<experiment_id> \
  --output-dir rc_trials/<experiment_id>/analysis

python tools/rc_eval/generate_paper_table.py \
  --analysis-dir rc_trials/<experiment_id>/analysis \
  --output-dir rc_trials/<experiment_id>/paper_outputs
```

Optional `--first-hazard-detection-s`, `--first-stop-command-s`, and
`--physical-stop-time-s` values are elapsed seconds from the trial logger start,
whose UTC timestamp is stored in `metadata.json`. Use `unknown` rather than
inventing an event that was not observable. Interrupted or emergency-intervention
trials are reported separately and excluded from TP/FN/FP/TN rates.

If a physical-stop label is available, TP/FN/FP/TN uses whether the vehicle
stopped before the predefined boundary. Otherwise the analyzer labels the
result explicitly as a stop-command proxy. Wilson 95% intervals do not remove
the correlation caused by repeating trials in the same scene.

## Mandatory human safety sequence

1. Assign a physical power-cut/emergency-stop operator.
2. Verify battery, steering neutral, command timeout, and emergency stop with
   propulsion disabled and wheels off the ground.
3. Confirm the exact model/checkpoint profile and that the logger publishes
   zero topics.
4. Freeze scene, start marker, obstacle marker, safety boundary, speed, and
   lighting notes before the first trial.
5. Record external video and rosbag; exclude interrupted or changed-condition
   trials with a reason rather than silently deleting them.
6. Begin low-speed physical trials only after the responsible human approves.
