# Contribution Guide

## Branching

- Use short, topic-based branches such as `feature/segformer-baseline` or `experiment/tensorrt-fp16`.
- Keep commits focused on one meaningful change.

## Folder Rules

- Put learning notes and one-off checks in `study/<member>/`.
- Put reproducible experiments in `experiments/<experiment-name>/`.
- Move reusable training, inference, mapping, or integration code into `src/`.
- Keep large files out of git. Use `data/` and `models/` only for README files, metadata, and path conventions.

## Experiment Checklist

Each experiment should include:

- goal and hypothesis
- dataset version and split
- model/config used
- metrics to compare
- hardware/runtime environment
- result summary and known failure cases
- one row in `results/benchmark-results.csv` when it produces benchmark numbers

## Documentation

- 작업 시작 전 루트 `ADOM_CONTEXT.md`와 `docs/status/README.md`를 읽는다.
- 범위, 인터페이스, 담당자, 성공 기준이 바뀌면 같은 PR에서
  `ADOM_CONTEXT.md`와 decision record를 함께 갱신한다.
- 프로젝트 진행률, blocker, 다음 hand-off는 `docs/status/`에 기록하고
  계획이나 계약을 중복 작성하지 않는다.
- Use `docs/meeting-notes/` for meeting logs.
- Use `docs/decision-records/` for decisions that affect repo structure, model choice, data policy, or deployment.
- AI 대화에서 나온 중요한 결론은 원문을 복사하지 않고
  `docs/ai-collaboration/`에 근거, 결정 후보, 미검증 사항, 후속 조치만 요약한다.
- Use `docs/system-architecture/` for diagrams and integration notes.
- Use `docs/setup-guides/` for reproducible setup instructions.

## Required Benchmark Metrics

For the first project phase, benchmark experiments should report:

- mIoU
- class IoU
- rare obstacle recall
- FPS
- latency p50/p95
- Jetson Orin Nano 8GB power draw
- costmap update rate, if the experiment touches mapping
