# Contribution Guide

## Branching

- Use short, topic-based branches such as `feature/segformer-baseline` or `experiment/tensorrt-fp16`.
- Keep commits focused on one meaningful change.

## Folder Rules

- Keep reusable training, inference, mapping, or integration code in `src/`.
- Keep ROS2 nodes in `ros2_ws/` as thin adapters over `src/` logic.
- Keep offline paper/RC evaluation scripts in `tools/`.
- Add or update `tests/` whenever a data, evaluation, or runtime contract changes.
- Keep large files out of git. `data/`는 split과 manifest만, `models/`는 런타임에 생성되는 checkpoint/export 배치 경로일 뿐이며 실제 파일은 커밋하지 않는다. 배치 규칙은 [`docs/setup-guides/jetson-model-checkpoint-handoff.md`](../docs/setup-guides/jetson-model-checkpoint-handoff.md)를 따른다.

## 외부 오픈소스 사용 규칙

- 전체 upstream repo를 그대로 복사해서 커밋하지 않는다.
- 수정이 필요한 외부 프로젝트는 개인 fork에서 작업한다.
- 빌드나 실행에 직접 필요해졌을 때만 submodule 또는 명확한 install guide로 연결한다.
- 외부 코드를 추가할 때는 upstream URL, license, commit hash 또는 release version, local modifications, 필요한 이유를 PR 본문과 관련 문서에 함께 남긴다.

## Experiment Checklist

Each experiment should include:

- goal and hypothesis
- dataset version and split
- model/config used
- metrics to compare
- hardware/runtime environment
- result summary and known failure cases

## Documentation

- 인터페이스나 데이터 계약이 바뀌면 같은 PR에서 관련 문서를 함께 갱신한다.
- Use `docs/decision-records/` for decisions that affect repo structure, model choice, data policy, or deployment. 기존 record는 지우거나 다시 쓰지 않고 새 record에서 `Supersedes` 관계를 남긴다.
- Use `docs/system-architecture/` for diagrams and integration notes.
- Use `docs/setup-guides/` for reproducible setup instructions.
- Use `docs/metrics/` for benchmark and metric definitions.

## Required Benchmark Metrics

For the first project phase, benchmark experiments should report:

- mIoU
- class IoU
- rare obstacle recall
- FPS
- latency p50/p95
- Jetson Orin Nano 8GB power draw
- costmap update rate, if the experiment touches mapping
