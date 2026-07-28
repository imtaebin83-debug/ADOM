# 0002. Initial Benchmark Scope

## Status

Superseded in part

The original hardware choice below was replaced by NVIDIA Jetson Orin Nano
8GB. The segmentation-first benchmark decision remains accepted.

## Context

착수 단계에서는 end-to-end 자율주행 demo보다 모델 성능과 온보드 가능성을 먼저 정량화해야 한다. 특히 오프로드 semantic segmentation에서 정확도와 runtime의 trade-off를 확인해야 이후 TensorRT 경량화, costmap 변환, ROS2 통합 방향을 결정할 수 있다.

## Decision

- 1차 목표는 segmentation benchmark까지로 제한한다.
- 첫 타깃 하드웨어는 Jetson Orin Nano 8GB로 둔다.
- ROS2 코드는 별도 repo로 분리하지 않고 이 mono repo 안의 `ros2_ws/`에서 관리한다.
- 필수 metric은 mIoU, class IoU, high_cost_or_obstacle recall, traversable precision, FPS, latency p50/p95, power draw, costmap update rate로 둔다.
- 실험 추적은 Markdown과 CSV로 시작한다.
- W&B나 MLflow는 실험 수가 늘어나고 자동 dashboard, artifact registry, remote run 비교가 필요해질 때 재검토한다.

## Rationale

Segmentation benchmark는 현재 연구 질문에 가장 직접적으로 답한다. Orin Nano 8GB의 제한된 연산력과 메모리 대역폭을 기준으로 정확도뿐 아니라 latency와 power를 함께 측정해야 실제 배치 가능성을 평가할 수 있다. Markdown/CSV는 초기 팀 규모에서 도입 비용이 낮고, git diff로 변경 이력을 보기 쉽다.

## Consequences

- 실험 문서는 `experiments/<name>/README.md`에 기록한다.
- 수치 결과는 `results/benchmark-results.csv`에 한 줄씩 누적한다.
- Benchmark 정의는 `docs/metrics/benchmark-protocol.md`를 기준으로 맞춘다.
- ROS2 integration은 나중에 진행하되, mono repo 안에서 `src/` 공통 로직을 재사용하도록 설계한다.
