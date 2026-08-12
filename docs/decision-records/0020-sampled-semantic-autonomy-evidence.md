# 0020. Sampled semantic autonomy evidence

- Status: Accepted
- Date: 2026-08-12
- Owners: ADOM team
- Supersedes: 0016의 mask와 semantic costmap 제외 범위

## Context

0016은 Jetson에서 관측된 recorder latency 증가와 timeout 위험 때문에 live autonomy
bag을 수치·상태 중심으로 줄였다. 실제 bag 분석에서는 조향과 정지는 확인할 수 있었지만,
어떤 Semantic20 ID가 어느 픽셀에 있었고 그 결과 costmap이 어떻게 구성됐는지 재현할 수
없었다. Full-rate 640x384 mono8 mask는 10 Hz 기준 payload만 약 2.46 MB/s이므로 이전
구성을 그대로 복원하지 않고 bounded evidence가 필요하다.

## Decision

Live autonomy bag에 다음 semantic evidence를 추가한다.

- `/adom/perception/semantic20_mask_evidence`: full-rate mask와 같은 header/payload를
  갖는 기본 2 Hz sample
- `/adom/navigation/semantic_costmap`: planner가 받은 작은 `OccupancyGrid`
- `/adom/perception/status`: inference frame마다 Semantic20 ID `0..18` 순서의 픽셀 수와
  전체 mask 대비 비율, present ID와 ignore `255` 통계

Costmap용 `/adom/perception/semantic20_mask` full-rate 토픽, camera, confidence, overlay,
path, IMU와 TF는 계속 제외한다. Evidence mask rate는 perception parameter로 조절하며
`0.0`이면 비활성화한다.

## Rationale and evidence

640x384 mono8 2 Hz는 약 0.49 MB/s이고 80x60 int8 costmap 10 Hz는 payload 기준 약
0.05 MB/s다. 합계 약 0.54 MB/s, 33 MB/min으로 full-rate mask만 기록할 때의 약 1/5이다.
픽셀 통계는 이미 생성된 mask에 선형 `bincount`를 적용하며 별도 raster를 만들지 않는다.
이 구성은 class 발생 시간은 full inference rate status로, 공간 근거는 2 Hz mask와
costmap으로 확인하는 절충안이다.

## Alternatives considered

- Full-rate mask 기록: 시간 해상도는 가장 좋지만 이전 Jetson latency 위험을 다시 키운다.
- 상태 통계만 기록: 매우 가볍지만 픽셀 위치와 costmap 투영 결과를 검증할 수 없다.
- Confidence/overlay/path까지 기록: 분석 편의보다 serialization과 disk 부하가 크며 mask와
  grid에서 후처리하거나 planner status로 대체할 수 있다.

## Consequences

2 Hz mask 사이의 짧은 공간 변화는 놓칠 수 있다. 대신 class count/ratio는 매 inference
frame에 남는다. `present_class_ids`는 픽셀이 하나 이상 존재한다는 뜻이며 instance 수나
검출 threshold를 의미하지 않는다. 예상 bandwidth는 계산값이며 target Jetson 실측 전에는
성능 안전성을 검증된 사실로 간주하지 않는다.

## Validation and rollback

Jetson의 같은 고정 장면에서 recorder off/on을 각각 60초 이상 실행해 perception
`processing_ms`, `capture_to_perception_output_ms`, controller `source_to_command_ms`의
p50/p95, overwritten frame 증가율, costmap/planner/controller watchdog 횟수를 비교한다.
안전 기준을 침해하거나 유의미한 tail latency 증가가 생기면 먼저
`evidence_mask_fps=0.0`으로 mask sample을 끄고 costmap+status만 유지한다. 문제가 남으면
0016의 numeric/status-only regex로 롤백한다.
