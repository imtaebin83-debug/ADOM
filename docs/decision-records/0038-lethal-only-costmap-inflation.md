# 0038. Lethal-only semantic costmap inflation

- Status: Accepted
- Date: 2026-08-12
- Owners: ADOM autonomy integration
- Supersedes: none

## Context

기존 semantic costmap은 `inflation_min_cost: 60` 이상인 모든 cell을 inflation seed로
사용했다. Inflation 중심의 계산 비용은 100이므로 bush, puddle, mud처럼 원래 60~89인
감속용 semantic cost도 hard-stop cell로 승격됐다. 실차에서 planner가 약 0.4 m 앞을
lethal로 판단하고 빈 path를 발행하는 동안 이 승격 동작이 원인 구분을 어렵게 했다.

## Decision

`inflation_seed_cost`를 별도 파라미터로 추가하고 기본값을 planner의 lethal 기준과 같은
90으로 둔다. Cost 90 이상인 cell과 geometric obstacle만 inflation을 시작한다.
`inflation_min_cost: 60`은 주변 inflation ring의 최소 감속 비용으로 계속 사용한다.

Rule planner는 `stopped`, `blocked`, `driving` 상태 또는 stop reason이 바뀔 때만 상태,
clearance, path point 수와 명령 속도를 콘솔에 기록한다. 기존 상태 topic은 유지한다.

## Rationale and evidence

Semantic cost 60~89는 경로 score와 감속에 반영할 위험 evidence이지 그 자체로 lethal을
의미하지 않는다. Seed 기준과 주변 최소 비용을 분리하면 실제 lethal cell의 정지 buffer를
유지하면서 non-lethal semantic 관측의 의도하지 않은 hard stop을 제거할 수 있다.
상태 전환 로그는 20 Hz 반복 출력을 피하면서 현장에서 fail-safe 원인을 바로 확인하게 한다.

## Alternatives considered

- `inflation_min_cost`를 90으로 변경: seed와 주변 ring의 최소 비용이 다시 결합돼 주변도
  lethal이 되므로 채택하지 않는다.
- Planner의 `lethal_cost` 또는 `stop_distance_m` 완화: 잘못된 cost 승격의 원인을
  가리고 실제 장애물 안전 여유도 함께 줄이므로 채택하지 않는다.
- Inflation 비활성화: 차량 footprint 주변의 회피·감속 여유를 잃으므로 채택하지 않는다.

## Consequences

- Cost 60~89인 cell은 원래 비용을 유지하며 inflation seed가 되지 않는다.
- Cost 90 이상과 높이 기반 geometric obstacle은 계속 lethal seed가 된다.
- 주변 inflation ring은 60~89 비용으로 planner score와 감속에 반영된다.
- Planner의 상태 전환은 콘솔에서도 보이지만 planning/control 출력 계약은 바뀌지 않는다.

## Validation and rollback

단위 테스트에서 non-lethal cost 85가 100으로 승격되지 않고 lethal cost 100은 주변을
inflation하는지 확인한다. Jetson에서는 빈 장면, 정적 장애물, wheels-off 순서로
`rule_status`, `local_path_status`, `/cmd_vel`과 `/drive`를 확인한다. 실제 lethal
장애물의 정지 buffer가 약해지면 `inflation_seed_cost`를 조정하거나 변경을 되돌린다.
