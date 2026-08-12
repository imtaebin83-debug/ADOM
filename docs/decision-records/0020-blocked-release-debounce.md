# 0020. BLOCKED release debounce

- Status: Accepted
- Date: 2026-08-12
- Owners: ADOM team
- Supersedes: none

## Context

Sparse semantic costmap의 lethal 관측이 frame마다 나타나고 사라져 planner가 약 0.1초
간격으로 BLOCKED와 DRIVING을 반복하는 현장이 확인됐다. 기존 gap hysteresis는 좌우
선택만 유지하며 BLOCKED 해제에는 적용되지 않았다.

## Decision

위험 또는 no-gap 판정은 기존처럼 즉시 BLOCKED로 적용한다. BLOCKED 이후에는 서로 다른
유효 costmap 3개에서 연속으로 안전 경로가 확인돼야 DRIVING으로 복귀한다. 50 Hz planner
timer가 동일 costmap을 반복 처리한 횟수는 clear frame으로 세지 않는다. empty costmap과
watchdog은 계속 즉시 정지한다.

## Rationale and evidence

정지 반응은 늦추지 않으면서 단일 clear frame으로 재출발하는 진동을 제거한다. 설정값
3은 현장 안정화를 위한 초기값이며 장시간 실차 검증 전 확정값으로 간주하지 않는다.

## Alternatives considered

- BLOCKED 진입 debounce: 장애물 정지를 지연하므로 채택하지 않았다.
- 시간 기반 debounce: costmap 주기가 불규칙할 때 실제 관측 수를 보장하지 못한다.

## Consequences

약 10 Hz costmap 기준 재출발이 약 0.3초 지연된다. 안전 정지, watchdog과 actuator
timeout 계약은 유지된다.

## Validation and rollback

`rule_status`의 `blocked_release_clear_count`로 연속 clear 수를 확인한다. 불필요한 재출발
지연이 크면 `blocked_release_clear_frames`를 조정하되 1 미만은 허용하지 않는다.
