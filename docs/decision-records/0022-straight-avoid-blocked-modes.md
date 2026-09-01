# 0022. Straight/avoid/blocked modes

- Status: Accepted
- Date: 2026-08-12
- Owners: ADOM team
- Supersedes: 0021의 상시 side-cost tree activation 부분

## Context

좌우 cost가 조금이라도 다르면 장애물이 없어도 25개 회피 tree가 항상 활성화됐다.
요구 동작은 기본 직진이며 실제 직진 경로에 장애물이 들어왔을 때만 회피하는 것이다.

## Decision

Semantic costmap은 registered depth가 투영된 결과이므로 직진 차량 corridor에서 첫 lethal
cell까지의 거리를 장애물 거리로 사용한다. 거리가 3.50 m보다 크면 STRAIGHT mode로
`[0, 0, 0]` 단일 경로만 평가한다. 0.30 m 초과 3.50 m 이하이면 AVOID mode로 전환해
0021의 좌우 전체 cost 비교 후 선택 방향의 25개 tree를 평가한다. 0.30 m 이하이면 즉시
BLOCKED다. 좌우 누적 cost가 같으면 왼쪽을 deterministic tie-break로 선택해 AVOID의
후보 수가 125개로 되돌아가지 않게 한다. `avoid_trigger_distance_m`은 현장 조정 가능한
초기 제안값이다.

## Rationale and evidence

장애물이 없을 때 costmap 비대칭이나 unknown 차이로 불필요하게 회전하는 것을 막는다.
동시에 이미 생성되는 depth-projected costmap을 사용하므로 별도 depth subscriber나 동기화
비용을 추가하지 않는다.

## Alternatives considered

- raw depth를 planner가 다시 구독: costmap과 중복 동기화 및 처리 경로가 생겨 기각했다.
- 항상 25개 tree: 기본 직진 요구와 맞지 않아 기각했다.

## Consequences

`rule_status`는 `planner_mode`, `straight_obstacle_distance_m`와 후보 수를 제공한다.
BLOCKED 해제에는 0040의 서로 다른 clear costmap 3개 debounce가 계속 적용된다.

## Validation and rollback

합성 costmap에서 먼 장애물 STRAIGHT 1개, 중거리 장애물 AVOID 25개, 0.30 m 이내
BLOCKED를 검사한다. 필요하면 `avoid_trigger_distance_m`을 고정 장면에서 조정한다.
