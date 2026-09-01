# 0023. Avoid trigger distance tuning

- Status: Accepted
- Date: 2026-08-12
- Owners: ADOM team
- Supersedes: 0022의 초기 `avoid_trigger_distance_m` 값

## Context

0022는 직진 장애물의 AVOID 진입 거리를 현장 조정 가능한 초기값 3.50 m로 두었다.
평상시 직진을 더 오래 유지하고 가까운 장애물에만 회피 tree를 활성화할 필요가 있다.

## Decision

`avoid_trigger_distance_m`을 1.50 m로 조정한다. 0.30 m 초과 1.50 m 이하에서는 AVOID,
1.50 m보다 멀면 STRAIGHT, 0.30 m 이하에서는 BLOCKED를 유지한다.

## Rationale and evidence

사용자 현장 판단에 따른 파라미터 조정이다. 알고리즘 구조와 후보 수는 변경하지 않는다.

## Alternatives considered

- 3.50 m 유지: 회피 모드가 너무 일찍 활성화되므로 채택하지 않았다.

## Consequences

회피 tree 생성 구간이 짧아지고 단일 직진 경로를 사용하는 구간이 길어진다.

## Validation and rollback

사용자 요청에 따라 이 파라미터 변경의 검증은 생략한다. 필요하면 현장 관찰 후 값을
재조정한다.
