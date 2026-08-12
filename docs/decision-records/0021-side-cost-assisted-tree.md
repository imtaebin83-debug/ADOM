# 0021. Side-cost-assisted direction tree

- Status: Accepted
- Date: 2026-08-12
- Owners: ADOM team
- Supersedes: 0019와 0020의 no-gap BLOCKED 입력 부분; 0020 release debounce는 유지

## Context

0019의 ray 기반 gap 폭·거리 판정은 sparse costmap에서 양쪽을 통과 불가로 해석해 기존
tree와 별개의 BLOCKED 조건을 만들었다. 요구사항은 gap을 안전 상태 판정기가 아니라
tree 방향 선택을 돕는 단순한 수단으로 제한하는 것이다.

## Decision

Robot-frame costmap 전체를 lateral center 기준 좌우 절반으로 나눈다. 관측 cell은 저장된
cost를, unknown cell은 tree와 동일한 `unknown_cost`를 사용해 각 절반의 총 cost를
계산한다. 총 cost가 낮은 쪽의 최대 첫 조향을 고정하고 남은 두 단계 25개 tree를 기존
score로 평가한다. 좌우 총 cost가 같으면 방향을 제한하지 않고 기존 125개 tree와 직진
선호를 유지한다.

좌우 보조 결과는 BLOCKED를 직접 만들지 않는다. BLOCKED는 6a4db6b 이전과 동일하게
선택된 최저-score tree path의 첫 lethal clearance가 `stop_distance_m` 이하일 때만
발생한다. watchdog과 empty costmap STOP, 0020의 3-frame clear release debounce는
유지한다.

## Rationale and evidence

거리·ray 수·최소 gap 폭 파라미터를 제거해 sparse 관측에 민감한 별도 통과 가능성 판정을
없앤다. 좌우 전체 cost는 이미 존재하는 semantic/geometric 위험도를 이용하면서 tree
후보를 25개로 줄이는 역할만 한다.

## Alternatives considered

- ray 기반 gap 폭 유지: sparse costmap에서 추가 BLOCKED를 발생시켜 기각했다.
- 좌우 cost로 BLOCKED 결정: tree의 실제 차량 가능 경로 검사를 우회하므로 기각했다.
- 항상 한쪽을 선택: 대칭 clear scene에서 불필요한 회전을 만들므로 동률에는 125개를
  유지한다.

## Consequences

`rule_status`는 좌우 총 cost, 선택 방향과 tree 후보 수를 제공한다. 좌우 누적 cost는
전역 목적지나 관측 범위 밖 막다른 길을 보장하지 않으며, 선택된 25개 tree의 기존
clearance/score가 최종 경로와 BLOCKED를 결정한다.

## Validation and rollback

합성 costmap에서 좌우 cost 우세 방향, 대칭 scene 125개, 선택 방향 25개와 기존
clearance BLOCKED 조건을 검사한다. 문제가 있으면 `side_cost_enabled: false`로 기존
125개 tree만 사용한다.
