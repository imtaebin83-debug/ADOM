# 0019. Gap-guided direction tree

- Status: Accepted
- Date: 2026-08-12
- Owners: ADOM team
- Supersedes: 0035의 중앙 장애물 방향 선택 부분

## Context

기존 planner는 5개 조향 action을 3단계로 전개한 125개 경로 중 순간 cost가 가장 낮은
경로를 선택했다. 좌·우 free-space 연결성을 먼저 비교하지 않아 근거리 score는 낮지만
안쪽이 좁거나 막힌 방향으로 회피할 수 있다. 사용자는 중앙 장애물 기준 좌·우 gap을
분석하되 planner 연산 P95가 기존보다 50 ms 이상 증가하면 변경을 기각하도록 승인했다.

## Decision

Semantic20 costmap의 전방 120도 영역을 41개 ray로 표본화한다. 차량 corridor 폭으로
각 ray를 검사하고 중앙 24도 안에서 3.5 m 이내 lethal obstacle이 발견되면 좌·우의
연속 free ray 폭, 전방 깊이와 unknown 비율을 평가한다. 최소 유효 gap 폭은 초기 제안값
0.45 m이며 실차 검증 전 확정 물리값으로 간주하지 않는다.

더 높은 gap score의 방향을 선택하고 gap 중심각에 가장 가까운 첫 steering action을
고정한다. 나머지 두 tree level만 전개해 25개 Ackermann path를 기존 risk, clearance,
steering score로 평가한다. 이전 방향과 반대 방향의 차이가 switch margin 이내이면 기존
방향을 유지한다. 양쪽 모두 최소 gap 폭을 만족하지 못하면 정지한다. 중앙 장애물이
없으면 기존 125개 tree와 직진 선호를 유지한다.

## Rationale and evidence

gap layer는 빠져나갈 좌·우 방향을 담당하고 tree는 선택 방향 안의 차량 가능 경로를
담당한다. `experiments/benchmark_gap_planner.py`로 80x60 costmap과 production tree
설정을 비교한 결과, P95 추가 비용 최댓값은 clear scene의 0.515 ms였다. 좌·우 회피
scene은 후보가 125개에서 25개로 줄어 P95가 각각 11.810 ms, 11.733 ms 감소했다. 이는
사용자가 정한 50 ms 기각 기준을 통과한다. 이 수치는 개발 호스트 측정이며 Jetson
실측값으로 간주하지 않는다.

## Alternatives considered

- 125개 tree score에 좌·우 bias만 추가: 계산량은 유지되지만 방향 결정을 명시적으로
  고정하지 않아 cycle별 좌우 전환을 충분히 막지 못한다.
- tree를 Follow-the-Gap 단일 목표점으로 대체: 가볍지만 Ackermann 가능 경로와 회피 후
  복귀 방향을 평가하는 기존 장점을 잃는다.
- 좌·우 선택 후 같은 쪽 두 첫 action을 유지한 50개 tree: 가능하지만 gap 중심각이 이미
  첫 조향 목표를 제공하므로 불필요한 후보를 남긴다.

## Consequences

`/adom/navigation/rule_status`에 장애물 감지, 선택 방향/각도, 좌·우 gap 폭·깊이와 실제
tree 후보 수가 추가된다. bag topic은 늘지 않는다. 2D local 관측 범위 밖의 막다른 길은
판단할 수 없으며, unknown 감점과 gap 파라미터는 고정 장면 및 저속 실차 시험이 필요하다.
watchdog, no-gap STOP, manual mode와 actuator timeout은 유지한다.

## Validation and rollback

합성 costmap에서 좌측 우세, 우측 우세, 양쪽 폐쇄, clear scene과 25개 후보 제한을
검증한다. 재현 성능 검사는 `PYTHONPATH=src python experiments/benchmark_gap_planner.py`로
실행하며 P95 증가가 50 ms 이상이면 `gap_enabled: false`로 rollback한다. 이후 shadow
mode, wheels-off, 저속 실차 순서로 방향 부호와 hysteresis를 확인한다.
