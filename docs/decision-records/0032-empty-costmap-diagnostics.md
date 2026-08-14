# 0032. Empty costmap projection diagnostics

- Status: Accepted
- Date: 2026-08-14
- Owners: ADOM team
- Supersedes: none

## Context

실차에서 정상 costmap과 모든 cell이 unknown인 empty costmap이 간헐적으로 교차했지만,
기존 상태의 `projected_points`와 `observed_cells`만으로는 depth 유효성, Semantic20 label,
TF 이후 높이 필터와 grid 경계 중 관측이 사라진 단계를 구분할 수 없었다. 근거 없이
costmap 크기나 안전 기준을 변경하면 원인을 가리거나 blind motion을 허용할 수 있다.

## Decision

투영 함수는 sampled pixel, finite/in-range depth, valid semantic label, depth-label 결합,
높이 필터 통과 point와 변환 후 Z 범위를 진단값으로 제공한다. Costmap ROS 상태에는 이
값들과 grid 내부 point 수를 싣고, 빈 grid를 `no_depth_in_range`,
`no_depth_with_semantic_label`, `height_filter`, `outside_costmap`, `rasterization` 중 하나로
분류한 `empty_reason`을 추가한다.

Empty costmap은 계속 발행하며 planner의 즉시 STOP, costmap watchdog과 actuator timeout은
변경하지 않는다.

## Rationale and evidence

한 status sample에서 손실 단계를 직접 식별하면 현장에서 depth 설정, mask ontology, TF와
높이 범위를 순서 없이 바꾸지 않아도 된다. 진단 count는 이미 costmap 생성에 쓰는 sampled
배열에서 계산하므로 추가 image copy나 ROS subscription이 필요하지 않다.

## Alternatives considered

- Empty frame 무시 또는 마지막 grid 유지: 센서 손실 때 blind motion 시간이 늘어 제외했다.
- 높이 범위나 costmap 길이를 즉시 확대: 원인이 확인되지 않았고 false obstacle 또는 계산량을
  늘릴 수 있어 제외했다.
- rosbag 후처리만 사용: 현장 반복 문제를 즉시 구분할 수 없어 상태 진단을 우선했다.

## Consequences

상태 JSON의 필드 수와 직렬화량이 소폭 증가한다. 기존 필드는 유지되며 planner/control
입력에는 변화가 없다. `empty_reason`이 null이면 grid에 하나 이상의 observed cell이 있다.

## Validation and rollback

합성 NaN depth와 below-ground TF 사례로 필터별 count와 Z 범위를 검사한다. Target Jetson에서
10초간 `costmap_status`를 수집해 empty frame의 reason이 일관되는지 확인하고, processing
p95 증가가 유의하면 고주기 상세 필드를 throttle하는 방식으로 조정한다.
