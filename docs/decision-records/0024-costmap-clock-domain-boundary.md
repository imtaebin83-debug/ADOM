# 0024. Costmap clock-domain boundary

- Status: Accepted
- Date: 2026-08-13
- Owners: ADOM team
- Supersedes: 0008b와 0009의 downstream camera timestamp 보존, 0012의 camera source-age 해석

## Context

Jetson에서 ZED RGB와 registered depth는 서로 동기화된 device/monotonic timestamp를
발행했지만 `rclpy` planner는 system ROS clock을 사용했다. Perception mask와 semantic
costmap이 camera stamp를 그대로 전달하자 planner가 Unix epoch 크기의 age를 계산해 모든
유효 costmap을 stale로 폐기했다. ZED RGB timestamp만 ROS publish time으로 바꾸면
depth와 clock domain이 갈라져 wrapper 내부 `DEPTH/RGB ASYNC`가 발생했다.

## Decision

ZED RGB, registered depth와 perception mask는 원본 camera timestamp를 유지한다. Semantic
costmap은 이 timestamp로 mask-depth matching과 TF lookup을 완료한 뒤, 완성된
`OccupancyGrid.header.stamp`를 costmap 노드의 ROS 현재 시각으로 기록한다. ZED wrapper의
timestamp 또는 transport 설정은 이 문제를 해결하기 위해 변경하지 않는다.

Planner와 local controller의 downstream age는 costmap 생성 이후 software freshness를
뜻한다. Camera-to-perception과 costmap 처리시간은 각 node status에서 별도로 관찰하며,
camera-to-costmap 수치는 camera와 ROS clock이 같은 domain일 때만 유효하다.

## Rationale and evidence

센서 timestamp는 RGB-depth pairing과 timestamped TF lookup에 필요하다. 반면 planner의
freshness 검사에는 같은 clock domain에서 생성된 timestamp가 필요하다. Costmap 출력이
센서 동기화가 끝나는 첫 경계이므로 여기서 ROS clock으로 전환하면 두 요구를 함께
만족한다. Costmap 수신 중단 시 0.20초 watchdog과 하위 command timeout은 유지된다.

## Alternatives considered

- ZED `debug.use_pub_timestamps` 활성화: RGB와 depth timestamp가 갈라져 wrapper 내부
  비동기가 발생해 기각했다.
- Planner source-age 검사 제거: clock 문제는 피하지만 freshness 방어가 약해져 기각했다.
- 매우 큰 `max_source_age_sec`: 잘못된 clock domain을 숨기고 stale data를 허용해 기각했다.

## Consequences

Costmap과 path header는 더 이상 원본 camera capture 시각을 나타내지 않는다. 기존
camera-to-action 숫자는 동일 clock domain이 검증된 별도 계측으로만 해석해야 한다.
Mask와 depth의 synchronization, TF lookup 및 ZED 기본 timestamp 동작은 바뀌지 않는다.

## Validation and rollback

정적 regression test로 sensor timestamp 기반 TF lookup 유지, costmap ROS restamp와
camera header 복사 제거를 확인한다. Jetson에서는 RGB/depth async warning이 없는지,
costmap stamp가 ROS time과 가까운지, planner stale rejection이 사라지는지 wheels-off로
확인한다. 문제가 있으면 costmap restamp만 되돌리되 ZED timestamp를 부분 변경하지 않는다.
