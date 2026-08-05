# adom_planning

Nav2 설정과 RTK waypoint 데이터 계약을 관리한다.

- Global: Smac Hybrid-A* (`minimum_turning_radius` 반영)
- Local: Regulated Pure Pursuit
- Localization input: `map -> odom -> base_link`
- Command output: `/cmd_vel`
- RTK input: `/fix`, 변환 결과는 `navsat_transform_node`의 map 좌표
- Sequential executor: `/fromLL` 변환 후 `/navigate_to_pose` goal을 하나씩 전송

단일 안테나 RTK는 heading을 주지 않는다. `heading: null`이면 executor가 다음 waypoint
방향을 map 좌표계에서 계산한다. 긴 GPS route는 40 m rolling global costmap에 한 번에
넣지 않고 가까운 waypoint를 순차 전송한다. 제자리 회전은 Ackermann 차량에서 불가능하므로
controller와 recovery에서 비활성화한다.

## Sequential GPS execution

Localization과 Nav2를 먼저 실행한 뒤 다음처럼 시작한다.

```bash
ros2 launch adom_planning sequential_gps.launch.py \
  waypoint_file:=/absolute/path/to/rtk_waypoints.yaml \
  start_nav2:=false
```

`start_nav2:=true`면 이 launch가 Nav2도 함께 시작한다. Executor는 유효한 `/fix`,
`/odometry/global`, `/odometry/gps`, `/fromLL`, `/navigate_to_pose`를 기다린다.
`/odometry/gps` 수신을 요구하므로 navsat geographic transform이 준비되기 전에 좌표를
변환하지 않는다. 상태는 다음 토픽으로 확인한다.

```text
/adom/planning/waypoint_status   std_msgs/String
/adom/planning/waypoint_index    std_msgs/Int32 (0-based)
```

기본 정책은 GNSS fix가 2초 이상 오래되면 활성 goal을 취소하고, goal 실패 시 전체 route를
중단하는 것이다. `NavSatFix`만으로 RTK FIX/FLOAT를 완전히 구분할 수 없으므로 실차에서는
수신기 전용 RTK 상태를 safety supervisor에 추가해야 한다.

현재 global costmap 폭은 40 m이므로 연속 waypoint 간격은 여유를 두고 15 m 이하로
구성하는 것을 권장한다. 더 긴 구간은 중간 waypoint를 추가하거나 global costmap 정책을
변경해야 한다.

차량 실측 후 `minimum_turning_radius`, footprint, 속도/가속도 제한을 수정한다.
