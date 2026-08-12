# adom_planning

Nav2 설정과 RTK waypoint 데이터 계약을 관리한다.

- Global: Smac Hybrid-A* (`minimum_turning_radius` 반영)
- Local: Regulated Pure Pursuit
- Localization input: `map -> odom -> base_link`
- Command output: `/cmd_vel`
- RTK input: `/fix`, 변환 결과는 `navsat_transform_node`의 map 좌표
- Sequential executor: `/fromLL` 변환 후 `/navigate_to_pose` goal을 하나씩 전송

단일 안테나 RTK는 heading을 주지 않는다. `heading_deg: null`이면 executor가 다음 waypoint
방향을 map 좌표계에서 계산한다. 명시적인 `heading_deg`는 ENU degree 단위로 0°가 동쪽,
+90°가 북쪽이다. 긴 GPS route는 40 m rolling global costmap에 한 번에
넣지 않고 가까운 waypoint를 순차 전송한다. 제자리 회전은 Ackermann 차량에서 불가능하므로
controller와 recovery에서 비활성화한다.

## Sequential GPS execution

Localization과 Nav2를 먼저 실행한 뒤 다음처럼 시작한다.

```bash
ros2 launch adom_planning sequential_gps.launch.py \
  waypoint_file:=ros2_ws/src/adom_planning/config/rtk_waypoints.example.yaml \
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

## Semantic20 low-level tree planning

이 로컬 플래너는 GPS 경로를 만들거나 입력으로 사용하지 않는다. 로봇 기준 semantic
costmap 위에서 `좌/약좌/직진/약우/우` 방향을 기본 3단계 tree로 전개하고, 각
root-to-leaf Ackermann corridor 중 충돌 비용과 평균 비용이 낮은 방향열을 선택해
`/adom/navigation/local_path` (`nav_msgs/Path`)를 발행한다. `local_path_control`이 이
경로와 IMU freshness 정보로 `/cmd_vel`을 만든다. 다음 planning cycle에는 최신
costmap에서 tree를 다시 만들기 때문에 첫 번째 방향만 실제로 실행하는 receding-horizon
방식이다. GPS는 `adom_logging`에서 이동 궤적 기록에만 사용한다.

Semantic20 costmap과 로컬 플래너는 다음 명령으로 함께 실행한다.

```bash
ros2 launch adom_planning semantic20_local_planning.launch.py
```

`rule_planner`는 로봇 중심 semantic costmap에서 휠베이스와 조향 한계를 만족하는
방향 tree를 평가한다. 중앙 장애물이 있으면 좌·우 gap 폭과 깊이를 비교해 방향과 첫
조향을 고정하고 남은 25개 tree path 중 가장 낮은 비용의 경로를 발행한다. 중앙
장애물이 없으면 기존 125개 tree와 직진 선호를 유지한다.
가까운 lethal cost, 0.20초 이상 갱신되지 않은 costmap, 0.80초 이상 오래된 센서
timestamp 또는 관측 cell이 없는 costmap에서는 반드시 정지한다.

Planner의 현재 속도 profile은 0.25..3.0 m/s이며
`/adom/navigation/planned_speed`로 local controller에 전달된다. Controller는 이를
`/cmd_vel`에 보존하고, gamepad control의 자율 모드(A 버튼)를 거쳐야만 `/drive`로
전달한다. PCA9685에 직접 연결하지 않는다. Downstream gamepad/PCA9685 nominal ceiling은
`vehicle.yaml`의 12.0 m/s이며, camera source stamp부터 controller command까지의 지연은
`/adom/control/local_path_status`의 `source_to_command_ms`로 확인한다. 엔코더가 없어
PWM 속도 대응은 nominal open-loop이다. Local controller는 정지 중 IMU bias를 학습하고
주행 중 short-horizon 적분 속도로 제한된 feedback 보정을 수행한다.

```bash
ros2 launch adom_planning rule_planning.launch.py
ros2 topic echo /adom/navigation/rule_status
```

`rule_status`의 `steering_sequence_deg`가 선택된 tree 방향열이고 첫 원소가 이번 cycle의
실행 조향이다. `gap_selected_side`, 좌·우 `gap_*_width_m`/`gap_*_depth_m`,
`gap_selected_goal_deg`와 `tree_candidate_count`로 gap 판단을 확인한다.
