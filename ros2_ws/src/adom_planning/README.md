# adom_planning

Nav2 설정과 RTK waypoint 데이터 계약을 관리한다.

- Global: Smac Hybrid-A* (`minimum_turning_radius` 반영)
- Local: Regulated Pure Pursuit
- Localization input: `map -> odom -> base_link`
- Command output: `/cmd_vel`
- RTK input: `/fix`, 변환 결과는 `navsat_transform_node`의 map 좌표

단일 안테나 RTK는 heading을 주지 않는다. waypoint의 위경도를 `map` pose로 바꾸는
executor는 datum/heading 검증 후 추가한다. 긴 GPS route는 40 m rolling global costmap에
한 번에 넣지 말고 가까운 waypoint를 순차 전송한다. 제자리 회전은 Ackermann 차량에서
불가능하므로 controller와 recovery에서 비활성화한다.

차량 실측 후 `minimum_turning_radius`, footprint, 속도/가속도 제한을 수정한다.

