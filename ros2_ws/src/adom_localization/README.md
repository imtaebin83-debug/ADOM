# adom_localization

`robot_localization`의 local EKF, global EKF, `navsat_transform_node`를 사용한다.

TF 소유권:

- local EKF: `odom -> base_link`
- global EKF: `map -> odom`
- URDF: `base_link -> zed_camera_link`, `base_link -> gnss_link`

ZED Wrapper의 odometry/map TF 발행은 꺼야 한다. 단일 안테나 RTK는 위치 정밀도만
높이고 절대 heading을 제공하지 않는다. ZED IMU yaw가 ENU 절대방향인지 검증하지
않은 상태에서 `navsat_transform_node`에 넣으면 지도 전체가 회전할 수 있다. 실차에서는
dual-antenna heading, 검증된 magnetometer 또는 surveyed datum/heading 중 하나를 사용한다.

