# ROS2 Humble Setup

## Repository Policy

ROS2 코드는 별도 repo로 분리하지 않고 이 mono repo 안의 `ros2_ws/`에서 관리합니다.

권장 역할 분리:

- `src/`: model loading, inference, post-processing, costmap conversion 등 ROS와 무관하게 재사용 가능한 공통 코드
- `ros2_ws/src/`: ROS2 package, node, launch file, message/service/action adapter

ROS2 node는 가능한 한 얇게 유지하고, 핵심 알고리즘은 `src/`의 공통 모듈을 호출하게 만듭니다.

## Target

- Ubuntu 22.04
- ROS2 Humble
- Jetson Orin Nano 8GB
- ZED 2i camera

## Future Package Candidates

```text
ros2_ws/src/
├── adom_perception_ros/
├── adom_costmap_ros/
├── adom_bringup/
└── adom_msgs/
```

아직 package skeleton은 만들지 않았습니다. 첫 ROS2 node의 입출력 topic과 message type이 정해지면 추가합니다.
