# ROS2 Workspace

ROS 2 Jazzy integration code lives here as part of the mono repo.

권장 원칙:

- `src/`의 공통 Python/ML 로직을 ROS2 node에서 호출한다.
- ROS2 package는 sensor I/O, topic/message 변환, launch/config, runtime wiring을 담당한다.
- 첫 package skeleton은 topic, message type, target node boundary가 정해진 뒤 추가한다.

예상 구조:

```text
ros2_ws/
├── src/
│   ├── adom_perception_ros/
│   ├── adom_costmap_ros/
│   ├── adom_bringup/
│   └── adom_msgs/
├── build/      # ignored
├── install/    # ignored
└── log/        # ignored
```
