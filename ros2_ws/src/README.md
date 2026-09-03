# ROS 2 Packages

- `adom_description`: vehicle URDF, sensor TF, chassis CAD mesh
- `adom_sensors`: ZED 2i and RTK GNSS launch/config adapters
- `adom_perception_ros`: TensorRT perception ROS adapter boundary
- `adom_costmap_ros`: semantic/depth costmap adapter boundary
- `adom_localization`: ZED VIO + RTK GNSS dual-EKF configuration
- `adom_planning`: Nav2 and RTK waypoint planning configuration
- `adom_control`: `/cmd_vel` to Ackermann/PCA9685 PWM
- `adom_logging`: logging-only GPS trail and bounded autonomy rosbag sessions
- `adom_bringup`: top-level launch

표준 ROS 메시지만 사용한다. `/emergency_stop`과 command timeout은 `adom_control`이
직접 처리한다. `pca9685_control`은 실제 I2C 하드웨어를 초기화하므로, PWM 없이 띄우려면
어떤 launch에서든 `start_pca9685:=false`를 명시한다. `low_level_autonomy.launch.py`만
이 값이 기본 `false`이고 나머지는 `true`다. top-level bringup은 센서, TF, localization이
검증될 때까지 autonomous planning을 비활성화한다.
