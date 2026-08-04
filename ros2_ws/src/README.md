# ROS 2 Packages

- `adom_description`: vehicle URDF and sensor TF
- `adom_sensors`: ZED 2i and RTK GNSS launch/config adapters
- `adom_perception_ros`: TensorRT perception ROS adapter boundary
- `adom_costmap_ros`: semantic/depth costmap adapter boundary
- `adom_localization`: ZED VIO + RTK GNSS dual-EKF configuration
- `adom_planning`: Nav2 and RTK waypoint planning configuration
- `adom_control`: `/cmd_vel` to Ackermann/PCA9685 PWM
- `adom_safety`: E-stop and safety-limit contract
- `adom_bringup`: top-level launch
- `adom_msgs`: reserved for unavoidable custom interfaces

The initial control launch is hardware-safe by default (`dry_run: true`) and the top-level
bringup keeps autonomous planning disabled until sensors, TF, and localization are verified.
