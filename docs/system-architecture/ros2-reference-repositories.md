# ROS 2 Reference Repositories

The ADOM ROS 2 package boundaries were informed by these upstream projects. Their source is
not copied into this mono repository.

- Navigation2: <https://github.com/ros-navigation/navigation2>
  - Nav2 lifecycle/bringup, Smac Hybrid-A*, Regulated Pure Pursuit, collision monitor
- robot_localization: <https://github.com/cra-ros-pkg/robot_localization>
  - dual-EKF and `navsat_transform_node` RTK GNSS pattern
- ZED ROS 2 Wrapper: <https://github.com/stereolabs/zed-ros2-wrapper>
  - ZED 2i image/depth/IMU/odometry interfaces
- nmea_navsat_driver: <https://github.com/ros-drivers/nmea_navsat_driver>
  - generic NMEA receiver to `sensor_msgs/NavSatFix`
- ros2_controllers: <https://github.com/ros-controls/ros2_controllers>
  - Ackermann controller interfaces and parameter conventions
- F1TENTH system: <https://github.com/f1tenth/f1tenth_system>
  - command mux, deadman, steering/throttle calibration patterns
- Adafruit CircuitPython PCA9685: <https://github.com/adafruit/Adafruit_CircuitPython_PCA9685>
  - Jetson-side PCA9685 API

Use ROS 2 Humble binary packages where available and pin external source dependencies outside
the ADOM packages. Do not vendor whole upstream repositories into `study/` or `src/`.
