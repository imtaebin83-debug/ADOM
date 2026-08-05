cmake_minimum_required(VERSION 3.8)
project(adom_planning)
find_package(ament_cmake REQUIRED)
install(DIRECTORY config launch DESTINATION share/${PROJECT_NAME})
install(PROGRAMS scripts/rtk_waypoint_executor.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME rtk_waypoint_executor
)
ament_package()
