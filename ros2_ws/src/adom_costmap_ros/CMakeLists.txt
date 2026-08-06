cmake_minimum_required(VERSION 3.8)
project(adom_costmap_ros)
find_package(ament_cmake REQUIRED)
install(DIRECTORY config launch DESTINATION share/${PROJECT_NAME})
install(PROGRAMS scripts/semantic_costmap_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME semantic_costmap_node
)
ament_package()
