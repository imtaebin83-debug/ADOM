cmake_minimum_required(VERSION 3.8)
project(adom_perception_ros)
find_package(ament_cmake REQUIRED)
install(DIRECTORY config launch DESTINATION share/${PROJECT_NAME})
install(PROGRAMS scripts/perception_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME adom_perception_node
)
install(PROGRAMS scripts/perception_cost4_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME adom_cost4_perception_node
)
install(PROGRAMS scripts/semantic20_colorizer_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME semantic20_colorizer_node
)
ament_package()
