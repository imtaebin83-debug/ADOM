# adom_costmap_ros

Segmentation mask와 registered depth를 결합해 Nav2 costmap에 주입하는 패키지다.
현재는 데이터 계약만 포함한다. 첫 구현은 별도 `OccupancyGrid` publisher로 검증한 뒤
성능이 필요할 때 `nav2_costmap_2d::Layer` C++ plugin으로 옮긴다.

