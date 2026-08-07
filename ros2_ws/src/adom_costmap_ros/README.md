# adom_costmap_ros

Segmentation mask, registered depth, camera intrinsics와 TF를 결합해 로봇 중심 semantic
`OccupancyGrid`를 발행한다. Cost4의 위험도가 높은 관측이 같은 cell에서 항상 우선하며,
차량 폭을 위한 inflation을 적용한다.

```text
/adom/navigation/semantic_costmap  nav_msgs/OccupancyGrid
/adom/navigation/costmap_status    std_msgs/String (JSON)
```

이번 구현은 rule planner와 RViz 검증을 위한 별도 grid다. 장기적으로 Nav2가 이 비용을 직접
합성해야 할 때는 동일한 투영/비용 계약을 `nav2_costmap_2d::Layer` C++ plugin으로 옮긴다.

Semantic20 perception과 연결할 때는 `semantic20_costmap.launch.py`를 사용한다. 이
launch는 `/adom/perception/semantic20_mask`, registered depth, camera info와 TF를
결합한다. `semantic20_costs.yaml`의 클래스 비용은 초기 후보이므로 실차 주행 전에
validation scene에서 동결해야 한다.
