# adom_costmap_ros

Segmentation mask, registered depth, camera intrinsics와 TF를 결합해 로봇 중심 semantic
`OccupancyGrid`를 발행한다. Cost4의 위험도가 높은 관측이 같은 cell에서 항상 우선하며,
차량 폭을 위한 inflation을 적용한다.

`inflation_seed_cost` 이상인 이미 lethal한 cell만 inflation을 시작한다. 주변 ring은
`inflation_min_cost` 이상의 감속 비용으로 유지하므로, 그 사이의 semantic cost가
inflation 활성화만으로 hard-stop cost 100으로 승격되지는 않는다.

```text
/adom/navigation/semantic_costmap  nav_msgs/OccupancyGrid
/adom/navigation/costmap_status    std_msgs/String (JSON)
```

RGB mask와 registered depth의 원본 camera timestamp는 depth matching과 TF lookup까지
그대로 사용한다. 완성된 `OccupancyGrid.header.stamp`는 costmap 노드의 ROS 현재 시각으로
새로 기록한다. ZED가 device/monotonic timestamp를 발행해도 planner가 system ROS clock과
직접 비교하지 않게 하며, downstream age는 costmap 생성 이후 software freshness를
뜻한다. Camera clock과 ROS clock이 같은 domain일 때만 status의
`source_to_costmap_output_ms`를 camera-to-costmap 지연으로 해석한다.

이번 구현은 rule planner와 RViz 검증을 위한 별도 grid다. 장기적으로 Nav2가 이 비용을 직접
합성해야 할 때는 동일한 투영/비용 계약을 `nav2_costmap_2d::Layer` C++ plugin으로 옮긴다.

Semantic20 perception과 연결할 때는 `semantic20_costmap.launch.py`를 사용한다. 이
launch는 `/adom/perception/semantic20_mask`, registered depth, camera info와 TF를
결합한다. `semantic20_costs.yaml`의 클래스 비용은 초기 후보이므로 실차 주행 전에
validation scene에서 동결해야 한다.

투영점은 먼저 camera optical frame에서 `base_link`로 변환된다. 기본 높이 범위
`-0.05..1.50 m`는 지면 아래로 5 cm보다 크게 overshoot한 stereo noise와 높은 허공
noise를 제거한다. 지면 자체는 semantic traversability 판단을 위해 남기며, 0.10 m
이상의 점만 geometric obstacle로 강제한다. 따라서 카메라가 pitch/roll된 경우에도
optical Y축 부호를 가정하는 별도 픽셀 필터를 추가하지 않는다.
