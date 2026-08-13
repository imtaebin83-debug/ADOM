# 0028. Semantic20 mask color monitor

- Status: Accepted
- Date: 2026-08-13
- Owners: ADOM team
- Supersedes: none

## Context

Semantic20 mask는 class ID `0..18`과 ignore `255`를 담은 `mono8` image라 일반 image viewer에서
거의 검게 보인다. Live inference와 rosbag replay 모두에서 class 위치를 사람이 즉시 확인할
수 있어야 하지만, 시각화를 위해 모델을 다시 실행하거나 ontology별 색을 복제해서는 안 된다.

## Decision

`adom_perception_ros`에 경량 colorizer node를 추가한다. 기본 입력은 2 Hz evidence mask이고
각 ID를 `src/adom/perception/semantic20.py`의 canonical palette로 변환해
`/adom/perception/semantic20_mask_color` `bgr8` image를 발행한다. ID, class name과 RGB 값은
`/adom/perception/semantic20_legend` JSON으로 발행한다. Ignore `255`는 검정이다.

Colorizer 출력은 진단 전용이며 costmap, planner와 controller 입력으로 사용하지 않는다.

## Consequences

Colorizer는 mask 크기의 BGR image를 추가 생성하므로 로컬 모니터링 CPU·메모리 비용이 조금
늘지만 모델 inference는 실행하지 않는다. 기본 2 Hz replay에서는 영향이 작다. Color output과
legend는 rosbag 기본 기록 대상에 추가하지 않는다.

## Validation

Synthetic ID mask로 canonical palette와 ignore black을 확인하고, rosbag replay에서 color
output의 header가 input mask와 동일한지 확인한다. `rqt_image_view`로 color topic을 열고 legend
topic의 19개 class와 ignore entry를 확인한다.
