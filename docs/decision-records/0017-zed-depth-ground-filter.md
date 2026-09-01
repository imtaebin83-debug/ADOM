# 0017. ZED depth and ground-frame filtering

- Status: Accepted
- Date: 2026-08-12
- Owners: 명섭
- Supersedes: none

## Context

Semantic costmap이 ZED registered depth를 사용하기 시작했지만 센서 설정은
`NEURAL_LIGHT`였고, SDK depth 범위와 costmap 범위가 분리돼 있었다. 지면 아래 stereo
overshoot가 geometric obstacle로 들어오는 것을 막으려면 카메라 optical Y축 부호를
고정 가정하기보다 실제 장착 TF를 거친 지면 기준 높이를 사용해야 한다.

ZED optical center의 지면 기준 높이는 2026-08-12에 정밀 재측정되어 0.21 m로
확인됐다. Mount pitch 기본값은 이 측정으로 검증되지 않았다.

## Decision

- ZED ROS 2 depth mode를 `NEURAL`로 설정한다.
- SDK와 costmap의 유효 범위를 0.30--8.0 m로 일치시킨다.
- `depth_confidence`와 `depth_texture_conf` 초기값은 각각 50으로 둔다. ZED confidence는
  0이 가장 신뢰도가 높고 100이 가장 낮으므로 threshold를 낮출수록 더 엄격하다.
- URDF의 ZED 지면 기준 높이를 검증된 0.21 m로 설정한다.
- Registered depth를 TF로 `base_link`에 변환한 뒤 Z가 -0.05--1.50 m인 점만 유지한다.
  0.10 m 이상의 관측은 기존대로 geometric obstacle로 처리한다.
- Depth stabilization은 positional tracking을 암묵적으로 켜지 않도록 0을 유지한다.

## Rationale and evidence

`NEURAL`은 `NEURAL_LIGHT`보다 저텍스처 영역과 물체 세부의 depth 품질을 우선하면서,
`NEURAL_PLUS`보다 Jetson GPU 경합 위험이 낮은 절충안이다. SDK에서 원거리 depth를 먼저
제한하면 costmap이 폐기할 8 m 밖 jitter를 publish하지 않는다.

높이 조건은 optical-frame Y가 아니라 ground-referenced `base_link` Z에 적용하므로
카메라 pitch와 ROS optical-axis convention을 TF가 처리한다. -0.05 m tolerance는 작은
TF/depth 오차를 허용하면서 물리적으로 지면 아래인 큰 overshoot를 제거한다.

## Alternatives considered

- `NEURAL_PLUS`: 품질은 높지만 Orin Nano에서 SegFormer와 함께 실행한 latency/GPU/RAM
  evidence가 없어 보류한다.
- Texture threshold 100: 낮은 texture 신뢰도 점을 거의 제거하지 않아 품질 개선 목적과
  맞지 않는다.
- SDK floor-plane detection: positional tracking/IMU 상태와 frame reset 동작을 추가로
  검증해야 하므로 현재 static-TF 기반 costmap 경로에는 넣지 않는다.
- Optical Y 픽셀 필터: mount pitch/roll과 좌표계 설정에 취약해 채택하지 않는다.

## Consequences

유효 depth pixel 수는 줄 수 있지만 지면 아래·저신뢰 오측정이 lethal obstacle로
투영될 가능성도 줄어든다. `NEURAL`의 추가 GPU 비용 때문에 live perception latency는
Jetson에서 다시 측정해야 한다. Camera pitch는 여전히 별도 검증 항목이다.

## Validation and rollback

고정된 1 m와 3 m reference, 평탄 노면, 작은 장애물을 같은 조명에서 촬영해 valid-pixel
비율, median absolute depth error, p95 latency와 false lethal cell 수를 비교한다.
SegFormer와 동시 실행할 때 watchdog 또는 latency budget을 만족하지 못하면
`NEURAL_LIGHT`로 되돌리되 범위·confidence·ground-frame height filter는 유지한다.
