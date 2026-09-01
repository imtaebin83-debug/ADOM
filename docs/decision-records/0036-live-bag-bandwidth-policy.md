# 0036. Live autonomy bag bandwidth policy

- Status: Accepted
- Date: 2026-08-12
- Owners: ADOM team
- Supersedes: 0035의 confidence/overlay live recording 부분

## Context

Jetson에서 autonomy recorder를 실행했을 때 camera→inference 시작과
camera→perception 출력 지연이 recorder를 끈 경우보다 각각 약 58 ms 증가했다. BGR
overlay와 confidence image는 진단에는 유용하지만 매 frame의 대용량 DDS 직렬화와
rosbag 쓰기를 발생시킨다.

## Decision

Live autonomy bag은 Semantic20 mask와 perception status를 기록하되 confidence image와
BGR overlay는 제외한다. confidence/overlay publisher는 유지하지만 구독자가 없을 때
confidence softmax·CPU 복사와 overlay colorize/blend/message 변환을 실행하지 않는다.
Costmap rasterization은 동일 cell의 최고 비용 규칙을 유지하며 NumPy로 벡터화한다.

## Rationale and evidence

Mask는 semantic 판단을 재현하는 핵심 근거인 반면 confidence/overlay는 mask에서 일부를
재생성하거나 별도 진단 세션에서 선택적으로 구독할 수 있다. 상태 JSON, path, command,
GPS는 image topic보다 훨씬 작다. Costmap은 mask보다 작지만 planner 입력 재현을 위해
유지한다.

## Alternatives considered

- 모든 image topic 유지: 분석 편의는 높지만 live safety latency가 증가한다.
- mask도 제외: 부하는 더 줄지만 perception 판단을 사후 재현하기 어렵다.
- planner timeout 완화: 처리량을 개선하지 않고 오래된 장면의 사용만 허용한다.

## Consequences

Live bag만으로 confidence/overlay를 직접 재생할 수 없다. 필요 시 actuator를 비활성화한
진단 세션에서 해당 topic을 별도로 기록한다. mask와 costmap의 실측 부하는 계속
관찰하며 camera→costmap status의 단계별 latency를 기준으로 추가 최적화를 결정한다.

## Validation and rollback

Bag topic 목록에 confidence/overlay가 없고 mask/status/costmap/path/command/GPS가 있는지
확인한다. perception status에서 두 image subscriber 수가 0인지 확인하고 recorder on/off
latency를 같은 장면에서 비교한다. 분석에 꼭 필요한 경우 별도 진단 config로 되돌린다.
