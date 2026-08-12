# 0018. Autonomous speed cap from bag latency

- Status: Accepted
- Date: 2026-08-12
- Owners: ADOM control integration
- Supersedes: speed-ceiling portions of 0015

## Context

실내 장애물 시험에서 차량 속도가 perception과 planning의 반응 속도보다 빠르게 느껴졌다.
정상 종료된 `autonomy_20260812_204613_+0900`과
`autonomy_20260812_204923_+0900` bag의 numeric/status topic을 분석했다. metadata가 없는
`autonomy_20260812_205016_+0900`은 불완전 bag으로 제외했다.

두 정상 bag에서 perception과 costmap status는 약 10.2 Hz였다. Camera capture부터
perception output까지의 중앙값은 약 162--163 ms, 95 percentile은 약 219--221 ms였다.
Camera source부터 local control command까지의 중앙값은 약 326--337 ms, 95 percentile은
약 425--431 ms였다. Costmap 자체 처리는 중앙값 약 4.3 ms이므로 perception inference와
frame 대기가 주된 계산 지연이다.

## Decision

Rule planner의 `max_speed_mps`를 현장 설정 1.0 m/s의 75%인 0.75 m/s로 낮춘다. IMU
feedback이 planner target보다 높은 command를 만들더라도 최종 autonomous command가 이
값을 넘지 않도록 local path controller의 `max_speed_mps`도 0.75 m/s로 둔다. Planner의
기존 `min_speed_mps: 0.10`과 STOP/watchdog 동작은 유지한다.

## Rationale and evidence

기존 bag의 clear-path command는 약 0.90 m/s였다. 동일한 risk scale이면 새 profile은 약
0.675 m/s가 되어 계산 지연 동안 이동하는 거리를 약 25% 줄인다. 속도 제한은 관측된
latency를 제거하지 않지만 실내 검증의 시간 여유를 늘리는 즉시 적용 가능한 안전 조치다.

## Alternatives considered

Perception FPS 또는 ZED 설정 변경은 latency와 GPU 부하를 다시 실측해야 하므로 이번 속도
제한과 분리한다. PWM calibration 변경은 STOP/watchdog와 수동 운전에도 영향을 줄 수 있어
autonomous speed cap 용도로 사용하지 않는다.

## Consequences

- Planner와 local controller의 autonomous hard ceiling은 0.75 m/s다.
- Manual gamepad ceiling과 PWM calibration은 변경하지 않는다.
- 약 0.33--0.43초의 source-to-command latency는 남아 있으므로 저속 wheels-off 및 폐쇄
  공간 검증이 계속 필요하다.

## Validation and rollback

설정 검사에서 planner와 local controller의 ceiling이 모두 0.75 m/s인지 확인한다.
재실행 후 `/adom/navigation/planned_speed`, `/cmd_vel`, `/drive`가 0.75 m/s를 넘지 않는지
rosbag으로 검증한다. 주행성이 지나치게 낮으면 planner 1.0 m/s와 local controller 3.0
m/s의 이전 field setting으로 rollback한 뒤 더 낮은 단계별 값을 재검토한다.
