# 0012. Planner source-age tolerance

- Status: Accepted
- Date: 2026-08-12
- Owners: ADOM autonomy integration
- Supersedes: none

## Context

Jetson 실차 파이프라인의 camera-to-costmap 지연은 정상 구간에서 약 174~255 ms였지만,
부하와 스케줄링 변화 중 0.40초를 넘는 간헐적 costmap이 관측됐다. 기존 planner는
camera source timestamp가 0.40초보다 오래되면 costmap을 즉시 폐기했다.

## Decision

`rule_planner.max_source_age_sec`의 기본값과 live config를 0.80초로 완화한다.
수신된 costmap 자체의 갱신 여부를 검사하는 `costmap_timeout_sec: 0.20`과 하위 control
및 actuator watchdog은 변경하지 않는다.

## Rationale and evidence

이번 변경은 계산 성능을 높이는 최적화가 아니라 간헐적인 source timestamp 지연을
현장에서 관찰하기 위한 허용 범위 조정이다. 최신 costmap 수신이 중단되면 기존
0.20초 watchdog이 계속 정지 명령을 발생시키므로, 데이터 정지 fail-safe는 유지된다.

## Alternatives considered

- 0.40초 유지: 더 엄격하지만 간헐적 지연 프레임을 모두 폐기한다.
- source-age 검사 제거: 오래된 장면을 제한 없이 사용할 수 있어 채택하지 않는다.
- 1초 이상 완화: 현재 요청 범위를 넘고 stale-scene 위험이 더 커 채택하지 않는다.

## Consequences

- 새 costmap이 계속 도착하는 상황에서는 최대 0.80초 된 camera 관측이 planning에
  사용될 수 있다.
- 이 값은 처리 latency를 줄이지 않는다.
- 실차 시험은 저속, wheels-off 검증, 물리 정지 담당자 조건을 유지한다.

## Validation and rollback

`/adom/navigation/rule_status`와 `/adom/control/local_path_status`에서 source age 및
camera-to-command 지연을 기록한다. 오래된 관측으로 인한 위험이나 불안정이 나타나면
config와 기본값을 0.40초로 되돌리고 원래 폐기 정책을 복원한다.
