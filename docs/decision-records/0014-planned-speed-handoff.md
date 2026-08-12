# 0014. Preserve planner speed through local control

- Status: Accepted
- Date: 2026-08-12
- Owners: ADOM planning/control integration
- Supersedes: speed handoff and 0.25 m/s controller-limit portions of 0010

## Context

Rule planner는 위험도, clearance, 조향량으로 `0.25..3.0 m/s` 속도를 계산했지만
`nav_msgs/Path`에는 geometry만 발행했다. Local path controller는 path를 받은 뒤 planner
속도를 알 수 없어 자체 `max_speed_mps: 0.25`로 속도를 다시 계산했다. 그 결과 조향은
계획대로 동작해도 PCA9685에는 control의 6.0 m/s 범위 중 약 1521 us만 전달될 수 있었다.

## Decision

Planner가 `/adom/navigation/planned_speed` (`std_msgs/Float32`)를 path와 함께 발행한다.
Local controller는 이 값을 freshness watchdog과 함께 받아 steering은 path에서 계산하고
speed는 planner 값을 보존해 `/cmd_vel`로 전달한다. Planner stop은 speed 0과 empty path를
함께 발행한다.

Planner와 local controller의 profile은 기존 `min_speed_mps: 0.25`,
`max_speed_mps: 3.0`을 유지한다. Downstream ceiling은 existing control 설정과 같은
6.0 m/s로 검증하며, gamepad mode mux와 PCA9685 watchdog은 계속 우회하지 않는다.

## Rationale and evidence

Path geometry와 속도 profile의 ownership을 planner에 함께 두면 중간 controller가 속도를
잃거나 재해석하지 않는다. 별도 speed freshness gate는 path만 갱신되거나 speed publisher가
끊긴 경우 zero command를 보장한다. Existing control 파라미터에 따르면 0.25 m/s는 약
1520.8 us, 3.0 m/s는 1750 us로 변환된다.

## Alternatives considered

- Planner가 `/cmd_vel`을 직접 발행: local path steering controller와 publisher ownership이
  충돌하므로 채택하지 않는다.
- Path pose의 비표준 field에 속도 저장: message 의미가 불명확하고 일반 ROS 도구와의
  호환성이 나빠 채택하지 않는다.
- Local controller에서 독립 속도 profile 재계산: planner가 계산한 위험 기반 속도를 다시
  잃으므로 채택하지 않는다.

## Consequences

- 새 runtime/evidence topic `/adom/navigation/planned_speed`가 추가된다.
- Path, speed 또는 IMU가 0.25초 이상 stale이면 local controller가 zero를 발행한다.
- `/cmd_vel` 이후에는 기존 A-button 승인, autonomous watchdog, `/drive`, PCA9685 mapping을
  그대로 사용한다.
- 엔코더가 없으므로 m/s와 실제 지상 속도의 대응은 open-loop이며 실측 검증이 필요하다.

## Validation and rollback

단위 테스트에서 planner/local/controller ceiling 일치, planner speed 보존, explicit stop을
검증한다. Jetson에서는 바퀴를 띄운 상태에서 planned speed, `/cmd_vel`, `/drive`, PWM을
동시에 관찰하고 B 버튼 및 각 topic timeout 시 1500 us neutral 복귀를 확인한다.
