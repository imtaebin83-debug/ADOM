# 0029. Autonomous 0.30..3.00 m/s speed profile

- Status: Superseded
- Date: 2026-08-14
- Owners: ADOM control integration
- Supersedes: speed-profile portions of 0039 and 0018

## Context

현재 저장소의 autonomous 속도 값은 실행 YAML, ROS node fallback, 테스트와 기준 문서가
서로 달랐다. 운영자는 자율주행 모드의 주행 중 명령 범위를 최소 0.30 m/s, 최대
3.00 m/s로 변경하도록 승인했다. 0018에서 확인한 약 0.33--0.43초의 camera
source-to-command 지연은 해소됐다고 검증되지 않았다.

## Decision

Rule planner와 local path controller의 `min_speed_mps`를 0.30, `max_speed_mps`를
3.00으로 통일한다. YAML과 ROS node fallback, reusable planner/controller defaults가
같은 값을 사용한다. 안전 판정으로 정지할 때의 0 명령은 최소 속도 적용 대상이 아니다.

STOP, command watchdog, 수동 reset, manual gamepad ceiling과 PCA9685의 nominal
12.0 m/s mapping은 변경하지 않는다.

## Rationale and evidence

Planner가 위험도와 clearance에 따라 계산한 양수 주행 명령을 0.30..3.00 m/s로 제한하고,
local controller가 동일한 범위를 보존하면 두 계층 사이의 상충하는 clamp를 제거할 수
있다. 이 값은 운영자 승인 command profile이며 실제 지상 속도 검증값은 아니다.

## Alternatives considered

- 0018의 0.10..0.75 m/s 유지: 기존 latency evidence에는 부합하지만 이번 운영 범위
  요청과 다르다.
- Planner만 변경: local controller의 별도 clamp로 planner profile이 보존되지 않는다.
- Manual/PWM ceiling도 3.0 m/s로 변경: 자율주행 범위를 넘어서는 별도 인터페이스
  변경이므로 제외했다.

## Consequences

- Clear-path autonomous command는 최대 3.00 m/s까지 증가할 수 있다.
- 양수 autonomous command는 최소 0.30 m/s지만 BLOCKED, timeout, E-stop과 mode 전환은
  계속 0/neutral을 명령한다.
- 실제 속도와 제동 거리는 미검증이며, 기존 latency evidence 때문에 실차 검증 전 안전
  여유를 주장할 수 없다.

## Validation and rollback

설정 테스트로 planner와 local controller의 최소·최대 값이 일치하는지 검사한다.
Wheels-off에서 STOP, timeout과 mode 전환 시 neutral을 확인한 뒤 폐쇄 공간에서 0.30 m/s부터
단계적으로 올리며 실제 속도, source-to-command 지연과 제동 거리를 rosbag으로 기록한다.
검증에 실패하면 planner와 local controller를 함께 0018의 0.10..0.75 m/s profile로
되돌린다.
