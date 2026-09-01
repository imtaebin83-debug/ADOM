# 0034. Semantic20 local gap planning

- Status: Accepted
- Date: 2026-08-07
- Owners: ADOM team
- Supersedes: none

## Context

Semantic20 perception을 조향과 주행 경로로 변환할 때 GPS 기반 2D path를 매 frame
생성할지, 우회 조향·속도만 control에 전달할지 결정해야 한다. GPS는 전역 위치와
목적지에는 유용하지만 근거리 장애물 경계와 camera latency를 표현하지 못한다.
`semantic20.py`는 class ID/색상/mapping 계약이며 3D geometry를 제공하는 모듈은 아니다.

## Decision

Semantic20 mask를 registered depth, camera intrinsics, TF와 결합해 robot-frame 2D
costmap을 만든다. local planner는 Ackermann-feasible corridor들을 costmap에서 평가해
Follow-the-Gap과 유사한 low-cost corridor를 선택한다. depth-projected 첫 장애물 거리와
근거리 비용을 score·속도에 반영하고 `nav_msgs/Path`를 발행한다. 별도 path controller가
IMU-aided GPS 속도 feedback으로 `/cmd_vel`을 생성한다. GPS/RTK는 후속 global layer에서
목적지와 선호 진행방향을 제공하며 local collision avoidance를 대체하지 않는다.

## Rationale and evidence

local path는 RViz, 기록, 후속 controller 교체에 사용할 수 있고 command는 기존 gamepad
mux와 watchdog에 바로 연결할 수 있다. 둘을 같은 plan에서 만들면 path와 command의
의미가 어긋나지 않는다. costmap timestamp를 path와 latency 측정까지 보존하면 오래된
scene에 대한 action을 탐지할 수 있다.

## Alternatives considered

- GPS-only 2D path: 전역 진행에는 적합하지만 근거리 gap 회피에는 부적합하다.
- steering/speed만 발행: 단순하지만 선택한 corridor를 검증·재사용하기 어렵다.
- Nav2 global/local stack 즉시 통합: 장기 방향이지만 현재 센서·TF·cost policy가
  미검증이라 단계적 검증에 불리하다.

## Consequences

`/adom/navigation/local_path`가 planning 출력이고 `/cmd_vel`은 path controller 출력이다.
GPS를 추가할 때는 global path 또는 relative desired heading으로 결합한다. 19개 Semantic20
class cost는 초기 후보이며 현장 검증 전 확정 사실이 아니다. depth, camera info 또는
TF가 끊기면 costmap/planner watchdog이 정지시켜야 한다.

## Validation and rollback

synthetic grid에서 직진, 좌우 회피, lethal stop, Semantic20 ID 10 log 투영과 ignore 255
제외를 검사한다. rosbag shadow mode에서 path와 command를 기록한 뒤 wheels-off와
0.3 m/s 이하 저속 순서로 검증한다. 문제가 있으면 planner를 정지하고 기존 manual mode와
watchdog neutral을 유지한다.
