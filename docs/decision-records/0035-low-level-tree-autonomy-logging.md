# 0035. Low-level direction-tree autonomy and session logging

- Status: Accepted
- Date: 2026-08-12
- Owners: ADOM team
- Supersedes: 0034의 single-corridor planning 및 GPS speed-feedback 부분

## Context

Semantic20 perception이 동작한 뒤의 다음 목표는 복잡한 global navigation이 아니라
콘솔에서 반복 실행할 수 있는 저속 직진·근거리 장애물 회피다. 기존 local planner는
한 개의 고정 조향 corridor만 비교했고 controller는 저속에서 잡음이 큰 GPS 위치 차분을
속도 feedback으로 요구했다. 또한 RGB 학습 데이터 rosbag과 자율주행 판단 evidence는
목적과 topic 구성이 다르다.

## Decision

Robot-frame semantic costmap에서 discrete Ackermann steering action을 3단계 tree로
전개한다. 기본 action은 -20, -10, 0, 10, 20도이며 3 m horizon에서 125개
root-to-leaf path를 평가한다. 위험도, lethal clearance, 조향 크기와 조향 변화가 가장
낮은 path를 선택하고 receding-horizon 방식으로 첫 action만 실행한다.

GPS `/fix`는 planning, localization, control에 연결하지 않는다. 최초 valid fix 기준의
짧은 local metric trail은 시각화·기록 목적으로만 발행한다. wheel-speed sensor가 없는
현재 controller는 기본 open-loop 속도 요청을 사용하고 IMU는 freshness watchdog과
진단 로그에 사용한다.

별도 `adom_logging` ROS package를 두고 autonomy launch와 함께 bounded rosbag을 자동
시작한다. perception mask/confidence/status, semantic costmap, selected local path와 tree
status, control command/status, final drive/PWM, emergency stop, IMU, raw GPS fix와 GPS
trail을 기록한다. 기존 `data/captures` RGB-only 수집 계약은 유지하고 autonomy bag은
`data/autonomy_bags`에 분리한다.

## Rationale and evidence

다단계 direction tree는 단일 고정 조향 arc보다 장애물 뒤의 복귀 방향까지 표현하면서
Nav2, global map 또는 GPS localization 없이 계산할 수 있다. 매 costmap에서 재계획하면
누적 localization 오차 없이 최신 근거리 관측에 반응한다. 모든 판단과 actuator 전단
상태를 같은 bag에 저장하면 camera timestamp부터 선택 path와 PWM까지 재현할 수 있다.

## Alternatives considered

- Nav2 + RTK global navigation: 현재 직진·근거리 회피 목표보다 범위와 검증 부담이 크다.
- 단일 steering corridor 선택: 단순하지만 회피 후 직진 복귀를 horizon 안에서 평가하지 못한다.
- GPS 차분 속도 feedback 유지: 0.25 m/s급 저속에서 위치 잡음 영향이 크다.
- 모든 데이터를 기존 RGB collection bag에 추가: 학습 데이터와 시스템 evidence의 목적,
  용량, 개인정보 범위를 불필요하게 결합한다.

## Consequences

planner는 전역 목표에 도달하지 않으며 local minima나 막힌 공간에서 정지할 수 있다.
GPS trail은 짧은 구간의 근사 기록이지 localization TF가 아니다. 실제 closed-loop speed
control은 wheel encoder 또는 검증된 vehicle odometry가 추가될 때까지 제공하지 않는다.
Semantic20 비용, PWM calibration, steering sign과 실차 속도는 wheels-off 및 저속 시험
전까지 미검증이다.

## Validation and rollback

synthetic grid에서 직진, 좌우 회피, lethal stop과 tree depth를 단위 테스트한다. ROS graph
및 bag topic은 shadow mode에서 확인하고 wheels-off 후 0.25 m/s 이하로 단계 상승한다.
문제가 있으면 `start_pca9685:=false` 또는 gamepad STOP을 사용하고 manual mode와 기존
0.25초 watchdog neutral을 유지한다.
