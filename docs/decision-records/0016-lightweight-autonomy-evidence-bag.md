# 0016. Lightweight autonomy evidence bag

- Status: Accepted
- Date: 2026-08-12
- Owners: ADOM team
- Supersedes: 0035와 0036의 live autonomy bag topic 범위

## Context

`rec`는 Semantic20 학습용 ZED RGB 수집이고 `t2`는 자율주행 동작 evidence 수집이지만,
기존 `t2` bag은 mask, costmap, path, IMU와 TF까지 기록했다. 이 토픽들은 크거나 주기가
높고, recorder 구독자가 추가되면 t5가 사용하는 동일 메시지의 DDS 직렬화·복사와 disk
I/O가 늘어난다. Jetson에서는 recorder 사용 시 perception latency 증가도 관측됐다.

사후 분석의 핵심 질문은 autonomous mode 진입 시점, blocked/driving 전환, 선택한 회피
조향 시퀀스, 계획/명령/추정 속도, 최종 drive/PWM, watchdog/E-stop 상태와 실제 GPS
이동경로다. 이 정보는 raster와 전체 path 없이 작은 상태·명령 메시지로 보존할 수 있다.

## Decision

`rec`의 RGB-only 수집 계약과 Y 버튼 동작은 변경하지 않는다. `t2`의 autonomy bag은
다음 경량 토픽만 자동 기록한다.

- perception/costmap/planner/controller status와 action latency
- planned speed, `/cmd_vel`, `/drive/autonomous`, 최종 `/drive`와 PWM
- control mode, emergency stop
- raw `/fix`와 GPS quality/status

카메라, Semantic20 mask/confidence/overlay, semantic costmap grid, local/rule path,
고주기 IMU, TF와 누적 `/adom/logging/gps_path`는 기록하지 않는다. GPS logger의 trail
publisher는 현장 시각화를 위해 유지하지만 bag의 GPS 경로는 raw `/fix` 시계열로
재구성한다.

## Rationale and evidence

`rule_status`는 blocked/driving, 첫 조향과 전체 선택 조향 시퀀스, 계획 속도, clearance와
score를 제공한다. `local_path_status`는 계획/명령/추정 속도, IMU 기반 수치, steering과
watchdog 원인을 제공한다. 따라서 회피 시점과 방법, 속도 변화를 분석하는 데 raster/path
메시지를 중복 기록할 필요가 없다. `/fix`는 각 위치 표본을 한 번씩 저장하는 반면 누적
`nav_msgs/Path`는 경로가 길어질수록 과거 점을 반복 직렬화하고 기록한다.

## Alternatives considered

- 기존 mask/costmap/path 유지: 판단 완전 재현성은 높지만 live t5 timeout 위험을 줄이는
  이번 목적과 맞지 않는다.
- t2에서 rosbag 완전 제거: 시스템 부하는 가장 작지만 시간축이 맞는 mode/회피/속도
  evidence를 잃는다.
- 누적 GPS Path 기록: 바로 시각화하기 쉽지만 동일한 과거 위치를 계속 중복 저장한다.
- CSV별도 logger 추가: 파일은 작지만 topic별 timestamp 정렬과 정상 종료 처리가 rosbag보다
  복잡하다.

## Consequences

경량 bag만으로 semantic mask나 costmap을 pixel/cell 단위로 재현할 수 없다. 해당 분석이
필요하면 actuator를 비활성화한 별도 진단 세션에서 선택적으로 기록한다. GPS trail은
bag 재생만으로 바로 나타나지 않을 수 있으며 `/fix`에서 다시 생성해야 한다.

## Validation and rollback

설정 테스트로 필수 numeric/status topic 포함과 mask/costmap/path/IMU/TF 제외를 확인한다.
Jetson에서는 `ros2 bag info`로 실제 topic을 확인하고 같은 장면에서 t2 on/off의 t5
callback 주기와 watchdog/timeout 발생 여부를 비교한다. timeout이 남으면 status 발행률과
storage write latency를 측정한 뒤 topic을 더 줄인다.
