# 0007. Camera-only data collection

- Status: Accepted
- Date: 2026-08-07
- Owners: ADOM team
- Supersedes: none

## Context

실차 수집 launch는 ZED 2i와 GNSS를 함께 시작했고, rosbag은 ZED 토픽뿐 아니라
`/fix`, `/joy`, `/drive`, 제어 모드도 기록했다. 현재 D-5 신규 데이터의 목적은
RGB 기반 semantic perception 학습 데이터 확보이며, 위치·주행 재구성 데이터까지
동기 수집하는 것은 PoC 범위를 넘는다.

## Decision

`adom_bringup data_collection.launch.py`는 GNSS를 시작하지 않는다. `data_recorder`는
ZED의 `/rgb` 하위 토픽만 rosbag에 기록하며 depth, point cloud, IMU, GNSS와 제어
토픽을 제외한다. 기본 capture 위치는 저장소 루트 기준 상대경로 `data/captures`로
둔다.

## Rationale and evidence

카메라 데이터만 수집하면 장치 의존성, 저장량, 현장 점검 범위를 줄이면서 현재
학습 데이터 목적을 충족한다. GPS trajectory나 제어 입력을 이용한 주행 재구성은
D-5 성공 조건이 아니다.

## Alternatives considered

- GNSS 노드는 실행하되 `/fix`만 rosbag에서 제외: 불필요한 장치 의존성이 남는다.
- 제어 토픽은 계속 기록: 주행 재구성에는 유용하지만 현재 데이터셋 목적에는
  필요하지 않다.

## Consequences

수집 세션으로 GPS trajectory나 운전자 명령을 사후 재구성할 수 없다. ZED wrapper가
발행하는 정확한 하위 토픽 집합은 설치 버전과 카메라 설정에 따르므로 현장에서
`ros2 topic list -t`와 `ros2 bag info`로 확인한다. 상대 capture 경로는
`ADOM_REPO_ROOT`가 설정되면 그 위치, 아니면 launch를 실행한 디렉터리를 기준으로
해석한다.

## Validation and rollback

launch description에서 GNSS가 비활성화되고 recorder 정규식이
`^/zed(/.*)?/rgb(/.*)?$`인지 정적 검사한다. 필요하면 새 decision record로 수집
범위를 다시 확장하고 launch와 recorder 설정을 함께 변경한다.
