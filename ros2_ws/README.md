# ROS 2 Workspace

이 mono repo 안에서 관리하는 ROS 2 **Jazzy** (Ubuntu 24.04 Noble) colcon 워크스페이스다.

## 원칙

- `src/adom`의 공통 Python/ML 로직을 ROS2 node에서 호출한다. 인지, costmap, 플래너의 실제
  알고리즘은 저장소 루트의 `src/`에 있고, 여기의 패키지는 sensor I/O, topic/message 변환,
  launch/config, runtime wiring을 담당한다.
- 표준 ROS 메시지만 사용한다. 커스텀 메시지 패키지를 두지 않는다.
- `/emergency_stop`과 command timeout은 `adom_control`이 직접 처리한다.
- 하드웨어를 초기화하는 launch는 기본값을 안전 쪽으로 둔다. PCA9685 PWM 출력은 명시적으로
  켜야 동작한다.

## 패키지

| 패키지 | 역할 |
| --- | --- |
| `adom_description` | 차량 URDF와 센서 TF |
| `adom_sensors` | ZED 2i, RTK GNSS launch/config 어댑터 |
| `adom_perception_ros` | Semantic20 퍼셉션 노드와 마스크 컬러라이저 |
| `adom_costmap_ros` | Semantic20 / Cost4 costmap 어댑터 |
| `adom_localization` | ZED VIO + RTK GNSS dual-EKF 구성 |
| `adom_planning` | Nav2 설정, RTK waypoint, 순차 GPS executor |
| `adom_control` | `/cmd_vel` → Ackermann → PCA9685 PWM, 게임패드 제어, 데이터 레코더 |
| `adom_logging` | GPS trail과 bounded autonomy rosbag 세션 (기록 전용) |
| `adom_bringup` | 상위 launch: 차량, 데이터 수집, rule autonomy, low-level autonomy |

패키지별 토픽 계약과 실행 방법은 각 패키지 README에 있다. 목록 요약은
[`src/README.md`](src/README.md)를 참고한다.

## 빌드

```bash
source /opt/ros/jazzy/setup.bash
cd ros2_ws
colcon build --symlink-install
source install/setup.bash
```

Jetson에서는 의존성 설치와 빌드를 한 번에 처리하는 진입점을 쓴다.

```bash
bash scripts/install_jetson_control.sh
```

`build/`, `install/`, `log/`는 git에서 제외된다.

## 실행

가장 안전한 순서는 센서 → TF/localization 확인 → 인지 → autonomy다.

```bash
# 1. 기본 차량 스택 (planning 비활성, control dry-run)
ros2 launch adom_bringup vehicle.launch.py

# 2. 인지 + costmap + 플래너 + 컨트롤러 + safety mux + rosbag
ros2 launch adom_bringup low_level_autonomy.launch.py \
  model_config:="$ADOM_MODEL_CONFIG" checkpoint:="$ADOM_CHECKPOINT"
```

프로세스가 올라와도 차량은 STOPPED로 시작하며, 게임패드 A 버튼을 눌러야 autonomous command가
`/drive`로 전달된다. 실제 PWM은 shadow / wheels-off 검증 후에만 `start_pca9685:=true`로 켠다.
dry-run은 `gamepad_control.launch.py start_pca9685:=false`를 사용한다.

자세한 Jetson 운영 절차는 [`SHORTCUT.md`](../SHORTCUT.md), 차량 ESC 설정은
[`RC_SETTING.md`](../RC_SETTING.md)를 따른다.
