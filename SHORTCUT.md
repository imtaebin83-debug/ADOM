# Jetson ADOM 단축 명령어

Jetson의 `~/.bashrc`에 정의한 ADOM용 Bash 함수 사용 안내다. 새 터미널을 열거나
함수를 수정한 뒤에는 다음 명령으로 설정을 다시 읽는다.

```bash
source ~/.bashrc
```

이 문서는 저장소가 `~/ADOM`, ROS 배포판이 ROS 2 Jazzy인 현재 함수 정의를 기준으로
한다. 여러 터미널에서 ROS 통신을 사용할 때는 모든 터미널의 `ROS_DOMAIN_ID`와 RMW
설정이 같아야 한다.

## 명령어 요약

| 단축어 | 빌드 대상 | 실행하거나 확인하는 항목 | 종료 방법 |
| --- | --- | --- | --- |
| `car` | `adom_control` | 게임패드 차량 제어 | `Ctrl-C` |
| `rec` | `adom_control`, `adom_bringup` | 데이터 수집 launch | `Ctrl-C` |
| `pwm` | 없음 | PWM 관련 토픽과 `/adom/control/pwm_us` | `Ctrl-C` |
| `zed` | 없음 | `/adom/recording/status` | `Ctrl-C` |
| `zedgui` | 없음 | ZED Explorer GUI | 창 닫기 |
| `gps` | 없음 | GPS 입력, 주기 및 기록용 trail 출력 | 자동 종료 |
| `up` | 없음 | 로컬 변경 삭제 후 `origin/jetson` 갱신 | 실행 전 주의 |
| `t0` | `adom_sensors` | ZED 2i 및 GNSS 센서 | `Ctrl-C` |
| `t1` | `adom_description` | 차량 URDF와 TF | `Ctrl-C` |
| `t2` | `adom_logging` 및 의존 패키지 | GPS trail과 autonomy rosbag 자동 기록 | `Ctrl-C` |
| `t3` | `adom_control` | 게임패드, safety mux와 PCA9685 실출력 제어 | `Ctrl-C` |
| `t4` | `adom_perception_ros` | Semantic20 CUDA perception | `Ctrl-C` |
| `t5` | `adom_costmap_ros`, `adom_planning`, `adom_control` | Semantic20 costmap, direction-tree planner, controller | `Ctrl-C` |

각 빌드 함수는 같은 `~/ADOM/ros2_ws/build`, `install`, `log`를 공유한다. 두 터미널에서
`colcon build`를 동시에 실행하지 않는다. 최초 전체 실행은 `t0`부터 `t5`까지 순서대로
빌드하고 실행한다. 새 구조에서는 `t2`가 localization을 실행하지 않으며 GPS는 주행
판단이 아닌 이동경로 기록에만 사용된다.

현재 권장 터미널 구성은 다음과 같다.

```text
t0  ZED RGB/depth/IMU + GNSS /fix
t1  vehicle description + camera/base_link TF
t2  GPS trail + autonomy rosbag recorder
t3  gamepad safety mux + PCA9685
t4  Semantic20 perception
t5  semantic costmap + 3-depth direction tree + local controller
```

전체 stack이 올라와도 `t3`는 STOPPED로 시작한다. `t0`~`t5`의 health check와 wheels-off
검증이 끝난 뒤에만 게임패드 A 버튼으로 autonomous mode를 승인한다.

`low_level_autonomy.launch.py`는 `t2`~`t5` 역할을 한 프로세스 그룹으로 묶은 대체
실행법이다. 기존 다중 터미널 방식에서는 이 launch를 추가로 실행하지 않는다. 함께
실행하면 perception, planner, controller, gamepad와 recorder가 중복된다. 통합 방식이
필요한 경우에만 `t0`, `t1`을 실행한 뒤 별도 터미널에서 다음을 사용한다.

```bash
export ADOM_REPO_ROOT="$HOME/ADOM"
ros2 launch adom_bringup low_level_autonomy.launch.py \
  model_config:="$ADOM_MODEL_CONFIG" \
  checkpoint:="$ADOM_CHECKPOINT" \
  start_pca9685:=true
```

## `car` — 차량 제어

```bash
car
```

`~/ADOM/ros2_ws`에서 ROS 2 Jazzy를 source하고 다음 작업을 수행한다.

```bash
colcon build --symlink-install --packages-select adom_control
source install/setup.bash
ros2 launch adom_control gamepad_control.launch.py
```

launch의 `start_pca9685` 기본값은 `true`다. 따라서 `car`는 PCA9685 노드까지 실행한다.
차량을 지면에 놓기 전에 LiPo를 분리하거나 바퀴를 띄운 상태에서 neutral, steering,
watchdog 동작을 먼저 확인한다.

## `rec` — 데이터 수집

```bash
rec
```

다음 두 패키지를 빌드하고 데이터 수집 launch를 실행한다.

```bash
colcon build --symlink-install \
  --packages-select adom_control adom_bringup
source install/setup.bash
ros2 launch adom_bringup data_collection.launch.py
```

현재 함수는 `~/ADOM/ros2_ws`에서 launch를 실행한다. 상대 경로로 지정된 결과물은 이
작업 디렉터리를 기준으로 생성될 수 있다.

게임패드의 Y 버튼으로 녹화를 시작하거나 중지한다. rosbag에는 ZED의 `/rgb` 하위
토픽만 기록하며 GPS, depth, point cloud, IMU, `/joy`, `/drive`는 기록하지 않는다.

`ADOM_REPO_ROOT`를 별도로 설정하지 않았다면 상대 저장 경로 `data/captures`는 다음
위치로 해석된다.

```text
~/ADOM/ros2_ws/data/captures
```

저장소 루트의 `~/ADOM/data/captures`를 사용하려면 `~/.bashrc`의 ADOM 환경 설정에
다음을 추가한다.

```bash
export ADOM_REPO_ROOT="$HOME/ADOM"
```

## `pwm` — PWM 출력 확인

```bash
pwm
```

ROS 2 Jazzy와 ADOM workspace overlay를 source한 다음, 이름에 `pwm`이 포함된 토픽을
출력하고 PWM 상태를 계속 표시한다.

```bash
ros2 topic list -t | grep pwm
ros2 topic echo /adom/control/pwm_us
```

출력은 `Ctrl-C`로 종료한다. 토픽이 없다면 먼저 `car` 또는 `t3`가 실행 중인지 확인한다.

## `zed` — 녹화 상태 확인

```bash
zed
```

다음 명령을 실행해 rosbag 녹화 상태를 계속 표시한다.

```bash
ros2 topic echo /adom/recording/status
```

`zed`라는 이름이지만 카메라 영상 토픽이 아니라 ADOM recorder의 상태 토픽을 확인하는
함수다. 출력은 `Ctrl-C`로 종료한다.

주요 JSON 필드는 다음과 같다.

| 필드 | 의미 |
| --- | --- |
| `recording` | 현재 녹화 중인지 여부 |
| `session` | 현재 session 디렉터리 |
| `size_bytes` | 현재 저장 크기 |
| `max_size_bytes` | session 최대 크기 |
| `reason` | 녹화 상태 또는 종료 사유 |

## `zedgui` — ZED Explorer

```bash
zedgui
```

다음 GUI 프로그램을 모든 장치 표시 옵션으로 실행한다.

```bash
/usr/local/zed/tools/ZED_Explorer -all
```

SSH에서 사용하려면 Jetson의 활성 디스플레이 세션 또는 X11 forwarding 설정이 필요할 수
있다.

## `gps` — GPS와 기록용 이동경로 점검

```bash
gps
```

ROS 2 Jazzy와 ADOM workspace overlay를 source한 뒤 다음 항목을 순서대로 확인한다.

1. `/fix` publisher/subscriber와 QoS 상세 정보
2. `/fix` 메시지 1개(최대 5초 대기)
3. `/fix` 수신 주기(6초 측정)
4. logging-only `/adom/logging/gps_path` 메시지 1개
5. `/adom/logging/gps_status` 메시지 1개

내부에서 실행하는 핵심 명령은 다음과 같다.

```bash
ros2 topic info /fix --verbose
timeout 5s ros2 topic echo /fix --once
timeout 6s ros2 topic hz /fix
timeout 5s ros2 topic echo /adom/logging/gps_path --once
timeout 5s ros2 topic echo /adom/logging/gps_status --once
```

기존 `gps` 함수가 `/odometry/gps`, `/odometry/global`, `/gps/filtered`를 확인한다면 위
명령으로 교체한다. `/adom/logging/gps_path`는 최초 valid fix를 원점으로 만든 기록용
local metric trail이며 TF localization이나 planning/control 입력이 아니다.

표준 `sensor_msgs/NavSatFix`만으로는 GNSS 수신기의 RTK `FIX`와 `FLOAT`를 완전히 구분할
수 없다. 해당 판정에는 실제 수신기가 발행하는 전용 status 토픽 확인이 추가로 필요하다.

## `up` — `jetson` 브랜치 강제 갱신

> **경고:** `up`은 커밋하지 않은 tracked 변경과 모든 untracked 파일·디렉터리를
> 삭제한다. `git clean -fd`로 삭제된 파일은 Git으로 복구할 수 없다.

```bash
up
```

내부적으로 다음 명령을 순서대로 실행한다.

```bash
cd ~/ADOM
git status --short
git reset --hard HEAD
git clean -fd
git switch jetson
git pull --ff-only origin jetson
git log -1 --oneline
```

아직 커밋하지 않은 코드, 설정, 문서 또는 보존해야 하는 untracked 결과물이 있으면
실행하지 않는다. 함수에는 중간 명령 실패 시 중단하는 `&&` 또는 `|| return`이 없으므로,
앞 명령이 실패해도 뒤 명령이 계속 실행될 수 있다는 점에도 주의한다.

## `t0` — 센서

```bash
t0
```

빌드와 실행 명령:

```bash
colcon build --symlink-install --packages-select adom_sensors
PYTHONNOUSERSITE=1 ros2 launch adom_sensors sensors.launch.py
```

ZED 2i wrapper와 GNSS serial driver를 실행한다. `PYTHONNOUSERSITE=1`은 이 launch
프로세스에만 적용된다.

## `t1` — 차량 description과 TF

```bash
t1
```

빌드와 실행 명령:

```bash
colcon build --symlink-install --packages-select adom_description
ros2 launch adom_description description.launch.py
```

차량 URDF를 읽고 `robot_state_publisher`를 실행한다.

## `t2` — GPS trail과 autonomy rosbag

```bash
t2
```

기존 `t2`의 `adom_localization localization.launch.py` 실행은 제거한다. `~/.bashrc`의
`t2` 함수를 다음 역할로 바꾼다.

```bash
t2() {
    cd "$HOME/ADOM/ros2_ws" || return
    source /opt/ros/jazzy/setup.bash
    colcon build --symlink-install --packages-up-to adom_logging || return
    source install/setup.bash
    export ADOM_REPO_ROOT="$HOME/ADOM"
    ros2 launch adom_logging autonomy_logging.launch.py \
      capture_root:=data/autonomy_bags
}
```

launch와 함께 rosbag이 자동으로 시작되며 `Ctrl-C` 시 metadata를 닫고 종료한다. 기본
결과 위치는 다음과 같다.

```text
~/ADOM/data/autonomy_bags/autonomy_<timestamp>/rosbag/
```

기록 대상은 perception mask/status, semantic costmap, local path와 선택 tree,
`/cmd_vel`, `/drive`, control/PWM/E-stop, IMU, raw `/fix`, GPS trail과 TF다. confidence와
BGR overlay는 live 성능을 위해 제외한다. 이 bag은 기존 RGB 학습 데이터용
`data/captures`와 분리된다.

## `t3` — 게임패드와 PCA9685 제어

```bash
t3
```

빌드와 실행 명령:

```bash
colcon build --symlink-install --packages-select adom_control
ros2 launch adom_control gamepad_control.launch.py \
  start_pca9685:=true start_data_recorder:=false
```

새 autonomy logging은 `t2`가 담당하므로 `start_data_recorder:=false`를 반드시 유지한다.
이를 빼면 RGB-only recorder와 autonomy recorder가 동시에 실행된다.

이 함수는 PCA9685 실출력을 명시적으로 활성화한다. 항상 STOPPED 상태에서 시작하고,
바퀴를 띄운 상태에서 neutral과 watchdog을 검증한 뒤에만 자율 모드를 승인한다.

## `t4` — Semantic20 perception

```bash
t4
```

`e49ad80`에서 perception 팀원이 기록한 Semantic20 SegFormer-B0 E0 모델 계약을
사용한다. 여기서 checkpoint는 목적지나 주행 경로가 아니라 학습된 perception 신경망의
가중치가 저장된 `.pth` 파일이다.

Git에는 checkpoint가 포함되지 않는다. perception 팀의 다음 파일을 Jetson으로 복사한다.

```text
best_mIoU_iter_6000.pth
```

권장 저장 위치는 다음과 같다. `models/checkpoints/`는 Git에서 제외된다.

```bash
mkdir -p "$HOME/ADOM/models/checkpoints/b0-e0"
# 전달받은 파일을 위 디렉터리에 복사한다.
```

`~/.bashrc`의 기존 `t4` 함수 전체를 다음 wrapper로 교체하고 설정을 다시 읽는다.

```bash
t4() {
    "$HOME/ADOM/scripts/run_jetson_t4.sh"
}

source ~/.bashrc
```

wrapper는 다음 B0-E0 config를 고정해 사용한다.

```text
configs/adom/export/segformer_b0_640x384_rellis3d.py
```

이 config는 `segformer_b0_stage2_e0_rellis.py`를 기반으로 하며 Semantic20 ID `0..18`,
ignore `255`, 640x384 resize/padding 계약을 사용한다. E1, B2 또는 Cost4 checkpoint를
같은 디렉터리에 넣지 않는다.

기본 디렉터리 밖의 파일을 사용하려면 실행 전에 정확한 B0-E0 파일을 지정한다.

```bash
export ADOM_CHECKPOINT="/absolute/path/to/best_mIoU_iter_6000.pth"
t4
```

script는 checkpoint가 없거나 여러 개면 launch를 시작하지 않고 원인을 출력한다.
별도 프로세스로 실행되므로 실패해도 현재 SSH shell은 종료되지 않는다. 빌드할 때
동일 workspace의 기존 install을 의도적으로 갱신하므로
`--allow-overriding adom_perception_ros`를 사용한다.

## `t5` — Semantic20 local planning

```bash
t5
```

빌드와 실행 명령:

```bash
colcon build --symlink-install \
  --packages-select adom_costmap_ros adom_planning
source install/setup.bash
ros2 launch adom_planning semantic20_local_planning.launch.py
```

`t3`가 먼저 최신 `adom_control`을 빌드하므로 `t5`에서 같은 package를 다시 빌드하지
않는다. 코드 변경 후에는 `t3`를 먼저 재실행하고 그 다음 `t5`를 재실행한다.

이 launch는 다음 세 요소를 함께 실행한다.

- `adom_costmap_ros`: Semantic20 mask, registered depth, camera info와 TF를 이용한 costmap
- `adom_planning`: 5방향을 3단계로 전개하는 Ackermann direction-tree planner
- `adom_control`: GPS를 사용하지 않는 local path controller와 `/cmd_vel` 출력

planner는 기본적으로 `[-20, -10, 0, 10, 20]°` 방향을 3단계 전개한 125개 path를
평가한다. `/adom/navigation/rule_status`의 `steering_sequence_deg`에서 선택한 방향열을
확인할 수 있으며, 매 cycle 첫 방향만 실행하고 최신 costmap에서 다시 계획한다.

`adom_control` 코드를 수정한 경우 `t3`와 `t5`를 모두 종료한 뒤 순서대로 다시 실행한다.

## 문제 발생 시 재빌드 원칙

특정 파트만 수정했다면 해당 launch를 `Ctrl-C`로 종료하고 담당 함수를 다시 실행한다.

| 수정한 경로 | 다시 실행할 함수 |
| --- | --- |
| `ros2_ws/src/adom_sensors` | `t0` |
| `ros2_ws/src/adom_description` | `t1` |
| `ros2_ws/src/adom_localization` | 현재 low-level autonomy stack에서는 사용하지 않음 |
| `ros2_ws/src/adom_logging` | `t2` |
| `ros2_ws/src/adom_control` | `t3`; local path controller 사용 시 `t5`도 재시작 |
| `ros2_ws/src/adom_perception_ros` | `t4` |
| `ros2_ws/src/adom_costmap_ros` | `t5` |
| `ros2_ws/src/adom_planning` | `t5` |
| `ros2_ws/src/adom_bringup` | `rec`; 통합 launch 사용 시 재빌드 |

## 자주 발생하는 문제

### `Package '<name>' not found`

해당 패키지 담당 함수를 순서에 맞게 먼저 실행하거나 직접 빌드한 뒤 overlay를 다시
source한다.

```bash
cd ~/ADOM/ros2_ws
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install --packages-select <PACKAGE_NAME>
source install/setup.bash
```

### 다른 터미널에서 토픽이 보이지 않음

모든 터미널에서 동일한 ROS domain과 RMW 구현을 사용해야 한다.

```bash
echo "$ROS_DOMAIN_ID"
echo "$RMW_IMPLEMENTATION"
```

### 함수 변경사항이 적용되지 않음

```bash
source ~/.bashrc
type car rec pwm zed zedgui gps up t0 t1 t2 t3 t4 t5
```
