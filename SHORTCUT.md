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
| `gps` | 없음 | GPS 입력, 주기 및 localization 출력 | 자동 종료 |
| `up` | 없음 | 로컬 변경 삭제 후 `origin/jetson` 갱신 | 실행 전 주의 |
| `t0` | `adom_sensors` | ZED 2i 및 GNSS 센서 | `Ctrl-C` |
| `t1` | `adom_description` | 차량 URDF와 TF | `Ctrl-C` |
| `t2` | `adom_localization` | local/global EKF와 NavSat 변환 | `Ctrl-C` |
| `t3` | `adom_control` | 게임패드와 PCA9685 실출력 제어 | `Ctrl-C` |
| `t4` | `adom_perception_ros` | Semantic20 CUDA perception | `Ctrl-C` |
| `t5` | `adom_costmap_ros`, `adom_planning` | Semantic20 costmap, local planner, controller | `Ctrl-C` |

각 빌드 함수는 같은 `~/ADOM/ros2_ws/build`, `install`, `log`를 공유한다. 두 터미널에서
`colcon build`를 동시에 실행하지 않는다. 최초 전체 실행은 `t0`부터 `t5`까지 순서대로
빌드하고 실행한다.

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

## `gps` — GPS와 localization 상태 점검

```bash
gps
```

ROS 2 Jazzy와 ADOM workspace overlay를 source한 뒤 다음 항목을 순서대로 확인한다.

1. `/fix` publisher/subscriber와 QoS 상세 정보
2. `/fix` 메시지 1개(최대 5초 대기)
3. `/fix` 수신 주기(6초 측정)
4. `/odometry/gps` 메시지 1개
5. `/odometry/global` 메시지 1개
6. `/gps/filtered` 메시지 1개

내부에서 실행하는 핵심 명령은 다음과 같다.

```bash
ros2 topic info /fix --verbose
timeout 5s ros2 topic echo /fix --once
timeout 6s ros2 topic hz /fix
timeout 5s ros2 topic echo /odometry/gps --once
timeout 5s ros2 topic echo /odometry/global --once
timeout 5s ros2 topic echo /gps/filtered --once
```

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

## `t2` — localization

```bash
t2
```

빌드와 실행 명령:

```bash
colcon build --symlink-install --packages-select adom_localization
ros2 launch adom_localization localization.launch.py
```

local EKF, global EKF와 `navsat_transform_node`를 실행한다. 주요 출력은
`/odometry/local`, `/odometry/global`, `/odometry/gps`, `/gps/filtered`다.

## `t3` — 게임패드와 PCA9685 제어

```bash
t3
```

빌드와 실행 명령:

```bash
colcon build --symlink-install --packages-select adom_control
ros2 launch adom_control gamepad_control.launch.py start_pca9685:=true
```

이 함수는 PCA9685 실출력을 명시적으로 활성화한다. 항상 STOPPED 상태에서 시작하고,
바퀴를 띄운 상태에서 neutral과 watchdog을 검증한 뒤에만 자율 모드를 승인한다.

## `t4` — Semantic20 perception

```bash
t4
```

다음 모델 설정을 사용하도록 정의되어 있다.

```bash
export ADOM_MODEL_CONFIG="$HOME/ADOM/configs/adom/phase1_semantic20/segformer_b0_stage2_e1_combined.py"
export ADOM_CHECKPOINT="<CHECKPOINT_ABSOLUTE_PATH>"
export PYTHONPATH="$HOME/ADOM/src${PYTHONPATH:+:$PYTHONPATH}"
```

빌드와 실행 명령:

```bash
colcon build --symlink-install --packages-select adom_perception_ros
python3 -c 'import torch, mmcv, mmseg, adom; print("Perception imports: OK")'
ros2 launch adom_perception_ros perception.launch.py \
  model_config:="$ADOM_MODEL_CONFIG" \
  checkpoint:="$ADOM_CHECKPOINT" \
  device:=cuda:0
```

> **실행 전 필수:** `~/.bashrc`의 `<CHECKPOINT_ABSOLUTE_PATH>`를 Jetson에 실제로 존재하며
> 위 Semantic20 config와 대응하는 checkpoint 절대 경로로 바꾼다.

함수는 config와 checkpoint가 읽히는지 검사하고 PyTorch, MMCV, MMSegmentation, ADOM
import를 검증한다. 현재 검사 실패 경로에 `exit 1`이 들어 있으므로 실패하면 `t4`만
끝나는 것이 아니라 현재 SSH shell 자체가 종료된다. shell을 유지하려면 `exit 1`을
`return 1`로 바꾸는 것이 안전하다.

## `t5` — Semantic20 local planning

```bash
t5
```

빌드와 실행 명령:

```bash
colcon build --symlink-install \
  --packages-select adom_costmap_ros adom_planning
ros2 launch adom_planning semantic20_local_planning.launch.py
```

이 launch는 다음 세 요소를 함께 실행한다.

- `adom_costmap_ros`: Semantic20 mask, registered depth, camera info와 TF를 이용한 costmap
- `adom_planning`: Ackermann corridor local planner
- `adom_control`: local path controller와 `/cmd_vel` 출력

`adom_control` 코드를 수정한 경우 `t3`에서 해당 패키지를 다시 빌드하고 `t3`와 `t5`를
모두 재시작한다.

## 문제 발생 시 재빌드 원칙

특정 파트만 수정했다면 해당 launch를 `Ctrl-C`로 종료하고 담당 함수를 다시 실행한다.

| 수정한 경로 | 다시 실행할 함수 |
| --- | --- |
| `ros2_ws/src/adom_sensors` | `t0` |
| `ros2_ws/src/adom_description` | `t1` |
| `ros2_ws/src/adom_localization` | `t2` |
| `ros2_ws/src/adom_control` | `t3`; local path controller 사용 시 `t5`도 재시작 |
| `ros2_ws/src/adom_perception_ros` | `t4` |
| `ros2_ws/src/adom_costmap_ros` | `t5` |
| `ros2_ws/src/adom_planning` | `t5` |
| `ros2_ws/src/adom_bringup` | `rec` |

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
