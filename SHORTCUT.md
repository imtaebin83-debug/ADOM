# Jetson ADOM 단축 명령어

Jetson의 `~/.bashrc`에 정의된 ADOM용 Bash 함수 사용 안내다. 새 터미널을 열거나
아래 명령을 실행한 뒤 사용할 수 있다.

```bash
source ~/.bashrc
```

기본 환경은 ROS 2 Jazzy와 `ROS_DOMAIN_ID=11`을 사용한다.

## 명령어 요약

| 명령어 | 용도 | 종료 방법 |
| --- | --- | --- |
| `car` | 게임패드 차량 제어 빌드 및 실행 | `Ctrl-C` |
| `cap` | ZED RGB 데이터 수집 빌드 및 실행 | `Ctrl-C` |
| `pwm` | PCA9685 PWM 출력값 확인 | `Ctrl-C` |
| `zed` | rosbag 녹화 상태 확인 | `Ctrl-C` |
| `zedgui` | ZED Explorer GUI 실행 | 창 닫기 |
| `up` | 로컬 작업을 폐기하고 원격 `main`으로 갱신 | 실행 전 주의 |

## `car` — 차량 제어

```bash
car
```

다음 작업을 순서대로 수행한다.

1. `~/ADOM/ros2_ws`로 이동
2. ROS 2 Jazzy 환경 활성화
3. `adom_control` 빌드
4. workspace overlay 활성화
5. `gamepad_control.launch.py` 실행

실행되는 원래 ROS 명령은 다음과 같다.

```bash
ros2 launch adom_control gamepad_control.launch.py
```

차량을 띄우기 전에는 LiPo를 분리하고 바퀴를 지면에서 띄운 상태에서 먼저 시험한다.

## `cap` — ZED RGB 데이터 수집

```bash
cap
```

`adom_control`과 `adom_bringup`을 빌드한 뒤 다음 launch를 실행한다.

```bash
ros2 launch adom_bringup data_collection.launch.py
```

게임패드의 Y 버튼으로 녹화를 시작하거나 중지한다. rosbag에는 ZED의 `/rgb` 하위
토픽만 기록하며 GPS, depth, point cloud, IMU, `/joy`, `/drive`는 기록하지 않는다.

현재 `cap` 함수는 `~/ADOM/ros2_ws`에서 launch를 실행한다. `ADOM_REPO_ROOT`를 별도로
설정하지 않았다면 상대 저장경로 `data/captures`는 다음 위치로 해석된다.

```text
~/ADOM/ros2_ws/data/captures
```

저장소 루트의 `~/ADOM/data/captures`를 사용하려면 `~/.bashrc`의 ADOM 환경 설정에
다음을 추가한다.

```bash
export ADOM_REPO_ROOT="$HOME/ADOM"
```

## `pwm` — PWM 디버깅

```bash
pwm
```

이름에 `pwm`이 포함된 ROS 토픽을 먼저 출력한 뒤 아래 토픽을 계속 표시한다.

```text
/adom/control/pwm_us    std_msgs/msg/Float64MultiArray
```

출력은 `Ctrl-C`로 종료한다. 토픽이 보이지 않으면 먼저 `car`를 실행하고
`ROS_DOMAIN_ID`가 다른 터미널과 동일한지 확인한다.

## `zed` — 녹화 상태 확인

```bash
zed
```

다음 상태 토픽을 계속 표시한다.

```text
/adom/recording/status    std_msgs/msg/String
```

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

ZED Explorer를 `-all` 옵션으로 실행한다. Jetson에 디스플레이 세션이 없거나 SSH에서
GUI forwarding을 사용하지 않으면 창이 열리지 않을 수 있다. ROS 녹화 상태 확인은
`zed`, 카메라 GUI 확인은 `zedgui`를 사용한다.

## `up` — 원격 main으로 강제 갱신

> **경고:** `up`은 커밋하지 않은 tracked 변경과 모든 untracked 파일을 삭제한다.
> 삭제된 파일은 Git으로 복구할 수 없을 수 있다.

```bash
up
```

내부적으로 다음 작업을 수행한다.

1. `git status --short`로 현재 변경 출력
2. `git reset --hard HEAD`로 tracked 변경 폐기
3. `git clean -fd`로 untracked 파일과 디렉터리 삭제
4. `main` 브랜치로 이동
5. `origin/main`을 fast-forward 방식으로 pull
6. 최신 커밋 출력

다음 항목이 하나라도 있으면 `up`을 실행하지 않는다.

- 아직 커밋하지 않은 코드나 설정
- 새로 수집했지만 Git에서 제외되지 않은 파일
- 보존해야 하는 untracked 문서나 실험 결과
- 다른 브랜치에서 작업 중인 변경

실행 전 최소한 다음 명령으로 상태를 직접 확인한다.

```bash
cd ~/ADOM
git status --short
```

변경을 보존해야 한다면 먼저 commit하거나 별도 위치에 백업한 뒤 실행한다.

## 자주 발생하는 문제

### `Package '<name>' not found`

workspace를 다시 빌드하고 overlay를 활성화한다.

```bash
cd ~/ADOM/ros2_ws
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install
source install/setup.bash
```

### 다른 터미널에서 토픽이 보이지 않음

모든 터미널에서 동일한 domain을 사용해야 한다.

```bash
export ROS_DOMAIN_ID=11
source /opt/ros/jazzy/setup.bash
source ~/ADOM/ros2_ws/install/setup.bash
```

### 함수 변경사항이 적용되지 않음

```bash
source ~/.bashrc
type car cap pwm zed zedgui up
```
