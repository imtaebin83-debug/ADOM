# adom_control

## Local path control

`local_path_control`은 `/adom/navigation/local_path`를 pure-pursuit 방식으로 추종하고
`/cmd_vel`을 발행한다. `/fix`의 연속 위치 차이로 지상속도를 구하고 ZED IMU의 종방향
가속도를 짧게 적분한 뒤 GPS 속도로 보정한다. IMU·GPS·path 중 하나라도 timeout이거나
path가 비어 있으면 zero command를 발행한다.

```bash
ros2 launch adom_control local_path_control.launch.py
ros2 topic echo /adom/control/local_path_status
```

기본 입력 topic은 `/zed/zed_node/imu/data`와 `/fix`이며 실제 장치 topic이 다르면
`local_path_control.yaml`에서 변경한다. IMU x축 bias와 GPS covariance 기준은 target
차량에서 정지·직진 로그로 보정하기 전 검증값이 아니다.

F1TENTH 호환 `/drive` (`AckermannDriveStamped`)를 PCA9685 CH0/CH1 PWM으로
출력한다. 8BitDo Ultimate C 2.4G 게임패드의 매뉴얼/자율 모드 전환을 지원한다.
PWM 노드는 항상 실제 PCA9685 하드웨어를 초기화하고 PWM을 출력한다.

## 게임패드 조작

- `X`: 매뉴얼 모드
- `A`: 자율주행 모드
- `B`: 정지 모드(소프트웨어 정지)
- `Y`: ZED 카메라 rosbag 수집 시작/중지
- 왼쪽 스틱(LTS) 상/하: 전진/가속 조절
- 오른쪽 스틱(RTS) 좌/우: 조향

프로그램은 항상 `stopped` 모드로 시작한다. `X`를 누른 뒤 두 스틱을 한 번
중앙에 놓아야 매뉴얼 입력이 활성화된다. 매뉴얼 모드에서 게임패드 메시지가
0.5초 끊기면 중립을 출력한다. `A`를 누르면 매뉴얼 명령을 즉시 버리고 새
자율주행 명령을 기다리며, 자율주행 명령도 0.25초 끊기면 중립이 된다.

현재 모드는 다음 토픽에서 확인한다.

```bash
ros2 topic echo /adom/control/mode
```

## ZED 2i 카메라 데이터 수집

`gamepad_control.launch.py`는 `data_recorder`도 기본 실행한다. Y 버튼을 한 번 누르면
수집을 시작하고 다시 누르면 중지한다. 한 세션이 10 GB에 도달해도 자동으로
중지한다. 결과는 기본적으로 저장소 기준 `data/captures/<시각>/`에 저장되며 이 경로는
git에서 제외된다. rosbag에는 ZED의 `/rgb` 하위 토픽만 기록하며 depth, point cloud,
IMU, GPS `/fix`, `/joy`, `/drive`, 제어 모드는 기록하지 않는다. 상태는 다음
명령으로 확인한다.

```bash
ros2 topic echo /adom/recording/status
```

Y의 기본 `record_button`은 4이다. `/joy`에서 실제 Y 인덱스가 다르면
`config/vehicle.yaml`의 `data_recorder.record_button`을 수정한 뒤 다시 빌드한다.
다른 상대 저장 위치가 필요하면 저장소 루트에서 다음처럼 지정한다.

```bash
ros2 launch adom_control gamepad_control.launch.py \
  capture_root:=data/alternate-captures
```

## Jetson에서 컨트롤러 연결 확인

2.4G USB 수신기를 Jetson에 연결하고 컨트롤러를 켠다. Ultimate C는 X-input과
D-input 모드를 지원한다. 기본 YAML은 일반적인 Linux X-input 배열을 사용하지만
커널/연결 모드에 따라 축과 버튼 번호가 달라질 수 있으므로 실차 구동 전에 반드시
확인한다.

컨트롤러 전원이 꺼진 상태에서 `X + Home`으로 켜면 X-input, `B + Home`으로
켜면 D-input 모드가 선택되고 마지막 모드가 저장된다. 먼저 X-input을 시험하고
Jetson에서 장치가 나타나지 않으면 D-input으로 다시 연결한다.

```bash
source /opt/ros/jazzy/setup.bash
source ros2_ws/install/setup.bash

lsusb
ls -l /dev/input/js* /dev/input/event* 2>/dev/null
ros2 run joy joy_node
```

`/dev/input` 권한 오류가 발생하면 다음 명령 후 로그아웃하고 다시 SSH로 접속한다.

```bash
sudo usermod -aG input "$USER"
```

다른 터미널에서 다음 명령을 실행하고 LTS 세로, RTS 가로, `X`, `A`, `B`, `Y`를
하나씩 움직이거나 누른다.

```bash
source /opt/ros/jazzy/setup.bash
ros2 topic echo /joy
```

기본 매핑은 다음과 같다.

| 입력 | YAML 파라미터 | 기본 인덱스 |
|---|---|---:|
| RTS 좌/우 | `right_stick_x_axis` | 2 |
| LTS 상/하 | `left_stick_y_axis` | 1 |
| X / 매뉴얼 | `manual_button` | 3 |
| A / 자율주행 | `autonomous_button` | 0 |
| B / 정지 | `stop_button` | 1 |
| Y / 데이터 수집 | `record_button` | 4 |

실제 `/joy` 배열과 다르면 `config/vehicle.yaml`의 `gamepad_control` 항목만
수정한다. 스틱 방향이 반대면 `steering_axis_scale` 또는
`throttle_axis_scale`을 `-1.0`으로 바꾼다. 확인이 끝나면 테스트용
`joy_node`를 `Ctrl-C`로 종료한다. launch 파일이 자체적으로 `joy_node`를
실행하므로 두 개를 동시에 띄우지 않는다.

`vehicle.yaml`의 조향각 설정(`*_deg`)은 degree 단위다. ROS 표준 메시지인
`AckermannDrive.steering_angle`과 `Twist.angular.z`는 radian 단위를 유지하며,
제어 노드가 메시지 경계에서 자동으로 변환한다.

## 빌드 및 ROS 2 활성화

```bash
cd ros2_ws
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install --packages-select adom_control
source install/setup.bash
```

LiPo를 분리하고 바퀴를 지면에서 띄운 상태에서 게임패드, 모드 선택기,
PCA9685 드라이버를 한 번에 실행한다.

```bash
ros2 launch adom_control gamepad_control.launch.py
```

다른 터미널에서 최종 명령과 계산된 PWM을 확인한다.

```bash
source /opt/ros/jazzy/setup.bash
source ros2_ws/install/setup.bash
ros2 topic echo /drive
ros2 topic echo /adom/control/pwm_us
```

USB 게임패드가 `/dev/input/js1`로 잡히는 등 장치 번호가 0이 아니면 다음처럼
실행한다.

```bash
ros2 launch adom_control gamepad_control.launch.py device_id:=1
```

이미 `vehicle.launch.py`가 `pca9685_control`을 실행 중이라면 중복 실행을 막는다.

```bash
ros2 launch adom_control gamepad_control.launch.py start_pca9685:=false
```

## 자율주행 입력 연결

기본 설정은 Nav2가 발행하는 `/cmd_vel` (`Twist`)을 자율주행 입력으로 받는다.
이 명령은 `A`를 눌러 자율주행 모드에 들어간 동안에만 `/drive`로 전달된다.

F1TENTH 형식의 planner가 `AckermannDriveStamped`를 출력한다면
`vehicle.yaml`을 다음처럼 변경하고 `/drive/autonomous`로 발행한다.

```yaml
gamepad_control:
  ros__parameters:
    autonomous_input_type: ackermann
    autonomous_drive_topic: /drive/autonomous
```

`pca9685_control.enable_cmd_vel`은 `false`로 유지해야 한다. `/cmd_vel`을 PWM
노드에 직접 연결하면 게임패드 모드 선택기를 우회하게 된다.

## 실차 활성화 전 점검

1. LiPo를 분리하고 I2C 주소 `0x40` 인식을 확인한다.
2. 바퀴를 지면에서 띄운다.
3. 조향 center/left/right 기계 한계를 측정한다.
4. XL-5 neutral/arming 및 최소 전진 PWM을 측정한다.
5. `B` 정지, 게임패드 연결 해제, 자율 명령 중단 시 중립을 확인한다.
6. 물리적 전원 차단 수단을 준비한 뒤 LiPo를 연결한다.

PCA9685는 Ubuntu의 `python3-smbus`를 통해 `/dev/i2c-N`에 직접 접근한다.
`config/vehicle.yaml`의 `i2c_bus`는 `i2cdetect -l`에서 확인한 40핀 헤더 버스와
일치해야 하며, 주소 `0x40` 또는 장치 권한을 확인하지 못하면 노드는 즉시 오류로
종료된다. 엔코더가 없는
open-loop throttle이므로 명령의 `m/s` 값은 실제 측정 속도를 보장하지 않는다.
소프트웨어 정지는 물리적인 LiPo/ESC 차단 장치를 대체하지 않는다.

Jetson 설치, I2C 배선 및 PWM 캘리브레이션 상세 절차는
`docs/setup-guides/jetson-console-control.md`를 참고한다.
