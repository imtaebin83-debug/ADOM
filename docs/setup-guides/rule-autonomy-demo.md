# ADOM Rule Autonomy Demo

이 데모는 ZED 2i RGB/depth로 Cost4 semantic costmap을 만들고, Ackermann corridor rule
planner가 `/cmd_vel`을 발행한다. RTK GNSS, ZED VIO와 IMU는 `robot_localization`의
local/global EKF에 들어간다. gamepad control이 자율 모드로 승인한 명령만 PCA9685로
전달한다.

현재 rule planner는 로봇 주변에서 안전한 진행 방향을 고르는 local reactive demo다.
`/odometry/global`은 RViz와 localization 상태 확인에 사용하지만 RTK waypoint를 목적지로
추종하지는 않는다. 목적지 주행은 `adom_planning`의 Nav2/RTK waypoint 경로를 별도로
활성화해야 한다.

## Prerequisites

- ZED 2i와 RTK GNSS의 ROS topic이 실제 장치에서 발행될 것
- ADOM Cost4 checkpoint가 존재할 것
- ROS Python 환경에서 `adom`, PyTorch, MMSegmentation 1.2.2와 MMCV 2.1.0을 import할 수 있을 것
- `base_link`에서 ZED optical frame까지 TF가 연결될 것
- PCA9685 I2C/PWM과 조향/ESC pulse를 바퀴를 띄운 상태에서 보정할 것

저장소 루트에서 실제 checkpoint 위치를 지정한 뒤, 모든 터미널에서 다음 setup을
실행한다. `ADOM_REPO`를 특정 팀원의 홈 디렉터리로 고정하지 않는다.

```bash
export ADOM_REPO="$(git rev-parse --show-toplevel)"
export ADOM_MODEL_CONFIG="$ADOM_REPO/configs/adom/runtime/segformer_b0_640x384_rellis3d.py"
export ADOM_CHECKPOINT="<CHECKPOINT_PATH>"
cd "$ADOM_REPO/ros2_ws"
source /opt/ros/jazzy/setup.bash
source install/setup.bash
```

새 터미널에서는 위 환경변수와 ROS setup을 다시 적용한다. checkpoint는 Git에
커밋하지 않으며 `ADOM_CHECKPOINT`에는 해당 장비에서 읽을 수 있는 파일 경로를 넣는다.

## Recommended: three terminals

### Terminal 0 — sensors, TF and localization

```bash
ros2 launch adom_bringup vehicle.launch.py \
  start_planning:=false start_control:=false
```

다음을 함께 시작한다.

- vehicle URDF and static sensor TF
- ZED 2i RGB, registered depth, point cloud, VIO and IMU
- RTK GNSS `/fix`
- local EKF `/odometry/local`
- global RTK EKF `/odometry/global`

### Terminal 1 — perception, semantic costmap, rule planning and RViz

```bash
ros2 launch adom_bringup rule_autonomy.launch.py \
  model_config:="$ADOM_MODEL_CONFIG" \
  checkpoint:="$ADOM_CHECKPOINT" \
  device:=cuda:0
```

RViz에는 segmentation overlay, Cost4 grid, selected rule path, ZED point cloud, TF,
`/odometry/local`, `/odometry/global`이 표시된다.

### Terminal 2 — gamepad safety selector and PCA9685 control

먼저 LiPo를 분리하고 바퀴를 띄운 채 출력 없이 mode chain을 시험한다.

```bash
ros2 launch adom_control gamepad_control.launch.py start_pca9685:=false
```

PCA9685/ESC/steering calibration이 끝난 뒤에만 실제 PWM을 활성화한다.

```bash
ros2 launch adom_control gamepad_control.launch.py start_pca9685:=true
```

항상 STOPPED로 시작한다. `A` 버튼을 눌러야 rule planner의 `/cmd_vel`이 `/drive`와
PCA9685로 전달되고, `B` 버튼은 소프트웨어 정지를 요청한다.

## Expanded: one responsibility per terminal

### Terminal 0 — vehicle TF and sensors

```bash
ros2 launch adom_bringup vehicle.launch.py \
  start_localization:=false start_planning:=false start_control:=false
```

### Terminal 1 — ZED VIO/IMU and RTK localization

```bash
ros2 launch adom_localization localization.launch.py
```

### Terminal 2 — MMSeg perception

```bash
ros2 launch adom_perception_ros perception.launch.py \
  model_config:="$ADOM_MODEL_CONFIG" \
  checkpoint:="$ADOM_CHECKPOINT" device:=cuda:0
```

### Terminal 3 — RGB/depth semantic costmap

```bash
ros2 launch adom_costmap_ros semantic_costmap.launch.py
```

### Terminal 4 — Ackermann corridor rule planner

```bash
ros2 launch adom_planning rule_planning.launch.py
```

### Terminal 5 — gamepad selector and hardware control

```bash
ros2 launch adom_control gamepad_control.launch.py
```

### Terminal 6 — RViz monitoring only

```bash
rviz2 -d "$(ros2 pkg prefix --share adom_bringup)/config/rule_autonomy.rviz"
```

통합 launch가 이미 RViz를 실행한다면 Terminal 6은 필요 없다. 호스트 PC에서 RViz를
실행할 때는 로봇과 같은 LAN, ROS distribution, RMW와 `ROS_DOMAIN_ID`를 사용하고 로봇의
통합 launch에는 `start_rviz:=false`를 전달한다.

## Health checks before pressing A

```bash
ros2 topic hz /zed/zed_node/rgb/color/rect/image
ros2 topic hz /zed/zed_node/depth/depth_registered
ros2 topic hz /adom/perception/semantic_mask
ros2 topic hz /adom/navigation/semantic_costmap
ros2 topic hz /cmd_vel
ros2 topic echo /adom/perception/status
ros2 topic echo /adom/navigation/costmap_status
ros2 topic echo /adom/navigation/rule_status
ros2 topic echo /adom/control/mode
```

다음 조건 중 하나라도 만족하면 자율 모드를 승인하지 않는다.

- mask/costmap이 끊기거나 source timestamp가 stale인 경우
- `/fix`, IMU 또는 odometry가 기대한 rate로 나오지 않는 경우
- RViz에서 ZED point cloud, semantic grid와 `base_link` TF가 어긋나는 경우
- neutral/left/right/forward PWM을 실측하지 않은 경우
