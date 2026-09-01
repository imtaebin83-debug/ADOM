# adom_bringup

## Low-level autonomy console

ZED와 `/fix` sensor driver가 실행된 상태에서 perception, Semantic20 costmap, 3-depth
direction tree planner, local controller, gamepad safety mux와 autonomy rosbag을 한 번에
시작한다. 실제 PWM은 기본적으로 꺼져 있다.

```bash
export ADOM_REPO_ROOT="$(git rev-parse --show-toplevel)"
ros2 launch adom_bringup low_level_autonomy.launch.py \
  model_config:="$ADOM_MODEL_CONFIG" checkpoint:="$ADOM_CHECKPOINT"
```

shadow/wheels-off 검증 뒤에만 `start_pca9685:=true`를 추가한다. 프로세스가 올라와도
차량은 STOPPED로 시작하며 게임패드 A 버튼을 눌러야 autonomous command가 `/drive`로
전달된다. rosbag은 `data/autonomy_bags/`에서 자동 시작되고 Ctrl-C 시 닫힌다.

기본 실행은 planning을 의도적으로 비활성화하고 control을 dry-run으로 시작한다.

```bash
ros2 launch adom_bringup vehicle.launch.py
```

센서/TF/localization 검증 후에만 `start_planning:=true`를 사용한다.

Cost4 PyTorch 인지, semantic costmap, rule planner와 RViz는 별도 안전 launch로 실행한다.
저장소 루트에서 다음 변수에 실제 checkout과 checkpoint 위치를 기록한다.

```bash
export ADOM_REPO="$(git rev-parse --show-toplevel)"
export ADOM_MODEL_CONFIG="$ADOM_REPO/configs/adom/runtime/segformer_b0_640x384_rellis3d.py"
export ADOM_CHECKPOINT="<CHECKPOINT_PATH>"
ros2 launch adom_bringup rule_autonomy.launch.py \
  model_config:="$ADOM_MODEL_CONFIG" \
  checkpoint:="$ADOM_CHECKPOINT"
```

이 launch는 모터나 게임패드를 시작하지 않는다. 실차 제어는 별도 터미널에서 바퀴를 띄운
상태로 `adom_control gamepad_control.launch.py`를 실행하고 A 버튼으로만 승인한다.

rosbag의 원본 sensor timestamp를 watchdog과 일치시키려면 clock도 함께 재생한다.

```bash
ros2 launch adom_bringup rule_autonomy.launch.py use_sim_time:=true \
  model_config:="$ADOM_MODEL_CONFIG" checkpoint:="$ADOM_CHECKPOINT"
ros2 bag play "<ROSBAG_DIRECTORY>" --clock
```

## 실차 데이터 수집

ZED 2i, 게임패드 제어와 10 GB 제한 recorder를 한 번에 실행한다. 이 수집
launch는 GNSS 노드를 시작하지 않으며 rosbag에는 ZED의 `/rgb` 하위 토픽만 기록한다.

```bash
ros2 launch adom_bringup data_collection.launch.py
```

저장소 루트에서 실행한다. X를 눌러 매뉴얼 모드로 전환하고 스틱을 중앙에 놓은 뒤
주행한다. Y는 데이터 수집 시작/중지 토글이다. 기본 저장 위치는 저장소 기준
`data/captures`이다.
