# adom_perception_ros

`src/adom`의 MMSegmentation 추론 로직을 ROS 2 I/O로 감싼다. ZED rectified RGB를
구독하고 Cost4 class ID mask, confidence, 색상 overlay를 원본 timestamp로 발행한다.

```text
/adom/perception/semantic_mask  sensor_msgs/Image (mono8, IDs 0..3)
/adom/perception/confidence     sensor_msgs/Image (mono8, 0..255)
/adom/perception/overlay        sensor_msgs/Image (bgr8)
/adom/perception/status         std_msgs/String (JSON)
```

ROS 환경에서 이 저장소의 Python package와 고정 OpenMMLab 환경을 먼저 설치한 뒤
실행한다. 아래 명령은 저장소 루트에서 실행하며, checkout 경로를 특정 사용자 홈에
고정하지 않는다.

```bash
export ADOM_REPO="$(git rev-parse --show-toplevel)"
export ADOM_MODEL_CONFIG="$ADOM_REPO/configs/adom/export/segformer_b0_640x384_rellis3d.py"
export ADOM_CHECKPOINT="<CHECKPOINT_PATH>"
python3 -m pip install -e "$ADOM_REPO"
ros2 launch adom_perception_ros perception.launch.py \
  model_config:="$ADOM_MODEL_CONFIG" \
  checkpoint:="$ADOM_CHECKPOINT" device:=cuda:0
```
