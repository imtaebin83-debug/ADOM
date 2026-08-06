# adom_perception_ros

`src/adom`의 MMSegmentation 추론 로직을 ROS 2 I/O로 감싼다. ZED rectified RGB를
구독하고 Cost4 class ID mask, confidence, 색상 overlay를 원본 timestamp로 발행한다.

```text
/adom/perception/semantic_mask  sensor_msgs/Image (mono8, IDs 0..3)
/adom/perception/confidence     sensor_msgs/Image (mono8, 0..255)
/adom/perception/overlay        sensor_msgs/Image (bgr8)
/adom/perception/status         std_msgs/String (JSON)
```

ROS 환경에서 이 저장소의 Python package와 고정 OpenMMLab 환경을 먼저 설치한 뒤 실행한다.

```bash
python3 -m pip install -e /home/myungsub/ADOM
ros2 launch adom_perception_ros perception.launch.py \
  model_config:=/home/myungsub/ADOM/configs/adom/export/segformer_b0_640x384_rellis3d.py \
  checkpoint:=/absolute/path/to/best_mIoU.pth device:=cuda:0
```
