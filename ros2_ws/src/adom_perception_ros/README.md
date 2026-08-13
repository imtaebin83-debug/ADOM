# adom_perception_ros

`src/adom`의 MMSegmentation 추론 로직을 ROS 2 I/O로 감싼 Semantic20 퍼셉션
노드다. 클래스 계약은 설치된 canonical
`data/semantic_20/config/bridge_mapping.yaml`에서 읽으며, 출력 ID는 `0..18` 또는
`255`만 허용한다. Cost4 mask와 혼용되지 않도록 별도 topic을 사용한다.
기존 Cost4 경로는 `perception_cost4.launch.py`, `config/perception.yaml`,
`adom_cost4_perception_node`로 별도 보존한다.

```text
/adom/perception/semantic20_mask  sensor_msgs/Image (mono8, IDs 0..18/255)
/adom/perception/semantic20_mask_evidence sensor_msgs/Image (mono8, 2 Hz default)
/adom/perception/confidence       sensor_msgs/Image (mono8, 0..255)
/adom/perception/overlay          sensor_msgs/Image (bgr8)
/adom/perception/status           std_msgs/String (JSON latency/counters)
```

## Semantic20 mask 컬러 모니터링

`mono8` mask의 ID를 canonical Semantic20 palette로 바꾸는 진단 노드다. 모델을 다시
추론하지 않으며 ignore `255`는 검정으로 표시한다.

```bash
ros2 launch adom_perception_ros semantic20_colorizer.launch.py
```

기본 입력은 `/adom/perception/semantic20_mask_evidence`, 출력은 다음과 같다.

```text
/adom/perception/semantic20_mask_color  sensor_msgs/Image (bgr8)
/adom/perception/semantic20_legend      std_msgs/String (ID/name/RGB JSON)
```

Full-rate live mask를 보려면 다음처럼 입력만 바꾼다.

```bash
ros2 launch adom_perception_ros semantic20_colorizer.launch.py \
  mask_topic:=/adom/perception/semantic20_mask
```

## Latest-frame 처리

이미지 subscription은 Best Effort, Keep Last 1이다. callback은 추론하지 않고 한 칸짜리
mailbox의 pending frame을 교체한다. 별도 worker가 현재 추론을 마치고 FPS limiter가
허용하는 즉시 mailbox에서 그 시점의 최신 frame 하나만 가져간다. 따라서 frame 01을
처리하는 동안 02, 03, 04가 도착하면 다음 추론 대상은 04다. 기본 `target_fps`는
30.0이며 이는 입력을 30 FPS로 만든다는 뜻이 아니라 추론 시작률의 상한이다.

`/adom/perception/status`의 핵심 필드는 다음과 같다.

- `queue_wait_ms`: ROS callback 수신부터 추론 worker 시작까지
- `capture_to_inference_start_ms`: 카메라 header timestamp부터 추론 시작까지
- `inference_ms`: MMSeg model 호출 시간
- `processing_ms`: 변환, 추론, overlay, ROS publish를 포함한 worker 시간
- `capture_to_perception_output_ms`: 카메라 timestamp부터 mask publish까지
- `overwritten_frames`: 추론하지 않고 더 최신 frame으로 교체된 누적 frame 수
- `class_pixel_counts` / `class_pixel_ratios`: 배열 index가 Semantic20 ID `0..18`인
  프레임별 픽셀 수와 전체 mask 대비 비율
- `class_names`: 위 배열과 같은 순서의 이름으로 bag 단독 분석 시 ID를 해석하는 계약
- `present_class_ids`: 픽셀이 하나 이상 나온 class ID 목록. instance/object 검출 수나
  신뢰도 threshold 통과를 뜻하지 않는다.
- `ignore_pixel_count` / `ignore_pixel_ratio`: ID `255` 영역 통계

Live autonomy bag은 costmap이 사용하는 full-rate mask를 직접 구독하지 않는다. `t2 mask`는
같은 header와 `mono8` payload를 갖는 `semantic20_mask_evidence`를 기록한다. `t2 evidence`는
별도 복제 image publisher 대신 t4의 full-rate source RGB topic을 직접 기록한다. Mask는
추론 입력 message의 header를 보존하므로 source RGB와 timestamp로 결합할 수 있다.
`evidence_mask_fps:=0.0`이면 mask sample 발행을 끌 수 있다. 클래스 통계는 sample 주기와
무관하게 inference frame마다 status에 남는다.

카메라 clock과 ROS clock이 같은 time domain이고 header stamp가 0이 아닐 때만
`capture_to_*` 값이 유효하다. downstream은 mask header를 path까지 그대로 보존한다.
path controller는 `/adom/control/local_path_status`의 `source_to_command_ms`로
camera→software command 시간을 발행한다. 이 값은 호환되는 costmap과 controller를
연결한 뒤에 유효하며 ESC/PCA9685의 물리 응답시간을 포함하지 않는다.

## 실행

ROS 환경에서 이 저장소의 Python package와 고정 OpenMMLab 환경을 먼저 설치한다.
Semantic20 config와 그 config로 학습한 checkpoint를 함께 사용해야 한다.

```bash
export ADOM_REPO="$(git rev-parse --show-toplevel)"
export ADOM_MODEL_CONFIG="$ADOM_REPO/configs/adom/export/segformer_b0_640x384_rellis3d.py"
export ADOM_CHECKPOINT="$ADOM_REPO/models/checkpoints/b0-e0/best_mIoU_iter_6000.pth"
python3 -m pip install -e "$ADOM_REPO"
ros2 launch adom_perception_ros perception.launch.py \
  model_config:="$ADOM_MODEL_CONFIG" \
  checkpoint:="$ADOM_CHECKPOINT" device:=cuda:0
```

이 기본 실행 쌍은 `e49ad80`에 기록된 Semantic20 SegFormer-B0 E0 legacy baseline이다.
checkpoint는 Git에 포함되지 않으므로 perception 담당자에게 전달받아 위 경로에 둔다.
E1, B2 또는 Cost4 checkpoint를 이 config와 혼용하지 않는다. Jetson `t4` 실행은
저장소 루트의 `scripts/run_jetson_t4.sh`를 사용한다.

다른 mapping 파일을 실험할 때만 `bridge_mapping:=<PATH>`를 넘긴다. 이 경우에도
노드는 19개 train ID와 ignore 255 계약을 검증한다. 실제 ZED topic, publisher QoS,
clock domain 및 Jetson 성능은 target 장치에서 검증해야 한다.
