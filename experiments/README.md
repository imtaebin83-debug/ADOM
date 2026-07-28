# Experiments

재현 가능한 실험 단위의 코드, 설정, 결과 요약을 보관합니다.

현재 계획된 실험 축:

- `baseline/`: YOLOv8-Seg, DeepLabV3+ 등 첫 기준 성능
- `segformer/`: SegFormer-B0/B2 fine-tuning and evaluation
- `quantization/`: ONNX/TensorRT FP16/INT8 export and benchmark
- `deployment/`: Jetson runtime, ROS2 node, camera integration checks
- `costmap/`: semantic mask to traversability/costmap conversion

각 실험은 최소한 다음을 기록합니다.

- goal
- dataset and split
- model/config
- command
- hardware
- metrics
- result summary
- next action

수치 benchmark 결과는 `results/benchmark-results.csv`에도 한 줄씩 추가합니다.

1차 필수 metric:

- mIoU
- class IoU
- high_cost_or_obstacle recall
- FPS
- latency p50/p95
- Jetson Orin Nano 8GB power draw
- costmap update rate, if applicable
