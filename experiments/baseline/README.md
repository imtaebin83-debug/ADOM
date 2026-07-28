# Baseline Experiments

YOLOv8-Seg, DeepLabV3+, 기타 첫 기준 모델을 재현하고 비교합니다.

초기 목표:

- RELLIS-3D 데이터 로딩과 split 확정
- 최소 1개 CNN baseline 재현
- mIoU, class IoU, high_cost_or_obstacle recall, traversable precision, FPS, latency p50/p95를 같은 환경에서 기록
- SegFormer 실험과 비교 가능한 evaluation protocol 정리
