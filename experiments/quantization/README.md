# Quantization Experiments

ONNX export, TensorRT FP16/INT8 변환, Jetson benchmark를 다룹니다.

초기 목표:

- PyTorch to ONNX export path 확정
- FP16 TensorRT engine 생성 및 정확도 저하 측정
- INT8 calibration dataset 기준 정의
- Jetson Orin Nano 8GB 기준 latency, FPS, memory, power 측정 양식 정리
