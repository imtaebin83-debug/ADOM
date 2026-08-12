# ADOM final presentation

밀리테크 최종 연구발표를 준비하는 문서의 진입점이다. 발표 대상은 국방·자율주행
연구 경험이 있는 박사급 청중으로 가정한다.

발표의 중심은 새로운 backbone을 제안했다는 주장이 아니다. 직접 수집한 off-road
데이터의 효과를 continued-training 통제군과 분리해 검증하고, 선택 모델의 의미가
ONNX와 target Jetson TensorRT까지 유지되는지 확인한 뒤, 실제 RC Car의 보수적인
안전 동작으로 연결하는 end-to-end 연구 과정이다.

## Documents

- [Storyboard](final-presentation-storyboard.md): 연구 주장, 슬라이드 흐름, 표현 경계,
  예상 질문과 문헌 근거
- [Evidence checklist](result-evidence-checklist.csv): 발표 전 확보해야 할 정량 결과,
  영상, 사진 및 재현성 정보

## Working claim

> 한국형 오프로드 장면에서 부족한 표적·희소 클래스 데이터를 직접 보강하고, 그
> 효과를 단순 추가 학습과 분리해 검증하며, 선택 모델을 서버 없이 Jetson에서
> 추론하고 저속 RC Car 안전 동작으로 연결하는 재현 가능한 end-to-end 연구
> 파이프라인을 구축했다.

아직 TA 비교와 실차 반복 검증이 완료되지 않았으므로 이 문장은 최종 claim 후보다.
완료된 결과와 계획은 발표 시각물에서도 서로 다른 색과 상태 표기로 구분한다.

## Presentation rules

- 과제 공식 명칭은 보존하되 현재 구현은 `SegFormer 기반 domain adaptation`으로
  정확히 기술한다. SegFormer-B0 자체를 새 Foundation Model이라고 부르지 않는다.
- FP16 TensorRT를 INT8 양자화라고 표현하지 않는다.
- file runtime reciprocal을 live-camera FPS라고 표현하지 않는다.
- semantic class를 traversability ground truth와 동일시하지 않는다.
- 동일 장면의 성공 영상뿐 아니라 전체 trial 수, 실패와 false stop을 함께 보고한다.
- 사진은 노력의 증거로, 실험 계약과 정량 결과는 연구 주장의 증거로 사용한다.
- checkpoint, ONNX, engine, 원시 로그와 개인 절대경로는 Git에 넣지 않는다.

## Current presentation-ready deployment evidence

| Item | Result | Meaning |
| --- | ---: | --- |
| PyTorch CPU↔ONNX Runtime CPU | 12 images, 100% argmax | ONNX graph preservation |
| Maximum PyTorch↔ONNX logit error | `0.0001034737` | FP32 same-backend diagnostic |
| ONNX↔TensorRT overall valid-region agreement | `99.9816%` | Target-device parity PASS |
| Worst per-image valid-region agreement | `99.9521%` | All references pass 99% gate |
| Maximum class-area difference | `0.0318%p` | Below `0.2%p` gate |
| TensorRT engine size | `9,244,628 bytes` (`8.82 MiB`) | Target-built artifact |
| File runtime total p95 | `22.11 ms` | H2D + engine + D2H only |
| Mean runtime reciprocal | `52.92 FPS` | Not a live-camera FPS result |

The engine SHA-256 is
`3e815c93a3f6b63265e449d85174c0a3164f636d5bce8e5fcbb1bc9ba272735d`.
The root context still describes this target validation as pending and must be updated in a
separate status/evidence change rather than silently treating this presentation note as the
project source of truth.
