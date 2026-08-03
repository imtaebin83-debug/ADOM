# 0004. RunPod MLOps 운영 정책

## Status

Accepted — 2026-08-03

## Decision

- RunPod Secure Cloud의 A100 80GB를 기본 학습 GPU로 사용한다.
- 하나의 Network Volume에 versioned 원본·전처리 데이터셋을 두고 여러 Pod가 공유 읽기한다.
- 각 Pod는 `/workspace/adom/runs/<run-id>`처럼 고유한 경로에만 결과를 쓴다.
- 학습 순서는 현재 계약인 B0 선행, B2 후속으로 유지한다.
- Weights & Biases를 핵심 실험 추적기로 사용하고 TensorBoard event를 같은 run의 로컬 백업으로 남긴다.
- 500 iteration마다 model, optimizer, parameter scheduler를 저장하고 `last_checkpoint`가 가리키는 checkpoint에서 재개한다.
- Docker Hub의 Git SHA 태그 이미지는 `/opt/adom`에 실행 코드를 포함한다. `/workspace`는 RunPod Network Volume 전용으로 사용한다.
- GitHub Actions의 자동 범위는 이미지 build, 동일 이미지 smoke test, Docker Hub push까지다. 비용이 발생하는 Pod 생성과 학습 시작은 별도 명시적 실행으로 둔다.

## Why

10일/20만원 연구 예산 안에서 A100 80GB는 SegFormer fine-tuning에 충분한 VRAM과 비용 효율을 제공한다. 공유 데이터셋은 약 100GB의 원본과 향후 전처리본 중복을 막고, run별 쓰기 경로 분리는 멀티 Pod 동시 실행의 충돌을 막는다. W&B와 TensorBoard 이중 기록, optimizer를 포함한 중간 checkpoint는 중단 후 비교 가능한 상태로 복구하기 위해 필요하다.
