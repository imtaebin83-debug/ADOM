# ADOM error log

RunPod, Docker, dataset, training 및 배포 과정에서 재현된 오류와 해결 근거를 누적한다.
새 항목은 최신 오류가 위에 오도록 추가하며, 추측이 아니라 로그와 재현 결과를
기준으로 작성한다.

## 기록 형식

각 오류에는 다음 정보를 남긴다.

- ID와 발생 시각(UTC)
- 상태: 조사 중, 수정 중, 로컬 검증 완료, 이미지 검증 완료
- 실행 환경과 immutable image Git SHA
- 영향받은 gate 또는 작업
- 증상과 핵심 traceback
- 직접 원인과 함께 드러난 관련 결함
- 수정 내용, 회귀 테스트 및 운영 복구 절차
- 남아 있는 검증 항목

## ERR-2026-08-05-001: W&B tag 64자 제한으로 Gate 1 초기화 실패

- 발생 시각: 2026-08-05 07:20 UTC 전후
- 상태: 로컬 targeted regression 통과, Docker image 검증 대기
- 환경: RunPod A100 80GB, Docker image Git SHA
  `4b3d33603c297c187dcbee84d6ef3c8dca71e291`
- 영향: E0 SegFormer B0 Gate 1 50-update smoke

### 증상

Runtime doctor는 `PASS`했지만 MMSeg runner가 생성되기 전 `wandb.init()`에서
다음 validation error로 종료됐다.

```text
Tag 'extra:runpod+a100+...+phase:e0-stage1-smoke' is 94 characters.
Tags must be between 1 and 64 characters.
```

학습 loop, dataloader 및 optimizer가 시작되기 전의 실패이므로 GPU OOM, dataset,
model forward 또는 loss 문제는 아니다.

### 원인

- tracking 환경이 개별 W&B tag를 `+`로 다시 합쳐 하나의 `extra:` tag로 만들었다.
- W&B run ID에는 64자 제한 처리가 있었지만 tag에는 같은 경계 처리가 없었다.
- smoke가 `WANDB_MODE=disabled`를 설정해도 Semantic20 config가
  `WandbVisBackend`를 항상 구성하여 로컬 `wandb.init()`과 tag validation을 실행했다.

### 수정

- 긴 tag를 사람이 읽을 수 있는 prefix와 SHA-256 8자리 suffix로 결정적으로 줄여
  최대 64자를 보장한다.
- 공통 tracking 환경에서 bounded `WANDB_EXTRA_TAG`를 한 번만 생성한다.
- `WANDB_MODE=disabled`이면 `WandbVisBackend`를 구성하지 않고 TensorBoard와 local
  backend만 유지한다.
- 짧은 tag 보존, 긴 tag의 길이/결정성/충돌 구분 및 disabled/online backend 선택을
  회귀 테스트한다.

### 운영 복구

- 실패한 output directory는 장애 증거로 보존한다.
- 컨테이너 내부 hot patch는 immutable image와 실제 코드가 달라지므로 사용하지 않는다.
- 수정된 새 Git SHA image를 빌드·검증한 후 새 run ID와 output directory에서
  `--micro-batch 16`으로 Gate 1을 다시 실행한다.

### 남은 검증

- [x] bounded tag unit test 및 전체 Python config compile 통과
- [ ] MMSeg/W&B가 설치된 Docker image에서 disabled/online backend integration test 통과
- [ ] GitHub Actions Docker build 및 image contract 통과
- [ ] 새 SHA image의 RunPod Gate 1 통과
