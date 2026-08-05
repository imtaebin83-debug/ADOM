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

## ERR-2026-08-05-007: best checkpoint에 resume state를 요구한 점검 오류

- 상태: 점검 조건 수정
- 환경: E0 B0 Gate 2 완료 후 full 직전 artifact audit
- 증상: mini의 `best_mIoU` checkpoint에 `optimizer`와 `param_schedulers`가 없어
  최종 점검 assertion이 실패했다.
- 원인: 모델 성능 기준 best checkpoint와 중단 복구용 periodic checkpoint의 역할을
  혼동했다. Full cycle은 Stage 1 best의 `state_dict`를 Stage 2 `load_from`으로 전달하며,
  Stage 2 optimizer/scheduler는 새로 시작한다.
- 판정: best checkpoint에는 `state_dict`를 요구한다. optimizer/scheduler 보존은
  `last_checkpoint`가 가리키는 periodic checkpoint와 Gate 3 resume 증거에서 확인한다.
- [x] Gate 2 periodic checkpoint의 optimizer/scheduler 직접 확인

## ERR-2026-08-05-006: W&B 경로 선생성으로 output 충돌

- 상태: 운영 복구 및 fresh Gate 2 검증 완료
- 환경: RunPod A100 80GB, E0 B0 Gate 2 재실행
- 증상: `Output exists: ...-b0-mini; use --resume explicitly`로 학습 시작 전 중단됨
- 원인: W&B writable 경고를 피하려고 Gate script가
  `mkdir -p "$MINI_ROOT/wandb"`를 먼저 실행했다. 이 명령이 output root까지 만들었고,
  Semantic20 runtime의 신규 실행 충돌 방지 검사가 이를 기존 run으로 판정했다.
- 복구: Gate output 내부를 선생성하지 않는다. 이미 존재하는 별도 로그 경로를
  `WANDB_DIR`로 지정하고, 실패 output은 삭제하지 않고 timestamp suffix로 보관한다.
- 주의: 실제 checkpoint가 없는 초기화 실패이므로 이 디렉터리에 `--resume`을 사용하지
  않는다. fresh output에서 Gate 2를 다시 시작한다.
- [x] fresh Gate 2 재실행 통과 (`aAcc=91.26`, `mIoU=38.81`, `mAcc=43.70`)

## ERR-2026-08-05-005: source된 strict mode가 tmux shell 종료

- 상태: 운영 복구 완료, 문서/스크립트 영구 반영 필요
- 환경: RunPod A100 80GB, 새 tmux interactive shell
- 증상: Gate 2 오류 직후 tmux session 자체가 종료되어 즉시 로그를 볼 수 없었음
- 원인: 공통 환경 파일의 `set -Eeuo pipefail`이 `source`를 통해 interactive shell에
  적용됐고, 자식 Gate script의 non-zero 종료가 interactive shell까지 종료시켰다.
- 복구: source 전용 환경 파일에는 strict mode를 넣지 않는다. strict mode는 실행되는
  Gate script 내부에만 둔다. Pod의 `gate-env.sh`에서 strict mode를 제거했고 로그는
  Network Volume에서 다시 검색해 복구했다.

## ERR-2026-08-05-004: W&B 기본 entity 404

- 상태: 운영 설정 수정 및 최소 online 검증 완료
- 환경: RunPod A100 80GB, image Git SHA
  `5c50bfdf2900596bcd447ed6c44ce7924bf10453`, E0 B0 Gate 2
- 증상: API key 인증은 성공하고 `Currently logged in as: imtaebin83`가 출력됐지만
  run 생성은 `entity imtaebin83 not found during upsertBucket` 404로 중단됐다.
- 원인: 인증 계정 username은 `imtaebin83`이지만 run을 소유할 실제 entity slug는
  `imtaebin83-seoul-national-university`였다. `WANDB_ENTITY`가 명시되지 않아 SDK가
  유효하지 않은 username을 대상으로 선택했다.
- 관련 경고: Network Volume 아래 W&B 디렉터리를 writable로 판단하지 못해 system
  temp directory로 fallback했다. fatal 원인은 아니지만 장기 학습 전 별도 수정한다.
- 복구: W&B workspace URL에서 실제 entity를 확인하고 API key를 재발급했다.
  `WANDB_ENTITY=imtaebin83-seoul-national-university`, `WANDB_PROJECT=adom`을 명시한
  최소 online run이 통과했다.
- [x] 올바른 W&B entity 확인
- [x] W&B directory write test 통과
- [x] 최소 online run 생성 통과
- [x] Gate 2 재실행 및 W&B online 동기화 통과

## ERR-2026-08-05-003: smoke 종료 시 전체 validation 실행

- 상태: 원인 확인 필요
- 환경: RunPod A100 80GB, image Git SHA
  `5c50bfdf2900596bcd447ed6c44ce7924bf10453`
- 영향: E0 B0 Gate 1이 50 update 뒤 RELLIS validation 900개를 추가 실행함
- 관찰: smoke는 validation interval을 51로 설정하지만 MMEngine이 마지막 50
  iteration에서 validation을 실행하고 `best_mIoU_iter_50.pth`를 생성했다.
- 결과: Gate 1 자체는 `PASS`했으며 `aAcc=88.99`, `mIoU=24.82`, `mAcc=27.85`였다.
- 후속: smoke에서 validation을 실제로 생략하도록 runner 종료 동작을 조사하고 회귀
  테스트를 추가한다. 현재 checkpoint는 smoke 산출물일 뿐 full 학습 입력으로 쓰지 않는다.

## ERR-2026-08-05-002: W&B secret 환경변수 대소문자 불일치

- 상태: RunPod Secret 매핑 및 online 검증 완료
- 환경: RunPod A100 80GB, E0 B0 Gate 1 직후
- 증상: Gate 1은 통과했지만 후속 스크립트가 `WANDB_API_KEY is missing`으로 중단됨
- 원인 1: RunPod에 `wandb_api_key`로 등록했지만 runtime은 Linux의 대소문자 구분에
  따라 별개인 `WANDB_API_KEY`를 요구한다.
- 원인 2: `read -rsp ... WANDB_API_KEY`는 현재 shell 변수만 만들며 자동으로 export하지
  않는다. 이 경우 현재 shell의 `test -n`은 통과하지만 자식 `bash` training script에는
  변수가 전달되지 않는다.
- 원인 3: browser terminal에서 일반 `Ctrl+V`를 사용해 붙여넣으면서 carriage return
  (`0x0d`)과 공백이 key 문자열 내부에 삽입됐다. 변수는 non-empty라 로컬 존재 검사는
  통과했지만 W&B server는 손상된 key를 401로 거부했다.
- 복구: `read` 직후 값을 다시 쓰지 않고 `export WANDB_API_KEY`를 별도로 실행하고,
  `bash -c`로 자식 process 상속 여부를 확인한다. 이후 Pod/template에서는 secret 값이
  uppercase 환경변수 `WANDB_API_KEY`에 연결되게 한다.
- 보안: secret 값 자체를 로그, shell history 또는 이 문서에 남기지 않는다.
- [x] uppercase 환경변수 존재 및 자식 process 전달 확인
- [x] 최소 W&B online run 생성 확인
- [x] Gate 2 W&B online run 생성 확인

## ERR-2026-08-05-001: W&B tag 64자 제한으로 Gate 1 초기화 실패

- 발생 시각: 2026-08-05 07:20 UTC 전후
- 상태: Docker image 및 RunPod Gate 검증 완료
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
- [x] MMSeg/W&B가 설치된 Docker image에서 disabled/online backend integration test 통과
- [x] GitHub Actions Docker build 및 image contract 통과
- [x] 새 SHA image의 RunPod Gate 1 통과
