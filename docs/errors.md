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

## ERR-2026-08-14-010: ZED live Semantic20 mask 세로 압축과 하단 sky 오탐

- 상태: 원인 확정, 팀 수정 완료 보고, config 정적 회귀 검증 완료
- 환경: Jetson Orin Nano, ROS 2 Jazzy, ZED 2i,
  Semantic20 SegFormer-B0 E0 PyTorch/MMSeg `t4`
- 영향: 실차 주행 중 저장한 live mask, 산악 환경 sky FP 및 logit 해석

### 증상과 재현 근거

- 동일 프레임의 사후 주입 mask는 화면 전체를 채웠지만, live mask는
  장면이 상단 약 203–204행으로 압축되고 나머지 하단이 대부분 sky였다.
- 사후 mask를 세로로 약 203–204 px로 압축하고 하단을 sky로 채운 결과는
  두 live mask와 픽셀 기준 약 88–90% 일치했다.
- 오류 export config에 640x360 가상 입력을 넣은 pipeline 정적 점검은
  tensor shape `(3, 640, 640)`, `ori_shape=(360,640)`,
  `img_shape=(640,640)`을 재현했다. `padding_size`/`img_padding_size`는
  metadata에 없었다.

### 직접 원인

`configs/adom/export/segformer_b0_640x384_rellis3d.py`의 기존
pipeline이 `(384,640)` 하나를 서로 다른 순서 계약에 재사용했다.

```python
export_size = (384, 640)
model = dict(data_preprocessor=dict(size=export_size))  # (H, W)
dict(type="Pad", size=export_size, ...)                # (W, H)
```

- `SegDataPreProcessor.size` 계약은 `(H,W)`이지만 MMCV `Pad.size`는
  `(W,H)`다.
- `Pad(size=(384,640))`는 width 384, height 640으로 해석됐다. 이미 640인
  width는 줄이지 못하고, 360인 height만 640까지 padding되어 실제
  입력이 `640x640`이 됐다.
- 기본 `PackSegInputs` metadata에 정확한 padding 크기가 없어 MMSeg
  postprocess가 하단 280행을 제거하지 않았다.
- 전체 640행을 원본 높이 360으로 복원하면 유효 영상은
  `360 * 360 / 640 = 202.5` 행이 된다. 이 값이 관측된 압축 높이와
  일치한다.
- black padding 영역에서 모델이 주로 sky를 출력해 화면 하단의
  거대한 sky FP처럼 보였다.

### 수정과 판정

- Jetson `t4` wrapper는 live runtime 전용
  `configs/adom/runtime/segformer_b0_640x384_rellis3d.py`를 사용하도록
  수정됐다.
- runtime pipeline은 explicit `Pad` transform을 제거하고 640x360을 유지한 뒤,
  `SegDataPreProcessor.test_cfg.size=(384,640)`에서 하단 24행을
  padding하도록 했다. 이 경로는 padding metadata를 기록해 postprocess가
  padding을 crop한 뒤 640x360으로 복원한다.
- runtime config 정적 점검은 pipeline tensor `(3,360,640)`,
  `ori_shape=img_shape=(360,640)`, preprocessor `test_cfg.size=(384,640)`을
  확인했다.
- `semantic20_colorizer_node` 및 `rqt_image_view`는 mask를 resize하지 않으므로
  원인에서 배제했다.
- `t4`는 TensorRT가 아닌 PyTorch/MMSeg `.pth` 추론 경로다. 사후 주입이
  TensorRT였다면 동일 frozen tensor의 PyTorch↔TensorRT parity는 별도로 검증한다.

### 실험 자료 주의사항

- 오류 config로 생성된 live mask의 하단 sky는 domain-shift FP, class
  imbalance 또는 logit uncertainty 근거로 사용하지 않는다.
- 수정된 runtime config로 동일 RGB를 재추론한 뒤에도 남는 FP만 실제
  산악 domain 오류로 분석한다.

## ERR-2026-08-07-009: Semantic20 ONNX export와 Jetson TensorRT hand-off 오류

- 상태: export/parity 및 target Jetson FP16 engine build 검증 완료
- 환경: RunPod A100, image Git SHA `e49ad806`; Jetson Orin Nano 8GB,
  JetPack 7.2, TensorRT 10.16.2
- 영향: B0 E0 Semantic20 ONNX export와 TensorRT engine build

### 재현된 오류와 원인

1. wheel 설치 MMDeploy에는 `.mim/tools/deploy.py`가 없었다. package 내부 경로를
   가정하지 않고 공개 API `mmdeploy.apis.torch2onnx`를 사용해야 한다.
2. opset 11 export는 `aten::unflatten`을 지원하지 않아 실패했다. opset 13에서
   export와 ONNX checker가 통과했다.
3. `(384, 640)` 하나를 MMSeg preprocessor와 MMCV pipeline에 공용으로 전달해
   `640x640` tensor가 만들어졌다. `SegDataPreProcessor.size`는 H,W이고
   `Resize.scale`/`Pad.size`는 W,H다. 각각 `(384,640)`과 `(640,384)`로 분리했다.
4. MMDeploy `onnx_config.input_shape=[640,384]`는 pipeline의 keep-ratio를 무시하고
   direct resize를 강제했다. `input_shape=None`으로 두고 pipeline이 resize와 static
   right/bottom padding을 수행하게 했다.
5. PyTorch CUDA(A100)와 ONNX Runtime CPU 비교는 최대 logits 오차가 약
   `0.01~0.1`이었지만, 동일 CPU backend 비교는 12장 모두 argmax 100% 및 최대
   절대오차 `0.0001034737`이었다. 공식 graph parity는 CPU↔CPU로 기록한다.
6. TensorRT build script의 `--memPoolSize=workspace:1024MiB`는 TensorRT 10.16에서
   `0.000976562 MiB`로 해석돼 attention tactic이 요구한 1280–1536 MiB를 모두
   제외했다. 숫자는 MiB 단위이므로 `workspace:2048`로 수정했다.

### 검증 결과와 영구 수정

- ONNX: FP32 raw logits, opset 13, input `1x3x384x640`, output
  `1x19x384x640`, embedded argmax 없음
- CPU parity: 12장, overall/minimum per-image argmax 100%, finite logits,
  최대 절대오차 `0.0001034737`
- TensorRT: target Jetson에서 FP16, workspace 2048 MiB로 build,
  `&&&& PASSED TensorRT.trtexec`, exit code 0
- H,W/W,H config, 공개 API exporter, 검증형 packager, 0-byte engine을 거부하는
  TensorRT builder를 추가했다.
- 전원 종료로 engine SHA, ONNX↔TensorRT parity 및 latency benchmark는 아직 미실측

## ERR-2026-08-06-008: code-smoke가 프로젝트 의존성을 설치하지 않음

- 상태: 수정 및 PR CI 검증 완료
- 환경: GitHub Actions `Code smoke tests`, PR #29, Python 3.10
- 증상: test collection 중
  `src/data/semantic_20/scripts/01_convert_bridge_sources.py`의 `import yaml`에서
  `ModuleNotFoundError: No module named 'yaml'`로 실패했다.
- 경로 판정: traceback의 `…/work/ADOM/ADOM/...`은 GitHub Actions의 정상적인
  `<workspace>/<repository>` checkout 구조이며 오류 원인이 아니다.
- 원인: `PyYAML`은 `pyproject.toml`과 Docker requirements에 이미 선언되어 있었지만,
  code-smoke workflow가 `numpy`와 `Pillow`만 수동 설치하고 프로젝트 자체는 설치하지
  않았다. 새 preprocessing test가 converter를 collection 단계에서 import하면서 잠재된
  CI 의존성 drift가 드러났다.
- 수정: commit `53cc030`에서 수동 의존성 목록 대신
  `python -m pip install --editable .`로 프로젝트의 선언된 의존성을 설치하도록 변경했다.
- [x] 기존 PR에 후속 commit push
- [x] GitHub Actions code-smoke 통과 확인

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
