# TODO: Phase 1 Semantic20 전처리·학습 안정화

> 아래 상세 내용은 안정화 과정에서 발견한 결함과 완료 조건을 보존한 기록이다.
> 실제 진행 여부는 이 체크리스트를 기준으로 한다.
> 현재 P0는 D-5 RC Car PoC이며, Clean v1 후속 실험은 발표 파이프라인 안정화까지
> 일시 중지한다. 프로젝트 우선순위는 [`docs/status/README.md`](status/README.md)를
> 따른다.

## 진행 현황 (2026-08-06)

- [x] 공식 RUGD RGB mask와 indexed mask 입력을 모두 지원한다.
- [x] unknown RGB, pair/크기, dtype/channel, target ID를 fail-fast 검증한다.
- [x] 중단 후 재개 가능한 canonical preprocess CLI와 원자적 publish를 구현한다.
- [x] legacy `study` runtime 의존성을 제거하고 Semantic20 package data를 이미지에 포함한다.
- [x] RELLIS/RUGD split parsing 및 missing/unexpected/duplicate 검사를 보강한다.
- [x] 관련 unit/integration test와 Docker image 내부 contract test를 통과한다.
- [x] E0/E1 processed dataset과 `final_check.json` 완료 조건을 통과한다.
- [x] GitHub Actions에서 immutable Git SHA Docker image 빌드·검사·push를 완료한다.
- [x] RunPod A100 80GB runtime/image preflight와 E0/E1 전체 dataset contract를 통과한다.
- [x] Gate 0: E0 SegFormer B0 micro-batch probe를 `16/1` effective batch 16으로 통과한다.
- [ ] 학습 중 `tqdm` 진행률 표시를 구현한다.
- [x] Gate 1: E0 B0 50-update smoke를 통과한다.
- [x] Gate 2: E0 B0 500-update mini-run과 W&B/validation을 통과한다.
- [x] Gate 3: checkpoint 및 optimizer/scheduler resume을 검증한다.
- [x] 위 gate 결과와 full 직전 artifact/config/runtime 점검을 모두 통과한 뒤
  E0 B0 full run 시작을 승인한다.
- [x] E0 B0 full Stage 1/2와 canonical test를 완료한다.
- [x] W&B 결과를 검토하고 B2 비교 실험 진행을 승인한다.
- [x] 동일 image SHA와 데이터 계약으로 E0 B2 probe/smoke/mini/resume/full을 완료했다.
- [x] B0/B2 canonical test의 동일 클래스 표를 작성했다. 다중 seed 비교는 별도 미완료다.

## P0: Jetson TensorRT hand-off 후속

- [x] E-ADOM selected checkpoint를 SHA256으로 동결했다.
- [x] locked canonical test와 export image/metadata/PyTorch-to-ONNX parity를 통과했다.
- [x] Jetson `t4 b0-e0`/`t4 eadom` profile을 checkpoint SHA까지 분리했다.
- [ ] E-ADOM export archive를 target Jetson으로 전송하고 `SHA256SUMS`를 검증한다.
- [ ] target Jetson의 실제 TensorRT 버전/GPU에서 B0-E0와 E-ADOM FP16 engine을 각각
  생성한다. RunPod에서 생성한 serialized engine은 전달하지 않는다.
- [ ] 각 profile에서 ONNX-to-TensorRT pixel argmax agreement 99.0% 이상과 reference
  image 10장 이상을 검증하고 latency/peak memory를 기록한다.
- [ ] `adom_perception_ros`에 `backend={mmseg,tensorrt}`와 profile별 `engine_path`를
  추가하고 `preprocess.json`의 resize, padding, channel order, mean/std를 그대로
  적용하는 native TensorRT backend를 구현한다.
- [ ] TensorRT backend도 현재 mask/header/topic/watchdog 계약을 보존하고 engine
  mismatch, deserialize 실패, shape/class 불일치 시 fail-closed하도록 테스트한다.
- [ ] 동일 recorded input으로 B0-E0/E-ADOM field A/B evidence를 만든 뒤 profile을
  선택한다. Canonical test를 다시 model-selection loop에 사용하지 않는다.

현재 `t4`의 두 profile은 모두 PyTorch/MMSeg CUDA backend다. TensorRT engine 생성과
standalone 검증은 진행할 수 있지만 ROS engine 연결이 완료된 것으로 간주하지 않는다.

## P0: E0 B2 통제 비교

- B0 기준 image Git SHA는
  `5c50bfdf2900596bcd447ed6c44ce7924bf10453`이다.
- B2에서도 dataset digest, split, seed 42, CE loss, optimizer, Stage 1/2 schedule과
  best-checkpoint test 정책을 유지한다.
- B2는 `16/1`, `8/2`, `4/4` 순서의 자동 micro-batch probe부터 실행한다.
- 단일 seed의 B2 차이가 2%p 미만이면 불확실, 2~4%p이면 seed 반복 필요,
  4%p 초과이면 유의미한 개선 후보로 판정한다. 이는 B0 500-update run 간 관찰된
  약 2.17%p 변동을 이용한 운영 기준이며 통계적 유의성 기준은 아니다.
- 최종 연구 주장에는 최소 3개 seed의 평균과 표준편차를 사용한다.

## P0: Phase 1 평가 계약 보강

B0 best Stage 2 validation은 6,000 update에서 `mIoU=51.07`이었고 canonical test는
`aAcc=89.78`, `mIoU=43.35`, `mAcc=67.22`였다. 학습은 정상이지만 canonical
validation/test에 일부 희소 클래스 GT가 없어 Phase 1 목표를 충분히 평가하지 못한다.

- [ ] train/val/test의 클래스별 image 수와 pixel support를 고정 artifact로 남긴다.
- [ ] GT가 존재하는 고정 클래스 집합의 `supported-class mIoU`를 추가한다.
- [ ] GT가 없는 클래스의 false-positive pixel/image rate를 별도 보고한다.
- [ ] pole, water, log 등 목표 희소 클래스의 macro IoU/Recall을 정의한다.
- [ ] 학습에 사용되지 않는 고정 rare-class challenge validation set을 확정한다.
- [ ] test의 `test_metrics.json`과 `confusion_matrix.json`을 W&B summary/artifact로
  업로드한다.
- [ ] test set으로 recipe를 반복 조정하지 않고 validation으로 선택한 최종 모델만
  canonical test에서 평가한다.

## P1: B2 이후 학습 recipe 후보

- [ ] B0의 6k 최고점과 40k 최종 하락을 근거로 18~20k 상한 또는 early stopping을
  검토한다.
- [ ] CE baseline 뒤 capped class-weighted CE를 첫 loss ablation으로 수행한다.
- [ ] 이후 CE+Lovasz 또는 Focal을 한 번에 하나씩 비교한다.
- [ ] pole 같은 소형·가느다란 클래스에는 class-aware crop sampling과 입력 해상도
  ablation을 loss 변경과 분리해 수행한다.

## 범위와 담당 구분

- 대상은 공식 RUGD RGB annotation을 RELLIS 기반 Semantic20 공간으로 변환하는
  Phase 1 전처리 경로다.
- `study/gahyung/Datasets_Repo`의 코드를 정식 `src` 경로로 승격하는 작업은 팀원이
  담당한다. 이 문서는 승격 위치를 다시 설계하지 않고, 승격되는 원본 코드에 반드시
  반영해야 할 결함과 완료 조건을 기록한다.
- 현재 RunPod에서 사용하는 임시 RGB 변환 어댑터는 복구용일 뿐 canonical
  pipeline으로 간주하지 않는다.

## P0: 공식 RUGD RGB mask 입력 지원

### 재현된 증상

`RUGD_annotations.zip`을 풀어 만든 mask는 `(H, W, 3)` RGB color mask다. 그러나
`01_convert_bridge_sources.py`의 `load_index_mask()`는 `(H, W)` 단일 채널 또는
모든 채널 값이 같은 mask만 허용하여 첫 샘플에서 다음 오류가 발생한다.

```text
ValueError: Expected indexed mask, got shape (550, 688, 3):
.../rugd-flat/indexLabel/park-2_00001.png
```

### 근본 원인

- flat index 생성 과정은 파일 이름과 pair만 검증하고 mask encoding은 검증하지
  않았다.
- 디렉터리 이름을 `indexLabel`로 만들었지만 그 안에는 RGB mask가 들어갔다.
- Semantic20 bridge는 RUGD source ID를 기대하지만, 공식 다운로드 경로는 RGB
  palette mask를 제공한다.
- 기존 RUGD `label_mapping.json`의 `RGB_TO_ADOM`은 Cost4 target을 위한 mapping이므로
  Semantic20 bridge에 그대로 사용할 수 없다.

### 필수 수정

- RUGD mask 입력 형식을 `auto`, `rgb`, `indexed` 중 하나로 명시하거나 자동
  판별한다.
- RGB 입력은 공식 RGB palette를 class name으로 해석한 뒤 Semantic20 bridge의
  `source_class -> target_id`와 합성한다.
- bridge에 없는 RUGD class는 명시적으로 `ignore_index=255`로 처리하고, 알려지지
  않은 RGB 값은 조용히 무시하지 말고 즉시 실패시킨다.
- 변환 전에 전체 palette coverage, image/mask pair, 크기, split 중복을 검사한다.
- RGB lookup은 per-pixel Python loop 대신 packed RGB LUT 또는 동등한 vectorized
  구현을 사용한다.
- 결과 mask는 `uint8`, 단일 채널이며 값의 집합이 Semantic20 `0..18, 255`의
  부분집합임을 보장한다.

## P0: 중단 후 재개와 원자적 publish

- 원격 연결 종료와 관계없이 tmux에서 실행할 수 있는 canonical entrypoint를 둔다.
- sample 단위 완료 manifest 또는 검증된 기존 출력 skip 기능을 제공해 RUGD
  7,436개 변환 중 중단되어도 처음부터 다시 시작하지 않게 한다.
- 진행 중 출력은 `<output>.partial` 또는 고유 staging run 아래에 기록한다.
- 모든 검증이 통과하기 전에는 `datasets/processed` 경로로 이동하지 않는다.
- 실패 시 stage, sample ID, 입력 경로, exception을 상태 JSON과 로그에 기록한다.
- 재실행 시 완성된 RELLIS 6,234개를 다시 변환하지 않고 실패한 stage부터 재개한다.
- `--overwrite`는 명시적으로 요청한 경우에만 허용하며 기본 동작은 fail-closed로
  유지한다.

## P1: Docker image 전처리 자산 패키징

- 현재 Dockerfile은 `src`, `configs`, `scripts`, `data`만 `/opt/adom`에 복사하고
  legacy `study`는 복사하지 않는다. 따라서 현 이미지에는 Semantic20 전처리
  스크립트와 RUGD split이 없다.
- `src/adom/runtime/semantic20_cycle.py`의 `REFERENCE_SPLITS`도 이미지에 없는
  `study/.../rellis3d_semantic20_v1/splits`를 참조한다. 현재 이미지에서는 E0/E1
  모두 GPU 학습 시작 전 dataset contract 검사에서 실패하므로, canonical split을
  `data/splits` 아래로 승격하고 runtime과 테스트가 같은 경로를 사용하게 수정한다.
- `COPY study`로 우회하지 않는다. 팀원이 승격한 canonical 코드, palette mapping,
  bridge mapping, RELLIS/RUGD split만 이미지에 포함한다.
- 이미지 빌드 후 다음 파일을 확인하는 smoke test를 추가한다.
  - Semantic20 preprocess CLI
  - RUGD RGB palette mapping
  - Semantic20 bridge mapping
  - RELLIS 및 RUGD train/val/test split
- 실행 로그에 image/source Git SHA와 dataset mapping version을 남긴다.

## P2: split 및 사전 검증 개선

- RUGD split 파일 세 개는 마지막 EOF newline이 없어 `wc -l`이 각각 1개씩 적게
  표시된다. 실제 레코드는 train 4,779, val 733, test 1,924로 총 7,436개다.
- split 검사 코드는 newline 개수가 아니라 비어 있지 않은 `splitlines()` 레코드
  수를 사용한다.
- 저장소의 text file은 가능하면 EOF newline을 추가하되, 입력 parser는 EOF
  newline 유무와 CRLF/LF 모두 처리한다.
- flat directory의 단순 파일 수뿐 아니라 expected split과 actual basename을
  양방향 비교하여 missing, unexpected, duplicate를 모두 검출한다.
- preflight에서 mask mode, shape, dtype, observed palette/ID를 표본이 아닌 전체
  또는 충분히 강한 deterministic audit으로 확인한다.

## P1: tmux 학습 진행률 가시화

- [ ] `tmux`의 interactive TTY에서 학습 runner iteration과 optimizer update 진행률을
  `tqdm` progress bar로 실시간 표시한다.
- [ ] 현재 gate, experiment, model, stage, update/total, loss, learning rate, ETA와
  CUDA peak memory를 한 줄에서 확인할 수 있게 한다.
- [ ] gradient accumulation을 사용할 때 runner iteration이 아니라 optimizer update
  기준 진행률이 정확히 표시되도록 한다.
- [ ] validation/test에는 처리한 sample 수와 전체 sample 수를 별도 progress bar로
  표시한다.
- [ ] non-TTY, GitHub Actions, 파일 redirection 환경에서는 progress bar를 자동으로
  비활성화하고 기존 MMEngine logger를 유지한다.
- [ ] `tqdm` 출력이 W&B, TensorBoard, MMEngine 로그 및 Network Volume에 보존되는
  텍스트 로그를 깨뜨리지 않도록 한다.
- [ ] 매 iteration의 불필요한 CUDA synchronize를 피하고 측정 가능한 학습 처리량
  저하가 없도록 한다.
- [ ] TTY/non-TTY 동작, gradient accumulation update 계산, 종료/예외 시 progress bar
  정리를 회귀 테스트로 검증한다.

## 테스트 요구사항

- 합성 RGB RUGD mask가 기대한 Semantic20 ID와 255로 변환되는 단위 테스트
- indexed mask 입력의 기존 동작을 보존하는 회귀 테스트
- unknown RGB, 중복 파일명, 누락 pair, image/mask 크기 불일치 실패 테스트
- 중간 중단 후 재개 시 완료 파일을 재작성하지 않는 테스트
- 빈 staging, 부분 staging, 이미 publish된 output에 대한 idempotency 테스트
- Docker image 안에서 preprocess `--help`와 필요한 mapping/split 존재 여부 검사
- Docker image와 동일한 파일 집합에서 E0/E1 `validate_semantic20_dataset()`을
  실행하여 legacy `study` 경로 의존성이 없음을 확인하는 테스트

## 실제 데이터 완료 조건

- RELLIS 변환: 6,234개
- RUGD 입력/출력: 7,436개
- YCOR 최종 포함: 751개
- combined manifest: 14,421개
- main split: train 9,868 / val 900 / test 899
- validation/test: RELLIS-only 정책 유지
- missing image/mask, size mismatch, non-single-channel mask: 모두 0
- 최종 `results/final_check.json`의 `status`가 `PASS`
