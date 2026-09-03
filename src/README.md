# Source Code

ROS 2와 독립적으로 재사용 가능한 로직만 둔다. ROS2 node, launch file, message adapter는
`ros2_ws/`에 있으며 여기의 함수를 감싸는 얇은 adapter 역할을 한다.

```text
src/
├── adom/            # 설치되는 메인 패키지
│   ├── data/        # 데이터셋 전처리, 검증, 패키징
│   ├── mmseg/       # MMSegmentation 확장 (dataset, metric, hook, sampler)
│   ├── perception/  # 추론 backend와 Semantic20 마스크 계약
│   ├── runtime/     # 학습·배포 사이클, 계약 게이트, export
│   ├── autonomy/    # costmap, 플래너, 경로 제어, 복구 로직
│   ├── analysis/    # 불확실성 분석
│   └── evaluation*.py  # Cost4 / Semantic20 지표 정의
├── adom_mmseg/      # MMSeg registry 등록용 별도 진입 패키지
└── data/            # 데이터셋별 변환 스크립트, 매핑 config, split
```

## `adom` 패키지

### `adom.data` — 데이터셋 전처리

원본 데이터셋을 Semantic20 계약(train ID `0..18`, ignore `255`)으로 변환하고 검증한다.

- `pipeline.py`, `io.py`, `schema.py`, `models.py` — 변환 파이프라인과 자료형
- `adapters/` — 데이터셋별 어댑터 (`rellis3d.py`, 공통 `base.py`)
- `semantic20.py` — Semantic20 패키지 생성 진입점
- `splits.py`, `validation.py`, `packaging.py` — split 생성, 무결성 검증, 배포 패키징
- `target_adaptation.py`, `transform_audit.py` — TA 패키지 구성과 transform 감사
- `preview.py`, `cli.py` — 시각화와 CLI

### `adom.mmseg` — 학습 확장

- `dataset.py` — `AdomSemantic20Dataset` 등 MMSeg dataset 등록
- `metrics.py` — `AdomSemantic20Metric`. stock `IoUMetric`이 CUDA `histc`를 쓰는 탓에
  strict deterministic 모드에서 거부되므로, confusion matrix를 CPU에서 계산한다. 체크포인트
  선택에 쓰이는 mIoU, recall, precision, absent-class FP 패널이 여기서 나온다.
- `hooks.py` — 학습 계약 검증 훅 (E0 SHA, seed, effective batch, phase update 검사)
- `samplers.py` — uniform / RCS 샘플러

### `adom.perception` — 추론

- `mmseg_backend.py` — MMSeg 체크포인트 추론 backend
- `semantic20.py` — 온톨로지, RGB 팔레트, 마스크 계약
- `latest_frame.py` — 최신 프레임 유지 버퍼

### `adom.runtime` — 사이클과 게이트

- `semantic20_cycle.py` — 학습 1-cycle 오케스트레이션. `--experiment`로 `e0`/`e1`/`e2`/
  `eadom`/`ta0`/`ta1`/`ta2`를 선택하며, 데이터셋 정체성 검증부터 Stage 2 hand-off까지 묶는다.
- `b5_gate.py`, `b5_capacity_domain_contract.py`, `b2_eadom_contract.py` — capacity 연구용
  사전등록 계약. B5는 go/no-go 판정 파일 없이 시작하지 않는다.
- `semantic20_export.py`, `onnx_parity.py`, `semantic20_tensorrt.py` — ONNX export와 parity,
  TensorRT engine 검증
- `semantic20_handoff.py`, `checkpoints.py`, `artifacts.py` — 체크포인트 SHA 검증과 hand-off
- `semantic20_aggregate.py`, `semantic20_logit_dump.py` — 결과 집계, raw logit 덤프
- `doctor.py`, `source_sampling.py`, `cycle.py` — 환경 진단, 소스 샘플링, Cost4 사이클

### `adom.autonomy` — 자율주행 로직

ROS 의존성 없이 순수 Python으로 구현해 오프라인 재현과 테스트가 가능하다.

- `costmap.py` — Semantic20 마스크를 traversability cost로 변환
- `rule_planner.py` — 3-depth 방향 트리 + gap-guided corridor 탐색
- `path_control.py`, `actuation.py` — 경로 추종과 Ackermann/PWM 명령 변환
- `imu_speed.py`, `stuck_recovery.py` — IMU 기반 속도 추정, 스턱 복구

### `adom.analysis`

- `semantic20_uncertainty.py` — SML, entropy, MSP, margin, energy 등 불확실성 지표 비교

## `adom_mmseg`

MMSeg registry에 Cost4/5-class 데이터셋을 등록하기 위한 별도 진입 패키지다. MMSeg config가
`custom_imports`로 이 경로를 직접 참조하므로 `adom` 패키지와 분리해 둔다.

## `src/data` — 데이터셋별 변환

각 소스 데이터셋의 원본 라벨을 Semantic20으로 옮기는 스크립트, 매핑 config, split을 둔다.
번호 접두사는 실행 순서다.

| 디렉터리 | 대상 |
| --- | --- |
| `rellis/` | RELLIS-3D 원본 ID 감사와 마스크 변환 |
| `rugd/` | RUGD 검사와 image/mask 정리 |
| `ycor/` | YCOR 원본 구조 확인, manifest 생성, 라벨 스캔, 쌍 검증 |
| `semantic_20/` | 소스 통합, YCOR RGB 분포 감사, 결합 패키지 생성과 검증 |
| `cost_4/` | 기존 Cost4 계약용 변환 |
| `adom_data/` | 자체 수집 데이터의 CVAT 마스크 정규화와 업로드 패키지 생성 |

`adom_data/`는 별도 계약이 있으므로 [해당 README](data/adom_data/README.md)를 따른다.

## 설치되는 CLI

`pip install -e .` 후 다음 명령을 쓸 수 있다.

| 명령 | 대상 |
| --- | --- |
| `adom-semantic20-preprocess` | Semantic20 패키지 생성 |
| `adom-semantic20-aggregate` | 실험 결과 집계 |
| `adom-semantic20-ta-package` | target adaptation 패키지 구성 |
| `adom-ta0-transform-audit` | crop/resize transform의 mask retention 감사 |
| `adom-semantic20-uncertainty` | 불확실성 지표 평가 |
| `adom-semantic20-logit-dump` | Jetson engine raw logit 덤프 |

## 규칙

- 이 패키지는 ROS 2에 의존하지 않는다. ROS 타입이 필요하면 `ros2_ws/`의 노드에서 변환한다.
- Semantic20 계약(`0..18` + `255`)을 바꾸는 변경은 전처리, config, ONNX 채널, ROS 토픽을
  동시에 건드리므로 decision record를 남긴다.
- 데이터·평가·런타임 계약을 바꾸면 `tests/`를 같은 PR에서 갱신한다.
