# ADOM: Autonomous Driving for Off-Road Military Vehicles

MUM-T(유무인 복합체계) 산악 오프로드 작전 환경을 위한 카메라 기반 파운데이션 모델 도메인 적응 및 온보드 인지 기술 연구.

**ADOM** is a research project focused on camera-based foundation model domain adaptation and on-board perception technology for MUM-T (Manned-Unmanned Teaming) in mountainous off-road environments.

## Project Goal

이 레포는 산악/비정형 오프로드 환경에서 카메라 기반 인지 모델을 학습, 경량화, 온보드 배치하고 최종적으로 ROS2 기반 주행 파이프라인과 연결하는 것을 목표로 합니다.

초기 연구 축은 다음과 같습니다.

- **Semantic segmentation baseline**: YOLOv8-Seg, DeepLabV3+, SegFormer 계열 비교
- **Domain adaptation / fine-tuning**: RELLIS-3D 등 오프로드 데이터셋 기반 성능 개선
- **Model optimization**: Jetson Orin Nano 8GB 온보드 구동을 위한 후속 TensorRT 최적화
- **Perception to control**: semantic costmap, ROS2 Nav2, high-level vehicle control 연동

## Current Operating Focus

현재 D-5 집중연구기간의 목표는 SegFormer-B0를 Jetson Orin Nano에 배포하고,
ZED 2i RGB 인지가 RC Car의 안전 정지로 이어지는 end-to-end PoC를 재현하는 것이다.
연구용 Clean Semantic20 비교와 Semantic23, depth/costmap, 웹 자동화는 보존하되
발표 시연 파이프라인이 안정화된 뒤 재개한다.

초기 benchmark 필수 metric:

- segmentation: mIoU, class IoU, high_cost_or_obstacle recall
- runtime: FPS, latency p50/p95
- edge deployment: power draw on Jetson Orin Nano 8GB
- mapping readiness: costmap update rate

## Repository Map

```text
.
├── configs/              # 학습, 평가, 배포 설정 파일
├── data/                 # 데이터셋 배치 규칙과 메타데이터, 실제 대용량 데이터는 미커밋
├── docs/                 # 시스템 아키텍처, 세팅 가이드, 데이터 계약, benchmark 정의
├── external/             # 필요 시 외부 오픈소스 fork/submodule 연결 지점
├── models/               # 체크포인트와 export 산출물 배치 규칙, 실제 모델 파일은 미커밋
├── ros2_ws/              # ROS2 패키지와 launch/config, mono repo 내부 colcon workspace
├── scripts/              # 학습/배포 사이클 실행 스크립트
├── src/                  # 학습, 추론, 평가, 데이터셋 전처리 코드
├── tests/                # 전처리·평가·런타임 계약 검증 테스트
└── tools/                # 논문용 평가, RC 주행 평가, 감사 도구
```

## How We Use This Repo

- `src/`는 실제 시스템에 들어갈 코드만 둡니다. 데이터셋 전처리, 학습 확장, 추론, 평가, 자율주행 로직이 여기에 있습니다.
- `ros2_ws/`는 이 mono repo 안에서 관리하는 ROS2 colcon workspace입니다. ROS2 node는 `src/`의 공통 로직을 감싸는 adapter 역할을 우선합니다.
- `tools/`는 논문 결과 재현과 실차 평가에 쓰는 오프라인 스크립트를 모아둡니다.
- `tests/`는 데이터 전처리, 평가, 런타임 계약을 검증합니다. CI에서 `python -m unittest discover -s tests`로 전부 실행됩니다.
- 대용량 데이터셋, 학습 결과, checkpoint, TensorRT engine은 git에 올리지 않습니다.

## Key Docs

- [Docs hub](docs/README.md)
- [Decision records](docs/decision-records/README.md)
- [System architecture overview](docs/system-architecture/overview.md)
- [Development setup guide](docs/setup-guides/development.md)
- [Benchmark protocol](docs/metrics/benchmark-protocol.md)
- [RELLIS-3D Cost4 data contract](docs/datasets/rellis3d-cost4.md)
- [RunPod training and DevOps guide](docs/devops.md)
- [RunPod one-cycle command](docs/runpod-one-cycle.md)
- [RC vehicle (Traxxas XL-5) setup](RC_SETTING.md)
- [Jetson shortcut commands](SHORTCUT.md)
- [Contribution guide](CONTRIBUTING.md)
