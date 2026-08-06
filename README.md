# ADOM: Autonomous Driving for Off-Road Military Vehicles

MUM-T(유무인 복합체계) 산악 오프로드 작전 환경을 위한 카메라 기반 파운데이션 모델 도메인 적응 및 온보드 인지 기술 연구.

**ADOM** is a research project focused on camera-based foundation model domain adaptation and on-board perception technology for MUM-T (Manned-Unmanned Teaming) in mountainous off-road environments.

> 현재 범위, 담당자, 성공 기준의 유일한 기준은
> [ADOM_CONTEXT.md](ADOM_CONTEXT.md)다. 작업을 시작하기 전에 이 문서와
> [프로젝트 상태판](docs/status/README.md)을 먼저 확인한다.

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
├── docs/                 # 회의록, 의사결정 기록, 시스템 아키텍처, 세팅 가이드
├── experiments/          # 재현 가능한 baseline/segformer/quantization/deployment/costmap 실험
├── external/             # 필요 시 외부 오픈소스 fork/submodule 연결 지점
├── models/               # 체크포인트와 export 산출물 배치 규칙, 실제 모델 파일은 미커밋
├── results/              # Markdown/CSV 기반 실험 결과 요약
├── ros2_ws/              # ROS2 Humble 패키지와 launch/config, mono repo 내부 colcon workspace
├── scripts/              # 반복 작업 자동화 스크립트
├── src/                  # 실제 프로젝트 코드
└── study/                # 팀원별 논문 정리, 학습 노트, 작은 테스트
```

## How We Use This Repo

- `study/`는 학습 기록과 작은 테스트를 위한 공간입니다. 외부 오픈소스 전체 코드를 그대로 커밋하지 않습니다.
- `experiments/`는 결과를 재현할 수 있는 실험 단위입니다. 실험이 여러 번 재사용되면 공통 로직을 `src/`로 옮깁니다.
- `src/`는 실제 시스템에 들어갈 코드만 둡니다. 임시 분석 코드와 노트북은 `study/` 또는 `experiments/`에 둡니다.
- `ros2_ws/`는 이 mono repo 안에서 관리하는 ROS2 Humble workspace입니다. ROS2 node는 `src/`의 공통 로직을 감싸는 adapter 역할을 우선합니다.
- `results/`는 가벼운 Markdown/CSV 실험 추적을 위한 공간입니다. W&B나 MLflow 같은 별도 플랫폼은 나중에 필요할 때 도입합니다.
- 대용량 데이터셋, 학습 결과, checkpoint, TensorRT engine은 git에 올리지 않습니다.

## Key Docs

- [Current Source of Truth](ADOM_CONTEXT.md)
- [Docs hub and collaboration workflow](docs/README.md)
- [Project status dashboard](docs/status/README.md)
- [Repo structure decision](docs/decision-records/0001-repo-structure.md)
- [Initial benchmark decision](docs/decision-records/0002-initial-benchmark-scope.md)
- [Benchmark protocol](docs/metrics/benchmark-protocol.md)
- [System architecture overview](docs/system-architecture/overview.md)
- [Development setup guide](docs/setup-guides/development.md)
- [RunPod training and DevOps guide](docs/devops.md)
- [RunPod one-cycle command](docs/runpod-one-cycle.md)
- [RELLIS-3D Cost4 data contract](docs/datasets/rellis3d-cost4.md)
- [Dataset preprocessing migration plan](docs/dataset-preprocessing-migration-plan.md)
- [Contribution guide](CONTRIBUTING.md)
