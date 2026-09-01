# ADOM Docs Hub

이 폴더는 ADOM 저장소를 재현하는 데 필요한 문서를 역할별로 보관한다.

## Start here

1. [System architecture](system-architecture/overview.md) — 인지에서 제어까지의 구성과 interface
2. [Development setup](setup-guides/development.md) — 환경 구축과 학습/평가 실행
3. [Benchmark protocol](metrics/benchmark-protocol.md) — metric 정의와 측정 조건
4. [RELLIS-3D Cost4 data contract](datasets/rellis3d-cost4.md) — 데이터셋 전처리 계약
5. [Decision records](decision-records/README.md) — 실험 설계와 주요 결정의 근거

## Document map

| 위치 | 용도 |
| --- | --- |
| `decision-records/` | 되돌리기 비싼 결정의 배경, 대안, 결과 (결정 시점 원문 보존) |
| `system-architecture/` | 안정화된 시스템 설계와 interface |
| `setup-guides/` | 재현 가능한 환경 구축·운영 절차 |
| `metrics/` | benchmark와 metric 정의 |
| `datasets/` | 데이터셋 전처리 계약과 클래스 매핑 |
| `devops.md`, `runpod-one-cycle.md` | RunPod 학습 이미지와 1-cycle 실행 절차 |
