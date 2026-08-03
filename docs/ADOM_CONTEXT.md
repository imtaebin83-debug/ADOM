# 🧠 ADOM Project Context & Decision Log

## [AI Agent Instructions] (DO NOT REMOVE)
당신은 ADOM 프로젝트의 AI 어시스턴트입니다. 코드를 작성하거나 아키텍처를 제안하기 전에 **반드시** 이 문서의 [Single Source of Truth]를 기준으로 판단하십시오. 
사용자와의 대화를 통해 새로운 기술적 결정이 내려졌거나 기존 방향이 수정되었다면, 사용자의 지시에 따라 **반드시 다음 두 가지 조치를 취하십시오:**
1. [Decision Log]의 최하단에 날짜, 결정 사항, 그리고 **'결정 사유(Why)'**를 기록합니다.
2. [Single Source of Truth]의 관련 항목을 최신 상태로 덮어씁니다.

---

## 🟢 [Single Source of Truth] (Current Project State)

**프로젝트 개요:** MUM-T 산악 오프로드 작전환경에서 무인차량(UGV)을 지원하는 '카메라 기반 온보드 인지 모델' 개발. (데모: Costmap 오버레이 시각화)

**1. 연구 핵심 목표 및 단계별 아키텍처**
- **Core Models:** SegFormer-B2 (연구적 목적 달성을 위한 베이스라인)
- **Framework:** OpenMMLab `mmsegmentation` (버전 1.2.2 고정)

- **Phase 1 (현재 집중): 20-Class 시맨틱 성능 개선 (Data Imbalance 해소)**
  - 기존 RELLIS-3D(20-Class) 베이스라인 체계를 그대로 유지하여 파인튜닝.
  - 절대적으로 부족한 '치명적 장애물(물, 통나무, 기둥 등)' 데이터를 외부 오픈 데이터(RUGD, YCOR 등)로 증강.
  - **목표:** 증강된 데이터셋을 통해 오프로드 특정 위험 객체의 인식률(Recall)이 비약적으로 향상됨을 20-Class 평가지표 기반으로 명확히 증명한다.

- **Phase 2 (고도화): Dual-Head Architecture (Semantic + Cost Map)**
  - Phase 1에서 검증된 하나의 인코더(Shared Backbone)에 두 개의 병렬 헤드 구성.
  - Head A (Ventral): 객체 외형 및 시맨틱 경계 인식 (20-Class 유지)
  - Head B (Dorsal): 주행 궤적 기반 데이터를 역투영한 Cost Map 직접 예측 (이 단계에서 5-Class Cost 체계 도입)
  - **가설 검증:** Head A+B 동시 학습 시 인코더의 Feature Sharing 시너지로 인해 Cost Map 추론 성능이 향상됨을 입증한다.

**2. 데이터 및 라벨링 정책**
- **Phase 1 (Semantic):** RELLIS-3D의 20개 클래스 체계와 라벨링 유지. 부족 클래스는 타 데이터셋에서 추출 후 20-Class ID에 맞게 매핑.
- **Phase 2 (Cost Map - Head B 전용 5-Class Cost Prior):**
  - 0: `paved` (포장/인공 지면)
  - 1: `natural_low` (흙길, 짧은 풀 등)
  - 2: `medium` (진흙, 물웅덩이, 덤불 등)
  - 3: `high_obstacle` (물, 통나무, 사람, 차량 등 회피 우선 객체)
  - 255: `ignore` (Loss 계산 시 제외)

**3. 하드웨어 및 시스템 통합**
- **개발 환경:** RunPod RTX 4090/A6000 + Docker (`nvcr.io/nvidia/pytorch:23.10-py3`)
- **엣지 배포:** NVIDIA Jetson Orin Nano 8GB (ROS2 Humble 연동). 
- *주의:* 현재는 하드웨어 스펙 업 가능성을 열어두고 있으므로, 30 FPS 방어 등의 엄격한 TensorRT 최적화보다는 **'연구적 가치 입증(데이터 증강, 듀얼 헤드 검증)'을 최우선**으로 둔다. 엣지 배포는 PoC(개념 증명) 목적으로 활용.

**4. DevOps / MLOps 인프라**
- OpenMIM 대신 `python -m pip` 직접 설치 및 `--no-deps` 덮어쓰기 정책 유지.
- CI/CD: GitHub Actions -> Docker Hub 자동 빌드 파이프라인 적용.
- Git 관리: 코드/config/Dockerfile/docs만 포함 (dataset/checkpoint/log 등은 제외).

---

## 📖 [Decision Log] (Changelog of Architectural Decisions)

> **포맷:** `[YYYY-MM-DD] 결정 사항 | 결정 사유(Why)`

- **[2026-08-03] 하드웨어 최적화 비중 축소 및 연구 가치(Data & Architecture) 중심 피벗**
  - *결정 사항:* Orin Nano 8GB 타겟의 엄격한 30 FPS 방어 및 TensorRT 최적화 태스크의 우선순위를 PoC 수준으로 낮춤. 
  - *결정 사유(Why):* 10일 스프린트 내에서 제어/하드웨어 배포에 시간을 쏟기보다, 인지 모델 본연의 연구적 독창성(Novelty)을 증명하는 것이 핵심 가치에 부합하기 때문.

- **[2026-08-03] Phase 1 20-Class 단일 학습 선행 및 5-Class 도입 시점 변경**
  - *결정 사항:* 초기부터 5-Class Cost Map으로 매핑하여 학습하는 대신, Phase 1에서는 RELLIS 기준 20-Class 시맨틱 모델로 부족 클래스 데이터 증강 실험을 선행. 5-Class Cost 체계는 Phase 2 듀얼 헤드(Dual-Head) 도입 시 Head B에만 적용.
  - *결정 사유(Why):* 부족 데이터 보완을 통한 인식률 향상(핵심 목표 1)을 명확하게 수치화하고 증명하기 위해서는 기존 RELLIS 데이터셋의 20-Class 평가 지표(mIoU, Recall)를 그대로 사용하는 것이 실험군/대조군 비교에 유리하기 때문. 이후 듀얼 헤드로 확장할 때 Cost Map을 추가하는 것이 실험적으로 훨씬 탄탄한 논리 구조를 가짐.