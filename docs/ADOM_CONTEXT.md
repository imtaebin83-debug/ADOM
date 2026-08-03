# 🧠 ADOM Project Context & Decision Log

## [AI Agent Instructions] (DO NOT REMOVE)
당신은 ADOM 프로젝트의 AI 어시스턴트입니다. 코드를 작성하거나 아키텍처를 제안하기 전에 **반드시** 이 문서의 [Single Source of Truth]를 기준으로 판단하십시오. 
사용자와의 대화를 통해 새로운 기술적 결정이 내려졌거나 기존 방향이 수정되었다면, 사용자의 지시에 따라 **반드시 다음 두 가지 조치를 취하십시오:**
1. [Decision Log]의 최하단에 날짜, 결정 사항, 그리고 **'결정 사유(Why)'**를 기록합니다.
2. [Single Source of Truth]의 관련 항목을 최신 상태로 덮어씁니다.

---

## 🟢 [Single Source of Truth] (Current Project State)

**프로젝트 개요:** MUM-T 산악 오프로드 작전환경에서 무인차량(UGV)을 지원하는 '카메라 기반 온보드 인지 모델' 개발. (데모: Costmap 오버레이 시각화)

**1. 연구 핵심 목표 및 아키텍처**
- **Core Models:** SegFormer-B2 (연구적 목적 달성을 위한 베이스라인)
- **Framework:** OpenMMLab `mmsegmentation` (버전 1.2.2 고정)
- **핵심 연구 1 (Data Imbalance 해소):** RELLIS-3D 기반에 부족한 '치명적 장애물(물, 통나무, 기둥 등)' 데이터를 외부 오픈 데이터(RUGD, YCOR 등)로 증강 파인튜닝하여 오프로드 특정 객체 인식률 향상 증명.
- **핵심 연구 2 (Dual-Head Architecture):** 하나의 인코더(Shared Backbone) + 두 개의 병렬 헤드 구성.
  - Head A (Ventral): 객체 외형 및 시맨틱 경계 인식 (20-Class 등)
  - Head B (Dorsal): 주행 궤적 기반 자가지도학습 데이터를 역투영한 Cost Map 직접 예측 (5-Class)

**2. 데이터 및 라벨링 (5-Class Cost Prior)**
- 0: `paved` (포장/인공 지면)
- 1: `natural_low` (흙길, 짧은 풀 등)
- 2: `medium` (진흙, 물웅덩이, 덤불 등)
- 3: `high_obstacle` (물, 통나무, 사람, 차량 등 회피 우선 객체)
- 255: `ignore` (Loss 계산 시 제외)

**3. 하드웨어 및 시스템 통합**
- **개발 환경:** RunPod RTX 4090/A6000 + Docker (`nvcr.io/nvidia/pytorch:23.10-py3`)
- **엣지 배포:** NVIDIA Jetson Orin Nano 8GB (ROS2 Humble 연동). 
- *주의:* 현재는 하드웨어 스펙 업 가능성을 열어두고 있으므로, 30 FPS 방어 등의 엄격한 TensorRT 최적화보다는 **'연구적 가치 입증(데이터 불균형 해소, 듀얼 헤드 검증)'을 최우선**으로 둔다. 엣지 배포는 PoC(개념 증명) 목적으로 활용.

**4. DevOps / MLOps 인프라**
- OpenMIM 대신 `python -m pip` 직접 설치 및 `--no-deps` 덮어쓰기 정책 유지.
- CI/CD: GitHub Actions -> Docker Hub 자동 빌드 파이프라인 적용.
- Git 관리: 코드/config/Dockerfile/docs만 포함 (dataset/checkpoint/log 등은 제외).

---

## 📖 [Decision Log] (Changelog of Architectural Decisions)

> **포맷:** `[YYYY-MM-DD] 결정 사항 | 결정 사유(Why)`

- **[2026-08-03] 하드웨어 최적화 비중 축소 및 연구 가치(Data & Architecture) 중심 피벗**
  - *결정 사항:* Orin Nano 8GB 타겟의 엄격한 30 FPS 방어 및 TensorRT INT8 양자화 최적화 태스크의 우선순위를 PoC 수준으로 낮춤. 대신 ①취약 클래스 데이터 증강(RELLIS + RUGD 등)과 ②Shared Backbone Dual-Head(Semantic + Cost Map) 멀티태스크 학습 검증에 집중하기로 함.
  - *결정 사유(Why):* 10일(총 300 Man-hours)이라는 제한된 스프린트 내에서 제어/하드웨어 배포에 시간을 쏟기보다는, 하드웨어 스펙 업 가능성을 열어두고 인지 모델 본연의 연구적 독창성(Novelty)과 성능 향상 가설을 데이터로 증명하는 것이 프로젝트의 핵심 가치에 부합하기 때문.