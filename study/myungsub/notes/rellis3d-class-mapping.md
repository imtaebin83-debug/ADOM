# RELLIS-3D Class Mapping & Dataset Strategy
**탄약 보급 비전투차량(한화 아리온스맷 / 현대로템 셰르파) 시나리오 맞춤형 오프로드 시맨틱 온톨로지 정의**

---

## 1. 개요 및 배경 (Context)

본 문서는 **대대급 이하 차륜형 다목적 무인차량(한화 에어로스페이스 아리온스맷 / 현대로템 HR-셰르파)을 활용한 탄약 및 보급품 지속지원 임무 수행** 시나리오를 기준으로, 카메라 기반 1차 인지모델(Semantic Segmentation) 학습에 활용할 [RELLIS-3D](https://github.com/unmannedlab/RELLIS-3D) 데이터셋 클래스를 재정의(Ontology Mapping)하는 지침 문서이다.

### 1.1 차륜형 무인차량(6x6 / 4x4) 기동성 고려사항
- **궤도형(Tracked)과의 차이**: 궤도형 차량에 비해 접지압이 높고 바퀴 간 슬립(Slip) 가능성이 존재하므로, 진흙(Mud)이나 깊은 수풀(Bush), 젖은 토사 지형에서 주행 비용(Traversability Cost)이 크게 증가한다.
- **탄약/중량 보급 수송 특성**: 적재 중량이 높을 경우 서스펜션 변형 및 제동 거리 증가로 인해, **물웅덩이(Puddle/Water), 쓰러진 나무(Log), 큰 바위(Boulder)**와 같은 희소 장애물(Rare Obstacle)에 대한 조기 탐지 및 높은 재현율(Recall)이 필수적이다.

---

## 2. RELLIS-3D 원본 클래스 분석

RELLIS-3D 데이터셋은 비정형 자연 지형(야지, 숲길, 수풀, 진흙 등)에서 취득된 RGB 이미지와 픽셀 단위 라벨을 제공한다. 기존 20개 내외의 세부 클래스를 **주행 가능성(Traversability)과 위험도** 기준으로 통폐합하여 인지모델 학습 효율과 속도(Jetson Orin NX 30+ FPS 목표)를 극대화한다.

---

## 3. 탄약 보급 시나리오 맞춤형 8-Class 온톨로지 매핑표 (Ontology Mapping Table)

아래 표는 RELLIS-3D의 원본 클래스 ID 및 명칭을 프로젝트 1차 벤치마크용 **8개 핵심 시맨틱 클래스**로 변환하는 매핑 기준이다.

| Target Class ID | Target Class Name | 포함되는 RELLIS-3D 원본 클래스 | 주행 속성 (Traversability) | ROS2 Costmap 변환 정책 |
| :---: | :--- | :--- | :--- | :--- |
| **0** | `Void / Background` | `void`, `sky` | 주행 무관 | Costmap 투영 제외 |
| **1** | `Traversable-Safe` | `dirt`, `grass`, `asphalt`, `concrete` | **정상 주행 가능** | Free Space (Cost: 0 ~ 10) |
| **2** | `Traversable-Rough` | `rubble`, `bush` | **주의 주행 영역 (감속 필요)** | Medium Cost (Cost: 100 ~ 150) |
| **3** | `High-Risk-Terrain` | `mud` | **슬립/고착 위험 영역** | High Cost (Cost: 200 ~ 230) |
| **4** | `Water / Puddle` | `puddle`, `water` | **침수/지형 미상 위험** | Lethal Obstacle (Cost: 254) |
| **5** | `Static-Obstacle` | `tree`, `pole`, `building`, `fence`, `barrier` | **정적 충돌 장애물** | Lethal Obstacle (Cost: 254) |
| **6** | `Rare-Obstacle` | `log` (쓰러진 나무), `boulder` (바위) | **치명적 희소 장애물** | Lethal Obstacle + Inflation |
| **7** | `Dynamic-Object` | `person`, `vehicle`, `object` | **동적/임시 객체** | Lethal Obstacle + 정지/우회 판단 |

> [!IMPORTANT]
> **희소 장애물(Rare Obstacle) 집중 관리**  
> `log`(쓰러진 나무)와 `boulder`(바위)는 야지 오프로드 시나리오에서 차량 하부 파손 및 전복을 유발하는 치명적 장애물이나, 데이터셋 내 출현 빈도가 낮다. 1차 벤치마크 평가 시 전체 `mIoU` 외에 **Class 6(Rare-Obstacle)의 Recall 지표를 핵심 KPI**로 추적한다.

---

## 4. 클래스 불균형(Class Imbalance) 해결을 위한 학습 전략

1. **Loss Function 최적화**:
   - 다수 클래스(`dirt`, `grass`, `tree`)가 손실함수를 지배하지 않도록 **Class-Balanced Focal Loss** 또는 **Lovasz-Softmax Loss**를 적용한다.
   - 희소 클래스(`Water/Puddle`, `Rare-Obstacle`)에 대해 인위적으로 높은 Class Weight를 부여한다.
2. **데이터 샘플링 전략**:
   - `log`, `puddle`, `mud` 픽셀 비율이 높은 프레임을 Oversampling하여 Epoch 구성 시 균형을 맞춘다.

---

## 5. 데이터셋 디렉토리 관리 규칙 (`data/README.md` 준수)

대용량 데이터셋 원본 및 전처리 라벨은 절대 Git 레포지토리에 커밋하지 않으며, 아래 권장 구조로 로컬 스토리지 또는 외부 NAS에 관리한다.

```text
data/
├── raw/
│   └── RELLIS-3D/
│       ├── Rellis-3D-images/
│       └── Rellis-3D-labels/
├── interim/
│   └── rellis_8class_mapped/       # 온톨로지 매핑 스크립트 실행 결과 (8개 ID 매핑 완료된 mask)
└── processed/
    └── benchmark_split_v1/         # Train / Val / Test 시퀀스 분할 데이터
```

- **표준 분할 규정**: RELLIS-3D 공식 시퀀스 분할을 준수하되, 평가의 신뢰성을 위해 Train 프레임과 Val/Test 프레임 간 **동일 시퀀스 혼합이 없도록 시퀀스 단위(Video Sequence-wise) 분할**을 엄수한다.
