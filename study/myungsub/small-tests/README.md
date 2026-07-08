# ADOM RELLIS-3D 8-Class Mapping & Costmap Demo (`small-tests/`)

첨부된 오프로드 레퍼런스 이미지(1~8번 영역)와 RELLIS-3D 클래스를 **탄약 보급 비전투차량(아리온스맷 / HR-셰르파) 시나리오 맞춤형 8-Class 시맨틱 온톨로지** 및 **ROS2 Nav2 Costmap**으로 변환하고 시각화하는 소형 테스트 코드입니다.

## 파일 구성

- **`demo_rellis_8class_mapping.py`**: 단일 실행 파이썬 스크립트 (`NumPy`, `Pillow` 사용)
- **`output_8class_comparison.png`**: 스크립트 실행 시 생성되는 3분할 비교 시각화 결과

## 실행 방법

### 1. 기본 레퍼런스 씬 테스트
```bash
cd ~/ADOM-Autonomous-Driving-for-Off-Road-Military-vehicles/study/myungsub/small-tests
python3 demo_rellis_8class_mapping.py --out output_8class_comparison.png
```

### 2. 커스텀 외부 PNG 이미지/마스크 입력 변환
다른 마스크나 RELLIS-3D 라벨 PNG 파일을 8-Class 및 Costmap으로 바꾸고 싶을 때 `--image` 옵션을 사용합니다:
```bash
python3 demo_rellis_8class_mapping.py --image path/to/custom_input.png --out my_result.png
```

## 변환 로직 (Reference 1~8 -> ADOM 8-Class)

| 레퍼런스 이미지 영역 | 매핑된 ADOM 8-Class ID 및 명칭 | ROS2 Nav2 Cost |
| :--- | :--- | :--- |
| **1: 하늘** | `0: Void / Sky` | `0` (비주행/제외) |
| **2: 도로 & 3: 왼쪽 지면** | `1: Traversable-Safe` | `0` (정상 주행 가능) |
| **5: 오른쪽 진경 초목** | `2: Traversable-Rough` | `128` (주의 주행 / 감속) |
| **6/7/8: 중경 초목 및 원경 숲** | `5: Static-Obstacle` | `254` (Lethal Obstacle) |
| **4: 쓰러진 통나무 (핵심 KPI)** | `6: Rare-Obstacle (Log)` | `254` (Lethal + 인플레이션) |
