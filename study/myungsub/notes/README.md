# Myungsub Study & Follow-up Log

탄약 보급 차륜형 다목적 무인차량(한화 아리온스맷 / 현대로템 HR-셰르파) 시나리오 기반 **데이터셋 구축(자료수집 및 기준선정)** 및 오픈소스 Follow-up 문서 저장소입니다.

## 학습 및 설계 노트 (`notes/`)

1. [RELLIS-3D Class Mapping & Dataset Strategy](notes/rellis3d-class-mapping.md)
   - RELLIS-3D 원본 데이터셋을 차륜형 무인차량의 오프로드 주행성(Traversability) 기준 8-Class 온톨로지로 재정의
   - 희소 장애물(`log`, `boulder`, `puddle`) 및 진흙(`mud`) 대응 전략
   - `data/README.md` 규칙에 따른 폴더 구조 및 시퀀스 분할 가이드
2. [ZED ROS2 Wrapper Setup & Sensor Pipeline Guide](notes/zed_ros2_setup.md)
   - Jetson Orin NX 탑재 기준 ZED 2i 카메라 ROS2 파라미터(`HD720 @ 30FPS`, `NEURAL` Depth)
   - 차륜형 무인차량 마운트 사양 및 ROS2 Nav2 Semantic Costmap 연동 아키텍처
