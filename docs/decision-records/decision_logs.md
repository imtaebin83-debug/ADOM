## 📖 [Decision Log] (Changelog of Architectural Decisions)

> **포맷:** `[YYYY-MM-DD] 결정 사항 | 결정 사유(Why)`
>
> 이 파일은 번호형 decision record 도입 전의 historical log다.
> 2026-08-06 이후의 결정 근거는 [`0006-d5-poc-pivot.md`](0006-d5-poc-pivot.md)를 우선한다.

- **[2026-08-03] 하드웨어 최적화 비중 축소 및 연구 가치(Data & Architecture) 중심 피벗**
  - *결정 사항:* Orin Nano 8GB 타겟의 엄격한 30 FPS 방어 및 TensorRT 최적화 태스크의 우선순위를 PoC 수준으로 낮춤.
  - *결정 사유(Why):* 10일 스프린트 내에서 제어/하드웨어 배포에 시간을 쏟기보다, 인지 모델 본연의 연구적 독창성(Novelty)을 증명하는 것이 핵심 가치에 부합하기 때문.

- **[2026-08-03] Phase 1 20-Class 단일 학습 선행 및 5-Class 도입 시점 변경**
  - *결정 사항:* 초기부터 5-Class Cost Map으로 매핑하여 학습하는 대신, Phase 1에서는 RELLIS 기준 20-Class 시맨틱 모델로 부족 클래스 데이터 증강 실험을 선행. 5-Class Cost 체계는 Phase 2 듀얼 헤드(Dual-Head) 도입 시 Head B에만 적용.
  - *결정 사유(Why):* 부족 데이터 보완을 통한 인식률 향상(핵심 목표 1)을 명확하게 수치화하고 증명하기 위해서는 기존 RELLIS 데이터셋의 20-Class 평가 지표(mIoU, Recall)를 그대로 사용하는 것이 실험군/대조군 비교에 유리하기 때문. 이후 듀얼 헤드로 확장할 때 Cost Map을 추가하는 것이 실험적으로 훨씬 탄탄한 논리 구조를 가짐.

- **[2026-08-03] B0 선행/B2 후속 학습 순서 유지**
  - *결정 사항:* 현재 실행기의 `B0 -> B2` 순서와 지원 모델을 B0/B2로 제한하는 계약을 유지한다.
  - *결정 사유(Why):* B0를 저비용 runtime·pipeline gate로 먼저 통과시킨 뒤 B2 실험을 수행하는 현재 흐름이 연구 일정에 충분하며, 현 단계에서는 임의 모델 병렬화보다 안정적인 반복 실행이 우선이기 때문이다.

- **[2026-08-03] RunPod A100 80GB 및 W&B 중심 MLOps 운영**
  - *결정 사항:* RunPod A100 80GB Secure Cloud와 공유 Network Volume을 사용한다. W&B를 핵심 실험 추적기로, TensorBoard를 로컬 백업으로 사용하며, 500 iteration 주기의 model/optimizer/scheduler checkpoint로 중단 학습을 재개한다. Docker 이미지는 코드까지 포함한 Git SHA 불변 이미지로 배포한다.
- *결정 사유(Why):* 10일/20만원 범위에서 A100 80GB가 VRAM과 비용의 균형이 좋고, 여러 Pod가 데이터셋을 중복 저장하지 않으면서도 실험별 로그와 결과를 격리할 수 있어야 하기 때문이다. 중간 checkpoint와 중앙 로그는 GPU 중단 시 손실을 제한하고 모델 버전 비교를 재현 가능하게 만든다.
