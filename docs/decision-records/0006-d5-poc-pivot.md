# 0006. D-5 Live Stop PoC Pivot

- Status: Accepted
- Date: 2026-08-06
- Owners: ADOM team
- Supersedes: 0002의 단계 순서, 0003의 overlay-only demo 범위,
  `decision_logs.md`의 2026-08-03 연구 우선순위

## Context

집중연구기간이 5일 남은 상태에서 SegFormer-B0/B2 E0 학습과 RC Car 하드웨어 조립은
완료됐지만 TensorRT, live perception ROS node, perception-to-control 검증, 자체 데이터
수집과 fine-tuning은 완료되지 않았다. 새 architecture나 통합 ontology의 연구 기여를
완성하기보다 발표에서 재현 가능한 end-to-end evidence를 확보하는 것이 우선이다.

## Decision

- D-5 최우선 목표를 `ZED RGB → SegFormer-B0 FP16 TensorRT → safety ROI → 정지`로 둔다.
- autonomy 범위는 저속 직진 Go/Stop이며 조향 회피, Nav2, depth, VIO, costmap은 제외한다.
- 기본 target은 Semantic20 `log` ID 10, `pole`은 stretch, 기존 B0/B2 `rubble`
  차이는 신규 학습 실패 시 fallback이다.
- 자체 데이터는 target만 라벨하고 나머지를 ignore 255로 두되, RELLIS full-label
  anchor와 섞어 short fine-tuning한다.
- TensorRT engine은 실제 Jetson에서 생성하며, input shape와 ROS topic은 담당자가
  실측하기 전까지 제안값으로 유지한다.
- 발표 성공은 모델 점수 하나가 아니라 동일 장면의 E0 실패/개선 모델 성공,
  live 또는 replay perception-to-stop evidence, watchdog 증거로 판정한다.

## Rationale and evidence

- B0는 이미 학습·평가됐고 Orin Nano 8GB에서 B2보다 배포 위험이 낮다.
- log는 B0/B2 IoU가 각각 40.33/40.17로 용량 증가만으로 개선되지 않았으며,
  pole보다 촬영·annotation·downsampling·ROI 검출 위험이 낮다.
- pole은 B0/B2 모두 IoU 0이지만 얇은 구조라 D-5의 유일한 성공 경로로 삼기 어렵다.
- water는 canonical test GT가 없고 RGB만으로 물리 위험을 판정하기 어려워 비교 근거가
  약하다.
- Jetson/ROS/ZED의 실제 설치 상태와 topic은 아직 audit 전이므로 문서 추정값을
  확정 계약으로 두면 integration 실패를 숨길 수 있다.

## Alternatives considered

- **LoRA 또는 새 decoder:** 구현·회귀 검증 비용에 비해 5일 내 이득이 불확실해 보류.
- **B2 우선 배포:** 정확도 이득이 class별로 불안정하고 latency 위험이 커 fallback으로 제한.
- **pole 단일 target:** 실패 사례는 강하지만 크기와 downsampling 위험 때문에 stretch로 제한.
- **depth/costmap/Nav2:** 군용 서사는 강하지만 GPU 경합과 integration 범위가 과도해 후속 단계로 이동.
- **recorded-only demo:** live 실패 시 보존해야 할 fallback이지만 primary goal로는 채택하지 않음.

## Consequences

- Phase 1 Clean Semantic20 연구 실험과 Semantic23 작업은 폐기하지 않고 발표 이후 재개한다.
- Day 1–2의 TensorRT/live ROS gate가 신규 데이터 학습보다 우선한다.
- 모델 개선이 작더라도 실패를 숨기지 않고 fallback ladder와 한계로 보고한다.
- 안전 시험은 actuator 미연결, wheels-off, watchdog, 저속 직선, soft target 순서를 따른다.

## Validation and rollback

- 상세 gate와 rollback 조건은 [D-5 status](../status/d5-poc.md)와
  [Source of Truth](../../ADOM_CONTEXT.md)를 따른다.
- Day 2 종료까지 live mask가 없으면 recorded ZED input으로 전환한다.
- 신규 fine-tuning이 개선되지 않으면 기존 rubble 비교 또는 E0 pipeline evidence를 쓴다.
- 발표 종료 후 연구 우선순위를 다시 결정할 때는 이 record를 덮어쓰지 않고 새 record를 만든다.
