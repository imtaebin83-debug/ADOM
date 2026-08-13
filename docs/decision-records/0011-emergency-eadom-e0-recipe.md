# 0011. Emergency E-ADOM model using the E0 training recipe

- Status: Accepted
- Date: 2026-08-13
- Owners: 태빈 및 perception training 담당자
- Supersedes: none; temporarily precedes the 0010 TA0 recipe-discovery sequence

## Context

Jetson에 자체 수집 데이터가 반영된 B0 모델을 다음 날 배포해야 한다. TA0 method-recipe
discovery의 모든 독립 ablation과 3-seed 결합 검증을 먼저 끝내면 배포 시한을 지킬 수
없다. 검증된 target-adaptation superset에는 canonical RELLIS train 4,435장과 ADOM
standalone train 133장으로 구성된 `ta1_train.txt`가 이미 포함돼 있다.

## Decision

긴급 배포 후보 `E-ADOM`을 별도 조건으로 추가한다. E-ADOM은 ImageNet MiT-B0
pretrained weight에서 시작하고 B0-E0와 동일한 학습 방법을 사용한다.

- 512x512 RandomResize/RandomCrop, horizontal flip, PhotoMetricDistortion
- ImageNet mean/std와 CE-only
- uniform `InfiniteSampler`, effective batch 16, seed 42
- Stage 1 head-only 4,000 optimizer updates
- Stage 2 full-model 40,000 optimizer updates
- canonical RELLIS validation과 constrained Clean v1 checkpoint selection

유일한 학습 입력 차이는 `splits/train.txt` 대신 RELLIS 4,435 + ADOM 133의
`splits/ta1_train.txt`를 사용하는 것이다. canonical RELLIS test는 tuning 중 잠근다.
ADOM diagnostic split은 checkpoint 선택에 사용하지 않고 선택 후 배포 적합성만 확인한다.

## Rationale and evidence

E-ADOM은 warm-start TA나 새 loss/sampler를 도입하지 않아 B0-E0 대비 데이터 추가라는
한 가지 차이만 남긴다. 기존 E1 dataset/config scaffold와 검증된 superset을 재사용해
새 ontology, mapping, model graph 또는 Jetson inference 계약을 만들지 않는다.

PyTorch 2.1 strict deterministic mode에서 MMSeg `IoUMetric`의 CUDA `histc`가 실패하므로
E-ADOM validation은 이미 authoritative metric인 CPU 기반 `AdomSemantic20Metric`만
사용한다. 학습 결정성, seed와 CUBLAS deterministic workspace는 유지한다.

## Alternatives considered

- TA0 I/O/B/L 전체 ablation 완료 후 TA1 실행: 연구적으로 우선이나 배포 시한 초과.
- B0-E0 warm-start short fine-tuning: 더 빠르지만 E0와 학습 방법이 달라 데이터-only
  조건이 아니므로 긴급 주 비교에서 제외.
- Lovasz/RCS/새 augmentation 결합: 원인 분리가 안 되고 새 image 검증 범위가 증가.

## Consequences

- E-ADOM은 긴급 단일-seed 배포 후보이며 0010의 최종 공통 TA recipe 결론을 대체하지 않는다.
- 모델 graph와 19-class output은 B0-E0와 동일해 기존 ONNX/Jetson 경로를 재사용한다.
- 내일 배포 후 TA0 recipe discovery와 3-seed 비교는 별도 연구 evidence로 재개한다.

## Validation and rollback

dataset/image/config/checkpoint SHA, finite loss, Stage 1 backbone freeze, Stage 2 backbone
update, canonical RELLIS `ValSupported13`/`RareRisk4`, absent-class FP를 기록한다. 선택 모델은
ADOM diagnostic과 PyTorch↔ONNX parity를 통과해야 한다. RELLIS 비열화 또는 ADOM 현장
실패 시 Jetson은 기존 B0-E0 package로 rollback한다.
