# 0010. TA0 common fine-tuning recipe selection

- Status: Accepted
- Date: 2026-08-12
- Owners: 태빈 및 perception training 담당자
- Amends: 0009의 TA0 해석과 실행 순서

## Context

0009는 TA0를 RELLIS-only continued-training control로 정의했다. 그러나 B0-E0의 현재
학습은 512x512 random crop, ImageNet normalization, CrossEntropy, head-only Stage 1과
full-model Stage 2를 사용하고, 실제 배포 후보는 640x384다. 앞으로 반복 사용할
fine-tuning 틀을 만들려면 TA0에서 crop, 전처리, 불균형 대응과 two-stage 최적화를 먼저
검증해야 한다. TA0만 다른 방법을 사용한 채 TA1/TA2를 실행하면 방법과 데이터 효과가
섞인다.

## Decision

TA0를 **공통 개선 recipe의 method control**로 재정의한다.

- Frozen E0: 기존 RELLIS 학습 checkpoint. 원본 보고 지표와 새 공통 evaluation
  pipeline에서의 재평가 지표를 모두 보존한다.
- TA0: E0에서 시작해 RELLIS-only로 선택된 개선 recipe를 적용한다.
- TA1: 같은 E0와 같은 TA0 recipe에 ADOM ZED2i train만 추가한다.
- TA2: 같은 E0와 같은 TA0 recipe에 RUGD, YCOR, ADOM ZED2i train을 추가한다.

TA0 recipe는 한 번에 여러 방법을 넣지 않는다. input/crop, two-stage optimization,
class-imbalance 대응을 순차 ablation하고, 각 단계에서 검증된 요소만 결합한 뒤 3-seed로
상호작용을 재검증한다. recipe 선택 중에는 canonical test를 열지 않는다. TA1/TA2 학습은
recipe가 commit SHA로 동결된 뒤 시작한다.

## Consequences

- 0009의 package, sampler, checkpoint SHA, 독립 output과 canonical evaluation 계약은
  유지된다.
- TA0 vs Frozen E0는 방법론+continued adaptation의 효과이고, TA1 vs TA0는 standalone
  데이터의 순효과, TA2 vs TA1/TA0는 public multi-source의 추가 효과다.
- source-aware sampler framework는 모두 사용하되 TA0는 RELLIS 1.0이다.
- no-crop 640x384, rare-class sampling, 복합 loss 또는 layer-wise LR은 아직 결과가 아닌
  후보이며 ablation 통과 전 공통 recipe로 주장하지 않는다.
- 최종 B0 graph를 바꾸지 않는 전처리·sampler·loss·optimizer 개선을 우선해 Jetson
  inference latency를 E0와 같은 등급으로 유지한다.

## Validation and rollback

세 input 후보의 transformed-mask class retention과 배포 parity를 먼저 정량 감사한다.
seed 42 mini에서 후보를 선별하고 seeds 42/43/44로 재검증한다. canonical RELLIS
OverallSupported와 RareRisk가 개선되지 않거나 기존 supported class가 허용 범위를 넘어
하락하면 해당 요소를 제외한다. 결합 recipe가 개별 개선을 재현하지 못하면 가장 단순한
통과 recipe로 rollback한다.
