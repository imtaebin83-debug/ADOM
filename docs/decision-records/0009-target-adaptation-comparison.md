# 0009. B0-E0 target-adaptation comparison

- Status: Accepted
- Date: 2026-08-12
- Owners: 태빈 및 perception training 담당자
- Supersedes: none

## Context

ZED 2i로 수집·라벨링한 standalone Semantic20 source package가 RunPod Network
Volume에 확인됐다. 이 데이터의 효과와 단순 추가 학습 효과를 분리하지 않으면
fine-tuning 개선을 데이터 구성의 효과로 해석할 수 없다. 기존 E1 package는
RELLIS, RUGD, YCOR를 포함하지만 canonical 비교군 B0-E0는 RELLIS-only다.

## Decision

Frozen B0-E0 selected checkpoint를 공통 초기점으로 사용해 세 condition을 독립적으로
실행한다.

- TA0: RELLIS-only 추가 학습. 추가 optimizer update 효과를 통제한다.
- TA1: RELLIS + `adom_zed2i` standalone train.
- TA2: RELLIS + RUGD + YCOR + `adom_zed2i` standalone train.

모든 condition은 B0, Semantic20 IDs `0..18`와 ignore `255`, canonical RELLIS
validation/test, 같은 optimizer-update budget과 seed를 사용한다. TA0/TA1/TA2는
공통 foundation commit에서 세 Git branch로 분기하며 RunPod Pod별 output directory와
W&B run identity를 분리한다. TA checkpoint를 서로 이어 학습하거나 weight averaging,
checkpoint merge 또는 ensemble하는 것은 이 비교에 포함하지 않는다.

## Rationale and evidence

- TA0와의 비교가 없으면 TA1/TA2의 개선이 신규 데이터가 아니라 continued training에서
  왔을 가능성을 배제할 수 없다.
- 동일 E0 초기점과 canonical evaluation split이 condition 간 인과 해석을 보존한다.
- 병렬 Pod 실행은 wall-clock을 줄이면서도 독립 output을 사용하면 재현성을 유지한다.
- standalone validation/test는 domain diagnostic이며 canonical checkpoint-selection
  metric을 대체하지 않는다.

## Alternatives considered

- TA1 checkpoint에서 TA2를 이어 학습: curriculum 효과가 섞이므로 별도 TA3 ablation
  없이는 채택하지 않는다.
- E0 대신 ImageNet initialization부터 재학습: target adaptation 질문과 다른 실험이며
  시간과 비용이 증가한다.
- 세 모델을 병합: 최종 배포 후보를 고르는 실험 목표와 맞지 않으며 추론·운영 계약을
  불명확하게 한다.

## Consequences

- 세 condition 모두 `--initial-checkpoint`와 expected SHA-256 검증이 필요하다.
- source-aware sampling 비중과 실제 draw count를 artifact로 남겨야 한다.
- RunPod에서는 같은 Network Volume을 read-only dataset source로 공유할 수 있지만
  output directory에 동시 쓰기를 허용하지 않는다.
- 모두 유효하면 결과를 합치는 대신 선택된 recipe로 E0에서 독립적인 TA-final을
  재학습한다.

## Validation and rollback

먼저 seed 42, 50-update smoke를 병렬로 실행해 dataset contract, checkpoint load,
sampler exposure와 artifact 분리를 검증한다. 이후 사용자 승인을 받고 mini 및 3-seed
full을 실행한다. contract 위반이나 데이터 QC 실패가 있으면 학습을 시작하지 않고
공통 foundation 또는 materialized package를 새 version으로 수정한다.
