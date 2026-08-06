# Clean Baseline v1 protocol

- Status: accepted
- Decision date: 2026-08-06
- Implementation: complete for evaluation/config/runtime contracts; E2 dataset
  materialization awaits the finalized GOOSE filesystem and split artifact
- Scope: Phase 1 Semantic20
- Semantic23 / PR #27: deferred

## Approved decisions

- Semantic20을 Phase 1 ontology로 유지한다.
- GOOSE 추가의 주된 검증 대상은 pole이다.
- checkpoint는 `ValSupported13-mIoU` 최고점에서 1.0%p 이내인 후보 중
  `RareRisk-4 mIoU`가 가장 높은 것을 선택한다.
- checkpoint 선택과 recipe 결정은 validation으로만 수행한다.
- test는 최종 확정 model만 한 번 평가한다.
- 4-source condition은 기존 E1을 재정의하지 않고 E2로 명명한다.
- main val/test는 RELLIS-only로 유지하고 source validation은 diagnostic으로 둔다.
- Clean v1 loss는 unweighted CrossEntropy와 `avg_non_ignore=True`를 사용한다.
- paired seeds는 42, 43, 44를 사용한다.
- 첫 Clean v1은 Stage 2 최대 40k를 유지하고 schedule 축소는 후속
  ablation으로 분리한다.

## Split policy

현재 E1 package의 main split은 다음과 같다.

| Split | Samples | Sources |
| --- | ---: | --- |
| train | 9,868 | RELLIS 4,435 + RUGD 4,779 + YCOR 654 |
| val | 900 | RELLIS only |
| test | 899 | RELLIS only |

RUGD val 733, RUGD test 1,924, YCOR val 97은 diagnostic split으로 별도 보존된다.
이 정책은 E0와 E1을 동일한 target-domain 표본에서 비교하게 해 외부 train data의
효과를 분리한다.

E2에서 GOOSE도 train source로 추가하되, sequence/site가 겹치지 않는 GOOSE
diagnostic validation을 별도로 잠근다. 이 diagnostic은 mapping/domain failure를
감시하지만 main checkpoint 선택에는 사용하지 않는다.

## Fixed metric sets

### OverallSupported

- ValSupported13: grass, tree, pole, sky, vehicle, log, person, bush,
  concrete, barrier, puddle, mud, rubble
- TestSupported11: grass, tree, pole, sky, log, bush, concrete, barrier,
  puddle, mud, rubble
- Core11: val/test 교차 비교용 공통 집합

GT support가 있는 class는 예측이 없어도 IoU/F1 0으로 포함한다. 그 class의
Precision 원시값은 undefined로 보존하되 macro Precision에서는 zero-division을
0으로 처리해 평균 분모가 줄지 않게 한다. GT support가 없는 class는 overall
평균에서 제외하고 `AbsentClassFP`로 평가한다. raw MMSeg mIoU는 legacy 호환용으로만
기록하며 평균 분모도 함께 저장한다.

### RareRisk-4 and AugmentedRisk-2

- RareRisk-4: pole, log, barrier, rubble
- AugmentedRisk-2: pole, rubble
- TerrainHazard diagnostic: water, puddle, mud

각 집합에 대해 macro IoU, Recall, Precision, F1/Dice와 class별 값을 기록한다.

### AbsentClassFP

- FP pixel count와 non-ignore pixel 대비 비율
- prediction area share
- FP image count와 image rate
- confusion source class

## Controlled comparison

| Factor | Fixed policy |
| --- | --- |
| Ontology | Semantic20, 19 trainable IDs + ignore 255 |
| Models | SegFormer B0 and B2 |
| Loss | unweighted CrossEntropy baseline |
| Seeds | proposed 42, 43, 44 |
| Resolution/augmentation | B0/B2와 E0/E1/E2에서 동일 |
| Optimizer/schedule | 같은 experiment version 안에서 동일 |
| Selection | approved constrained validation rule |
| Test | final selected model only |

## Dataset conditions

- E0: RELLIS
- E1: RELLIS + RUGD + YCOR
- E2: E1 + GOOSE direct Semantic20 mapping

E1을 4-source 의미로 재정의하면 기존 run과 이름이 충돌하므로 E2가 권장된다.
GOOSE conditional/no-match mapping은 첫 baseline에서 사용하지 않는다.

## Success gates for E2 versus E1

- paired 3-seed `AugmentedRisk-2 mIoU` 평균이 최소 +3.0%p
- pole IoU가 3 seeds 중 최소 2개에서 개선
- `ValSupported13-mIoU` 평균 하락이 1.0%p 이내
- RareRisk-4 중 log/barrier의 치명적 비열화가 없음
- absent-class FP가 사전 합의한 허용 범위를 넘지 않음

조건을 만족하지 않으면 loss를 즉시 변경하지 않고 source sampling, crop 내 class
노출량, mapping 품질과 domain shift를 먼저 진단한다.

## Resolved decision set

1. 4-source condition: `E2`
2. Safety panels: RareRisk-4 and AugmentedRisk-2
3. Main evaluation: RELLIS-only; source evaluation: diagnostic
4. Loss: unweighted CE with `avg_non_ignore=True`
5. Repetition: paired seeds 42, 43, 44
6. Schedule: Stage 2 maximum 40k retained for the first Clean v1 series
