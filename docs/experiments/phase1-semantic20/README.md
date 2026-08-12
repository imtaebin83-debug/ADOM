# Phase 1 Semantic20 experiment registry

## Research objective

Semantic20을 유지하면서 외부 데이터셋의 직접 대응 label을 학습에 추가해
RELLIS target domain의 전체 성능을 보존하고 희소 위험 class를 개선한다.

## Dataset conditions

| Condition | Main train | Main validation | Main test | Role |
| --- | --- | --- | --- | --- |
| E0 | RELLIS | RELLIS | RELLIS | 원본 baseline |
| E1 | RELLIS + RUGD + YCOR | RELLIS | RELLIS | 기존 3-source 증강 |
| E2 | E1 + GOOSE Semantic20 direct mapping | RELLIS | RELLIS | GOOSE의 추가 효과 분리 |
| TA0 | RELLIS, B0-E0 warm start | RELLIS | RELLIS | 공통 개선 recipe의 method control |
| TA1 | RELLIS + ADOM ZED2i | RELLIS | RELLIS | TA0 recipe에서 신규 target-domain data 순효과 |
| TA2 | RELLIS + RUGD + YCOR + ADOM ZED2i | RELLIS | RELLIS | TA0 recipe에서 multi-source 상보 효과 |

기존 `E1`의 의미는 config와 decision record에 이미 고정돼 있으므로 GOOSE를
추가한 4-source 조건은 `E2`로 관리하는 것을 권장한다. RUGD/YCOR/GOOSE의
source validation은 main checkpoint 선택에 섞지 않고 diagnostic으로 보관한다.

## Metric panels

1. `OverallSupported`: GT support로 고정한 전체 class macro metric
2. `RareRisk-4`: pole, log, barrier, rubble의 안전 성능 감시
3. `AugmentedRisk-2`: 실제 외부 데이터 보강 효과를 검증하는 pole, rubble
4. `AbsentClassFP`: GT가 없는 class의 false-positive rate

`RareRisk-4`는 다음 사전 기준을 모두 만족한다.

- RELLIS E0 train pixel share가 1% 미만이다.
- 우세한 배경 지형이 아니라 충돌 가능한 compact/discrete hazard이다.
- canonical RELLIS val과 test 모두 GT가 있다.
- val과 test 각각 최소 50개 image에서 GT가 관측된다.

water는 위험하지만 val/test GT가 없어서 challenge-only로 분리한다. vehicle,
person, object, fence는 canonical test coverage가 없어 main RareRisk 평균에 넣지
않는다. puddle과 mud는 compact obstacle이 아닌 terrain hazard panel 후보이다.

## External-source hypotheses

| Source | Semantic20 contribution | Phase 1 interpretation |
| --- | --- | --- |
| RUGD | rubble(rock-bed), water, person 및 공통 class | rubble 개선의 주된 원인 후보 |
| YCOR | puddle, grass | terrain hazard와 domain diversity 보조 |
| GOOSE | pole 및 직접 대응 공통 class | pole 개선의 주된 원인 후보 |

따라서 4-source 학습의 핵심 주장은 `AugmentedRisk-2` 개선이다. log와 barrier는
RareRisk-4에 남겨 비열화 여부를 감시하지만, 현재 source mapping으로는 개선을
강하게 기대하지 않는다.

## Versions

- [v0-legacy](versions/v0-legacy/README.md)
- [v1-clean-baseline](versions/v1-clean-baseline/README.md)
- [Clean Baseline v1 protocol](protocols/clean-baseline-v1.md)
- [TA0/TA1/TA2 parallel RunPod runbook](target-adaptation-runbook.md)
- [TA method-recipe selection plan](ta-method-recipe-selection.md)
