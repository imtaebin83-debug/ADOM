# v0 Legacy baseline

최초 Semantic20 B0/B2 실험 계약을 보존한다. 학습 가능성과 two-stage fine-tuning
효과를 확인한 탐색적 baseline이지만, raw MMSeg mIoU의 model-dependent 평균
분모와 반복 test 실행 때문에 confirmatory baseline으로 사용하지 않는다.

기존 checkpoint는 폐기하지 않고 fixed-support, RareRisk, absent-class FP metric을
재산출한다. 재산출 결과는 원래 W&B metric을 덮어쓰지 않고 `legacy_rescored_v1`
namespace로 기록한다.

## Runs

- [B0 E0 seed 42](b0-e0-seed42.md)

검토 가능한 artifact가 확보되는 순서대로 B2와 E1 run 문서를 추가한다.
