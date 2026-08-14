# B0-E0-seed42

## Identity

| Field | Value |
| --- | --- |
| Status | complete, audited legacy baseline |
| Experiment version | v0-legacy |
| Model | SegFormer-B0 |
| Dataset condition | E0, RELLIS-only |
| Ontology | Semantic20: IDs 0..18, ignore 255 |
| Seed | 42 |
| Deterministic | false |
| Loss | unweighted CrossEntropy, legacy ignore normalization |
| Container/image revision | `5c50bfdf2900...` |
| Main split | train 4,435 / val 900 / test 899 |

## Purpose

- Semantic20 two-stage training pipeline이 정상 동작하는지 확인한다.
- frozen-backbone Stage 1 대비 full fine-tuning Stage 2의 효과를 확인한다.
- 이후 B2 및 multi-source data condition의 기준점을 만든다.

## Runs and artifacts

| Stage | W&B run | Selected checkpoint |
| --- | --- | --- |
| Mini Stage 1 | [run](https://wandb.ai/imtaebin83-seoul-national-university/adom/runs/20260805T083456Z-5c50bfdf2900-b0-mini-b0-e0-stage1-mini) | iter 500 |
| Full Stage 1 | [run](https://wandb.ai/imtaebin83-seoul-national-university/adom/runs/20260805T122006Z-5c50bfdf2900-b0-full-b0-e0-stage1-full) | iter 1,000 by raw mIoU |
| Full Stage 2 | [run](https://wandb.ai/imtaebin83-seoul-national-university/adom/runs/20260805T122006Z-5c50bfdf2900-b0-full-b0-e0-stage2-full) | iter 6,000 by raw mIoU |
| Canonical test | [run](https://wandb.ai/imtaebin83-seoul-national-university/adom/runs/20260805T122006Z-5c50bfdf2900-b0-full-b0-e0-test) | Stage 2 iter 6,000 |

원본 `summary.json`, `test_metrics.json`, `confusion_matrix.json`은 repository에
추적되지 않는다. W&B/local output artifact의 영구 보관 상태를 별도로 보완해야 한다.

### Deployment checkpoint provenance

- Selected checkpoint: Stage 2 `best_mIoU_iter_6000.pth`
- Network Volume path reported on 2026-08-07:
  `/workspace/adom/runs/semantic20/e0/20260805T122006Z-5c50bfdf2900-b0-full/b0/stage2/best_mIoU_iter_6000.pth`
- Selection rule used by this legacy run: maximum raw validation MMSeg mIoU
- Selection evidence: validation mIoU `51.07` at iter 6,000; canonical test was
  executed with the same checkpoint
- `checkpoint_selection.json`: not produced by this pre-Clean-v1 legacy run
- Checkpoint SHA256:
  `d76229ff623eb382fd48011decf54c342d88a113bcbe650fb58cc20e42cabe73`

The immutable image revision, W&B run, exact checkpoint path, selection rule, and
SHA256 must all be copied into the ONNX export report. The later Clean-v1
`checkpoint_selection.json` contract must not be retroactively attributed to this run.

## Aggregate results

단위는 percent이다. `Fixed mIoU`는 model prediction이 아니라 GT support로 고정한
class 집합을 사용한 사후 재산출 값이다.

| Split/checkpoint | aAcc | raw MMSeg mIoU | Fixed mIoU | mAcc | Note |
| --- | ---: | ---: | ---: | ---: | --- |
| Mini Stage 1 iter 500 | 91.26 | 38.81 | - | 43.70 | smoke result |
| Val Stage 1 iter 1,000 | 91.79 | 41.28 | 50.81 | 57.82 | Stage 1 raw best |
| Val Stage 2 iter 6,000 | 93.23 | 51.07 | 58.93 | 66.79 | selected legacy checkpoint |
| Val Stage 2 iter 14,000 | - | - | 60.11 | - | logged checkpoints 중 fixed-support best |
| Val Stage 2 iter 40,000 | 93.74 | 44.62 | 58.35 | 63.82 | final checkpoint |
| Test Stage 2 iter 6,000 | 89.78 | 43.35 | 59.11 | 67.22 | raw denominator 15, GT-supported denominator 11 |

## Canonical test class results and coverage

`Acc`는 일반 accuracy가 아니라 class Recall이다. GT가 없는 class의 Recall은 N/A다.

| Class | E0 train images | Val images | Test images | Test IoU | Test Recall | Interpretation |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| dirt | 32 | 0 | 0 | N/A | N/A | 평가 불가 |
| grass | 4,433 | 900 | 899 | 83.92 | 96.76 | strong |
| tree | 4,044 | 900 | 899 | 76.09 | 94.68 | strong |
| pole | 1,528 | 211 | 255 | 0.00 | 0.00 | 완전 실패, 우선 보강 대상 |
| water | 85 | 0 | 0 | N/A | N/A | canonical 평가 불가 |
| sky | 4,367 | 900 | 899 | 95.29 | 96.75 | strong |
| vehicle | 1,379 | 382 | 0 | N/A | N/A | test 평가 불가 |
| object | 1,108 | 0 | 0 | 0.00 | N/A | GT 부재 false positive |
| asphalt | 608 | 0 | 0 | N/A | N/A | 평가 불가 |
| building | 1,329 | 0 | 0 | 0.00 | N/A | GT 부재 false positive |
| log | 94 | 79 | 147 | 40.33 | 63.97 | RareRisk, 불안정 |
| person | 546 | 192 | 0 | 0.00 | N/A | GT 부재 false positive |
| fence | 657 | 0 | 0 | 0.00 | N/A | GT 부재 false positive |
| bush | 4,319 | 900 | 899 | 69.97 | 76.32 | good |
| concrete | 2,088 | 314 | 298 | 59.17 | 61.55 | moderate |
| barrier | 2,021 | 157 | 100 | 56.68 | 57.66 | RareRisk, moderate |
| puddle | 1,045 | 369 | 221 | 70.93 | 89.58 | strong terrain hazard |
| mud | 2,334 | 683 | 844 | 44.51 | 47.23 | weak recall |
| rubble | 351 | 370 | 231 | 53.34 | 54.88 | RareRisk, 보강 여지 |

## Interpretation

### Supported conclusions

- Stage 2 full fine-tuning은 Stage 1보다 명확히 개선됐다.
- 전체 학습은 발산하지 않았고 pipeline 및 best-checkpoint 저장이 동작했다.
- dominant class는 강하지만 pole은 canonical test에서 완전히 실패했다.
- RUGD의 rubble과 GOOSE의 pole 보강 효과를 비교할 기준점으로 사용할 수 있다.

### Metric correction

raw MMSeg mIoU는 GT가 없는 class에서 prediction이 없으면 NaN, false positive가
있으면 IoU 0이 되어 평균 분모가 model-dependent였다. 이 때문에 test raw mIoU
43.35와 GT-supported11 mIoU 59.11 사이에 큰 차이가 생겼다. Stage 2의 legacy
6k 선택도 fixed-support 기준 최적점과 일치하지 않았다.

### Overfitting interpretation

train loss는 약 0.20에서 0.10으로 계속 감소했고 raw validation mIoU는 6k 이후
하락했다. 하지만 fixed-support mIoU는 6k 58.93, 14k 60.11, 40k 58.35로 훨씬
안정적이다. 따라서 raw 하락 전체를 overfitting으로 해석하면 과장이다. 다만
log처럼 개별 희소 class가 후반에 악화된 증거는 있어 class-level early stopping이
필요하다.

## Legacy limitations

- raw mIoU가 checkpoint 선택 metric이었다.
- deterministic=false이며 단일 seed 결과다.
- test가 자동 cycle 안에서 실행돼 untouched final test 원칙을 충족하지 못했다.
- Precision/F1 및 absent-class FP rate가 표준 artifact로 남지 않았다.
- 40k schedule은 B0에 과도할 가능성이 있지만 schedule 효과가 별도 통제되지 않았다.

## Decision

- 이 run은 폐기하지 않고 `Legacy Baseline v0`로 유지한다.
- 기존 confusion matrix와 checkpoint를 Clean v1 metric으로 재평가한다.
- 새 confirmatory claim은 Clean Baseline v1에서만 수행한다.
