# B0-E0 한국 현장 domain-shift 분석과 논문용 정리

> 상태: 2026-08-25 fresh direct-inference 결과에 기반한 해석 초안
> 주 평가 root: `/workspace/adom/paper_eval_outputs/20260824T152720Z`
> 전체 자체수집 B0 보조평가: `supplemental/b0_self_collected`
> 주의: 이 문서는 새로운 프로젝트 결정을 승인하지 않는다. 수치는 아래 artifact에
> 추적되며, 인과관계가 아니라 관찰된 성능과 정성 증거를 해석한다.

## 1. `native`와 `common`의 정확한 의미

클래스 `c`의 IoU를 \(\mathrm{IoU}_c\), 데이터셋 `D`에서 GT가 한 픽셀이라도
있는 클래스 집합을 \(S_D\)라고 하면 다음과 같다.

- **Native supported mIoU**:
  \(\frac{1}{|S_D|}\sum_{c\in S_D}\mathrm{IoU}_c\). 해당 데이터셋에서 실제로
  라벨된 모든 GT-present 클래스를 평균한다. 데이터셋 자체 성능 요약에는 적합하지만,
  두 데이터셋의 지원 클래스가 다르면 값끼리 직접 비교하면 안 된다.
- **Common-supported mIoU**:
  두 비교 데이터셋에서 GT가 모두 존재하고 label mapping도 같은 교집합
  \(S_{\mathrm{RELLIS}}\cap S_{\mathrm{Korean}}\)만 평균한다. 이번 canonical
  RELLIS–Korean 비교의 교집합은 **`log`, `rubble`**이다.

Korean held-out은 `log`, `rubble`만 라벨되어 있으므로 native와 common이 같은
0.00%다. 반면 전체 conflict-free 자체수집 집합은 `log`, `person`, `rubble`을
지원한다. 이때 native 13.31%는 `person` 39.87%가 평균을 올린 값이고, 실제 희소
위험 클래스만 보는 common(`log`, `rubble`)은 0.036%다. 따라서 이 분석의 임무
질문에는 native보다 common과 클래스별 IoU/recall이 더 직접적이다.

한국 mask는 target-only partial annotation이다. `255 ignore` 영역의 예측은 FP로
계산하지 않았으며, GT가 없는 클래스는 0이 아니라 N/A로 처리했다.

## 2. 논문 주 비교표: 동일 조건 fresh 2×2 평가

**Table 1.** Canonical test 성능. 모든 값은 새로 누적한 pixel confusion matrix에서
계산했으며 단위는 %다. Common set은 `log`, `rubble`이다.

| Evaluation dataset | Model | Native mIoU | Common mIoU | log IoU | log Recall | rubble IoU | rubble Recall |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RELLIS canonical test | B0-E0 | 59.11 | 46.83 | 40.33 | 63.97 | 53.33 | 54.88 |
| RELLIS canonical test | E-ADOM | 58.04 | 51.85 | 40.57 | 63.37 | 63.12 | 65.47 |
| Korean held-out test | B0-E0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| Korean held-out test | E-ADOM | 56.96 | 56.96 | 71.93 | 100.00 | 41.98 | 41.98 |

핵심 해석은 다음과 같다.

1. B0-E0의 Korean held-out 0은 평균 방식의 산물이 아니다. `log` GT 207,627 px와
   `rubble` GT 1,584,510 px에서 **두 클래스 모두 TP가 정확히 0**이었다.
2. E-ADOM은 Korean held-out에서 두 희소 위험을 회복했다. 다만 RELLIS에서
   native mIoU는 1.08 pp 낮아졌고, 개선은 클래스별로 균일하지 않다.
3. RELLIS에서 E-ADOM의 가장 큰 이득은 `rubble` IoU +9.79 pp/recall +10.60 pp다.
   `log` IoU는 +0.24 pp로 사실상 유지되고 recall은 -0.61 pp다.
4. 같은 RELLIS에서 `barrier` IoU는 -14.71 pp, `mud` IoU는 -6.53 pp다. 따라서
   E-ADOM을 “전반적으로 우수한 모델”로 표현하기보다 클래스별 trade-off를 가진
   현장 적응 모델로 표현해야 한다.

**Table 2.** E-ADOM − B0-E0의 주요 변화(%p).

| Dataset | Common mIoU | log IoU | log Recall | rubble IoU | rubble Recall | barrier IoU | mud IoU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RELLIS | +5.02 | +0.24 | -0.61 | +9.79 | +10.60 | -14.71 | -6.53 |
| Korean held-out | +56.96 | +71.93 | +100.00 | +41.98 | +41.98 | N/A | N/A |

Korean common-mIoU paired bootstrap 95% CI는 [41.98, 100.00] pp지만, 클래스별
독립 positive sequence 수는 충분하지 않아 개별 클래스 CI는
`INSUFFICIENT_SUPPORT`다. RELLIS도 sequence-aware CI를 주장할 독립 support가
충분하지 않았다. 이 CI는 평가 표본 불확실성만 반영하며 단일 training seed의
불확실성을 반영하지 않는다.

## 3. B0-E0를 자체수집 전체에 적용한 결과

B0-E0는 한국 데이터를 학습에 사용하지 않았기 때문에 train/val/test를 모두 외부
도메인 진단에 사용할 수 있다. 반대로 E-ADOM은 Korean train에 노출되었으므로 같은
전체 합집합을 E-ADOM 일반화 성능으로 보고하면 안 된다.

원본 manifest는 215행이지만 RGB SHA-256 기준 203장이다. 이 중 12개 train/val
중복 RGB 그룹은 동일 픽셀에 `rubble`과 `log`가 충돌하여, 두 annotation을 모두
제외한 **191장 conflict-free unique union**을 전체 자체수집의 주 결과로 사용했다.

**Table 3.** B0-E0의 자체수집 데이터 성능(%). `person`이 없는 split은 N/A다.

| Cohort | Images/rows | GT-present classes | Native mIoU | Common hazard mIoU | log IoU | person IoU | rubble IoU |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| Korean train | 133 | log, person, rubble | 12.6688 | 0.0424 | 0.0607 | 37.9214 | 0.0241 |
| Korean val | 21 | log | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| Korean held-out test | 61 | log, rubble | 0.0000 | 0.0000 | 0.0000 | N/A | 0.0000 |
| All rows, duplicate-weighted sensitivity | 215 | log, person, rubble | 12.5322 | 0.0354 | 0.0487 | 37.5257 | 0.0221 |
| **Conflict-free unique union (primary)** | **191** | **log, person, rubble** | **13.3141** | **0.0365** | **0.0533** | **39.8693** | **0.0197** |

Conflict-free union의 GT support는 `log` 3,469,036 px, `person` 760,472 px,
`rubble` 14,576,929 px다. 그러므로 `log`/`rubble`의 거의 0인 IoU는 GT pixel이
너무 적어서 우연히 생긴 0으로 설명할 수 없다. 10개 sequence별로도 `log`가 있는
8개 중 6개가 정확히 0이고 최고도 0.099%였으며, `rubble`이 있는 2개 sequence는
0과 0.022%였다.

전체 conflict-free union에서 B0의 GT→prediction 분포는 다음과 같다.

| GT class | Correct | Dominant wrong predictions |
| --- | ---: | --- |
| log | 0.0533% | bush 58.79%, puddle 16.10%, mud 11.54%, grass 6.83%, tree 4.95% |
| rubble | 0.0197% | puddle 40.99%, bush 36.47%, mud 14.13%, grass 3.71%, tree 2.51% |
| person | 52.44% recall | barrier 13.09%, puddle 12.01%, sky 11.90%, bush 6.74% |

## 4. 실제 RGB·GT·예측을 본 원인 분석

그림은 모델 결과를 본 뒤 임의로 고른 것이 아니다. RELLIS `log`/`rubble`은
GT-positive 이미지의 per-image IoU 중앙값, Korean `log`/`rubble`은 held-out에서
해당 GT 면적이 가장 큰 이미지, `person`은 positive train 이미지의 IoU 중앙값,
annotation conflict는 충돌 pixel 수가 가장 큰 그룹을 사용했다. 정확한 sample ID는
`figures/domain_shift_v1/selection_manifest.json`에 저장했다.

### 4.1 log: 가는 물체와 배경 질감의 결합이 바뀜

- RELLIS 대표 장면은 밝고 개방된 초지/차량로다. GT 4,719 px에서 B0는 `log`
  56.05%, `rubble` 30.94%, `grass` 10.70%로 예측했고 frame log IoU는 53.85%다.
- Korean 대표 장면은 수관 아래의 어두운 숲길이며, `log`는 낙엽·흙 위에 놓인 가늘고
  불규칙한 붉은 선형 물체다. 가장 큰 log GT 25,104 px에서도 B0는 `puddle`
  76.46%, `mud` 18.26%, `tree` 3.53%로 예측하여 log TP가 0이었다.
- 이는 단순한 작은-object support 부족만이 아니다. 충분히 큰 GT에서도 모델이 물체
  모양보다 그림자, 지면 색과 주변 식생 texture에 반응하는 장면이 확인된다.

### 4.2 rubble: source의 개방형 노면에서 한국의 음영 자갈길로 전환

- RELLIS 대표 장면의 GT 48,394 px에서 `rubble` 예측 비율은 69.56%이고 frame IoU는
  66.28%다.
- Korean 대표 장면은 그늘, 밝은 자갈, 풀 경계와 구조물이 함께 나타난다. GT
  181,129 px의 큰 자갈 영역을 B0는 `puddle` 66.52%, `mud` 33.40%로 분류했고
  rubble TP는 0이었다.
- 두 Korean 예측 mask 모두 source 장면보다 지면이 여러 vegetation/terrain class로
  강하게 파편화된다. 시각적으로는 조명, 지면 재질, 배경 식생과 촬영 시점의 복합
  shift가 source에서 학습한 class-context 결합을 깨뜨린 것으로 해석할 수 있다.

### 4.3 person이 native 평균을 올리는 이유

Korean person 중앙값 장면에서 사람의 실루엣은 source와 비교적 공통된 형태 cue를
유지한다. GT 52,125 px 중 44.45%를 person으로 회복했지만 나머지는 `sky` 19.76%,
`bush` 13.39%, `puddle` 8.58%, `barrier` 8.30%였다. 전체 union의 person IoU
39.87%가 native 평균을 올리므로, native 13.31%를 희소 위험 인지 성능으로 읽으면
안 된다. 또한 partial mask에서 ignore 영역의 person FP는 평가되지 않으므로 이
수치는 완전 라벨 장면의 full precision을 의미하지 않는다.

### 4.4 0의 원인이 아닌 것: 클래스 수와 train/val 중복

- 4개 ID가 관찰된다는 사실 자체가 IoU 0을 만들지 않는다. 그중 `255`는 ignore이고,
  실제 semantic label은 `log`, `person`, `rubble` 세 개다. held-out은 그중
  `log`, `rubble` 두 개이며 둘 모두 충분한 GT를 갖지만 TP가 0이었다.
- 동일 RGB 중 가장 큰 충돌 예시는 39,910 non-ignore pixels를 train에서는
  `rubble`, val에서는 `log`로 라벨했다. 이런 12개 그룹은 전체-data 집계의 신뢰성을
  떨어뜨리지만 held-out과는 image/sequence hash overlap이 없고, conflict-free 191장
  결과에서도 실패가 유지된다. 따라서 held-out 0의 원인은 이 중복이 아니다.

관찰된 shift의 정성 증거는 강하지만, 식생·지형·조명·class frequency 각각의 독립적
인과 효과를 이 단일 실험으로 분리할 수는 없다. 논문에서는 “복합 domain shift에서
관찰된 failure”라고 쓰고 특정 한 요인이 원인이라고 단정하지 않는 것이 안전하다.

## 5. 논문에 바로 넣을 결과 문단

### 한국어 초안

공개 오프로드 데이터로만 학습된 B0-E0은 RELLIS canonical test에서 log와 rubble의
common-supported mIoU 46.83%를 기록했으나, 독립적인 한국 현장 held-out 61장에서는
두 클래스 모두 true positive를 하나도 생성하지 못해 IoU와 recall이 0%였다. 이
현상은 held-out split에 한정되지 않았다. train/validation annotation까지 포함하되
중복 RGB의 상충 라벨을 제거한 자체수집 191장에서도 log와 rubble IoU는 각각
0.053%와 0.020%에 그쳤으며, positive sequence 대부분에서 동일한 실패가 관찰되었다.
정성 분석에서 한국의 가는 통나무는 주로 puddle과 mud로, 음영 아래의 자갈 지면은
puddle과 mud로 오분류되었다. 이는 공개 벤치마크 성능만으로 식생·지면 재질·조명과
클래스 분포가 함께 달라진 실제 운용 환경의 희소 위험 인지를 보장할 수 없음을
보여주는 구체적 failure case다.

Korean train 데이터를 추가한 E-ADOM은 held-out에서 log IoU 71.93%와 rubble IoU
41.98%를 회복했다. 그러나 RELLIS native mIoU는 1.08 pp 감소했고 barrier와 mud
IoU도 각각 14.71 pp와 6.53 pp 낮아졌다. 따라서 결과는 전반적 성능 향상보다
**현장 희소 위험 적응과 source-domain 클래스별 trade-off**로 해석해야 한다.

### English draft

> Although B0-E0, trained only on public off-road datasets, achieved a
> common-supported mIoU of 46.83% for log and rubble on the canonical RELLIS
> test set, it produced no true-positive pixels for either class on the
> independent 61-image Korean held-out set, yielding zero IoU and recall.
> The failure was not confined to the held-out split: after removing
> contradictory duplicate annotations, log and rubble IoUs remained 0.053%
> and 0.020%, respectively, across 191 unique self-collected images. Visual
> inspection showed that thin logs and shaded gravel surfaces were instead
> assigned predominantly to puddle, mud, and vegetation classes. These
> results provide a concrete failure case showing that performance on public
> benchmarks alone does not guarantee stable recognition of rare hazards
> under a compound shift in vegetation, ground appearance, illumination, and
> class distribution.

> E-ADOM, which incorporated Korean training data, recovered 71.93% log IoU
> and 41.98% rubble IoU on the Korean held-out set. This target-domain gain was
> class-specific rather than uniform: RELLIS native mIoU decreased by 1.08
> points, with barrier and mud IoUs decreasing by 14.71 and 6.53 points. We
> therefore frame E-ADOM as rare-hazard field adaptation with explicit
> source-domain trade-offs, rather than as a generally superior segmentation
> model.

## 6. 논문용 figure caption

**Figure A — domain shift.**

> Deterministically selected source- and target-domain examples for log and
> rubble. RELLIS panels show the median per-image class IoU among positive
> images, whereas Korean panels show the held-out image with the largest
> ground-truth class area. B0-E0 recognizes both classes in the open RELLIS
> scenes but maps thin logs and shaded gravel in Korean field scenes mainly to
> puddle and mud, producing zero class IoU in the shown target examples. Black
> pixels in Korean ground truth denote ignored, unlabeled regions.

**Figure B — partial person success.**

> A median person-positive Korean training frame. Person shape is partially
> recovered, explaining why native supported mIoU is higher than the
> log/rubble-only common mIoU. Metrics are conditional on partial annotations;
> predictions in ignored regions are not counted as false positives.

**Figure C — annotation conflict audit.**

> An identical RGB frame appears in the training and validation exports with
> conflicting rubble and log labels on 39,910 non-ignore pixels. All 12 such
> duplicate groups were excluded from the primary 191-image union; the
> duplicate-weighted and train-/validation-preference variants are reported
> only as sensitivity analyses.

## 7. 권장 headline과 주장 범위

가장 정직한 주 framing은 **“system demonstration with class-specific
trade-offs”**다. 보조 framing은 **“rare-hazard refinement, with log as the
deployment scenario and rubble as the strongest RELLIS offline gain”**이다.

권장 주장:

> 특정 지역의 공개 데이터로 학습한 모델이 식생·지면 재질·조명과 클래스 분포가
> 함께 달라진 실제 운용 환경에서도 임무상 중요한 희소 위험을 안정적으로 인지한다고
> 보장할 수는 없다. B0-E0의 RELLIS–Korean 격차는 이를 보여주는 구체적 반례다.

피해야 할 주장:

- “모든 공개 데이터 학습 모델은 한국 환경에서 일반화하지 못한다.” 단일 architecture,
  checkpoint, seed로 보편 명제를 증명하지 못한다.
- “B0-E0가 한국 데이터에 적응하지 못했다.” B0-E0에는 적응 절차가 없었으므로 정확한
  표현은 **zero-shot cross-domain generalization failure**다.
- “E-ADOM이 모든 클래스에서 개선됐다.” RELLIS barrier/mud와 native mIoU regression이
  존재한다.
- “class frequency 차이가 실패의 원인으로 입증됐다.” 현재 실험은 복합 shift의 결과를
  보였지만 각 요인의 인과 기여는 분리하지 않았다.

## 8. 필수 한계 공개

1. 두 checkpoint는 단일 seed이며 training-seed uncertainty가 없다.
2. Korean mask는 target-only partial label이다. ignore 영역의 false positive와
   negative-scene false alarm은 이 평가로 측정하지 못한다.
3. Korean held-out의 클래스별 independent positive sequence가 적어 개별 CI는
   insufficient support다.
4. 전체 자체수집 집합에는 동일 RGB의 상충 annotation 12그룹이 있었으며, 주 결과는
   이를 모두 제외한 191장이다.
5. frame-level segmentation 성능만 측정했다. 연속 영상에서의 temporal stability와
   실제 Go/Stop false-stop/miss rate는 별도 system evaluation이 필요하다.
6. 정성 장면은 사전 규칙으로 선택했지만, 식생·지형·조명·빈도의 원인을 분리하는
   controlled ablation은 아니다.

## 9. 추적 가능한 근거 artifact

- 4행 표: `paper_table.{csv,md}`
- paired delta/CI: `paired_deltas.{csv,md}` 및 `metrics/*__paired_bootstrap.json`
- 직접 누적 confusion: `metrics/*__confusion_matrix.npy`, `*__per_class.csv`
- 전체 자체수집 요약: `supplemental/b0_self_collected/summary.json`,
  `cohort_metrics.csv`, `per_sequence_metrics.csv`, `prediction_distribution.csv`
- 중복 라벨 감사: `supplemental/b0_self_collected/duplicate_label_conflicts.csv`
- 정성 그림: `supplemental/b0_self_collected/figures/domain_shift_v1/`
- 선정 규칙/sample ID: 위 figure 디렉터리의 `selection_manifest.json`
