# Semantic20 field uncertainty and dominant-class attraction experiment

> 상태: **제안 / 구현 완료 / 실제 raw-logit·GT 실행 전**
> 작성일: 2026-08-13
> 모델: frozen SegFormer-B0 E0, Semantic20 IDs `0..18`, ignore `255`
> 용도: 산악 field failure의 원인 후보를 선별하고 다음 수집·라벨 정책을 정하는 진단
> 금지: 이 점수나 threshold를 RC Car 제어 명령에 직접 연결하지 않는다.

## 1. 관찰과 주장 경계

실제 RC Car/ZED 산악 주행의 사용자 제공 5-frame 합성 화면에서는 RGB의 흙·낙엽
주행면이 mask 하단의 넓은 파란 영역으로 반복 출력됐다. 사용자가 적용한 RELLIS
palette가 canonical Semantic20 palette와 동일하다는 전제에서 이는 `sky` 예측 실패
후보이다. 한 장면은 하단이 초록 계열로도 출력되어 `sky`뿐 아니라 `grass` 등
dominant class 쪽으로 prediction이 끌리는 가설도 함께 검정한다.

이 화면은 정성 evidence다. screenshot/overlay는 resize, 압축, RGB/BGR 또는 alpha
blend가 개입할 수 있으므로 class ID 판정과 픽셀 수는 반드시 원본 mono8 argmax mask로
다시 확인한다. 합성 화면의 5개 frame이 한 연속 주행에서 나온 경우 모두 같은
`sequence_id`와 같은 split에 둔다.

RELLIS-3D는 class imbalance와 지형 다양성을 명시적 난점으로 보고하고, 첨부된 논문
그림에서도 sky, grass, tree, bush가 대부분의 image pixel을 차지한다. 그러나 이 사실은
"sky pixel이 많기 때문에 field sky FP가 발생했다"는 인과 증명이 아니다. 다음 경쟁
설명을 분리한다.

1. **class-prior/dominant attraction:** 학습 분포의 다수 클래스가 현장 FP propensity와
   연관된다.
2. **appearance/context shift:** 한국 산악의 밝은 흙, 낙엽, motion blur와 camera pose가
   RELLIS의 class-conditional appearance/context와 다르다.
3. **miscalibration/overconfidence:** 틀린 field pixel에도 softmax가 과도하게 뾰족하다.
4. **preprocess/runtime 오류:** RGB/BGR, normalization, padding 또는 palette 표시가
   잘못됐다. 이 가설은 동일 frozen tensor의 ONNX↔TensorRT parity로 먼저 배제한다.

## 2. 원래 variance 아이디어의 수정

19-class 확률 `p`의 평균은 항상 `1/19`이다. 따라서

```text
V_p = (1/19) * sum_c (p_c - 1/19)^2
```

는 uniform prediction에서 `0`이고 one-hot prediction에 가까울수록 커진다. 즉
**큰 softmax class-wise variance는 일반적으로 불확실성이 아니라 높은 확신**이다.
raw-logit variance도 logit scale에 의존하며 epistemic uncertainty가 아니다. 모델 간
또는 stochastic pass 간 variance와 단일 pass의 19-class 축 variance를 혼동하지 않는다.

따라서 두 variance의 음수를 비교군으로만 남기고, primary score는 class별 max-logit
분포 차이를 보정하는 Standardized Max Logits(SML)로 둔다.

```text
L(x)       = max_c z_c(x)
y_hat(x)   = argmax_c z_c(x)
SML(x)     = (L(x) - mu[y_hat]) / sigma[y_hat]
U_SML(x)   = -SML(x)                # 클수록 의심
```

`mu_c`, `sigma_c`는 field image가 아닌 frozen RELLIS reference split에서, class `c`로
예측된 valid pixel의 max logit으로 계산한다. SML 외에 entropy, `1-MSP`, top1-top2
margin uncertainty, energy를 같은 방향(클수록 의심)으로 비교한다. Softmax는 OOD에
과신할 수 있으므로 어떤 단일 score도 unknown 보장을 제공하지 않는다.

## 3. 사전 등록 가설

### H1. dominant-class attraction association

RELLIS reference GT pixel share와 held-out field test의 class별 다음 지표 간 Spearman
association을 본다.

- 절대 FP pixel 수: 화면 면적과 prediction 빈도의 영향을 크게 받는 기술통계
- FPR: `FP_c / GT가 c가 아닌 pixel`
- FDR: `FP_c / c로 예측한 pixel`; `1 - precision_c`

FP 절대 개수만으로 가설을 채택하지 않는다. 이 분석은 19개 class의 관찰적 연관이며
인과 추론이 아니다. appearance shift를 통제한 재가중/재학습 ablation 없이는
"픽셀 비중 때문에 발생"이라고 결론내리지 않는다.

### H2. dirt-to-sky failure detection

Primary contrast는 다음 두 pixel stratum이다.

```text
positive: GT=dirt(0), prediction=sky(5)
control:  GT=sky(5),  prediction=sky(5)
```

`U_SML`의 positive median이 control보다 크고, validation/test 양쪽에서 AUROC와 AP가
chance보다 높으며 validation threshold가 test에서도 유지되는지 확인한다. 함께
`all_error` 대 correct pixel task도 평가한다.

중요한 반증 가능성은 **과신한 dirt→sky**다. 이 경우 FP의 max probability와 SML이
실제 sky만큼 높아 H2가 실패할 수 있다. 그것도 중요한 결과이며, 단일-pass softmax
점수가 field pain point detector로 불충분하다는 뜻이다.

### H3. sequence-held-out generalization

threshold는 validation sequence에서만 F1로 선택하고 고정한다. test sequence에는 한 번
적용해 precision, recall, F1, FPR을 보고한다. test 결과를 보고 threshold나 score를
다시 고르지 않는다. 최종 주장을 하려면 장소·조명·주행 방향이 다른 독립 test
sequence가 필요하다.

## 4. 데이터 수집 및 라벨링

권장 최소 단위는 독립 sequence이며 인접 frame 수가 아니다.

| Split | 권장 sequence | 역할 |
| --- | ---: | --- |
| `reference` | RELLIS train/val의 고정 표본 3개 이상 | class별 SML `mu/sigma`, class prior |
| `validation` | 산악 3개 이상 | score 비교와 threshold 선택 |
| `test` | 다른 장소/방향/시간 3개 이상 | 잠긴 threshold 최종 평가 |

각 산악 sequence에서 실패가 보이는 frame만 고르지 말고 고정 간격으로 frame을 먼저
추출한다. 이후 다음 영역을 Semantic20 ID로 라벨한다.

- 최소안: `dirt`, `sky`와 평가 불가 영역 `255`
- 권장: `dirt`, `sky`, `grass`, `tree`, `bush`, `mud`, `rubble` 등 보이는 전체 class
- 경계 ambiguity는 억지로 class를 정하지 않고 `255`

logit만으로 "그 객체가 실제 무엇인지" 알아낼 수 없다. logit score는 검토할 영역을
순위화하며, 실제 class는 사람이 RGB를 보고 Semantic20로 라벨하거나 ontology 밖이면
별도 `unknown candidate` metadata로 기록한다. Semantic23 확장 class는 Semantic20 mask
ID에 임의로 넣지 않는다.

## 5. Jetson raw-logit 수집

기존 hand-off preprocess와 동일하게 만든 frozen input tensor를 target Jetson에서
재추론한다. raw logits는 frame당 약 18.7 MiB이므로 모든 영상 frame이 아니라
sequence별 고정 추출 frame에만 저장한다.

```bash
adom-semantic20-logit-dump \
  --engine <B0_E0.engine> \
  --input-dir <FROZEN_INPUT_NPY_DIR> \
  --output-dir <GENERATED_LOGIT_DIR>
```

출력은 `*_logits.npy` (`1x19x384x640`)와 `*_mask.png` (mono8 ID)다. engine, input tensor,
logits, mask의 SHA-256과 실제 preprocessing metadata를 run record에 남긴다. raw image만
있고 input tensor가 없다면 기존 MMDeploy task processor로 먼저 tensor를 만들거나,
실제 Jetson perception executable에 bounded diagnostic dump를 추가해야 한다. 화면을
캡처해 RGB 값을 역추정하는 방법은 사용하지 않는다.

## 6. 분석 manifest와 실행

CSV path는 manifest 기준 상대경로를 권장한다. `image_path`는 overlay 생성용 선택
field다.

```csv
sample_id,sequence_id,split,logits_path,label_path,image_path
rellis_001,rellis_seq00,reference,logits/rellis_001_logits.npy,labels/rellis_001.png,images/rellis_001.jpg
mount_val_001,mount_val_seq01,validation,logits/mount_val_001_logits.npy,labels/mount_val_001.png,images/mount_val_001.png
mount_test_001,mount_test_seq01,test,logits/mount_test_001_logits.npy,labels/mount_test_001.png,images/mount_test_001.png
```

```bash
adom-semantic20-uncertainty \
  --manifest <manifest.csv> \
  --output-dir <GENERATED_ANALYSIS_DIR> \
  --true-class 0 \
  --predicted-class 5
```

산출물:

- `uncertainty-report.json`: reference SML 통계, confusion matrix, validation/test AUROC·AP,
  validation threshold와 locked-test 결과, class prior/FP 진단
- `frame-strata-summary.csv`: pixel 상관을 숨기지 않기 위한 frame별 positive/control 수와
  median
- `visualizations/`: test SML heatmap, review mask, 선택적 RGB overlay

분석기는 각 frame/stratum에서 최대 5,000 pixel을 결정론적으로 표본화한다. pixel-level
AUROC/AP는 탐색 지표이며 독립 표본 검정으로 해석하지 않는다. 최종 논문 수치는
sequence별 effect와 bootstrap confidence interval을 추가하고, 최소 3개보다 많은
독립 sequence로 반복한다.

## 7. 결과에 따른 action

| 결과 | 해석 | 다음 action |
| --- | --- | --- |
| dirt→sky에서 SML/entropy 높음, test 재현 | 기존 출력으로 검토 영역 ranking 가능 | 해당 sequence/appearance를 우선 수집·라벨링 |
| dirt→sky가 낮은 uncertainty로 과신 | softmax-only detector 실패 | ensemble/TTA/feature-distance 또는 학습 기반 OOD를 별도 ablation |
| sky뿐 아니라 grass FP도 reference share와 연관 | dominant attraction 가설 지지(비인과) | source-aware sampling/logit adjustment를 독립 실험 |
| prior와 FP propensity 연관 없음 | 단순 class-count 설명 기각 | camera/preprocess/context/texture shift를 우선 분석 |
| 특정 unlabeled region 반복 상위 | ontology coverage pain point 후보 | RGB 검토 후 standalone 추가 수집 또는 Semantic23 proposal |

현재 standalone 수집이 주로 `log`인 것은 2주 sprint에서 안전 시연 target을 잠정
선정한 결과다. 이번 진단은 이를 소급해 정당화하는 도구가 아니다. 대신 다음 수집
cycle에서 어떤 class/appearance가 실제 field weakness인지 정량적으로 우선순위화하고,
그 결과를 standalone 데이터 또는 별도의 Semantic23 decision에 반영하는 도구다.

## 8. 근거 문헌

- RELLIS-3D는 off-road class imbalance와 environmental topography를 핵심 난점으로
  보고한다: <https://arxiv.org/abs/2011.12954>
- SML은 semantic segmentation의 predicted-class별 max-logit 분포를 표준화한다:
  <https://openaccess.thecvf.com/content/ICCV2021/html/Jung_Standardized_Max_Logits_A_Simple_yet_Effective_Approach_for_Identifying_ICCV_2021_paper.html>
- Maximum Softmax Probability는 error/OOD detection의 고전 baseline이지만 완전한
  보장은 아니다: <https://openreview.net/forum?id=Hkg4TI9xl>
- Energy score는 softmax overconfidence 문제를 완화하기 위한 비교 기준이다:
  <https://proceedings.neurips.cc/paper/2020/hash/f5496252609c43eb8a3d147ab9b9c006-Abstract.html>
- modern neural network confidence는 calibration이 필요할 수 있다:
  <https://proceedings.mlr.press/v70/guo17a.html>
- Fishyscapes는 semantic segmentation blind-spot/uncertainty 평가 benchmark다:
  <https://arxiv.org/abs/1904.03215>
- DARPA RACER는 비정형 off-road 환경에서 반복적인 field develop-test cycle과 데이터
  축적의 필요성을 설명한다:
  <https://www.darpa.mil/research/programs/robotic-autonomy-in-complex-environments-with-resiliency>

RACER는 반복 field cycle의 **운용 동기**를 지지하지만 SML이나 logit variance의
수학적 타당성을 직접 지지하지 않는다. 그 방법론 근거는 SML/OOD/calibration 문헌과
본 실험의 held-out 검증에서 확보한다.
