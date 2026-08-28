# B2-E-ADOM capacity × domain study

> 상태: **사전 계획 / 미실행**
> 작성일: 2026-08-25
> ontology: Semantic20 IDs `0..18`, ignore `255`
> primary question: Korean field failure가 B0 용량 한계인지, source-to-field
> domain shift인지, 또는 두 요인의 상호작용인지 구분한다.
> 범위: offline training/evaluation. D-5 Go/Stop 안전 계약과 Jetson 배포 모델은
> 이 실험만으로 변경하지 않는다.

## 1. 배경과 현재 evidence

현재 seed 42의 관측값은 다음과 같다. 값은 사실로 검증된 기존 artifact만 적고,
B2-E-ADOM 값은 아직 존재하지 않는다.

| Model | Train data | RELLIS log IoU | RELLIS rubble IoU | Korean log IoU | Korean rubble IoU |
| --- | --- | ---: | ---: | ---: | ---: |
| B0-E0 | RELLIS 4,435 | 40.33 | 53.33 | 0.00 | 0.00 |
| B0-E-ADOM | RELLIS 4,435 + Korean 133 | 40.57 | 63.12 | 71.93 | 41.98 |
| B2-E0 | RELLIS 4,435 | 40.17 | 66.87 | 미평가 | 미평가 |
| B2-E-ADOM | RELLIS 4,435 + Korean 133 | 미실행 | 미실행 | 미실행 | 미실행 |

B2-E0는 RELLIS에서 B0보다 rubble IoU가 13.53 pp 높지만 log IoU는 0.16 pp
낮다. 따라서 capacity 증가는 class별로 다르게 작용하며, B0의 작은 용량만으로
Korean zero-shot failure 전체를 설명할 수는 없다. B2-E0의 Korean 성능과
B2-E-ADOM의 양쪽 domain 성능을 같은 계약으로 측정해야 한다.

기존 fresh paper evaluation은 다음 두 test manifest를 분리해 사용했다.

- RELLIS canonical test: 899 images, full Semantic20 labels
- Korean held-out: 61 images, target-only partial labels
  - log: 10 images, independent positive sequence 1개
  - rubble: 51 images, independent positive sequence 1개
  - co-occurrence/negative sequence: 0개

Korean held-out은 이미 결과와 RGB를 사람이 검토했으므로 완전히 blind한 submission
test로 간주하지 않는다. 이번 seed 42 실험에서는 고정 diagnostic test로만 사용하고,
강한 일반화 주장은 새 sequence-disjoint heldout-v2에서 재검증한다.

## 2. 주요 가설

### H1. domain-data necessity

모델 용량을 B0에서 B2로 키워도 Korean 데이터를 학습하지 않은 E0의 현장 성능은
낮게 유지되고, Korean train을 포함한 E-ADOM에서만 log/rubble가 회복될 것이다.

- primary statistic:
  `Delta_target(B2) = Korean common mIoU(B2-E-ADOM) - Korean common mIoU(B2-E0)`
- 강한 운영상 지지 기준: `Delta_target(B2) >= 20 pp`이고 log/rubble recall이 모두
  0보다 크다.
- 해석: 이 결과는 B0 용량만이 failure의 충분한 설명이라는 가설을 반박하지만,
  모든 architecture에 대한 보편적 domain-generalization 주장을 증명하지 않는다.

### H2. capacity-only contribution

B2-E0가 B0-E0보다 Korean held-out에서 유의미하게 높다면, source-only 모델의
표현력도 zero-shot transfer에 기여한다.

- statistic: `Korean common mIoU(B2-E0) - Korean common mIoU(B0-E0)`
- 기대: RELLIS에서 확인된 class별 차이 때문에 rubble가 log보다 더 개선될 가능성이
  있다.
- 반증: B2-E0에서도 log/rubble TP가 0이면 capacity-only 설명은 이 held-out에서
  지지되지 않는다.

### H3. capacity × adaptation interaction

adaptation 효과가 backbone 용량에 따라 달라지는지 다음 difference-in-differences로
측정한다.

```text
Interaction =
  [B2-E-ADOM - B2-E0] - [B0-E-ADOM - B0-E0]
```

metric별로 Korean common mIoU, log/rubble IoU와 recall에 적용한다.

- `|Interaction| < 10 pp`: single-seed 운영 기준에서 뚜렷한 interaction 없음
- `Interaction >= 10 pp`: B2와 field data 사이 positive synergy 후보
- `Interaction <= -10 pp`: B2에서 adaptation 이득이 감소하는 diminishing-return 후보

10 pp는 통계적 유의성 기준이 아니라 사전 등록한 탐색적 효과크기 기준이다.
submission 주장에는 seeds 42/43/44의 평균, 표준편차와 paired effect가 필요하다.

### H4. source-domain retention and class trade-off

B2-E-ADOM은 target gain을 얻으면서 B2-E0의 RELLIS 성능을 대체로 보존하지만,
B0에서 관측된 barrier/mud 같은 class-specific regression이 다시 나타날 수 있다.

- source guardrail: RELLIS native-supported mIoU 감소가 2 pp를 넘지 않는다.
- 반드시 보고할 class: log, rubble, barrier, mud, puddle, concrete
- worst-class delta를 평균에 숨기지 않는다.
- target gain이 크더라도 source guardrail을 넘으면 “전반적으로 우수”가 아니라
  target adaptation with source trade-offs로 해석한다.

### H5. rare-class response is class dependent

기존 B2-E0 결과를 근거로 rubble는 capacity의 이득을 더 받고, 가늘고 불규칙한
log는 field data의 이득을 더 받을 것으로 예상한다. 이는 물체 크기나 texture가
원인이라는 인과 가설이 아니라, 두 class의 관측된 반응 차이에 대한 사전 기대다.

## 3. 기대 결과와 판정표

가장 가능성이 높은 사전 기대는 **domain data가 주효하고 capacity는 rubble에
선택적으로 도움을 주는 결과**다. 실제 결과가 다르면 아래 규칙대로 해석한다.

| 관측 결과 | 지지되는 설명 | 결론 범위 | 다음 행동 |
| --- | --- | --- | --- |
| B2-E0도 Korean 0, B2-E-ADOM만 회복 | domain-data necessity | 이 checkpoint/held-out의 구체적 반례 | clean multi-seed와 heldout-v2 반복 |
| B2-E0가 Korean에서 크게 회복 | capacity contribution | B0 capacity도 failure 요인 | B2-E-ADOM의 추가 이득과 interaction 계산 |
| B2-E-ADOM target gain + RELLIS 유지 | 유용한 capacity-aware adaptation | offline rare-hazard refinement | latency/Jetson 평가는 별도 gate |
| target gain과 함께 RELLIS 큰 하락 | source-target trade-off | 일반적 성능 향상 주장 금지 | sampling/loss 변경을 별도 ablation |
| B2-E-ADOM도 Korean 실패 | data quality/recipe 병목 | capacity 확대 불충분 | label policy, exposure, clean split 우선 감사 |
| log와 rubble 반응이 반대 | class-specific interaction | 평균 하나로 요약 금지 | class별 error/qualitative 분석 |

## 4. 실험 조건

### 4.1 Primary matched-legacy run

첫 run은 architecture 외의 차이를 만들지 않는다.

| 항목 | 고정 계약 |
| --- | --- |
| Conditions | B0/B2 × E0/E-ADOM |
| Primary new condition | B2-E-ADOM seed 42 |
| Initialization | official MiT-B2 pretrained backbone; B2 model config |
| E-ADOM train | frozen `ta1_train.txt`, 4,568 rows = RELLIS 4,435 + Korean 133 |
| Validation | canonical RELLIS-only 900 images |
| Tests | RELLIS canonical 899; Korean held-out 61, 항상 분리 보고 |
| Input/augmentation | existing 512×512 crop and E-ADOM pipeline 그대로 |
| Loss | CrossEntropyLoss, `avg_non_ignore=True`, class weight 없음 |
| Schedule | Stage 1 head 4k optimizer updates + Stage 2 full 40k updates |
| Effective batch | 16; micro-batch 차이는 gradient accumulation으로만 보정 |
| Seed | 42, deterministic contract |
| Selection | Korean test를 보지 않고 canonical RELLIS validation rule로 freeze |
| TTA | off |

필수 config diff는 B0 E-ADOM의 model base를 `segformer_b2.py`로 바꾸는 것뿐이다.
dataset, split, schedule, optimizer, loss, augmentation 또는 selection rule을 동시에
바꾸지 않는다. B2 decoder/backbone 차이는 architecture condition 자체에 포함된다.

기존 E-ADOM train에는 diagnostic val export와 상충하는 12개 train RGB가 있다.
이 12개는 Korean held-out과 겹치지 않고 RELLIS-only checkpoint selection에도
들어가지 않았지만, 데이터 품질 sensitivity에는 영향을 줄 수 있다. architecture를
고립하는 primary run은 기존 B0-E-ADOM과 같은 4,568-row split을 사용한다.

### 4.2 Publication-quality clean sensitivity

primary run 뒤 stronger claim이 필요할 때만 수행한다.

- audit가 제안한 conflict-free train 4,556 rows를 별도 immutable split으로 materialize
- B0-E-ADOM과 B2-E-ADOM을 모두 같은 clean split에서 재학습
- E0 conditions는 같은 RELLIS split을 유지
- seeds 42/43/44를 네 condition 모두에서 맞춤
- 기존 61장 diagnostic 결과와 새 heldout-v2 결과를 분리
- heldout-v2는 class당 독립 positive sequence 3개 이상, co-occurrence와 negative
  sequence를 포함

clean run 결과를 legacy run과 섞어 하나의 architecture delta로 계산하지 않는다.

## 5. 실행 순서와 gate

현재 확인된 RunPod GPU는 RTX 4090 24 GiB다. 전체 학습 전에 다음 순서를 지킨다.

1. **Protocol freeze**
   - 이 문서와 config-only diff를 먼저 commit한다.
   - code Git SHA, container identity, dataset/split digest, checkpoint SHA를 기록한다.
   - Korean test inference는 recipe와 selection rule이 고정된 뒤에만 실행한다.
2. **Static contract**
   - B2-E-ADOM Stage 1/2 config import
   - resolved config에서 train/val/test split, 19 classes, ignore 255 확인
   - B0-E-ADOM과 architecture 이외 diff를 machine-readable하게 저장
3. **Memory probe**
   - `16/1 -> 8/2 -> 4/4` 순서로 micro-batch/accumulation probe
   - effective batch 16 유지; OOM이 아닌 첫 조합을 freeze
4. **Training gates**
   - 50-update smoke
   - 500-update mini-run과 validation
   - checkpoint/optimizer/scheduler resume 검증
   - Stage 1 full 4k, Stage 2 full 40k
5. **Checkpoint freeze**
   - RELLIS validation artifact만으로 선택
   - 선택 iteration, metric, checkpoint SHA-256 기록
   - Korean diagnostic val은 train과의 상충 duplicate 때문에 selection에 사용하지 않음
6. **Final direct evaluation**
   - B2-E0와 B2-E-ADOM을 동일한 RELLIS/Korean ordered manifests로 fresh inference
   - B0의 기존 2×2와 evaluation contract가 같은지 검사
   - manifest 또는 preprocessing contract가 다르면 비교 중단
7. **Analysis**
   - 2×2 table, paired deltas, interaction, confusion matrix, per-sequence 결과
   - log/rubble 대표 failure와 source regression qualitative panel
   - single-seed 및 partial-label 한계를 표와 본문에 함께 기록

권장 output root:

```text
/workspace/adom/runs/semantic20/eadom/seed42/full/b2/
/workspace/adom/paper_eval_outputs/<UTC>-b0-b2-capacity-domain/
```

dataset, checkpoint, prediction, W&B cache와 output은 Git에 넣지 않는다.

## 6. 평가 지표

### Primary target-domain metrics

- Korean common-supported mIoU (`log`, `rubble`)
- log/rubble IoU, recall, precision, F1
- GT→prediction confusion distribution
- sequence별 metric; independent support가 부족하면 CI를
  `INSUFFICIENT_SUPPORT`로 유지

Korean mask는 target-only partial annotation이므로 ignore 영역의 false positive는
측정되지 않는다. 현재 log/rubble IoU를 full-scene precision 또는 false-stop rate로
해석하지 않는다.

### Source guardrail metrics

- RELLIS native-supported mIoU
- RELLIS common-supported mIoU (`log`, `rubble`)
- log, rubble, barrier, mud, puddle, concrete IoU/recall
- worst supported-class IoU delta

### Capacity and interaction metrics

- B2-E0 − B0-E0
- B2-E-ADOM − B0-E-ADOM
- E-ADOM − E0 within B0 and B2
- difference-in-differences interaction

runtime/VRAM은 RunPod 진단값으로 기록할 수 있지만 Jetson latency나 실차 배포
가능성을 대신하지 않는다. B5는 현재 config/runtime 지원이 없으므로 이 primary
matrix에 넣지 않는다.

## 7. 중단 조건

다음 중 하나면 full run 또는 비교를 중단하고 blocker artifact를 남긴다.

- dataset/split digest가 frozen 값과 다름
- Semantic20 class order 또는 ignore index 불일치
- B0 E-ADOM 대비 architecture 외 config diff 발견
- effective batch 16을 유지할 수 없음
- non-finite loss, deterministic/resume contract 실패
- checkpoint 선택 전에 Korean test를 recipe 조정에 사용
- RELLIS와 Korean을 하나의 pooled mIoU로 합치려는 평가
- test manifest/contract가 두 모델에서 다름

## 8. B5 진입 결정

B5는 B2 결과를 본 뒤 별도 protocol과 branch에서만 검토한다.

- B2-E0가 B0-E0보다 Korean에서 개선되어 capacity contribution이 관측됨
- 또는 B2 interaction이 `|10 pp|` 이상으로 capacity 축이 연구 결론을 바꿈
- B5 pretrained/config/runtime, 24 GiB memory probe와 deployment relevance를 별도 검증
- B5를 추가하면 B0/B2와 동일 seed/data/update/evaluation 계약을 유지

B2가 domain-data necessity 결론을 바꾸지 않고 compute/deployment 비용만 증가시키면
B5는 수행하지 않는다. 2-page paper에서는 더 큰 backbone 하나보다 독립 Korean
sequence와 negative/co-occurrence annotation 확장이 우선이다.

## 9. 결과 보고 문장 틀

가설이 지지될 때 사용할 수 있는 보수적 문장:

> Increasing the SegFormer backbone from B0 to B2 did not by itself eliminate
> the observed Korean field failure, whereas adding the target-domain training
> subset recovered log and rubble recognition under the same evaluation
> contract. The result is consistent with a dominant data-domain effect, with
> class-specific capacity interactions and explicit source-domain trade-offs.

capacity-only 효과가 관측되면 다음처럼 수정한다.

> The larger B2 backbone improved zero-shot transfer for selected classes, but
> target-domain training provided an additional class-dependent gain. We
> therefore attribute the result to an interaction between model capacity and
> domain-specific supervision rather than either factor alone.

금지할 표현:

- “큰 모델이면 domain shift가 해결된다.”
- “E-ADOM은 모든 클래스와 환경에서 우수하다.”
- “국가 차이가 원인으로 입증됐다.”
- “61 frames가 독립 표본 61개다.”
- “partial-mask IoU가 실제 주행 safety success를 보장한다.”
