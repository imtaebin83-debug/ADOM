# Capacity × domain adaptation paper analysis and B5 proposal

> 상태: **B0/B2 결과 검증 완료; B5는 제안이며 실행 승인 아님**
> 작성일: 2026-08-28
> ontology: Semantic20 IDs `0..18`, ignore `255`
> primary run: legacy-matched train 4,568, RELLIS val 900/test 899,
> Korean held-out 61
> source of truth:
> [B2 seed 42 run record](b2-eadom-capacity-domain-seed42-run.md)

이 문서는 검증된 B0/B2 2×2 결과를 논문 주장, 실무 모델 크기 선택 지침과
B5 후속 가설로 연결한다. 수치가 없는 B5 항목은 모두 proposal이며 결과처럼
표현하지 않는다. B5 config, checkpoint 또는 run을 이 문서가 승인하지 않는다.

## 1. 한 문장 결론

이 고정된 seed 42/held-out에서는 **모델을 B0에서 B2로 키우는 것만으로 Korean
field failure가 해결되지 않았고, target-domain supervision을 추가한 뒤에야 큰
모델의 이점이 나타났다.** 따라서 주효과는 domain data이며, capacity는 supervision이
있을 때 그 효과의 크기를 class-dependent하게 증폭하는 조건부 요인으로 해석한다.

논문에 사용할 수 있는 보수적 영문 문장:

> Increasing model capacity from SegFormer-B0 to B2 did not materially improve
> zero-shot recognition on the fixed Korean diagnostic set. Target-domain
> supervision produced the dominant gain, while the larger backbone amplified
> that gain after adaptation and preserved the source-domain aggregate metric.

이 문장은 모든 환경, architecture 또는 deployment safety에 대한 보편 명제가 아니다.

## 2. 검증된 2×2 결과

### 2.1 Korean held-out: target-only partial masks

단위는 percent다. `common mIoU`는 log와 rubble의 평균이다.

| Capacity | Train condition | common mIoU | log IoU | log recall | rubble IoU | rubble recall |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| B0 | E0, RELLIS only | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| B0 | E-ADOM, RELLIS + Korean train | 56.9586 | 71.9334 | 100.0000 | 41.9838 | 41.9838 |
| B2 | E0, RELLIS only | 0.1156 | 0.2090 | 0.2090 | 0.0222 | 0.0222 |
| B2 | E-ADOM, RELLIS + Korean train | 95.4852 | 96.7669 | 100.0000 | 94.2035 | 94.2035 |

### 2.2 RELLIS canonical test: source retention

| Capacity | Train condition | native-supported mIoU | log/rubble common mIoU | aAcc |
| --- | --- | ---: | ---: | ---: |
| B0 | E0 | 59.1118 | 46.8318 | 89.7847 |
| B0 | E-ADOM | 58.0352 | 51.8470 | 89.4871 |
| B2 | E0 | 58.4762 | 53.5174 | 89.5352 |
| B2 | E-ADOM | 61.4450 | 56.0096 | 89.5953 |

B2-E-ADOM은 B2-E0보다 RELLIS native-supported mIoU가 `+2.9688 pp` 높아
사전등록한 source guardrail을 통과했다. 그러나 supported class 전체가 개선된 것은
아니다. bush `-1.9217 pp`, tree `-0.5589 pp`의 IoU 감소를 평균에 숨기지 않는다.

## 3. 효과 분해: capacity와 domain data를 분리해서 읽기

다음 네 효과를 구분해야 한다.

```text
Adaptation effect at size S = E-ADOM(S) - E0(S)
Capacity effect in recipe R = B2(R) - B0(R)
Interaction = Adaptation effect(B2) - Adaptation effect(B0)
```

### 3.1 Korean target 효과

| Metric | B0 adaptation | B2 adaptation | interaction | capacity effect in E0 | capacity effect after adaptation |
| --- | ---: | ---: | ---: | ---: | ---: |
| common mIoU | +56.9586 | +95.3696 | +38.4111 | +0.1156 | +38.5267 |
| log IoU | +71.9334 | +96.5579 | +24.6245 | +0.2090 | +24.8336 |
| log recall | +100.0000 | +99.7910 | -0.2090 | +0.2090 | 0.0000 |
| rubble IoU | +41.9838 | +94.1813 | +52.1976 | +0.0222 | +52.2197 |
| rubble recall | +41.9838 | +94.1813 | +52.1975 | +0.0222 | +52.2197 |

이 표가 가장 중요한 결과다.

- E0에서 B0→B2 capacity 증가는 거의 효과가 없다.
- E-ADOM에서는 B0→B2가 큰 차이를 만든다.
- 따라서 `B2 is better`가 아니라 `B2 can use target supervision better on
  this diagnostic`가 관측에 더 충실한 표현이다.
- log recall은 두 adapted model이 모두 100%라 ceiling에 걸려 interaction을
  분해할 수 없다. 이때 IoU와 precision을 함께 봐야 한다.

B2 adaptation common-mIoU gain은 `+95.3696 pp`이고 sequence-aware paired
bootstrap 95% CI는 `[94.1813, 99.7910]`이다. 단, 이는 log/rubble 두 sequence
unit을 재표집한 구간이다. 각 class는 positive unit이 하나뿐이므로 class-level CI는
`INSUFFICIENT_SUPPORT`다.

### 3.2 RELLIS source 효과

| Metric | B0 adaptation | B2 adaptation | interaction | capacity effect in E0 | capacity effect after adaptation |
| --- | ---: | ---: | ---: | ---: | ---: |
| native-supported mIoU | -1.0766 | +2.9688 | +4.0454 | -0.6356 | +3.4098 |
| log/rubble common mIoU | +5.0152 | +2.4922 | -2.5230 | +6.6856 | +4.1626 |
| aAcc | -0.2976 | +0.0602 | +0.3578 | -0.2495 | +0.1082 |

source-only에서 B2가 B0보다 native mIoU가 높지 않다는 사실도 중요하다. 모델 크기
증가는 평균 성능을 자동으로 보장하지 않는다. 반면 RELLIS log/rubble common
mIoU는 E0에서도 B2가 높아, capacity response가 metric과 class에 의존함을 보인다.

## 4. 이 결과는 얼마나 의미가 있는가

### 4.1 강하게 말할 수 있는 것

1. **고정 checkpoint/held-out에 대한 반례는 강하다.** B2-E0도 Korean log/rubble를
   실질적으로 검출하지 못했다. 작은 B0 용량만으로 zero-shot failure 전체를
   설명할 수 없다.
2. **domain data의 practical effect는 매우 크다.** B2에서 `+95.3696 pp`는 metric
   noise 수준이 아니라 운영적으로 큰 차이다.
3. **source aggregate를 희생한 결과가 아니다.** 같은 B2에서 RELLIS native mIoU가
   하락하지 않았다.
4. **architecture-only matched comparison의 내부 타당성은 높다.** B0-E-ADOM과
   B2-E-ADOM은 architecture/capacity 외 학습 계약을 같게 유지했고, B2 두 모델은
   같은 inference config, manifest와 evaluation contract로 평가했다.

### 4.2 제한적으로만 말할 수 있는 것

| Claim | 현재 신뢰도 | 이유 |
| --- | --- | --- |
| 이 고정 Korean diagnostic에서 domain data가 필요했다 | 높음 | E0 두 용량이 모두 거의 0, adapted 두 모델이 회복 |
| B2가 B0보다 target supervision을 더 잘 활용했다 | 중간 | 효과는 크지만 seed 42와 sequence 두 개뿐 |
| capacity×adaptation synergy가 일반적으로 존재한다 | 낮음~중간 | `+38.4111 pp` interaction은 크지만 multi-seed/heldout-v2 없음 |
| B2가 모든 source class를 보존한다 | 낮음 | bush/tree regression 존재, RELLIS 독립 sequence unit도 하나로 기록됨 |
| B2-E-ADOM이 실제 주행에서 안전하다 | 주장 불가 | negative/co-occurrence 및 full-scene annotation/online trial 없음 |

### 4.3 외적 타당성을 제한하는 항목

- seed 42 한 개뿐이다.
- Korean held-out은 log 10 frames의 positive sequence 1개, rubble 51 frames의
  positive sequence 1개다. 61 independent samples가 아니다.
- negative sequence와 log/rubble co-occurrence sequence가 없다.
- target-only partial mask라 ignore 영역의 false positive와 false STOP을 측정하지
  못한다.
- B0/B2 adapted 차이가 true capacity effect인지 sequence-specific fitting, target
  exposure, optimization stability 또는 세 요소의 조합인지 분리되지 않았다.
- 기존 matched-legacy train에는 diagnostic validation export와 충돌하는 RGB 12개가
  있다. primary 비교에는 공정하게 동일 split을 썼지만 publication-quality 주장은
  별도 clean-split sensitivity가 필요하다.

따라서 현재 결과는 **강한 engineering decision evidence**이지만, 아직 보편적인
scaling law 또는 deployment-safety evidence는 아니다.

## 5. 왜 이런 결과가 나왔을 가능성이 있는가

아래는 관측 사실이 아니라 결과와 일치하는 mechanism hypothesis다. 논문에서는
원인으로 단정하지 않고 후속 ablation 대상으로 표현한다.

### M1. Missing target support

RELLIS-only E0는 Korean 촬영 조건에서 log/rubble의 appearance, background와 sensor
statistics를 학습하지 못했다. B2가 더 많은 feature capacity를 가져도 target support가
없으면 올바른 decision boundary를 만들 정보가 없다. 두 E0 모델이 거의 0인 결과가
이 설명과 일치한다.

### M2. Capacity is conditional on supervision

Korean train 133장이 target feature 방향을 제공한 뒤에는 B2의 더 깊고 넓은
representation이 그 신호를 활용할 수 있었을 수 있다. E0 capacity effect가 0에
가깝고 adapted capacity effect가 `+38.5267 pp`인 패턴이 이 가설과 일치한다.
그러나 feature visualization이나 controlled representation probe가 없으므로
인과적으로 입증된 것은 아니다.

### M3. Class geometry and context

RELLIS E0에서 B2의 B0 대비 효과는 log `-0.1600 pp`, rubble `+13.5312 pp`였다.
넓은 rubble region은 multi-scale context와 texture aggregation의 도움을 받고, 가늘고
불규칙한 log는 resize/crop과 boundary error에 더 민감할 가능성이 있다. Korean
adaptation 뒤에는 두 class 모두 회복하지만 rubble의 capacity gain이 더 크다.
이 역시 class shape/size annotation을 추가한 뒤 검증해야 하는 설명이다.

### M4. RELLIS-only selection acted as a retention constraint

checkpoint를 Korean test가 아니라 RELLIS validation만으로 고른 절차가 catastrophic
source regression을 제한했을 수 있다. B2-E-ADOM의 RELLIS aggregate가 개선된 결과는
이 selection policy와 일치한다. 어떤 checkpoint selection rule이 원인이었는지는
별도 ablation 없이는 알 수 없다.

### M5. Diagnostic ceiling and sequence homogeneity

B2-E-ADOM의 Korean common mIoU `95.4852`는 실제 field distribution 전체의
ceiling을 의미하지 않는다. positive sequence 두 개와 partial masks는 같은 장면의
연속 frame에서 높은 점수를 만들 수 있고, negative false positive를 벌점으로 주지
않는다. B5에서 추가 개선이 작아도 model saturation인지 metric saturation인지 먼저
분리해야 한다.

## 6. 논문 구성안

### 6.1 추천 연구 질문과 제목

연구 질문:

> Under matched optimization and evaluation contracts, does increasing semantic
> segmentation capacity resolve field-domain failure, or does capacity become
> useful only after target-domain supervision?

제목 후보:

1. **Capacity Does Not Replace Domain Supervision: A Matched SegFormer Study for
   Rare Off-Road Hazards**
2. **When Does a Larger Segmentation Backbone Help? Separating Capacity and
   Domain Adaptation in Off-Road Perception**
3. **A 2×2 Capacity–Domain Study of Rare-Hazard Segmentation under Matched
   Training Contracts**

### 6.2 Introduction 흐름

1. off-road rare hazards는 source benchmark 성능과 실제 field transfer가 다를 수 있다.
2. 현장 failure를 보면 흔히 더 큰 backbone 또는 target fine-tuning 중 하나를 선택하지만,
   둘의 효과가 confounded되어 있다.
3. 동일 recipe/split/update/evaluation에서 B0/B2 × E0/E-ADOM 2×2를 구성한다.
4. contribution은 최고 점수 자체보다 capacity-only, adaptation, interaction을 분리한
   설계와 실무용 초기 model-sizing protocol이다.

### 6.3 Methods에 반드시 포함할 내용

- Semantic20 IDs `0..18`, ignore `255`와 mapping identity
- E0 RELLIS train 4,435; E-ADOM RELLIS 4,435 + Korean 133
- matched-legacy train 4,568, RELLIS val 900/test 899, Korean held-out 61
- Stage 1 4k + Stage 2 40k updates, effective batch 16, seed 42
- B0↔B2 architecture-only mechanical diff
- RELLIS validation-only checkpoint selection
- Korean held-out test-only lock
- direct inference, TTA off, exact checkpoint/manifest/evaluation hashes
- target common mIoU, class IoU/recall, RELLIS source guardrail와 DiD 정의

### 6.4 Main paper tables and figures

| Item | 내용 | 목적 |
| --- | --- | --- |
| Table 1 | B0/B2 × E0/E-ADOM dataset, parameters/FLOPs, checkpoint identity | 조건과 capacity 축 명확화 |
| Table 2 | 본 문서의 Korean 2×2 table | target-domain 주결과 |
| Table 3 | RELLIS native/common와 mandatory class deltas | source retention과 trade-off |
| Figure 1 | x축 B0/B2, E0/E-ADOM 두 선의 Korean common mIoU interaction plot | capacity가 supervision 뒤에만 드러나는 패턴 |
| Figure 2 | x축 RELLIS native mIoU, y축 Korean common mIoU Pareto plot | source-target trade-off 시각화 |
| Figure 3 | log/rubble IoU와 recall의 effect decomposition | class-dependent response |
| Figure 4 | 동일 frame의 B0/B2 × E0/E-ADOM mask grid | failure/recovery 유형 제시 |

실제 parameter count, MACs/FLOPs, peak VRAM과 latency는 같은 input shape/runtime에서
기계적으로 측정해 Table 1에 넣는다. 공개 문헌의 대략적 숫자를 현재 config의 값처럼
복사하지 않는다.

### 6.5 Results 문단 초안

> On the fixed Korean diagnostic set, increasing capacity alone had negligible
> effect: common mIoU changed from 0.0000 for B0-E0 to 0.1156 for B2-E0.
> Adding target-domain training data increased common mIoU to 56.9586 for B0 and
> 95.4852 for B2. The adaptation gain was therefore 56.9586 points at B0 and
> 95.3696 points at B2, yielding a preregistered difference-in-differences
> interaction of +38.4111 points. On canonical RELLIS test data, B2 adaptation
> increased native-supported mIoU from 58.4762 to 61.4450, passing the source
> retention guardrail.

### 6.6 Discussion 문단 초안

> The experiment does not support the view that a larger backbone alone resolves
> the observed domain failure. Instead, the larger model became advantageous
> after target-domain supervision was introduced, suggesting that capacity and
> domain support play different roles. The effect was class dependent and was
> larger for rubble than for log. Because the Korean diagnostic contains only one
> positive sequence per class and target-only partial masks, the interaction is
> an engineering signal rather than a population-level scaling law.

### 6.7 Limitation 문단 초안

> The Korean evaluation comprises two independent positive sequence units, one
> per target class, with no negative or co-occurrence sequence. Its 61 frames must
> not be treated as 61 independent samples, and ignored regions prevent a
> full-scene false-positive analysis. Results are from one seed and the
> matched-legacy training split. Multi-seed clean-split experiments and a new
> sequence-disjoint heldout set are required before claiming general capacity
> scaling or deployment safety.

## 7. 다른 연구자를 위한 초기 모델 크기 선택 protocol

### 7.1 핵심 권고

처음부터 가장 큰 모델 하나만 fine-tune하지 않는다. 최소한 **작은 모델과 중간 모델의
2×2 pilot**을 동일 계약으로 실행한다.

```text
small E0       small adapted
medium E0      medium adapted
```

이 네 점이 있어야 다음을 구분할 수 있다.

- domain data가 필요한가?
- capacity만으로 zero-shot transfer가 개선되는가?
- capacity가 adaptation 뒤에만 유용한가?
- 더 큰 모델의 target gain이 source regression을 동반하는가?

### 7.2 권장 순서

1. **Data audit first**
   - ontology/mapping, split leakage와 exact RGB collision 감사
   - target diagnostic을 frame 수가 아니라 independent sequence 수로 설계
   - positive, negative, co-occurrence와 partial/full-scene mask 범위를 기록
2. **Cheap sentinel run**
   - 작은 모델의 E0/adapted short-budget pilot
   - domain-data effect가 있는지 빠르게 확인
3. **Matched medium factorial**
   - 중간 모델의 E0/adapted를 같은 update/effective batch/selection으로 실행
   - architecture 외 resolved config diff를 fail-closed로 검사
4. **Choose the smallest Pareto-valid model**
   - target metric, source guardrail, VRAM/latency를 동시에 만족하는 가장 작은 모델 선택
5. **Scale only to resolve a decision**
   - medium 결과가 ceiling인지 capacity bottleneck인지 불명확할 때만 large model 실행

### 7.3 제안하는 pilot 판정 규칙

아래 5/10 pp 값은 이번 결과에서 도출한 practical heuristic이며 보편적 통계 기준이
아니다. 응용 위험과 metric noise에 맞춰 사전등록해야 한다.

| 관측 | 초기 해석 | 권장 선택 |
| --- | --- | --- |
| E0 두 크기 모두 낮고 adapted 두 크기 모두 `+20 pp` 이상 | domain supervision essential | adapted 조건 중 guardrail을 만족하는 최소 모델 |
| medium E0 − small E0 `>=10 pp` | capacity-only transfer 후보 | medium을 초기 baseline에 포함 |
| medium adapted − small adapted `<5 pp` | adapted performance plateau 후보 | 비용이 작은 모델 우선 |
| medium adapted − small adapted `>=10 pp`이고 source decline `<=2 pp` | capacity is useful after supervision | medium full run 우선 |
| target gain은 크지만 source decline `>2 pp` | source-target trade-off | 모델 확대보다 sampling/loss/selection ablation |
| class별 부호가 다름 | class-specific bottleneck | 평균 모델 크기 결론 금지; class별 error audit |

이번 결과에 적용하면 B0는 domain-shift sentinel로 유용하지만 최종 adapted candidate로는
B2가 우세하다. 이유는 B2-E0가 아니라 B2-E-ADOM이 B0-E-ADOM보다 Korean common
mIoU `+38.5267 pp` 높고 RELLIS guardrail도 통과했기 때문이다.

## 8. B5 후속 실험 제안

### 8.1 B5를 실행할 연구 질문

> Does the supervision-conditional capacity benefit observed from B0 to B2
> continue at B5, or has target performance reached a metric/data ceiling?

B2 Korean interaction `+38.4111 pp`가 사전등록한 `+10 pp` 진입 기준을 넘었으므로
B5 제안의 근거는 있다. 그러나 B5가 필요하다는 결론은 아니다. B2-E-ADOM이 이미
현재 diagnostic에서 `95.4852`이므로 B5는 최고 점수 경쟁이 아니라 **plateau와
generalization을 구분하는 decision experiment**여야 한다.

### 8.2 B5 matched conditions

| Field | Proposed frozen value |
| --- | --- |
| New conditions | B5-E0 and B5-E-ADOM |
| Baselines | frozen B0/B2 four-condition results; never overwrite |
| Initialization | official MiT-B5 pretrained checkpoint, exact SHA recorded |
| E0 train | same RELLIS 4,435 |
| E-ADOM primary train | same matched-legacy 4,568 = RELLIS 4,435 + Korean 133 |
| Validation/selection | same RELLIS-only 900 and constrained rule |
| Tests | same RELLIS 899 and Korean test-only 61, reported separately |
| Schedule | same Stage 1 4k + Stage 2 40k optimizer updates |
| Effective batch | 16; micro-batch difference only through accumulation |
| Seed A | 42 decision run |
| Seed B | if B5 changes the decision, matched 42/43/44 across B0/B2/B5 |
| TTA | off |
| Clean split | separate sensitivity only; never mix with primary |
| Hardware | A100 or RTX PRO A6000; exact GPU/driver/container recorded |

B5 model base/config/runtime support는 현재 repository에 없다. 먼저 official pretrained
identity, model resolution, parameter/MAC count와 memory contract를 추가해야 한다.
B0/B2와 architecture 외 차이가 없음을 새로운 allowlist test로 증명한다.

### 8.3 B5 hypotheses

#### B5-H1. Capacity alone remains insufficient

Expected:

```text
Korean common mIoU(B5-E0) < 20
B5-E0 - B2-E0 < 10 pp
```

이 조건이면 더 큰 backbone도 target supervision을 대체하지 못한다. 반대로
B5-E0가 20 이상이고 log/rubble recall이 함께 회복되면, capacity-only contribution이
B5 scale에서 나타났으므로 현재 결론을 수정해야 한다.

#### B5-H2. Adapted target performance is near a ceiling

Expected:

```text
Korean common mIoU(B5-E-ADOM) - B2-E-ADOM < 5 pp
```

5 pp 미만이면 현재 diagnostic에서 B2 이후 실용 이득이 plateau라고 본다. 단, metric
ceiling과 model ceiling을 구분하려면 heldout-v2가 필요하다. 5 pp 이상이면서 source
guardrail도 통과하면 B5의 추가 capacity가 의미 있는 후보가 된다.

#### B5-H3. Adaptation gain plateaus after B2

Define:

```text
A0 = B0-E-ADOM - B0-E0 = 56.9586 pp
A2 = B2-E-ADOM - B2-E0 = 95.3696 pp
A5 = B5-E-ADOM - B5-E0
```

- `|A5 - A2| < 10 pp`: adaptation-capacity interaction plateau
- `A5 - A2 >= 10 pp`: continued positive interaction candidate
- `A5 - A2 <= -10 pp`: diminishing return, optimization instability 또는 overfit 후보

이 10 pp는 기존 사전등록의 exploratory effect-size threshold를 계승하며 통계적
유의성 기준이 아니다.

#### B5-H4. Source retention remains mandatory

```text
RELLIS native mIoU(B5-E-ADOM) - B5-E0 >= -2 pp
```

log, rubble, barrier, mud, puddle, concrete와 worst supported class delta를 모두
보고한다. target score가 올라도 source decline이 2 pp를 넘으면 B5를 전반적 개선으로
선택하지 않는다.

#### B5-H5. The preferred model is Pareto-efficient, not the largest

B0/B2/B5 각각에서 실제 parameter count, MACs/FLOPs, peak training VRAM,
inference latency와 target/source metrics를 측정한다. B5가 B2보다 target `+5 pp`
미만이고 source/latency/VRAM Pareto frontier를 개선하지 못하면 B2를 권장 모델로
유지한다.

#### B5-H6. Model-size ranking must survive heldout-v2

현재 Korean set에서 B5가 B2보다 높더라도, class당 independent positive sequence
3개 이상, negative와 co-occurrence를 포함한 heldout-v2에서 ranking이 유지되지 않으면
general capacity scaling evidence로 인정하지 않는다.

### 8.4 Sequential execution and gates

1. **Protocol/config commit before GPU work**
   - B5-E0/E-ADOM config와 architecture-only tests
   - official pretrained URL/SHA, resolved parameter/MAC count
2. **Static data/selection contract**
   - B0/B2와 동일 ontology, split, mapping, update와 RELLIS-only selection
3. **Memory probe preserving effective batch 16**
   - `16/1 → 8/2 → 4/4 → 2/8`; OOM이 아닌 첫 조합 고정
4. **50-update smoke, 500-update mini, exact resume test**
5. **Seed 42 B5-E0 and B5-E-ADOM full matched run**
6. **Checkpoint freeze before Korean test unlock**
7. **Fresh direct RELLIS/Korean evaluation under one B5 contract**
8. **Decision gate**
   - B5가 H1/H2/H3 중 결론을 바꾸거나 Pareto frontier를 개선할 때만 seeds 43/44
   - 결론이 같고 gain `<5 pp`면 추가 B5 seeds보다 heldout-v2/clean-split 우선

### 8.5 B5 stop conditions

- official B5 pretrained/config identity가 불명확함
- B0/B2 대비 architecture 외 resolved-config diff 발견
- effective batch 16 또는 deterministic resume 불가
- Korean test를 checkpoint/threshold/recipe 선택에 사용하려는 시도
- B5-E0와 B5-E-ADOM의 manifest/evaluation contract 불일치
- primary matched-legacy와 clean-split 결과를 하나의 capacity curve로 혼합
- B5 단일 seed 결과만으로 일반적인 model-size law를 주장

## 9. 권장 연구 순서

논문 메시지를 가장 효율적으로 강화하는 순서는 다음과 같다.

1. Korean heldout-v2: class당 positive sequence 3개 이상 + negative + co-occurrence
2. matched seeds 43/44 for B0/B2
3. conflict-free 4,556-row B0/B2 clean sensitivity
4. 위 결과가 capacity interaction을 유지하면 B5 seed 42 decision run
5. B5가 결론/Pareto frontier를 바꿀 때만 B5 seeds 43/44

현재 데이터에서는 B5 하나를 즉시 추가하는 것보다 independent target support를
늘리는 것이 논문의 외적 타당성을 더 크게 개선한다. B5의 가장 좋은 역할은
`bigger is better`를 보여주는 것이 아니라, **domain supervision 뒤에 나타난 B2의
capacity benefit이 계속되는지 또는 B2에서 충분히 포화되는지 판별하는 것**이다.
