# ADOM-v2 라벨 및 Korean held-out 후보 감사

> 감사일: 2026-08-31  
> 상태: **제안 및 QC 기록**. 이 문서는 기존 Semantic20 ontology, split 또는
> benchmark 결정을 변경하지 않는다.  
> 고정 계약: Semantic20 train ID `0..18`, ignore `255`

## 1. 결론

현재 `adom_output`은 **초벌 라벨 또는 weak-label 학습 후보로는 활용 가능하지만,
논문의 독립 Korean held-out 정답으로 바로 사용할 품질은 아니다.** 실제 RGB가
보이는 overlay를 검토했을 때 경계의 거친 정도뿐 아니라 `log`/`rubble`의 의미가
프레임마다 달라지는 문제, 큰 사각형/다각형 형태의 `sky`와 `bush`, 누락된
`grass`가 함께 관찰됐다. 이 상태로 평가하면 모델 성능보다 annotator 정책과
polygon 방식의 차이가 점수를 지배할 수 있다.

`0811_3`과 `0811_8`은 기존 Korean held-out의 두 test sequence가 맞다.
canonical split은 다음과 같다.

- `260811_3/20260811_111223_+0900`: rubble positive sequence
- `260811_8/20260811_174256_+0900`: log positive sequence
- 합계 61장: rubble 51장, log 10장

따라서 이 두 sequence의 재라벨은 **새 독립 test가 아니라 기존 test의 재주석 또는
legacy anchor**다. 특히 현재 로컬 `adom_output(0811_3)`은 `color`, `mask`,
`overlay`가 모두 0장이고, `adom_output(0811_8)`에만 10쌍이 있다.

P/N/C 구성 방향은 적절하다. 다만 논문의 primary heldout-v2는 모델 개발에 쓰인
기존 capture를 제외하고 **P2 + P4 + N1 + N2 + C2의 새 125장**으로 고정해야
한다. P1 + P3 + C1의 33장은 train/val/기존 test가 섞인 legacy audit set으로만
별도 보고해야 한다.

## 2. 확인한 자료와 한계

### 2.1 로컬 자료

다음 네 output 폴더와 기존 held-out visual-review montage를 읽기 전용으로
확인했다. 데이터와 이미지는 저장소에 복사하거나 commit하지 않았다.

| output | mask/RGB-overlay pair | 비고 |
|---|---:|---|
| `adom_output(0810_1)` | 39 | 39개 frame에 mask, color mask, overlay 존재 |
| `adom_output(0810_2)` | 43 | 43개 frame에 mask, color mask, overlay 존재 |
| `adom_output(0811_3)` | 0 | 세 하위 폴더가 비어 있음 |
| `adom_output(0811_8)` | 10 | frame 18--23, 60, 61, 63, 64 |
| 합계 | 92 | 사용자가 설명한 P/N/C 원본 158장과 동일한 묶음이 아님 |

12개의 서로 다른 overlay를 확대 검토하고, 기존 held-out의 log montage 1개와
rubble montage 3개를 함께 비교했다. 92개 mask 전체에 대해서는 크기, ID,
class presence, pixel count와 인접 frame IoU를 계산했다.

### 2.2 Drive 자료

공유 Drive의 158장 구성은 사용자가 제공한 아래 manifest를 기준으로 감사했다.
현재 Codex 브라우저 런타임은 Windows kernel asset 경로 오류로 열리지 않았고,
일반 읽기 전용 웹 접근에서도 Drive folder를 열 수 없었으므로 **Drive의 각 원본
preview와 실제 파일명은 독립적으로 재확인하지 못했다.** 아래 P/N/C 수량은
검증 완료 사실이 아니라 제공된 manifest다.

| cohort | 설명 | 수량 | primary-v2 적격성 |
|---|---|---:|---|
| P1 | 기존 capture의 log-only | 15 | 부적격; legacy anchor |
| P2 | 새 log-only | 30 | 조건부 적격 |
| P3 | 기존 `260811_3`, `260811_4` rubble-only | 15 | 부적격; legacy anchor |
| P4 | 새 rubble-only | 30 | 조건부 적격 |
| N1 | 새 일반 비포장도로 | 30 | 조건부 적격 |
| N2 | 새 가지/돌 hard negative | 30 | 조건부 적격 |
| C1 | 기존 `260811_4` log+rubble | 3 | 부적격; legacy anchor |
| C2 | 새 log+rubble | 5 | 조건부 적격 |

`조건부 적격`은 각 묶음이 기존 train/val/test와 RGB 중복이 없고, capture 단위로
분리되며, 아래 annotation/QC 계약을 통과한다는 뜻이다.

## 3. 기계적 mask 감사

모든 mask는 `1280 x 720`, 8-bit single-channel이고 관찰된 ID는
`0, 2, 5, 10, 13, 18, 255`다. 이는 기존 canonical Semantic20 ID 중
`dirt, tree, sky, log, bush, rubble, ignore`에 해당한다. **선언한 7개 class 중
`grass` ID 1은 92장 전체에서 0 pixel이다.** 실제로 grass가 전혀 없는 capture인지,
ground vegetation을 `dirt` 또는 `bush`로 흡수했는지 annotator 확인이 필요하다.

| output | dirt | grass | tree | sky | log | bush | rubble | ignore | valid coverage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `0810_1` | 39 | 0 | 38 | 39 | 19 | 39 | 20 | 39 | 88.2% |
| `0810_2` | 43 | 0 | 43 | 43 | 17 | 43 | 0 | 43 | 66.5% |
| `0811_8` | 10 | 0 | 10 | 10 | 4 | 10 | 5 | 10 | 72.0% |

표의 class 값은 해당 class가 1 pixel 이상 있는 frame 수다. `0810_2`의 rubble
0은 장면상 정상일 수 있지만, `grass` 0은 7-class complete annotation이라는
설명과 맞는지 반드시 확인해야 한다.

연속 frame의 mask IoU는 넓은 background class에서 매우 높고 target class의
등장/소멸이 급격했다. 예를 들어 `0811_8`의 median adjacent IoU는 dirt 0.950,
bush 0.933, sky 0.886인 반면 log는 0.662였다. `0810_1`의 log와 rubble median은
각각 0.000이었다. 이것만으로 복사 라벨을 단정할 수는 없지만, 연속 영상에서
target instance identity를 추적하는 temporal QC가 필요하다는 신호다.

## 4. 실제 이미지 기반 품질 판정

### 4.1 반복해서 관찰된 문제

1. **`sky` polygon이 수관/가지 위를 가로지름**
   - `0810_1` frame 14, 17, 33, 38과 `0811_8` frame 18, 21, 60, 64에서
     큰 사각형 또는 직선 polygon이 나뭇가지와 잎을 포함한다.
   - 이는 작은 boundary noise가 아니라 sky/tree class의 false label이다.
2. **`tree`가 trunk와 canopy에서 서로 다른 규칙으로 적용됨**
   - 큰 trunk는 비교적 잘 잡지만 가는 trunk/branch는 누락되고, 일부 canopy는
     넓게 tree로 칠하면서 다른 곳은 ignore 또는 bush로 남는다.
3. **`bush`가 개체가 아닌 큰 영역 polygon으로 칠해짐**
   - `0810_1`과 `0811_8`의 경사면 vegetation에서 실제 잎/관목 경계를 크게
     벗어나고 tree/ground와 겹치는 의미가 생긴다.
4. **`dirt`가 leaf litter, 작은 돌, 얕은 뿌리까지 포괄함**
   - traversable ground를 넓게 잡은 정책 자체는 가능하지만, 돌/가지/뿌리의
     어떤 크기부터 rubble/log인지 사전 규칙이 없어 target boundary가 흔들린다.
5. **`log`와 `rubble`의 의미 충돌**
   - `0810_1` frame 17/33에서는 돌무더기처럼 보이는 영역이 rubble로, 굵은
     노출 뿌리는 log로 보이지만 작은 파편은 frame별로 빠진다.
   - `0811_8` frame 18과 60/64에서는 동일한 종류의 지면 돌출물/뿌리가
     rubble 색과 log 색으로 갈린다. 특히 가늘고 구부러진 노출 뿌리를 rubble로
     표시한 사례는 class 의미와 맞지 않는다.
6. **polygon 경계가 성능 차이를 가릴 정도로 거침**
   - `0810_2` frame 26/33의 전경 log/뿌리와 지면 경계는 큰 덩어리로 합쳐지고,
     물체 사이의 dirt가 log에 포함되거나 반대로 log 내부가 빠진다.

### 4.2 등급

| 용도 | 판정 | 이유 |
|---|---|---|
| 논문 primary test ground truth | **사용 불가, 재QC 필요** | 의미 충돌과 systematic polygon error가 있음 |
| model/recipe 선택용 validation | 사용 금지 | 새 heldout을 tuning에 쓰면 독립성이 사라짐 |
| weak-label 추가 학습 | 조건부 가능 | train-only 분리, high-confidence region만 사용, 255 확대 필요 |
| annotator 초벌 mask | 가능 | RGB를 보며 2차 교정할 출발점으로는 유용 |
| legacy test 재주석 분석 | 가능 | 기존 결과와 별도 버전으로 보고하고 점수를 섞지 않아야 함 |

## 5. ADOM-v2 annotation 계약 제안

Semantic20 ID는 바꾸지 않는다. 아래 규칙은 ADOM-v2 제안이며 승인 후 별도
decision record와 versioned manifest로 고정해야 한다.

| class (ID) | 포함 | 제외/ignore 기준 |
|---|---|---|
| dirt (0) | 노출 토양과 통행 가능한 흙길/leaf-litter ground | 분리 가능한 돌, 목재, 돌출 뿌리 |
| grass (1) | 지면의 초본성 식생 | 관목 잎, 나무 수관, 낙엽만 있는 지면 |
| tree (2) | 살아 있고 지면에 뿌리내린 trunk와 연결된 woody structure | 떨어진 통나무/가지, bush |
| sky (5) | 실제로 보이는 하늘 pixel | 수관, 잎, 가지 사이를 polygon으로 덮지 않음 |
| log (10) | 떨어진/쓰러진 목재 또는 주행면 위로 명확히 돌출된 woody root/branch | 평평하게 묻힌 얇은 뿌리와 크기 판정 불가 물체는 255 |
| bush (13) | 낮은 woody shrub의 잎/줄기 덩어리 | tree canopy, grass, 배경 전체 polygon |
| rubble (18) | 분리 가능한 돌, 자갈 집합 또는 주행 위험이 되는 rock patch | 흙의 단순 texture, 목재/뿌리 |

`log`의 “가지/뿌리”와 N2 hard negative를 동시에 정의하려면 촬영 전에 최소
크기/돌출 조건을 고정해야 한다. 예를 들어 실제 폭/높이를 측정할 수 없다면
annotation에서 임의 pixel 길이로 대체하지 말고, annotator가 일치할 수 있는
운영 규칙과 `ambiguous -> 255` 원칙을 둔다. 같은 물체를 P에서는 log, N2에서는
negative로 부르면 benchmark 자체가 모순된다.

평가용 7-class mask는 **화면에 보이는 7개 class를 모두 exhaustively label**해야
한다. scene 이름이 `log-only`여도 log만 칠하고 나머지를 255로 두어서는 안 된다.
다만 Semantic20의 나머지 12 class와 진짜 모호 영역은 255로 유지할 수 있다.
각 frame의 valid coverage를 기록하고 지나치게 낮은 frame은 제외 사유를 남긴다.

### 5.1 최소 QC 절차

1. 원본 RGB SHA-256, capture ID, 연속 frame 범위, 장소/시간을 manifest에 고정한다.
2. 기존 RELLIS/Korean train/val/test와 exact RGB 및 near-duplicate를 검사한다.
3. annotator A가 7-class complete mask를 만들고, annotator B가 RGB+overlay로
   target class와 sky/tree/bush 경계를 전수 검수한다.
4. log/rubble positive와 N2 hard negative는 두 annotator가 독립 판정한다.
   불일치는 합의하거나 255로 보낸다.
5. 같은 capture의 인접 frame에서 instance identity와 label이 갑자기 바뀌는지
   temporal strip으로 검수한다.
6. class ID, shape, unknown ID, empty class, valid coverage와 split leakage gate를
   기계적으로 통과시킨다.
7. manifest와 mask SHA를 동결한 뒤 B0/B2/B5 checkpoint를 처음 평가한다.

## 6. Korean heldout-v2 split

### 6.1 primary와 legacy를 분리해야 하는 이유

기존 split과 제공된 P/N/C 설명을 교차하면 다음 leakage가 있다.

- P1은 기존 Korean train, validation, test capture가 섞여 있다.
- P3의 `260811_3`은 기존 test지만 `260811_4`는 기존 train이다.
- C1의 `260811_4`는 기존 train이다.
- `260811_3`과 `260811_4`는 각각 continuous capture이며 frame 단위로 다시
  쪼개거나 같은 독립 sequence처럼 세면 안 된다.

따라서 권장 구조는 다음과 같다.

| 평가층 | cohort | image 수 | 용도 |
|---|---|---:|---|
| Primary heldout-v2 | P2, P4, N1, N2, C2 | 125 | 새 일반화 평가, test-only |
| Legacy/relabel anchor | P1, P3, C1 | 33 | 기존 장면과의 연결 및 label-version 영향 |
| 기존 benchmark | `260811_3`, `260811_8` 원래 partial mask 61 | 61 | 이전 B0/B2 결과의 재현용; 점수 혼합 금지 |

125장은 좋은 시작이지만 통계 단위는 image 수가 아니라 **독립 capture sequence
수**다. 각 cohort의 30장이 한 연속 sequence라면 유효 표본은 30이 아니라 1에
가깝다. primary claim을 위해 권장하는 최소 구성은 log positive, rubble positive,
negative 각각 3개 이상의 독립 sequence이며, co-occurrence도 2--3개 독립
sequence가 바람직하다. 현재 C2 5장이 한 sequence라면 co-occurrence 결론은
descriptive로 제한한다.

## 7. 평가 지표와 논문 사용법

기존 Korean 95.4852%는 log 10장 한 sequence와 rubble 51장 한 sequence의
positive-only partial mask에서 나온 target common mIoU다. 이는 “고정된 두
positive sequence에서 B2-E-ADOM이 target을 잘 회복했다”는 근거이지, 실제 Korean
field distribution 전체의 95% 성능이나 false-positive 안전성을 뜻하지 않는다.

heldout-v2에서는 다음을 함께 보고한다.

| scenario | primary metric | 보조 metric |
|---|---|---|
| P2 log positive | log IoU, recall | boundary F1, per-sequence detection rate |
| P4 rubble positive | rubble IoU, recall | boundary F1, per-sequence detection rate |
| N1 easy negative | log/rubble FP pixel rate | FP image rate, largest FP component |
| N2 hard negative | log/rubble FP pixel rate | branch/stone subgroup FPR |
| C2 co-occurrence | log/rubble 각각 IoU/recall | log↔rubble confusion, both-detected rate |
| 전체 valid region | 7-class macro mIoU | per-class IoU, valid coverage |

모든 평균과 bootstrap은 먼저 frame을 sequence 안에서 집계하고 **sequence-macro**로
계산한다. 서로 인접한 frame 30장을 독립 표본으로 bootstrap하지 않는다. B0-E0,
B0-E-ADOM, B2-E0, B2-E-ADOM은 동일한 frozen manifest와 inference contract에서
비교하고, threshold/checkpoint/recipe는 이 test를 본 뒤 바꾸지 않는다.

### 7.1 논문에서 가능한 주장

- 기존 heldout: capacity-only가 target-domain 실패를 해결하지 못했고,
  domain supervision 뒤 B2의 이점이 드러났다는 **고정 diagnostic 결과**.
- heldout-v2: positive recall뿐 아니라 negative false positive와 co-occurrence
  confusion까지 포함한 더 어려운 일반화 결과.
- 새 set에서 성능이 95.5%보다 낮아지는 것은 실패가 아니라 기존 set의 ceiling과
  coverage 부족을 해소했다는 증거다.

다음 주장은 아직 불가능하다.

- 현재 92개 초벌 mask로 계산한 수치를 paper ground-truth 성능으로 주장
- P1/P3/C1을 primary-v2에 섞어 독립 일반화로 주장
- 158 images를 158 independent samples로 간주
- 새 test를 보고 B2/B5 recipe 또는 threshold를 고른 뒤 test-only라고 표현
- N2의 실제 가지/돌을 명시적 class 규칙 없이 negative로 간주

## 8. 권장 다음 단계

1. 팀 총괄 manifest에 각 파일의 P/N/C cohort, capture ID, sequence ID, 원본 SHA를
   추가하고 158장 수량을 재검증한다.
2. `0811_3`의 누락된 51개 output을 복구하되 새 primary가 아니라 legacy relabel로
   분류한다.
3. 우선 target-heavy 158장 전체에서 log/rubble 의미를 재판정하고, `grass=0`의
   원인을 확인한다.
4. sky 사각형, tree/bush 대형 polygon, log/rubble 충돌을 전수 교정한다.
5. P2/P4/N1/N2/C2만으로 frozen primary-v2 manifest를 만든 뒤 split/leakage 및
   label gate를 실행한다.
6. B0/B2의 frozen checkpoint 네 개를 같은 evaluator로 먼저 평가한다. 그 결과가
   capacity 결론에 불확실성을 남길 때만 사전등록한 B5 실험을 진행한다.

