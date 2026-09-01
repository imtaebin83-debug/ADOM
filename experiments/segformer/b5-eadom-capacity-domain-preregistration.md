# B5-E0 / B5-E-ADOM capacity × domain preregistration

> 상태: **사전등록 / 설계·config만 준비됨 / B5 미실행**
> 작성일: 2026-08-27
> 출발점: `codex/b2-eadom-capacity-domain` commit `7cafc31`
> ontology: Semantic20 IDs `0..18`, ignore `255`
> primary split: matched-legacy 4,568
> 범위: offline training/evaluation only; D-5 Go/Stop 또는 Jetson 배포 계약 변경 없음

이 문서는 기존
[B2 capacity × domain study](b2-eadom-capacity-domain-study.md)를 B5까지 확장한다.
B5는 더 큰 모델을 단순 추가하는 실험이 아니다. B2 결과가 capacity 축에 관해
결론을 바꿀 만한 불확실성을 남기고, 고정된 GO artifact가 그 이유를 증명할 때만
실행한다. 현재 GO artifact와 B2-E-ADOM 최종 결과가 없으므로 모든 B5 GPU gate는
닫혀 있다.

## 1. 비교 질문과 H1–H5 연계

두 신규 condition은 `B5-E0`와 `B5-E-ADOM`이다. 아래 기호에서 `Y(m,d)`는 model
capacity `m ∈ {B0,B2,B5}`, data condition `d ∈ {E0,E-ADOM}`의 같은 metric이다.
Korean과 RELLIS metric은 서로 합쳐 pooled mIoU를 만들지 않는다.

### H1 — domain-data necessity at B5

```text
A5 = Y(B5,E-ADOM) - Y(B5,E0)
```

Korean common-supported mIoU에서 `A5 >= 20 pp`이고 log/rubble recall이 모두 0보다
크면 B5에서도 domain-data necessity를 강하게 지지한다. B5-E0만으로 회복되면
capacity-only 설명의 지지가 커지지만, 한 Korean positive sequence/class 결과를
보편적 domain generalization으로 확대하지 않는다.

### H2 — B0→B2→B5 capacity trend

E0와 E-ADOM을 분리해 아래 두 increment를 모두 보고한다.

```text
C02(d) = Y(B2,d) - Y(B0,d)
C25(d) = Y(B5,d) - Y(B2,d)
```

`C02`와 `C25`가 같은 방향이라고 선형 scaling을 주장하지 않는다. B0→B2와 B2→B5는
parameter 간격이 같지 않으며 single seed다. 같은 방향의 두 increment는 monotonic
trend 후보, 반대 방향은 saturation/architecture-specific response 후보로만 해석한다.

### H3 — capacity × adaptation interaction

capacity별 adaptation effect와 두 difference-in-differences를 계산한다.

```text
A0 = Y(B0,E-ADOM) - Y(B0,E0)
A2 = Y(B2,E-ADOM) - Y(B2,E0)
A5 = Y(B5,E-ADOM) - Y(B5,E0)

DID02 = A2 - A0
DID25 = A5 - A2
```

Korean common mIoU와 log/rubble IoU·recall 각각에 적용한다. `|DID| < 10 pp`는
single-seed 운영 기준에서 뚜렷한 interaction 없음, `>= 10 pp`는 positive synergy,
`<= -10 pp`는 diminishing return 후보로 해석한다. 10 pp는 통계적 유의성 기준이
아니다. `DID25 - DID02`도 보고하되 curvature의 인과 증거로 부르지 않는다.

### H4 — source retention

B5-E-ADOM 대비 B5-E0의 RELLIS native-supported mIoU 감소가 2 pp를 넘지 않는지
검사한다. log, rubble, barrier, mud, puddle, concrete와 worst supported-class delta를
반드시 보고한다. guardrail을 넘으면 target gain과 함께 source trade-off로 기술한다.

### H5 — class-dependent response

log와 rubble의 capacity/adaptation delta를 따로 보고한다. rubble가 capacity에,
log가 field data에 더 반응할 것이라는 기존 기대를 유지하지만, 반대 결과도 그대로
보고한다. object size나 texture를 원인으로 단정하지 않는다.

## 2. 고정 condition과 provenance

### 2.1 기존 condition — 변경 금지

| Condition | Frozen provenance | Selection/evaluation note |
| --- | --- | --- |
| B0-E0 | image Git `5c50bfdf...`; selected iter 6,000; checkpoint `d76229ff623eb382fd48011decf54c342d88a113bcbe650fb58cc20e42cabe73` | legacy raw RELLIS-val mIoU selection |
| B0-E-ADOM | image Git `9d4f08e4...`; selected iter 26,000; checkpoint `f4cc41fd91e9df8e7aa3f726498e80636b736dfadf0e1baf338fe7c82a83399c` | constrained RELLIS-val selection |
| B2-E0 | image Git `5c50bfdf...`; selected iter 14,000; checkpoint `c47288019185e18fffdb856d2f47f56936adb06db7579416271ab468b3849f4f` | legacy raw RELLIS-val selection; Korean fresh evaluation pending in B2 study |
| B2-E-ADOM | B2 study branch/config; final checkpoint 미존재 | 진행 중인 다른 작업의 B2 run 결과를 그대로 인용하며 이 작업에서 건드리지 않음 |

B0/B2 E0의 historical selected checkpoint와 E-ADOM의 constrained selection은
selection provenance가 완전히 같지 않다. 따라서 historical 3-capacity table은
**provenance-aware exploratory trend**다. publication-quality causal capacity trend에는
같은 selection rule로 B0/B2/B5를 fresh rerun/reselect한 별도 multi-seed matrix가 필요하다.
기존 checkpoint를 바꾸거나 재명명해 이 차이를 숨기지 않는다.

### 2.2 신규 B5 condition — architecture/capacity만 변경

| 항목 | B5-E0 | B5-E-ADOM |
| --- | --- | --- |
| Reference config | corresponding B2-E0 Stage 1/2 | corresponding B2-E-ADOM Stage 1/2 |
| Model | MiT-B5 + same 19-class SegFormer head | 동일 |
| Initialization | official MiT-B5 ImageNet checkpoint | official MiT-B5 ImageNet checkpoint |
| Train split | RELLIS 4,435 | RELLIS 4,435 + Korean train 133 = 4,568 |
| Validation | canonical RELLIS 900 only | canonical RELLIS 900 only |
| Canonical test | RELLIS 899, final checkpoint freeze 후 1회 | 동일 |
| Korean held-out | 61, test-only | 61, test-only |
| Stage 1 | head-only 4,000 optimizer updates | 동일 |
| Stage 2 | Stage 1 selected checkpoint에서 full 40,000 updates | 동일 |
| Effective batch | 16 | 16 |
| Seed | 42 | 42 |
| TTA | off | off |

공식 MMSegmentation v1.2.2 MiT-B5 config의 depth `[3,6,40,3]`와 ImageNet checkpoint
`mit_b5_20220624-658746d9.pth`를 사용한다. B2→B5 allowlist는 checkpoint 변수,
backbone `init_cfg.checkpoint`, `num_layers` 세 경로뿐이다. B2/B5가 이미 공유하는
embed dims, decoder input/channels는 바꾸지 않는다.

B0-E0, B2-E0, B0-E-ADOM 또는 B2-E-ADOM experiment checkpoint를 B5 초기값으로
사용하지 않는다. Stage 2의 유일한 handoff는 같은 B5 condition의 Stage 1
RELLIS-val-selected checkpoint다.

공식 config 근거:

- <https://github.com/open-mmlab/mmsegmentation/blob/v1.2.2/configs/segformer/segformer_mit-b5_8xb2-160k_ade20k-512x512.py>
- <https://github.com/open-mmlab/mmsegmentation/blob/v1.2.2/configs/segformer/segformer_mit-b2_8xb2-160k_ade20k-512x512.py>

## 3. 데이터·mapping·evaluation lock

Primary matched-legacy E-ADOM identity는 B2 run record의 값을 그대로 사용한다.
2026-09-01 raw `/workspace` artifact 재감사에서 기존 65-hex identifier마다 중복된
문자 하나가 확인됐다. 아래 64-hex 값은 문자열을 임의로 자른 결과가 아니라 B2 full
run의 `dataset_contract.json` 필드와 fresh-evaluation manifest summary를 다시 읽어
확인한 값이다. 이 provenance amendment는 decision record 0041을 따른다.

| Field | Re-audited frozen SHA-256 |
| --- | --- |
| train / val / test | 4,568 / 900 / 899 |
| train composition | RELLIS 4,435 + Korean train 133 |
| verified manifest pairs | 14,636 |
| split contract | `fab9c136c81081464d9db099656680dac3bf2921a4ae2bbd7605c383b309ab93` |
| manifest | `183dda705e76b451dc383a81f517d36df3d6032f00002ab225421b9ae316b9dd` |
| image content | `ce06265e6146bcd37692938786386cbd9b844e9742f831284ee5d26aedd15305` |
| mask content | `5ae15ab1eff69921168b15811683edab41472456a439b58aa6384c6d472c377e` |
| combined content | `a70c6b9467b692a4797976659c6dcd501c80938626226000a6c214efcdec5e42` |
| canonical mapping SHA-256 | `ecfa61662ddbf16c801bcac22db11b0e7ee2408d635e3018a21dd389933a6bc55` |

Evaluation lock은 model-resolved contract와 ordered data manifest를 구분한다. B0와 B2
fresh evaluation은 같은 evaluator policy를 사용하지만 resolved model architecture가
contract payload에 포함되므로 contract SHA가 다르다. B5 GO artifact는 B2 결과를
근거로 하므로 B0 contract를 대입하지 않고 B2 evidence contract를 고정한다.

| Artifact | Role | Re-audited SHA-256 |
| --- | --- | --- |
| B0 resolved evaluation contract | B0 reference metrics only | `096467321246732da9d2f4a31ad8f75626b1aba0500e0680ba4ddd778241635e` |
| B2 resolved evaluation contract | B5 GO evidence lock | `4adfcb3ae550274ed3436c695c872e030c804bb8c16c09025958797312d8d592` |
| ordered RELLIS test 899 manifest | shared B0/B2 evidence | `2e078a3ac89d870b4dfb5838f8cc2772e788ecdd7cb011c309d59b4ca6a66918` |
| ordered Korean held-out 61 manifest | shared B0/B2 evidence | `1eb86ff65620fb5c0afc1d58c572c517cacc937468ebd865375aaa26d81eb782` |

Korean held-out은 recipe, loss/sampler, memory plan, threshold, early stopping,
checkpoint selection, GO artifact의 metric 재계산에 사용하지 않는다. RELLIS-val로
checkpoint와 모든 학습 결정을 먼저 freeze하고 SHA-256을 기록한 뒤 direct test만 한다.

알려진 12개 train-RGB/diagnostic-val conflict를 제거한 4,556 clean split은 primary가
아니다. 수행한다면 `matched-legacy-4568` 결과와 별도 표·output root·digest를 사용하고,
B0/B2/B5 E-ADOM을 모두 4,556으로 재학습한다. 두 split의 값을 한 capacity delta나
DiD에 섞지 않는다.

## 4. 기계적 사전 실행 계약

`python -m adom.runtime.b5_capacity_domain_contract`는 optimizer update 없이 다음을
검사한다.

- B2→B5 architecture-only allowlist와 non-architecture fingerprint
- B5-E0와 B5-E-ADOM Stage 1/2 네 config의 split, manifest, loss, crop, seed, update 수
- official B5 initialization과 top-level `load_from=None`
- Stage 1 `FreezeBackboneHook`, Stage 2 `BackboneAuditHook`
- constrained RELLIS-val selection과 `CanonicalTestLockHook`
- Semantic20 19 classes, IDs `0..18`, ignore `255`
- frozen E-ADOM split/manifest/mapping/content digest
- effective batch 16과 profile별 proposed fallback order

향후 static gate 예시는 다음과 같다. 이 명령은 학습하지 않지만 실제 dataset을 전부
decode/hash하므로 immutable training image에서 실행한다.

```bash
python -m adom.runtime.b5_capacity_domain_contract \
  --dataset /workspace/adom/datasets/processed/adom_semantic20_target_adaptation_v1 \
  --gpu-profile a100-80gb \
  --micro-batch 16 \
  --accumulative-counts 1 \
  --output /workspace/adom/runs/semantic20/b5-capacity-domain/protocol/static.json
```

Stage 2 진입 직전 runtime은 `stage1/checkpoint_selection.json`의 selected path와 실제
`load_from` path가 같고 Stage 1 output 내부에 있는지 확인하고 checkpoint SHA-256을
`stage2_handoff.json`에 기록한다.

## 5. GPU profile과 memory probe

사용자가 말한 “PRO A6000 계열”은 제품명이 모호하다. runtime의 실제
`torch.cuda.get_device_name(0)`과 physical memory를 보고 아래 하나로만 선택한다.
`RTX A6000`, `RTX 6000 Ada Generation`, `RTX PRO 6000 Blackwell`은 서로 다른 제품이며
이 문서는 RTX 6000 Ada profile을 암묵적으로 대체 사용하지 않는다.

| Profile ID | Exact family | Official nominal VRAM | Proposed probe order | Status before probe |
| --- | --- | ---: | --- | --- |
| `a100-40gb` | A100 name에 40GB 명시 | 40 GB | `8/2 → 4/4 → 2/8 → 1/16` | proposal |
| `a100-80gb` | A100 name에 80GB 명시 | 80 GB | `16/1 → 8/2 → 4/4 → 2/8 → 1/16` | proposal |
| `rtx-a6000-48gb` | NVIDIA RTX A6000 | 48 GB | `8/2 → 4/4 → 2/8 → 1/16` | proposal |
| `rtx-pro-6000-blackwell-96gb` | RTX PRO 6000 Blackwell, edition 기록 | 96 GB | `16/1 → 8/2 → 4/4 → 2/8 → 1/16` | proposal |
| `rtx-pro-4500-blackwell-32gb` | RTX PRO 4500 Blackwell, full device | 32 GB | `8/2 → 4/4 → 2/8 → 1/16` | proposal; PTX-JIT preflight required |
| `rtx-5090-32gb` | NVIDIA GeForce RTX 5090 | 32 GB | `8/2 → 4/4 → 2/8 → 1/16` | proposal; PTX-JIT preflight required |

표의 `micro-batch/accumulation`은 실제 수용량 주장이 아니다. 각 GPU의 첫 실행에서
2 runner-iteration memory probe를 큰 micro-batch부터 수행하고, 첫 non-OOM 조합을
freeze한다. OOM만 다음 fallback을 허용한다. non-OOM error, effective batch 16 붕괴,
MIG/vGPU slice, profile name/VRAM mismatch는 즉시 중단한다. accumulation을 늘리면
max/warmup/validation/checkpoint runner iteration도 함께 배율 조정해 optimizer update
수가 유지돼야 한다.

RTX PRO 4500 Blackwell Server Edition은 16GB MIG 두 개로 분할할 수 있으므로
`rtx-pro-4500-blackwell-32gb`는 runtime에서 약 32GB 전체 장치와 compute capability
12.0을 함께 확인한다. RTX 5090도 별도 name pattern과 32GB range를 사용해 PRO 4500
profile과 서로 대체하지 않는다. 현재 NGC 23.10/PyTorch 2.1 image가 native `sm_120`
binary를 포함하지 않으면 doctor는 PTX-JIT provisional warning을 기록하고, 실제 B5
forward/backward 및 2-iteration memory probe를 통과하기 전에는 학습을 허용하지 않는다.

runtime doctor는 marketing GB와 driver GiB 표기의 차이를 허용하는 좁은 VRAM range와
정확한 name pattern을 함께 검사한다. 예를 들어 A100 40GB를 A100 80GB profile로,
RTX 6000 Ada를 RTX A6000 profile로 통과시키지 않는다.

공식 hardware 근거:

- A100 40/80GB: <https://www.nvidia.com/en-us/data-center/a100/>
- RTX A6000 48GB: <https://www.nvidia.com/en-gb/products/workstations/rtx-a6000/>
- RTX PRO 6000 Blackwell 96GB: <https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-6000-family/>
- RTX PRO 4500 Blackwell 32GB: <https://www.nvidia.com/en-us/data-center/rtx-pro-4500-blackwell-server-edition/>
- GeForce RTX 5090 32GB: <https://marketplace.nvidia.com/en-us/consumer/graphics-cards/geforce-rtx-5090-founders-edition/>

### 시간·비용 산정 — probe 전 수치 금지

각 profile에서 smoke와 500-update mini의 로그로 stage별 optimizer updates/s,
validation wall time, checkpoint overhead를 측정한다. 예상치는 다음 식으로 계산한다.

```text
train_hours(stage) = target_optimizer_updates / measured_updates_per_second / 3600
validation_hours(stage) = number_of_validations × measured_validation_seconds / 3600
checkpoint_hours(stage) = number_of_checkpoints × measured_checkpoint_seconds / 3600
total_hours = stage1 + stage2 + fixed final evaluations
estimated_cost = total_hours × provider_quote_per_hour
```

quote에는 provider, region/secure-cloud 여부, 정확한 GPU profile, 조회 UTC, storage/egress
포함 여부를 기록한다. probe 전 시간·비용은 모두 `proposal/unverified`이며 결과표에
관측값처럼 쓰지 않는다.

## 6. B5 GO / NO-GO

B5는 B2-E0와 B2-E-ADOM의 RELLIS/Korean fresh evaluation이 같은 frozen manifest에서
끝나고 checkpoint SHA가 기록된 뒤에만 검토한다. 다음 중 하나가 실제 B2 metric으로
충족될 때만 GO artifact를 만들 수 있다.

1. `|Korean common mIoU(B2-E0) - Korean common mIoU(B0-E0)| >= 10 pp`
2. `|DID02| >= 10 pp`
3. log/rubble capacity effect의 부호가 반대이고 두 effect의 차이가 `>= 10 pp`

이 trigger들은 B2가 capacity 축에 결론 변경 가능성을 남겼다는 운영 기준이다.
GO artifact는 B2-E0/B2-E-ADOM checkpoint SHA, frozen evaluation/manifest SHA,
`matched-legacy-4568`, `korean_heldout_used_for_selection=false`를 포함해야 한다.
runtime은 `adom-b5-capacity-domain-go-v1` artifact를 검증하지 못하면 probe조차 시작하지
않는다. [`b5-go-decision.template.json`](b5-go-decision.template.json)은 의도적으로
`NO_GO`와 invalid placeholder로 저장돼 있으며, raw artifact 재감사와 B2 해석 뒤
별도 run artifact로 복사·작성한다. template 자체를 `GO` evidence로 쓰지 않는다.

2026-09-01 fresh artifact 재감사 결과는 다음과 같아 두 번째 trigger를 충족한다.
단위는 모두 percentage point다.

| Metric | Re-audited value |
| --- | ---: |
| capacity-only common mIoU, B2-E0 minus B0-E0 | 0.11560489282459831 |
| A0 | 56.95856191044892 |
| A2 | 95.36961496716020 |
| DID02 | 38.41105305671128 |
| log capacity effect within E-ADOM | 24.833576042586145 |
| rubble capacity effect within E-ADOM | 52.219739856485596 |

따라서 GO artifact의 trigger는
`abs_b2_difference_in_differences_ge_10pp`로 고정한다. checkpoint SHA는 B2 fresh
audit의 `checkpoint_manifest.json`, metric은 B0/B2 Korean summary와 per-class CSV,
ordered manifest SHA는 B2 `dataset_manifest_summary.json`에서 직접 복사한다.

다음은 NO-GO다.

- 위 trigger가 모두 불충족이고 B2가 B0의 domain-data 결론을 바꾸지 않음
- B2-E0 또는 B2-E-ADOM fresh evaluation/checkpoint provenance 미완료
- dataset/evaluation digest mismatch 또는 Korean held-out leakage
- B2에서 data quality/label/split 문제가 먼저 발견됨
- B5 compute가 독립 sequence/negative/co-occurrence annotation보다 결론 가치가 낮음
- 정확한 GPU profile, immutable image SHA 또는 effective batch 16을 확보하지 못함

## 7. 실행 중단 조건

향후 승인된 실행에서도 다음 중 하나면 즉시 중단하고 blocker artifact만 보존한다.

- architecture allowlist 또는 non-architecture fingerprint mismatch
- official B5 ImageNet init가 아닌 B0/B2 experiment checkpoint 사용
- primary split 4,568, manifest/mapping/content digest 불일치
- Semantic20 class order, 19 outputs 또는 ignore 255 변경
- Stage 1 backbone hash 변경, Stage 2 handoff mismatch, Stage 2 backbone 미변경
- RELLIS-val 이외 signal이 checkpoint/early-stop/recipe/threshold 선택에 유입
- canonical test unlock이 checkpoint freeze 전 발생
- effective batch 16 불가, non-finite loss, resume/determinism 실패
- hardware profile name/VRAM 불일치 또는 profile 밖 GPU로 자동 대체
- RELLIS/Korean pooled metric, primary/clean split 혼합, 다른 ordered manifest 사용

## 8. 결과 보고 틀

모든 metric은 B0/B2/B5 × E0/E-ADOM 3×2 table로 보고하고, 아래 delta를 같은 표에 둔다.

- `C02(E0)`, `C25(E0)`, `C02(E-ADOM)`, `C25(E-ADOM)`
- `A0`, `A2`, `A5`
- `DID02`, `DID25`, exploratory `DID25-DID02`
- RELLIS source guardrail과 worst-class delta
- log/rubble individual IoU/recall/precision/F1

Korean 61장은 class당 independent positive sequence가 하나이고 target-only partial
mask다. 61 independent samples, full-scene precision/false-stop, deployment safety,
통계적 유의성 또는 일반적 scaling law를 주장하지 않는다.

## 9. 이번 설계 작업의 명시적 비실행 범위

- RunPod/SSH 접속 및 Pod 생성
- W&B run 생성/조회/수정
- Docker build/push
- B5 memory probe, smoke, mini, resume, full training
- B5 RELLIS/Korean evaluation
- dataset/checkpoint/logits/masks/run output 생성 또는 Git 추가
