# TA method-recipe selection plan

## Goal and comparison

목표는 방법을 많이 넣는 것이 아니라, 앞으로 TA1/TA2와 후속 target adaptation에
재사용할 **증거가 있는 최소 recipe**를 고르는 것이다. 성능 상승을 보장하는 전처리나
loss는 없으므로 한 번에 결합하지 않고 다음 비교 사슬을 유지한다.

```text
Frozen E0
  -> TA0-C0 continued-training reference
  -> TA0-I  input/crop winner
  -> TA0-O  optimization winner
  -> TA0-B  imbalance winner
  -> TA0-R  combined recipe, 3 seeds
  -> TA1/TA2 data conditions, same recipe
```

모든 TA0 ablation은 같은 E0 SHA, RELLIS train, canonical RELLIS val, effective batch,
seed와 **총 optimizer update 수**를 사용한다. Stage 1을 추가한 실험은 Stage 1+2 합이
direct fine-tuning control과 같아야 한다.

## Gate A: crop and resolution audit

현재 pipeline은 `RandomResize(1024x512, 0.5..2.0)` 뒤 512x512 crop을 사용한다.
`cat_max_ratio=0.75`는 dominant class crop을 줄일 뿐 log/pole을 crop 안에 보장하지
않는다. 다음 후보를 offline transform audit 후 비교한다.

| ID | Train/eval spatial policy | Question |
| --- | --- | --- |
| I0 | 현재 512x512 random crop | E0와 같은 augmentation의 continued-training reference |
| I1 | 640x384 keep-ratio resize + ignore pad, no crop | 배포 shape와 full-scene context가 유리한가 |
| I2 | 640x480 keep-ratio resize + ignore pad, no crop | RELLIS 원본 aspect/context 보존이 640x384보다 유리한가 |

mask interpolation은 nearest만 허용하고 RGB는 bilinear를 사용한다. 각 source·class별로
최소 20회 transform Monte Carlo를 실행해 다음을 artifact로 남긴다.

- 원본에 있던 class가 transformed mask에 남을 확률
- class별 retained pixel/connected-component 비율
- non-ignore 비율, pad 비율, aspect distortion
- log, pole, barrier, rubble의 image exposure와 crop miss rate
- train/eval/deployment tensor shape, resize와 padding 좌표

I1은 유력 후보일 뿐 자동 채택하지 않는다. RELLIS는 약 688x550이고 640x384와 aspect가
달라 letterbox pad가 커질 수 있다. Frozen E0도 선택된 evaluation pipeline으로 다시
평가해 모델 개선과 evaluation resize 변화가 섞이지 않게 한다.

## Gate B: preprocessing audit

| Item | Default decision | Critical check |
| --- | --- | --- |
| normalization | ImageNet mean/std 유지 | E0 pretrained feature 계약이므로 source별 재정규화는 별도 ablation |
| horizontal flip | 유지 | 좌우 의미가 대칭인지 overlay 확인 |
| photometric distortion | 약한 범위 유지 후보 | ZED 노출/색 분포 밖의 비현실적 sample과 rare-class 소실 여부 |
| vertical flip/large rotation | 제외 | 주행 장면 geometry를 깨뜨림 |
| direct aspect warp | 제외 후보 | thin object·log shape 왜곡을 letterbox와 비교 |
| MixUp/CutMix/ClassMix | 첫 recipe에서 제외 | mask boundary와 source ontology noise를 증폭할 수 있음 |
| source normalization | 제외 | source identity shortcut을 줄일 가능성보다 E0 drift 위험이 큼 |

추가로 sequence leakage, near-duplicate/pHash, image-mask alignment, ignore boundary,
class별 pixel/image/sequence support와 source별 RGB 통계를 package digest에 연결한다.

## Gate C: two-stage optimization

현재 공통 scaffold의 head-only 1,000 + full-model 5,000 updates는 최종 결론이 아니다.
E0 head도 이미 학습됐으므로 다음을 같은 총 budget으로 비교한다.

1. **O0 direct full fine-tuning:** 작은 LR로 전체 B0를 바로 update하는 최소 control.
2. **O1 LP-FT style:** 500~1,000 update head-only 뒤 남은 budget full fine-tuning.
3. **O2 discriminative LR:** early MiT stage는 낮은 LR, late stage와 head는 높은 LR.
4. **O3 partial-to-full unfreeze:** late encoder+head 후 전체 모델. O1/O2 결과가 불안정할
   때만 진행한다.

warmup, AdamW, gradient accumulation과 checkpoint selection은 optimizer-update domain으로
고정한다. L2-SP/E0 teacher distillation은 catastrophic forgetting이 실제 관측될 때 E2
후속 ablation으로 둔다. LoRA는 B0에서 메모리 이익이 작고 MMSeg/TensorRT merge 검증
비용이 있으므로 core TA recipe가 아니다. AdaptFormer는 module이 graph에 남아 latency와
export 계약을 바꾸므로 더 뒤로 둔다.

## Gate D: class imbalance

동시에 여러 rebalancing을 켜면 과보정 원인을 알 수 없다. 우선순위는 다음과 같다.

1. **Source-aware sampling:** 이미 구현. TA0=RELLIS 1.0, TA1/TA2는 고정 source 비율.
2. **Image-level Rare Class Sampling:** mask class presence 통계로 rare-risk image를 더 자주
   뽑는다. 실제 post-transform class exposure를 기록하고 source quota 안에서 적용한다.
3. **CE + Lovasz-Softmax:** CE-only와 독립 비교한다. Lovasz는 평가 지표인 IoU의 surrogate라
   후보 가치가 있지만 작은/noisy class에서 변동성을 확인해야 한다.
4. **Class-balanced CE 또는 focal/OHEM:** RCS와 Lovasz가 부족할 때만 각각 독립 ablation.
   inverse-pixel weight를 그대로 쓰지 말고 weight cap과 gradient norm을 기록한다.

RCS, class weight, focal과 OHEM을 한 번에 결합하지 않는다. rare class IoU만 올리고
false positive나 common terrain을 무너뜨리는 후보는 실패다.

## Selection gates

seed 42, 500-update mini는 명백한 실패 제거용이며 최종 선택 근거가 아니다. 살아남은
후보와 결합 recipe는 seeds 42/43/44 full에서 다음을 만족해야 한다.

- canonical RELLIS `OverallSupported`와 `RareRisk-4` 평균 개선
- log/pole/barrier/rubble의 image exposure 및 per-class IoU/recall 보고
- 기존 supported class의 허용 범위 밖 하락 없음
- absent-class false-positive와 ADOM diagnostic false-stop 악화 없음
- source/class exposure가 요청 비율과 일치하고 resume 재현성 통과
- 단일 B0 graph 유지; export 후 ONNX parity와 Jetson FP16 p50/p95 latency 회귀 통과

개별로 성공한 요소를 합쳤더라도 결합 결과가 다시 통과해야 한다. 모두 효과가 있으면
checkpoint를 병합하는 것이 아니라 선택 recipe로 동일 E0에서 TA-final 하나를 새로
학습한다.

## Primary literature

- SegFormer, NeurIPS 2021: <https://proceedings.neurips.cc/paper_files/paper/2021/hash/64f1f27bf1b4ec22924fd0acb550c235-Abstract.html>
- LP-FT/OOD feature distortion, ICLR 2022: <https://arxiv.org/abs/2202.10054>
- DAFormer Rare Class Sampling, CVPR 2022: <https://openaccess.thecvf.com/content/CVPR2022/html/Hoyer_DAFormer_Improving_Network_Architectures_and_Training_Strategies_for_Domain-Adaptive_Semantic_CVPR_2022_paper.html>
- Lovasz-Softmax, CVPR 2018: <https://openaccess.thecvf.com/content_cvpr_2018/html/Berman_The_LovaSz-Softmax_Loss_CVPR_2018_paper.html>
- Class-Balanced Loss, CVPR 2019: <https://openaccess.thecvf.com/content_CVPR_2019/html/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.html>
- L2-SP, ICML 2018: <https://proceedings.mlr.press/v80/li18a.html>
- AdaptFormer, NeurIPS 2022: <https://papers.nips.cc/paper_files/paper/2022/hash/69e2f49ab0837b71b0e0cb7c555990f8-Abstract-Conference.html>
- LoRA, ICLR 2022: <https://openreview.net/pdf?id=nZeVKeeFYf9>
