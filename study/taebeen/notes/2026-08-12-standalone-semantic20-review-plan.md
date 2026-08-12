# ADOM standalone Semantic20 데이터 검토 및 target-adaptation 실험안

- 작성일: 2026-08-12
- 상태: **제안 / 실제 RunPod 데이터 정성·정량 QC 전 / 학습 미승인**
- 검토 대상: PR #32 (`dataset/adom_1`, `e1174682`)
- 현재 기준 모델: legacy B0-E0, RELLIS-only, selected Stage 2 iter 6,000
- ontology: Semantic20 train ID `0..18`, ignore `255`

이 문서는 코드 검토와 실험 제안이다. 기존 E0/E1/E2의 의미나 프로젝트 결정을
변경하지 않는다. 실제 데이터 검증과 사용자 승인 전에는 학습을 시작하지 않는다.

## 1. 결론

PR #32의 standalone 변환 결과는 **Semantic20 mask 자체의 형식은 적합**하지만,
현재 `semantic20_cycle`에 바로 넣어 재현 가능한 fine-tuning을 실행할 수 있는 상태는
아니다.

통과한 코드 계약:

- CVAT RGB mask를 `L`/`uint8` train-ID mask로 변환한다.
- valid ID는 `0..18, 255`, `reduce_zero_label=False`다.
- image/mask 크기, 상대경로 pair, unknown RGB, split sample 중복과 선언된 sequence의
  split 중복을 검사한다.
- `manifest.csv`의 `sample_key,image_path,mask_path`는
  `AdomSemantic20Dataset(manifest=...)`가 읽을 수 있다.

학습 전 blocker:

1. PR은 standalone package만 만들며 RELLIS anchor와 결합하는 Semantic20 builder를
   제공하지 않는다. 같이 추가된 `src/data/cost_4/` 코드는 다른 ontology이므로 이
   실험에 사용할 수 없다.
2. 현재 runtime은 `e0/e1/e2`의 source와 정확한 sample count를 hard-code한다.
   standalone 또는 RELLIS+standalone condition은 contract 단계에서 거부된다.
3. PR package에는 runtime이 combined condition에 요구하는 `_SUCCESS`와
   `results/final_check.json`이 없다.
4. 현재 cycle의 Stage 1은 B0-E0 checkpoint가 아니라 ImageNet MiT-B0 weight에서
   시작한다. 따라서 현재 cycle을 그대로 실행하면 요청한 “B0-E0에서 다시
   fine-tuning” 실험이 아니다.
5. 변환 통계는 전 split 합계뿐이고 validator도 per-split/per-sequence class pixel 및
   image support를 재계산해 저장하지 않는다. class imbalance 기여도를 아직 판정할
   수 없다.
6. `all_ignore_masks`를 세지만 실패시키거나 negative set으로 분리하지 않는다.
   partial-label random crop은 full image가 유효해도 all-ignore crop을 만들 수 있다.
7. preview/overlay, mask 경계 QC, near-duplicate/동일 장소 누수 검사와 사람의 승인
   기록이 없다.
8. upload package의 SHA-256 manifest를 Semantic20 converter가 다시 검증하지 않아
   업로드 source와 training package 사이 provenance가 끊긴다.

따라서 PR #32는 **전처리 기반으로는 유용하지만 training-ready 통합은 미완료**로
판정한다.

## 2. 데이터 형식 비교

| 항목 | RELLIS E0 | RUGD/YCOR E1 | standalone PR #32 | 판정 |
| --- | --- | --- | --- | --- |
| ontology | Semantic20 | Semantic20 direct bridge | Semantic20 partial | 일치 |
| mask | single-channel uint8 | single-channel uint8 | `L` uint8 | 일치 |
| train IDs | `0..18`, `255` | direct subset + `255` | direct subset + `255` | 일치 |
| image extension | JPEG | PNG/JPEG 혼합, manifest | PNG, manifest | loader 호환 |
| split | sequence-level | source split + RELLIS main val/test | declared capture sequence | 구조상 적합, 실제 scene 누수 미검증 |
| main val/test | canonical RELLIS | canonical RELLIS | standalone val/test | 역할이 다름 |
| current runtime | E0 지원 | E1 지원 | 미지원 | blocker |

standalone에서 black을 `255 ignore`로 바꾸는 것은 partial label 학습에는 맞다. 단,
“black frame = target이 없는 negative”라는 의미는 아니다. 사람이 target 부재를
확인한 별도 metadata 없이는 false-stop 평가에 사용할 수 없다.

## 3. 현재 확인 가능한 규모와 불균형 가설

### 3.1 image exposure

| train condition | RELLIS | RUGD | YCOR | standalone | standalone uniform share |
| --- | ---: | ---: | ---: | ---: | ---: |
| RELLIS + standalone | 4,435 | 0 | 0 | 133 | 2.91% |
| E1 train + standalone | 4,435 | 4,779 | 654 | 133 | 1.33% |

단순 concat+uniform sampling이면 standalone 영향은 매우 작다. partial mask의
non-ignore 비율까지 고려하면 실제 gradient 기여는 더 작을 수 있다. source-aware
batching 또는 rare-class sampling 없이 “데이터를 추가했다”만으로 효과를 기대하면
안 된다.

### 3.2 기존 source의 알려진 class 특성

- RELLIS 전체 mask 기준 pixel share: log 약 0.033%, rubble 약 0.186%, pole 약
  0.021%로 희소하다. bush는 약 17.45%라 standalone bush는 imbalance 해소의 핵심이
  아니다.
- RUGD+YCOR bridge 전 split non-ignore pixel 중 rubble은 약 2.565%다. 기존 E1은
  rubble을 이미 강하게 보강한다.
- RUGD+YCOR bridge에는 log 직접 mapping이 없다. standalone log가 충분하고 정확하면
  가장 독립적인 기여가 예상된다.
- standalone mapping은 dirt, log, person, bush, rubble을 포함한다. PR 설명의 표에는
  dirt/bush가 빠져 있으므로 실제 mask에서 두 색이 존재하는지 확인해야 한다.
- PR README에 따르면 person은 train sequence 한 곳에만 있어 독립 val/test 성능을
  보고할 수 없다.

standalone의 실제 `pixel_counts_by_target_id`, split별 image presence와 resize 뒤
target area를 얻기 전에는 “불균형 해소에 유의미하다”고 결론 내리지 않는다.

## 4. 필수 데이터 QC gate

### Gate Q0: provenance와 package integrity

- PR head SHA, source upload manifest SHA, mapping SHA, split SHA, 최종 manifest SHA를
  하나의 report에 저장한다.
- upload `manifest.json`의 모든 source image/mask SHA를 재계산한다.
- output은 새 경로에 만들고 `_SUCCESS`는 모든 검증 통과 뒤에만 생성한다.

### Gate Q1: 구조와 label 통계

- 215/133/21/61 sample 수를 실제 manifest에서 재계산한다.
- split/source-sequence별 class pixel count, image presence, non-ignore ratio를
  저장한다.
- all-ignore image와 random-crop 뒤 all-ignore 비율을 측정한다.
- 각 val/test target class가 최소 두 독립 sequence에 존재하는지 확인한다.
- standalone sample과 기존 source 사이 SHA/pHash 중복 및 standalone split 간
  near-duplicate를 검사한다.

### Gate Q2: 정성 검토

다음 strata에서 원본/mask/overlay/640x384 deployment preview를 만든다.

- 모든 sequence에서 최소 3장
- 모든 observed class의 pixel area 상/중/하 대표
- target이 작거나 화면 가장자리에 있는 장면
- 경계가 복잡한 log/rubble/person 장면
- all-ignore 및 명시적으로 target-absent인 장면
- class별 최소 10장, 전체 최소 50장

사람이 아래를 확인한다.

- RGB 색상 mapping과 class 의미가 맞는가
- 같은 class instance가 일부만 누락되지 않았는가
- polygon 경계가 물체를 과도하게 침범하거나 잘라내지 않는가
- log와 rubble, person과 pole/vegetation 같은 혼동 가능 class가 일관적인가
- 640x384 resize/pad 뒤 target이 사라지지 않는가
- train/val/test가 사실상 같은 위치·시점의 인접 장면이 아닌가

### Gate Q3: 평가 가능성

- standalone val이 한 sequence뿐이면 grouped stratified split을 다시 설계한다.
- test는 checkpoint/loss/sampler 선택에 사용하지 않는다.
- target-absent negative scene은 mask의 ignore와 별도로 `target_absent_verified=true`
  metadata를 갖는다.
- small dataset 결과는 per-sequence bootstrap과 paired 3 seeds로 보고한다.

## 5. RunPod에서 확인하고 내려받는 절차

아래 명령은 read-only discovery부터 시작한다. 실제 root를 찾은 뒤 placeholder를
바꾼다.

```bash
find /workspace/adom/datasets -type f \
  -path '*/metadata/conversion_summary.json' -print
find /workspace/adom/datasets -type f -name manifest.csv \
  -path '*adom*' -print
```

```bash
export ADOM_STANDALONE_ROOT=<FOUND_PROCESSED_ROOT>
test -f "$ADOM_STANDALONE_ROOT/manifest.csv"
cat "$ADOM_STANDALONE_ROOT/metadata/conversion_summary.json"
sha256sum "$ADOM_STANDALONE_ROOT/manifest.csv" \
  "$ADOM_STANDALONE_ROOT/metadata/label_mapping.json" \
  "$ADOM_STANDALONE_ROOT/metadata/split_sequences.json"
```

PR code가 immutable image에 아직 없다면 `/opt/adom`을 수정하지 말고 PR head를
별도 임시 checkout에서 검증한다. head는 `e11746826ba70ccdbfe587381ab61324fabbd17e`
이어야 한다.

```bash
python3 <PR32_CHECKOUT>/src/data/adom_data/scripts/validate_semantic20_package.py \
  --input-root "$ADOM_STANDALONE_ROOT"
```

정성 검토용 preview packager는 training branch에서 추가한다. 생성물은 Git이 아닌
Network Volume의 versioned QC 경로에 둔다. QC archive와 checksum만 내려받는다.

```bash
export ADOM_QC_ARCHIVE=/workspace/adom/exports/adom-standalone-qc-v1.tar
tar -C /workspace/adom/exports -cf "$ADOM_QC_ARCHIVE" adom-standalone-qc-v1
sha256sum "$ADOM_QC_ARCHIVE" > "$ADOM_QC_ARCHIVE.sha256"
runpodctl send "$ADOM_QC_ARCHIVE" "$ADOM_QC_ARCHIVE.sha256"
```

Pod가 출력한 one-time code를 로컬에서 `runpodctl receive <CODE>`로 받는다. 큰 전체
dataset은 SSH/SCP 또는 rsync를 사용하고, credential/API key는 채팅이나 저장소에
남기지 않는다.

## 6. 권장 실험: 기존 E0/E1/E2와 분리한 target-adaptation family

저장소에서 E1은 `RELLIS+RUGD+YCOR`, E2는 `E1+GOOSE`로 이미 고정돼 있다. 새 실험을
E1로 재정의하지 않고 `TA0/TA1/TA2`로 분리한다.

모든 condition은 동일한 B0-E0 selected checkpoint에서 시작하고, 동일한 update 수,
augmentation, optimizer와 seed `42,43,44`를 사용한다.

| ID | initialization | train exposure | 목적 |
| --- | --- | --- | --- |
| Frozen E0 | E0 | 없음 | 배포 baseline |
| TA0 | E0 | RELLIS-only continued | 추가 update 효과 통제 |
| TA1 | E0 | RELLIS + standalone | 자체 데이터의 직접 효과 |
| TA2 | E0 | RELLIS + RUGD + YCOR + standalone | 외부 domain diversity와 상호작용 |

GOOSE는 기존 E2 연구와 혼동을 피하기 위해 첫 target-adaptation 비교에서 제외한다.

### 6.1 단계

1. Q0-Q3 통과
2. 50-update smoke: finite loss, backbone freeze/update, sampler exposure 확인
3. seed 42로 500-update learnability mini-run
4. TA0/TA1/TA2를 3 seeds로 2,000-5,000 updates
5. validation으로 checkpoint와 recipe 선택
6. 선택된 하나만 standalone held-out test와 canonical RELLIS test를 1회 실행

초기 recipe는 root context의 short fine-tuning 제안을 따른다.

- E0 checkpoint에서 시작
- head 중심 500-1,000 updates
- full model 2,000-5,000 updates
- backbone LR = head LR의 0.1배
- target class weight는 sampler/crop 진단 뒤에도 학습되지 않을 때만 최대 3배

### 6.2 sampling

naive concat은 금지한다. mini-run에서는 RELLIS:standalone 1:1 exposure로
learnability를 확인한다. confirmatory comparison은 shared source의 exposure를
고정한 source-aware batch를 사용한다. 예시 micro-batch 16:

- TA0: RELLIS 16
- TA1: RELLIS 12 + standalone 4
- TA2: RELLIS 7 + RUGD 4 + YCOR 1 + standalone 4

이는 제안값이며 실제 class pixel/crop support를 본 뒤 동결한다. TA1/TA2에서
standalone 25%를 같게 유지해 희석 차이를 줄인다.

### 6.3 metric과 성공 gate

Primary:

- standalone validation의 log/rubble class IoU와 Recall
- target safety corridor의 instance/ROI Recall

Secondary:

- Precision/F1, false-positive area, connected-component false alarm
- 명시적 target-absent clip의 false-stop event rate

Non-degradation:

- canonical RELLIS `ValSupported13-mIoU` 평균 하락 1.0%p 이내
- RELLIS RareRisk-4(log/pole/barrier/rubble) 치명적 하락 없음
- paired seed 3개 중 2개 이상에서 같은 방향의 개선

test는 최종 선택 모델에 한 번만 사용한다. pixel을 독립 표본으로 간주한 과도하게
좁은 confidence interval은 사용하지 않고 sequence 단위로 보고한다.

## 7. 예상 결과와 연구적 의미

### TA1: RELLIS + standalone

예상:

- log가 충분히 라벨돼 있으면 가장 큰 개선 가능성이 있다. 기존 external bridge에는
  log 직접 mapping이 없기 때문이다.
- ZED 2i와 실제 운용 장소 domain에 맞아 custom recall은 개선될 가능성이 높다.
- 데이터가 작고 장면 다양성이 낮으면 custom test에는 좋아도 RELLIS/general domain은
  유지 또는 소폭 하락할 수 있다.

의미:

- 현장 센서·장면에 대한 적은 수의 partial label이 배포 실패를 고칠 수 있는지 검증한다.
- TA0와 비교하면 단순한 추가 update 효과를 분리할 수 있다.

### TA2: RELLIS + RUGD + YCOR + standalone

예상:

- rubble과 일반 outdoor feature는 RUGD의 도움을 받을 수 있다.
- YCOR의 주요 직접 class는 grass/puddle이어서 log 개선에는 직접 기여가 작다.
- source balancing이 없으면 standalone이 1.33%로 희석돼 TA1보다 target improvement가
  작을 수 있다.
- source-aware sampling이 적절하면 target 성능과 broader robustness를 함께 유지할
  가능성이 있다. 반대로 mapping noise/domain conflict가 있으면 precision과
  RELLIS mIoU가 하락할 수 있다.

의미:

- 자체 데이터와 public off-road datasets가 상보적인지, 또는 작은 target domain
  signal을 희석하는지 검증한다.
- TA1 대비 TA2의 차이는 “데이터가 많을수록 좋다”가 아니라 source mixture와
  mapping 품질의 효과다.

## 8. Jetson latency 해석

제공된 두 snapshot의 최신 값:

- `average_fps=11.57`
- `capture_to_receive_ms=81.51`
- `queue_wait_ms=32.84`
- `capture_to_inference_start_ms=114.35`
- `inference_ms=74.82`
- `processing_ms=80.51`
- `capture_to_perception_output_ms=194.81`
- `overwritten_frames/received_frames=708/11509=6.15%` cumulative

관계:

```text
capture_to_inference_start
  = capture_to_receive + queue_wait
  = 81.51 + 32.84 = 114.35 ms

capture_to_perception_output
  ~= capture_to_inference_start + processing
  = 114.35 + 80.51 = 194.86 ms
```

`processing_ms - inference_ms`는 약 5.69 ms이므로 worker 내부 병목의 대부분은
TensorRT inference다. 그러나 inference 밖의 frame age가 이미 약 114 ms라 engine만
최적화해도 end-to-end latency가 충분히 줄지 않을 수 있다.

현재 평균 수치만으로 p95를 알 수 없다. tracked repository의
`adom_perception_ros/scripts/perception_node.py`도 이 상세 field를 발행하지 않으므로,
실제 Jetson executable의 source SHA/경로를 먼저 확보해야 “실제 코드 기준” audit가
완료된다.

### 8.1 제안 latency gate

최종 상한은 실제 0.3 m/s braking distance와 camera-to-command p95로 동결한다. 그
전까지 다음을 제안값으로 사용한다.

| 항목 | 최소 live gate | engineering target |
| --- | ---: | ---: |
| output frequency | >=10 Hz | >=15 Hz |
| inference p95 | <=75 ms | <=55-60 ms |
| processing p95 | <=100 ms | <=70 ms |
| capture-to-perception-output p95 | <=200 ms | <=150 ms |
| camera-to-command p95 | <=250 ms | <=200 ms |

현재 `capture_to_perception_output` 평균이 이미 약 195 ms라 p95 200 ms gate는 통과할
가능성이 낮다. 평균 195 ms 동안 0.3 m/s 차량은 약 5.85 cm 이동하며, 여기에
costmap/controller와 실제 braking distance가 추가된다. watchdog 250 ms는 command
loss fail-safe이지 target detection 반응시간 보장이 아니다.

### 8.2 Jetson에서 추가로 수집할 명령

```bash
git -C <JETSON_REPO> rev-parse HEAD
ros2 node info /adom/perception
ros2 param dump /adom/perception
ros2 topic info <ACTUAL_ZED_RGB_TOPIC> --verbose
ros2 topic info /adom/perception/semantic20_mask --verbose
```

```bash
timeout 60s ros2 topic hz <ACTUAL_ZED_RGB_TOPIC>
timeout 60s ros2 topic hz /adom/perception/semantic20_mask
timeout 60s ros2 topic bw <ACTUAL_ZED_RGB_TOPIC>
```

```bash
nvpmodel -q --verbose
sudo jetson_clocks --show
tegrastats --interval 1000
```

```bash
timeout 120s ros2 topic echo /adom/perception/status \
  --field data --full-length > /tmp/adom-perception-status.txt
timeout 120s ros2 topic echo /adom/navigation/costmap_status \
  --field data --full-length > /tmp/adom-costmap-status.txt
timeout 120s ros2 topic echo /adom/control/local_path_status \
  --field data --full-length > /tmp/adom-control-status.txt
```

추가 필요 정보:

- 각 latency의 p50/p95/p99와 5분 이상 thermal steady-state
- ZED actual publish resolution/FPS, QoS, encoding과 timestamp 기준
- TensorRT engine precision, input shape, workspace, power mode
- output subscriber on/off별 latency
- 0.3 m/s 실제 braking distance와 허용 stop corridor margin

## 9. fine-tuning과 경량화 roadmap

| 방법 | 기대 효과 | 구현 난이도 | 학습 비용 | inference 영향 | 우선순위 |
| --- | --- | --- | --- | --- | --- |
| source-aware/rare-class sampling | 희소 target 노출 증가 | 중 | 낮음 | 없음 | P0 |
| target-aware crop + resize QC | all-ignore crop 감소 | 중 | 낮음 | 없음 | P0 |
| E0 warm-start differential LR | 안정적 domain adaptation | 낮음 | 낮음 | 없음 | P0 |
| class weight <=3x | target gradient 증가 | 낮음 | 낮음 | 없음 | P1, sampler 후 |
| CE+Lovasz 또는 focal ablation | IoU/희소 class 개선 가능 | 중 | 중 | 없음 | P1 |
| copy-paste/ClassMix | log/rubble 위치 다양성 | 중 | 중 | 없음 | P1, mask QC 후 |
| LoRA on MiT attention/MLP | trainable parameter/optimizer memory 감소 | 중-상 | 낮음-중 | merge하면 없음 | P2 ablation |
| AdaptFormer/VPT | parameter-efficient adaptation | 상 | 중 | module/token overhead 가능 | P3 |
| QLoRA | training memory 감소 | 상 | 중 | 배포 속도 이점 없음 | 보류 |
| B2 teacher -> B0 distillation | B0 정확도 개선 가능 | 상 | 높음 | B0 runtime 유지 | P2 |
| lower static input shape | latency 직접 감소 | 중 | 중 | 정확도와 latency trade-off | P1 |
| TensorRT explicit INT8 PTQ | latency/전력 개선 가능 | 중 | 낮음-중 | 감소 기대 | P1 after FP16 baseline |
| INT8 QAT | PTQ accuracy 회복 | 상 | 높음 | 감소 기대 | P2 if PTQ fails |
| structured pruning | 실제 FLOP/latency 감소 가능 | 상 | 높음 | 감소 가능, 보장 안 됨 | P3 |

LoRA는 A100 메모리가 충분하고 B0가 이미 작은 현재 조건에서 첫 선택이 아니다. 적용할
경우 adapter를 base weight에 merge한 뒤 기존 ONNX raw-logit 계약으로 export해야
TensorRT inference overhead가 없다. optimizer state와 checkpoint 크기 감소는 장점이지만
MMSeg/MiT integration, merge, parity 검증 비용이 추가된다.

경량화 순서:

1. FP16 engine과 ROS frame-age 병목을 분리하고 p95 profile 확정
2. 640x384, 512x320, 512x288/320-pad static shape A/B
3. subscriber 없는 confidence/overlay 계산·publish 제거 여부 확인
4. explicit-quantized INT8 PTQ를 target Jetson에서 build하고 class/ROI parity 평가
5. PTQ가 target recall/FP gate를 깨면 QAT
6. 그 뒤에도 latency가 부족할 때만 distillation 또는 TensorRT-realizable structured
   pruning

Orin Nano의 Ampere GPU에서는 FP8/FP4를 목표로 잡지 않는다. INT8 calibration set은
train/validation 선택과 분리하고 ZED 실제 장면, 각 target, negative, 조명/거리 strata를
대표해야 한다. engine은 계속 target Jetson에서 생성한다.

## 10. 승인 후 구현 범위 제안

새 branch는 최신 `main`과 PR #32 반영 상태에서 만든다. 현재 다른 작업의 dirty branch를
재사용하지 않는다.

1. PR #32 review blocker 수정 또는 후속 patch
2. Semantic20 QC/statistics/preview tool
3. RELLIS+standalone 및 E1+standalone combined package builder
4. `TA0/TA1/TA2` config/runtime contract와 E0 checkpoint warm-start
5. source-aware sampler와 exposure artifact
6. synthetic tests, full unit tests, Git artifact guard
7. RunPod Q0-Q3 실데이터 검증
8. 사용자에게 실제 통계·preview와 최종 run cost를 보고
9. 명시 승인 후 smoke -> mini -> full 3-seed fine-tuning

## 11. 2026-08-12 RunPod discovery 결과

사용자가 실행한 discovery 결과:

```text
*/metadata/conversion_summary.json: 결과 없음
*/manifest.csv:
/workspace/adom/datasets/processed/adom_semantic20_rellis_rugd_ycor_v1/manifest.csv
```

현재 확인되는 것은 기존 E1 processed package뿐이다. standalone package가
`/workspace/adom/datasets/processed`에 없으므로 다음 중 하나다.

1. RGB-mask upload source만 있고 PR #32 변환을 아직 실행하지 않음
2. 다른 `/workspace` 경로에 upload/processed package가 있음
3. upload가 다른 Network Volume 또는 Pod volume에 있음

TA 구현·학습 전에 아래 read-only discovery를 실행한다.

```bash
find /workspace -maxdepth 6 -type d \
  \( -iname '*standalone*' -o -iname '*adom*data*' \
     -o -iname '260810*' -o -iname '260811*' \) -print | sort
```

```bash
find /workspace -maxdepth 7 -type f \
  \( -name manifest.json -o -name conversion_summary.json \
     -o -name split_sequences.json -o -name label_mapping.json \) \
  -print | sort
```

```bash
find /workspace/adom/datasets -maxdepth 5 -type f \
  \( -name '*.tar' -o -name '*.zip' \) \
  -printf '%p\t%s bytes\n' | sort
du -h --max-depth=3 /workspace/adom/datasets 2>/dev/null \
  | sort -h | tail -50
```

원본 위치가 확인되면 PR #32 변환과 Q0-Q3를 먼저 수행한다. 현재 상태에서 TA1/TA2
학습을 시작하면 안 된다.

## 12. 114 ms pre-inference latency 진단

114 ms는 SegFormer의 필수 지연이 아니며 불가피하다고 볼 근거가 없다. 두 부분을
따로 진단한다.

### 12.1 capture-to-receive 약 82 ms

가능한 원인:

- ZED exposure/readout/rectification 및 wrapper 내부 buffering
- depth/VIO/point-cloud 등 불필요 module과 GPU/CPU contention
- RGB publish resize/encoding과 DDS serialization/copy
- publisher/subscriber QoS queue
- header stamp와 subscriber clock의 기준 또는 동기 오차

raw RGB subscriber에서 `now - header.stamp`를 직접 측정하고, status node의 값과
대조한다. `ros2 topic delay`가 설치돼 있으면 함께 사용한다.

```bash
ros2 topic delay <ACTUAL_ZED_RGB_TOPIC>
ros2 topic info <ACTUAL_ZED_RGB_TOPIC> --verbose
ros2 param dump /zed/zed_node
```

ZED를 RGB-only, 실제 필요한 publish resolution/FPS, Best Effort + Keep Last 1로 둔 뒤
전후 p95를 비교한다. clock mismatch가 있으면 이 수치는 pipeline latency가 아니라
timestamp offset이므로 먼저 바로잡는다.

### 12.2 queue wait 약 33 ms

30 Hz frame period 한 개와 거의 같다. inference가 약 75 ms라 subscriber callback과
worker가 직렬이거나 one-slot frame이 worker 시작까지 기다리는 구조일 가능성이 있다.
latest-frame slot 자체는 옳지만 다음을 확인한다.

- ROS callback은 frame reference/sequence만 교체하고 즉시 반환
- worker가 완료 즉시 가장 최신 frame을 가져오며 오래된 frame을 처리하지 않음
- queue depth 1, old frame overwrite, backlog 증가 없음
- image conversion/preprocess가 queue wait와 processing 중 어디에 계측되는지
- Python executor/GIL 또는 mutex가 callback과 worker를 불필요하게 직렬화하는지

이론적으로 inference가 input period보다 길어도 최신 frame을 worker 완료 시점에
가져오면 queue wait는 0-33 ms 범위가 될 수 있다. 평균 33 ms 고정은 최적 상태라고
보기 어렵다. 실제 상세 status emitter source가 repository에 없으므로 해당 파일과
Git SHA를 확보한 뒤 수정 여부를 결정한다.

## 13. TA 병렬 실행과 최종 통합 원칙

### 13.1 비교 구조

- primary deployment baseline: frozen B0-E0 selected checkpoint
- continued-training control: TA0
- standalone effect: TA1 대 E0 및 TA0
- external bridge interaction: TA2 대 TA1

E0는 고정 checkpoint라 seed가 없다. TA0/TA1/TA2는 같은 E0 checkpoint에서 각각
독립 시작하고 seed `42,43,44`를 paired comparison으로 사용한다. TA1 checkpoint에서
TA2를 이어 학습하면 update 수와 순서 효과가 섞이므로 첫 비교에서는 금지한다.

### 13.2 병렬화

가능한 실행 방식:

1. 3 Pods: TA0/TA1/TA2의 같은 seed를 동시에 실행
2. 최대 9 Pods: 3 conditions x 3 seeds를 동시에 실행
3. 1 multi-GPU Pod: GPU별로 독립 process와 output root 사용

첫 실행은 3 Pods로 seed 42 smoke/mini를 병렬 실행하고, gate 통과 뒤 seeds 42-44를
wave로 돌리는 것을 권장한다. 9 Pods는 wall time은 줄지만 dataset validation I/O,
비용 감시와 장애 추적이 복잡하다.

Network Volume은 여러 Pod에 attach할 수 있지만 동시 write는 충돌 위험이 있다.
다음 계약을 강제한다.

- dataset package는 content SHA로 고정하고 모든 Pod에서 read-only로 취급
- Pod마다 완전히 다른 output root
- W&B run ID/name과 cache directory도 Pod별 분리
- 같은 `last_checkpoint`, `status.json`, summary, TensorBoard directory 공유 금지
- 가능하면 dataset tar를 각 Pod의 `/tmp/data`로 checksum 검증 후 cache
- full dataset audit는 한 번만 실행하고 training Pod는 frozen audit SHA를 확인

제안 output:

```text
/workspace/adom/runs/semantic20/target-adaptation-v1/
  ta0/seed42/<run-id>/
  ta1/seed42/<run-id>/
  ta2/seed42/<run-id>/
```

### 13.3 모두 효과가 있을 때

TA0/TA1/TA2 checkpoint를 ensemble하거나 평균하지 않는다. 현재 latency budget에도
ensemble은 맞지 않는다. 세 결과가 모두 좋아도 다음과 같이 결정한다.

1. TA0 개선: E0가 under-trained였거나 short continued training 자체가 유효
2. TA1이 TA0보다 추가 개선: standalone의 순효과
3. TA2가 TA1보다 추가 개선하며 non-degradation 통과: public source의 상보 효과
4. validation에서 가장 좋은 source weights와 recipe를 동결
5. `TA-final`을 동일 E0 checkpoint에서 새로 학습
6. TA-final 하나만 standalone held-out 및 canonical RELLIS test에 1회 평가

TA1 후 TA2 순차 curriculum이 필요하면 `TA3-curriculum`이라는 별도 ablation으로
취급하고 동일 update budget의 control을 둔다.

## 14. TA0-TA2 코드 보완 계획

### Workstream A: standalone materialization과 QC

1. PR #32를 최신 main 위에서 반영하고 standalone upload source를 찾는다.
2. converter가 upload SHA manifest를 실제 파일과 교차 검증하게 한다.
3. split/source-sequence별 pixel count, image presence, non-ignore, all-ignore 통계를
   생성한다.
4. deployment resize preview, class/sequence stratified overlay, pHash duplicate report를
   만든다.
5. all-ignore train frame을 제거하거나 explicit negative diagnostic으로 분리한다.
6. `_SUCCESS`, immutable manifest/content SHA와 approval record를 Q0-Q3 통과 후 생성한다.

예상 변경:

- `src/data/adom_data/scripts/convert_semantic20.py`
- `src/data/adom_data/scripts/validate_semantic20_package.py`
- 신규 reusable QC/preview module 또는 script
- `tests/test_adom_data_preprocessing.py`

### Workstream B: 하나의 immutable target-adaptation package

E1 package와 standalone package를 결합한 superset 하나를 만든다. source 이름은
`adom_zed2i`로 고정한다.

```text
splits/ta0_train.txt              RELLIS
splits/ta1_train.txt              RELLIS + ADOM ZED2i train
splits/ta2_train.txt              RELLIS + RUGD + YCOR + ADOM ZED2i train
splits/val.txt                    canonical RELLIS val
splits/test.txt                   canonical RELLIS test
splits/adom_val_diagnostic.txt    standalone val
splits/adom_test_diagnostic.txt   standalone test
```

모든 source pair는 package-root-relative manifest에 기록하고 val/test canonical split을
바꾸지 않는다. TA condition마다 dataset을 복제하지 않아 content 차이를 막는다.

예상 변경:

- 신규 `src/data/semantic_20/scripts/05_build_target_adaptation_package.py`
- 신규 `src/data/semantic_20/scripts/06_validate_target_adaptation_package.py`
- canonical source/mapping digest metadata
- `tests/test_semantic20_preprocessing.py`

### Workstream C: source-aware sampler

현재 `InfiniteSampler` uniform concat을 대체하는 deterministic weighted infinite sampler를
MMSeg registry에 추가한다. sample key prefix로 source를 판별하고 requested weight와
실제 draw count를 artifact로 남긴다.

초기 confirmatory weight:

- TA0: RELLIS 1.00
- TA1: RELLIS 0.75, ADOM ZED2i 0.25
- TA2: RELLIS 0.4375, RUGD 0.25, YCOR 0.0625, ADOM ZED2i 0.25

mini learnability gate에서는 TA1에 한해 RELLIS:ADOM 1:1을 먼저 시험할 수 있지만,
confirmatory 비교에는 위 fixed weight를 사용한다.

예상 변경:

- 신규 `src/adom/mmseg/samplers.py`
- `src/adom/mmseg/__init__.py`
- exposure audit hook/artifact
- sampler determinism, weight, resume unit tests

### Workstream D: E0 warm-start TA runtime

1. experiment enum에 `ta0/ta1/ta2`를 추가한다.
2. `--initial-checkpoint`와 `--expected-initial-checkpoint-sha256`를 TA에서 필수화한다.
3. B0, 19-class head와 known E0 SHA를 preflight에서 검증한다.
4. Stage 1에도 E0 checkpoint를 `load_from`으로 전달한다.
5. TA config에서는 ImageNet `init_cfg` download를 끄고 E0 load만 사용한다.
6. Stage 1 500-1,000 updates, Stage 2 2,000-5,000 updates 전용 schedule을 분리한다.
7. dataset/content/checkpoint/sampler digest를 run context에 저장해 unsafe resume을 막는다.

예상 변경:

- `src/adom/runtime/semantic20_cycle.py`
- `configs/adom/phase1_semantic20/_base_/datasets/ta*.py`
- `configs/adom/phase1_semantic20/_base_/schedules/target_adapt_*.py`
- `configs/adom/phase1_semantic20/segformer_b0_*_ta*.py`
- `tests/test_semantic20_training.py`
- `tests/test_semantic20_clean_contract.py`

### Workstream E: evaluation과 parallel launch contract

- standalone val/test metric을 canonical RELLIS checkpoint-selection metric과 분리
- E0/TA0/TA1/TA2 동일 scene prediction artifact
- source/sequence별 metric과 false-stop negative diagnostic
- condition/seed별 unique W&B/run directory generator
- 3-Pod seed-42 smoke command와 3-seed full command 문서화
- final test lock 유지

### 구현 순서와 승인 gate

1. A 완료: 실제 data Q0-Q3 report와 preview 사용자 검토
2. B-C 완료: package/sampler synthetic test와 exposure report
3. D-E 완료: local unit suite, Git artifact guard, container config parse
4. RunPod dataset contract + B0 2-update probe
5. TA0/TA1/TA2 seed42 50-update smoke 병렬
6. 결과·예상 GPU-hour 보고 후 사용자 full-run 승인
7. 500-update mini 병렬
8. 3-seed full, validation selection, TA-final 결정

당장 코드를 보완할 수는 있지만 standalone data root가 확인되기 전에는 4단계 이후로
진행하지 않는다.

## 참고

- PR #32: <https://github.com/imtaebin83-debug/ADOM/pull/32>
- LoRA: <https://arxiv.org/abs/2106.09685>
- AdaptFormer: <https://papers.nips.cc/paper_files/paper/2022/hash/69e2f49ab0837b71b0e0cb7c555990f8-Abstract-Conference.html>
- DAFormer rare class sampling: <https://openaccess.thecvf.com/content/CVPR2022/papers/Hoyer_DAFormer_Improving_Network_Architectures_and_Training_Strategies_for_Domain-Adaptive_Semantic_CVPR_2022_paper.pdf>
- Lovasz-Softmax: <https://openaccess.thecvf.com/content_cvpr_2018/html/Berman_The_LovaSz-Softmax_Loss_CVPR_2018_paper.html>
- TensorRT quantization: <https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/work-quantized-types.html>
- RunPod transfer: <https://docs.runpod.io/pods/storage/transfer-files>
