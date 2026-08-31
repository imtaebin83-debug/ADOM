# ADOM Semantic20 통합 전처리 가이드

이 가이드는 팀 GitHub에 이미 존재하는 **RELLIS-3D / RUGD / YCOR 전처리 코드와 mapping을 그대로 재사용**해,
`RELLIS-3D + RUGD + YCOR + ADOM-v2`를 한 번에 학습용 Semantic20 패키지로 만드는 방법입니다.

## 1. 기준 label

최종 마스크는 single-channel PNG이며 다음 ID만 허용합니다.

| ID | class |
|---:|---|
| 0 | dirt |
| 1 | grass |
| 2 | tree |
| 3 | pole |
| 4 | water |
| 5 | sky |
| 6 | vehicle |
| 7 | object |
| 8 | asphalt |
| 9 | building |
| 10 | log |
| 11 | person |
| 12 | fence |
| 13 | bush |
| 14 | concrete |
| 15 | barrier |
| 16 | puddle |
| 17 | mud |
| 18 | rubble |
| 255 | ignore |

`num_classes=19`, `ignore_index=255`, `reduce_zero_label=false`입니다.

---

## 2. 필요한 Python 패키지

Ubuntu 예시:

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install numpy pillow pyyaml
```

---

## 3. 팀 GitHub clone

예시:

```bash
cd ~
git clone https://github.com/imtaebin83-debug/ADOM.git
cd ADOM
```

전처리 스크립트는 이 repository 안에 이미 있는 다음 파일들을 재사용합니다.

```text
study/gahyung/Datasets_Repo/
├── RELLIS-3D/
│   └── rellis3d_semantic20_v1/
├── RUGD/
├── YCOR/
└── ADOM-Semantic20/
    └── adom_semantic20_rellis_rugd_ycor_v1/
```

---

## 4. 원본 데이터 폴더 만들기

예를 들어 모든 원본 데이터를 `~/ADOM_data`에 둔다고 하면:

```bash
mkdir -p ~/ADOM_data/raw
```

최종적으로 아래처럼 둡니다.

```text
~/ADOM_data/
└── raw/
    ├── RELLIS-3D/
    │   ├── 00000/
    │   │   ├── pylon_camera_node/
    │   │   └── pylon_camera_node_label_id/
    │   ├── 00001/
    │   ├── 00002/
    │   └── ...
    │
    ├── RUGD/
    │   └── .../
    │       ├── image/
    │       └── indexLabel/
    │
    ├── YCOR/
    │   ├── train/
    │   │   ├── <sample_folder>/
    │   │   │   ├── rgb.jpg
    │   │   │   └── labels.png
    │   │   └── ...
    │   └── valid/
    │       └── <sample_folder>/
    │           ├── rgb.jpg
    │           └── labels.png
    │
    └── ADOM-v2/             # 완성 후 추가
        ├── images/
        │   ├── train/
        │   ├── val/
        │   └── test/
        └── masks/
            ├── train/
            ├── val/
            └── test/
```

### RELLIS-3D

각 sequence 폴더 안에 최소한 다음 두 폴더가 있어야 합니다.

```text
pylon_camera_node/
pylon_camera_node_label_id/
```

### RUGD

통합 스크립트가 `RUGD/` 아래를 재귀 탐색하여 이름이 `image`, `indexLabel`인 폴더를 찾습니다.

즉 원본 압축을 푼 뒤 실제 RGB가 들어있는 `image/`, index mask가 들어있는 `indexLabel/` 구조만 유지하면 됩니다.

### YCOR

기존 코드에서 확인한 원본 구조를 그대로 사용합니다.

```text
YCOR/
├── train/<sample>/rgb.jpg
├── train/<sample>/labels.png
├── valid/<sample>/rgb.jpg
└── valid/<sample>/labels.png
```

### ADOM-v2

ADOM-v2는 아직 URL/최종 package가 확정되지 않았으므로, 통합 코드는 다음 **input contract**만 요구합니다.

```text
ADOM-v2/
├── images/train/
├── images/val/
├── images/test/
├── masks/train/
├── masks/val/
└── masks/test/
```

이미지와 mask는 **동일 stem**이어야 합니다.

예:

```text
images/train/frame_000123.png
masks/train/frame_000123.png
```

ADOM-v2 mask는 CVAT export 원본 palette/overlay가 아니라 반드시 **class-ID mask PNG**여야 합니다.

```text
single channel
dtype: uint8
allowed IDs: 0..18, 255
```

---

## 5. 통합 스크립트 위치

이 파일을 예를 들어 다음 위치에 둡니다.

```text
~/ADOM/scripts/data/prepare_adom_semantic20_v2.py
```

현재 전달한 `prepare_adom_semantic20_v2.py`를 해당 위치에 복사하면 됩니다.

---

## 6. 실행

### ADOM-v2가 아직 없을 때

```bash
cd ~/ADOM

python scripts/data/prepare_adom_semantic20_v2.py \
  --repo-root ~/ADOM \
  --data-root ~/ADOM_data \
  --output-root ~/ADOM_data/processed/adom_semantic20_v2 \
  --skip-adom-v2 \
  --overwrite
```

### ADOM-v2까지 준비된 뒤

```bash
cd ~/ADOM

python scripts/data/prepare_adom_semantic20_v2.py \
  --repo-root ~/ADOM \
  --data-root ~/ADOM_data \
  --output-root ~/ADOM_data/processed/adom_semantic20_v2 \
  --adom-eval-policy diagnostic \
  --overwrite
```

`~/ADOM_data/raw/ADOM-v2`에 존재한다면 `--adom-v2-root`는 생략해도 됩니다.

다른 위치에 있다면:

```bash
python scripts/data/prepare_adom_semantic20_v2.py \
  --repo-root ~/ADOM \
  --data-root ~/ADOM_data \
  --adom-v2-root /path/to/ADOM-v2 \
  --output-root ~/ADOM_data/processed/adom_semantic20_v2 \
  --adom-eval-policy diagnostic \
  --overwrite
```

---

## 7. evaluation policy

기존 3-source Semantic20 package는 다음 정책을 사용했습니다.

```text
train = RELLIS train + RUGD train + YCOR train
val   = RELLIS val only
test  = RELLIS test only
```

이 정책을 보존하기 위해 기본값은:

```bash
--adom-eval-policy diagnostic
```

입니다.

ADOM-v2가 `133 / 21 / 61` train/val/test split으로 만들어진 경우:

```text
main train
  = 기존 main train + ADOM-v2 train(133)

main val
  = 기존 RELLIS val

main test
  = 기존 RELLIS test

diagnostic
  = adom_v2_val_diagnostic.txt  (21)
  = adom_v2_test_diagnostic.txt (61)
```

이렇게 두면 **canonical RELLIS 평가**와 **자체 데이터 target 평가**를 분리할 수 있습니다.

만약 정말 하나의 mixed val/test가 필요하면:

```bash
--adom-eval-policy mixed
```

를 사용합니다.

이 경우:

```text
val  = RELLIS val + ADOM-v2 val
test = RELLIS test + ADOM-v2 test
```

가 됩니다.

---

## 8. 출력 구조

정상 완료되면:

```text
~/ADOM_data/processed/adom_semantic20_v2/
├── images/
│   ├── rellis3d/
│   ├── rugd/
│   ├── ycor/
│   └── adom_v2/
├── masks/
│   ├── rellis3d/
│   ├── rugd/
│   ├── ycor/
│   └── adom_v2/
├── splits/
│   ├── train.txt
│   ├── val.txt
│   ├── test.txt
│   ├── rugd_val_diagnostic.txt
│   ├── rugd_test_diagnostic.txt
│   ├── ycor_val_diagnostic.txt
│   ├── adom_v2_val_diagnostic.txt
│   └── adom_v2_test_diagnostic.txt
├── results/
│   ├── ...
│   ├── adom_v2_class_statistics.csv
│   ├── build_info.json
│   └── final_unified_check.json
└── manifest.csv
```

ADOM-v2를 아직 넣지 않았다면 `adom_v2/`와 ADOM-v2 diagnostic split은 생성되지 않습니다.

---

## 9. 스크립트가 내부적으로 하는 일

한 번 실행하면 아래 순서로 자동 처리합니다.

```text
1. RELLIS raw mask
   → 기존 rellis3d_semantic20_v1 mapping
   → 0..18 / 255 Semantic20 mask

2. RUGD
   → image / indexLabel 자동 탐색
   → 기존 train/val/test split 기준 staging
   → 기존 bridge_mapping.yaml로 Semantic20 변환

3. YCOR
   → RGB palette labels.png 판독
   → 기존 YCOR label_mapping.json + bridge mapping
   → Semantic20 변환

4. RELLIS + RUGD + YCOR 결합

5. 기존 3-source validator 실행

6. ADOM-v2가 있으면
   → image/mask pairing 검사
   → single-channel 검사
   → 0..18/255 class ID 검사
   → split 반영

7. 최종 전체 validation
   → image-mask 존재
   → 해상도 일치
   → class ID
   → duplicate sample_key
   → train/val/test overlap
   → non_ignore_ratio
```

---

## 10. 완료 확인

마지막에 다음과 같이 나오면 완료입니다.

```text
[PASS] Unified dataset build completed.
```

그리고:

```bash
cat ~/ADOM_data/processed/adom_semantic20_v2/results/final_unified_check.json
```

에서 최소한:

```json
{
  "status": "PASS"
}
```

를 확인합니다.

---

## 11. GitHub에 올릴 것 / 올리지 않을 것

### 올릴 것

```text
prepare_adom_semantic20_v2.py
README 또는 본 가이드
mapping/config 변경사항이 있다면 해당 config
최종 split 목록
작은 validation summary
```

### 올리지 않을 것

```text
원본 RGB
전체 mask
ZIP
processed 전체 dataset
checkpoint
ONNX
TensorRT engine
개인 PC 절대경로가 박힌 설정
```

데이터 자체는 기존 프로젝트 규칙처럼 Git 외부 저장소에서 관리하는 것이 맞습니다.

---

## 12. ADOM-v2가 완성된 뒤 수정할 부분

ADOM-v2 URL이 확정되어도 이 통합 스크립트의 핵심 코드는 바꿀 필요가 없습니다.

수정할 것은 사실상 두 가지입니다.

1. 가이드에 ADOM-v2 다운로드 URL 추가
2. 다운로드/압축 해제 후 아래 contract로 정리

```text
raw/ADOM-v2/images/train
raw/ADOM-v2/images/val
raw/ADOM-v2/images/test
raw/ADOM-v2/masks/train
raw/ADOM-v2/masks/val
raw/ADOM-v2/masks/test
```

만약 실제 ADOM-v2 배포 구조가 이와 다르면, `resolve_adom_pair_roots()`에 실제 구조 하나만 추가하면 됩니다.
