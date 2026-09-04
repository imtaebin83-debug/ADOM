# Self-contained ADOM Semantic20 통합 전처리

이 버전은 **ADOM GitHub 내부의 기존 전처리 Python/YAML/JSON/split TXT를 전혀 호출하지 않는 독립 실행 버전**입니다.

필요한 것은 다음뿐입니다.

1. `prepare_adom_semantic20_standalone.py`
2. 사용자가 각 공식 배포처에서 직접 받은 원본 `RELLIS-3D`, `RUGD`, `YCOR`
3. 나중에 추가할 `ADOM-v2`
4. Python 패키지 `numpy`, `Pillow`

> 여기서 "외부 의존성 제거"는 팀 GitHub의 기존 전처리 코드/설정 파일 의존성을 제거했다는 뜻입니다.  
> 이미지와 배열 처리를 위해 `numpy`와 `Pillow`는 설치해야 합니다.

---

## 1. 폴더 배치

예:

```text
~/ADOM_data/
├── RELLIS-3D/
├── RUGD/
├── YCOR/
└── ADOM-v2/     # 아직 없으면 생략 가능
```

또는 기존처럼:

```text
~/ADOM_data/
└── raw/
    ├── RELLIS-3D/
    ├── RUGD/
    ├── YCOR/
    └── ADOM-v2/
```

두 구조 모두 자동 인식합니다.

### RELLIS-3D

```text
RELLIS-3D/
├── 00000/
│   ├── pylon_camera_node/
│   └── pylon_camera_node_label_id/
├── 00001/
├── 00002/
├── 00003/
└── 00004/
```

split은 코드 내부에서 다음처럼 결정합니다.

```text
00000, 00001, 00002 -> train
00003               -> val
00004               -> test
```

### RUGD

기본적으로 아래 이름의 폴더를 재귀적으로 찾습니다.

```text
image/
indexLabel/
```

또한 일부 repackaged 구조의 `RUGD_frames-with-annotations`, `RUGD_annotations`도 fallback으로 탐색합니다.

RUGD split 역시 별도 TXT 없이 sequence 이름으로 자동 결정합니다.

```text
train:
  park-2
  trail
  trail-3
  trail-4
  trail-6
  trail-9
  trail-10
  trail-11
  trail-12
  trail-14
  trail-15
  village

val:
  park-8
  trail-5

test:
  creek
  park-1
  trail-7
  trail-13
```

### YCOR

```text
YCOR/
├── train/
│   └── <sample>/
│       ├── rgb.jpg
│       └── labels.png
└── valid/
    └── <sample>/
        ├── rgb.jpg
        └── labels.png
```

### ADOM-v2

현재 코드는 다음 구조를 input contract로 사용합니다.

```text
ADOM-v2/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── masks/
    ├── train/
    ├── val/
    └── test/
```

mask는 single-channel class-ID PNG이고 값은 `0..18, 255`여야 합니다.

---

## 2. 설치

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install numpy Pillow
```

---

## 3. ADOM-v2 없이 현재 실행

```bash
python prepare_adom_semantic20_standalone.py \
  --data-root ~/ADOM_data \
  --output-root ~/ADOM_data/processed/adom_semantic20 \
  --skip-adom-v2 \
  --overwrite
```

---

## 4. ADOM-v2까지 준비된 후

`~/ADOM_data/ADOM-v2` 또는 `~/ADOM_data/raw/ADOM-v2`에 넣었다면:

```bash
python prepare_adom_semantic20_standalone.py \
  --data-root ~/ADOM_data \
  --output-root ~/ADOM_data/processed/adom_semantic20 \
  --adom-eval-policy diagnostic \
  --overwrite
```

다른 경로라면:

```bash
python prepare_adom_semantic20_standalone.py \
  --data-root ~/ADOM_data \
  --adom-v2-root /path/to/ADOM-v2 \
  --output-root ~/ADOM_data/processed/adom_semantic20 \
  --adom-eval-policy diagnostic \
  --overwrite
```

---

## 5. 내장된 Semantic20 mapping

최종 mask:

```text
0  dirt
1  grass
2  tree
3  pole
4  water
5  sky
6  vehicle
7  object
8  asphalt
9  building
10 log
11 person
12 fence
13 bush
14 concrete
15 barrier
16 puddle
17 mud
18 rubble
255 ignore
```

### RELLIS

기존 프로젝트의 `rellis_to_target` mapping을 코드 안에 그대로 포함했습니다.

### RUGD

기존 프로젝트 bridge 정책을 유지합니다.

사용:

```text
grass     -> 1
tree      -> 2
water     -> 4
sky       -> 5
asphalt   -> 8
building  -> 9
person    -> 11
bush      -> 13
rock-bed  -> 18 (rubble)
```

그 외 RUGD class는 `255`로 처리합니다.

특히 RUGD는 palette-index 값이 배포본에 따라 class 의미와 직접 일치하지 않을 수 있으므로,
`P/RGB/RGBA` mask는 **palette RGB -> class 의미 -> Semantic20** 순서로 변환합니다.

숫자 ID만 들어있는 grayscale mask도 다음을 지원합니다.

```text
--rugd-index-scheme auto
--rugd-index-scheme legacy
--rugd-index-scheme official
--rugd-index-scheme compact
```

일반적으로 `auto`를 사용합니다.

### YCOR

RGB palette를 코드 내부에 모두 포함했습니다.

```text
RGB
 -> YCOR source class
 -> Semantic20
```

사용되는 class는:

```text
traversable_grass -> 1 grass
puddle            -> 16 puddle
```

나머지는 v1 bridge 정책에 따라 `255`입니다.

Puddle이 존재하는 YCOR sample은 non-ignore 비율이 1% 미만이어도 보존합니다.

---

## 6. main split 정책

기본 `diagnostic` 정책:

```text
train
 = RELLIS train
 + RUGD train
 + YCOR train
 + ADOM-v2 train (존재 시)

val
 = RELLIS val only

test
 = RELLIS test only
```

다음은 diagnostic split으로 별도 저장됩니다.

```text
RUGD val
RUGD test
YCOR valid
ADOM-v2 val
ADOM-v2 test
```

ADOM-v2까지 main val/test에 합치고 싶다면:

```bash
--adom-eval-policy mixed
```

---

## 7. 출력

```text
OUTPUT_ROOT/
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
│   ├── target_class_statistics.csv
│   ├── build_summary.json
│   └── final_check.json
└── manifest.csv
```

ADOM-v2가 없으면 관련 폴더는 비어 있고 diagnostic TXT는 생성되지 않습니다.

---

## 8. 완료 판정

마지막 출력:

```text
[PASS] Unified standalone preprocessing completed.
```

그리고:

```bash
cat OUTPUT_ROOT/results/final_check.json
```

에서:

```json
{
  "status": "PASS"
}
```

를 확인합니다.

---

## 9. GitHub에 공개할 것

공개:

```text
prepare_adom_semantic20_standalone.py
README
requirements_standalone.txt
```

공개하지 않음:

```text
RELLIS-3D 원본/가공 데이터
RUGD 원본/가공 데이터
YCOR 원본/가공 데이터
통합된 전체 images/masks
원본 ZIP
모델 artifact
```

이 standalone 파일은 실행 시 인터넷이나 ADOM GitHub repository를 참조하지 않습니다.
