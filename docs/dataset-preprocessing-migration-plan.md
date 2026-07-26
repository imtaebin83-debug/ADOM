# Dataset Preprocessing Migration Plan

## 목적

`study/gahyung/Datasets_Repo`에 있는 개인 작업 결과를 팀 공용 파이프라인으로
승격한다. 이동 후에는 로컬 Windows 절대경로 없이 RunPod와 Docker Compose에서
RELLIS-3D, RUGD, YCOR 전처리를 동일한 방식으로 재현할 수 있어야 한다.

이 문서는 이동 계획이며 실제 raw/processed 데이터는 Git에 포함하지 않는다.

## 최종 배치

```text
scripts/data_preprocessing/
├── rellis3d/
│   ├── inspect_raw.py
│   ├── convert_masks.py
│   ├── make_splits.py
│   └── validate.py
├── rugd/
│   ├── inspect_raw.py
│   ├── convert_masks.py
│   ├── make_splits.py
│   └── validate.py
└── ycor/
    ├── common.py
    ├── inspect_raw.py
    ├── convert_masks.py
    ├── make_splits.py
    └── validate.py

configs/datasets/
├── rellis3d/
│   └── cost4_mapping.yaml
├── rugd/
│   └── cost4_mapping.json
└── ycor/
    └── cost4_mapping.json

data/
├── splits/<dataset>/
├── manifests/<dataset>/
└── external/<dataset> -> /workspace/adom/datasets/<dataset>

results/datasets/<dataset>/
├── class_statistics.csv
├── qc_report.csv
├── final_check.txt
└── previews/

docs/datasets/
├── data-access.md
├── rellis3d.md
├── rugd.md
└── ycor.md
```

`study/gahyung`에는 개인 학습 노트와 작은 실험만 남긴다. 동일한 전처리 코드의
복사본을 `study`와 `scripts`에 동시에 유지하지 않는다.

## Runtime 경로 계약

`init_workspace.sh`와 Compose가 제공하는 입력 경로는 다음으로 고정한다.

```text
/workspace/adom/datasets/rellis3d/raw
/workspace/adom/datasets/rugd/raw
/workspace/adom/datasets/ycor/raw
```

전처리 산출물은 다음에 저장한다.

```text
/workspace/adom/outputs/preprocessing/rellis3d
/workspace/adom/outputs/preprocessing/rugd
/workspace/adom/outputs/preprocessing/ycor
```

공용 스크립트는 최소한 다음 인자를 받아야 한다.

```text
--input-root
--output-root
--mapping
```

선택 인자는 데이터셋에 따라 `--split-root`, `--workers`, `--overwrite` 등을
추가할 수 있다. 기본값을 제공하더라도 개인 홈 디렉토리를 포함하면 안 된다.

## 단계별 작업

### 1. 누락 원본 복구

- 팀원의 원래 저장소에서 YCOR `common.py`를 복구한다.
- 실제 전처리에 사용한 commit SHA와 mapping 파일을 확인한다.
- 누락 파일 내용을 추측하거나 새로 생성하지 않는다.

### 2. 순수 파일 이동

별도 브랜치에서 `git mv`만 수행하고 첫 커밋으로 분리한다.

```text
refactor(data): move shared preprocessing files out of study
```

이 커밋에서는 로직, mapping, 생성 결과를 수정하지 않는다. Git rename 추적과
이후 리뷰를 쉽게 하기 위한 단계다.

### 3. 경로 및 설정 리팩터링

- `C:\Users\...`와 다른 개인 절대경로를 제거한다.
- 입력/출력/mapping 경로를 CLI 인자로 받는다.
- mapping은 `configs/datasets/<dataset>`의 파일만 source of truth로 사용한다.
- RUGD처럼 Python과 JSON에 mapping을 중복 정의하지 않는다.
- manifest에는 dataset root 기준 상대경로만 기록한다.

권장 커밋:

```text
fix(data): parameterize preprocessing paths and restore missing modules
```

### 4. 검증을 fail-closed로 변경

다음 중 하나라도 발생하면 non-zero exit code로 종료한다.

- 처리한 sample이 0개
- RGB-only 또는 mask-only sample 존재
- 중복 sample ID 또는 split overlap
- 이미지와 mask 크기 불일치
- mapping되지 않은 source label
- 허용되지 않은 target ID
- 필수 QC 컬럼 누락
- 변환 예외 또는 빈 QC report

검증 스크립트와 final check는 동일한 QC schema를 사용해야 한다. 검사 항목을
찾지 못한 경우 PASS로 처리하지 않는다.

권장 커밋:

```text
fix(data): make preprocessing validation fail closed
```

### 5. 작은 synthetic test 추가

실제 데이터셋 대신 직접 생성한 소형 RGB/mask fixture로 다음을 검사한다.

- source ID 또는 palette RGB가 올바른 Cost4 ID로 변환됨
- `255` ignore index 보존
- unknown label에서 실패
- 단일 채널 `uint8` PNG 생성
- pair 누락과 split overlap 탐지

fixture에는 원본 데이터셋 이미지나 annotation을 사용하지 않는다.

### 6. 산출물 재생성

수정된 코드로 split, manifest, class statistics, QC, final check를 다시 생성한다.
기존 결과 파일을 손으로 고치지 않는다. Git에 반영하기 전 다음을 제거한다.

- 개인 PC 절대경로
- raw/processed 파일
- source dataset 재배포가 제한된 이미지
- checkpoint, ONNX, TensorRT artifact

preview는 라이선스를 확인하고 대표 샘플만 유지한다.

## 담당 범위

### DevOps 담당

- 최종 디렉토리 생성과 순수 `git mv`
- `init_workspace.sh`, Compose, ignore 및 CI artifact guard
- RunPod mount와 cache 경로 검증

### 데이터 전처리 담당

- 누락 파일 복구
- mapping과 변환 로직 확인
- CLI 경로 인자화
- 데이터셋별 결과 재생성

### 리뷰어

- split 중복/교차 확인
- QC가 실제 오류에서 실패하는지 확인
- 추적 파일에 raw 데이터와 로컬 절대경로가 없는지 확인

## 하지 말아야 할 작업

- `main`에 직접 push하거나 force push하지 않는다.
- `git add .`로 검토되지 않은 파일을 일괄 stage하지 않는다.
- raw RGB, 전체 mask, archive를 Git에 추가하지 않는다.
- `.pth`, `.pt`, `.onnx`, `.engine`, `.plan`을 Git에 추가하지 않는다.
- 누락된 코드를 추측해 복원하지 않는다.
- mapping을 합의 없이 변경하지 않는다.
- 기존 QC/final check를 손으로 PASS로 수정하지 않는다.
- `study`와 공용 경로에 같은 코드 복사본을 남기지 않는다.

## 완료 조건

- 세 데이터셋이 `/workspace/adom/datasets/<dataset>/raw`에서 실행된다.
- 전처리 결과가 `/workspace/adom/outputs/preprocessing/<dataset>`에 생성된다.
- 코드와 추적 CSV/TXT/MD에 개인 절대경로가 없다.
- mapping 파일과 변환 코드가 일치한다.
- synthetic test가 통과한다.
- 오류 fixture에서 validator가 non-zero로 실패한다.
- split에 중복과 교차가 없다.
- Git 추적 파일에 raw/processed/model artifact가 없다.
- Pull Request에 실행 명령, 검증 결과, 제한 사항이 기록된다.
