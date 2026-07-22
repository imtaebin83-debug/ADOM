# ADOM Dataset Preprocessing

이 저장소는 ADOM 프로젝트에서 모델 학습에 사용할 오프로드 데이터셋의 전처리 코드와 결과를 공유하기 위한 저장소입니다.

현재 포함하는 데이터셋은 다음과 같습니다.

- RELLIS-3D
- RUGD
- YCOR

전체 RGB 이미지와 전체 마스크는 GitHub에 직접 업로드하지 않습니다.  
GitHub에는 전처리 코드, 실제 매핑 기준, split, 통계, 검증 결과, 대표 preview만 저장합니다.  
모델 학습용 대용량 데이터는 팀 전용 Google Drive 또는 Shared Drive에서 별도 버전으로 관리합니다.

---

## 1. Repository Structure

```text
ADOM/
├── README.md
├── .gitignore
├── DATA_ACCESS.md
├── RELLIS-3D/
│   ├── README.md
│   ├── scripts/
│   ├── config/
│   ├── splits/
│   ├── results/
│   └── previews/
├── RUGD/
│   ├── README.md
│   ├── scripts/
│   ├── config/
│   ├── splits/
│   ├── results/
│   └── previews/
└── YCOR/
    ├── README.md
    ├── scripts/
    ├── config/
    ├── splits/
    ├── results/
    └── previews/
```

각 폴더의 의미는 다음과 같습니다.

| Folder | Contents |
|---|---|
| `scripts/` | 실제 최종 전처리에 사용한 Python 코드 |
| `config/` | 실제 변환에 사용한 source class → ADOM class 매핑 |
| `splits/` | `train.txt`, `val.txt`, `test.txt` 등의 학습 split |
| `results/` | class statistics, QC, final check, preprocessing summary |
| `previews/` | RGB·변환 마스크·overlay 대표 시각화 |

빈 템플릿이나 실제로 사용하지 않은 매핑 파일은 업로드하지 않습니다.

---

## 2. ADOM Cost4 Label Schema

| ID | Class | Description |
|---:|---|---|
| 0 | `paved_low_cost` | 포장도로 또는 안정적인 인공 지면 |
| 1 | `natural_low_cost` | 흙길, 짧은 풀 등 일반적인 저비용 오프로드 지면 |
| 2 | `medium_cost` | 진흙, 덤불, 물웅덩이 등 감속 또는 주의가 필요한 영역 |
| 3 | `high_cost_or_obstacle` | 물, 구조물, 사람, 차량, 통나무 등 회피 우선 영역 |
| 255 | `ignore` | 하늘, void, 불확실 영역 등 학습 제외 |

최종 학습 마스크 형식:

```text
Format: PNG
Channel: single-channel
Data type: uint8
Valid IDs: 0, 1, 2, 3, 255
Ignore index: 255
```

---

## 3. Dataset Contents

### 3.1 RELLIS-3D

```text
RELLIS-3D/
├── README.md
├── scripts/
│   ├── 실제 최종 전처리 스크립트
│   ├── preview 생성 스크립트
│   └── final check 생성 스크립트
├── config/
│   └── class_mapping.yaml
├── splits/
│   ├── train.txt
│   ├── val.txt
│   └── test.txt
├── results/
│   ├── class_statistics.csv
│   ├── qc_report.csv
│   └── final_check.txt
└── previews/
    ├── train_preview_*.png
    ├── val_preview_*.png
    └── test_preview_*.png
```

주의사항:

- 빈 `source_mapping.csv`는 업로드하지 않습니다.
- 실제 변환에 사용한 `class_mapping.yaml`만 유지합니다.
- `metadata.csv`는 실제 전처리에서 생성되지 않았다면 추가하지 않습니다.
- 별도 `preprocessing_summary.md`가 없어도 `RELLIS-3D/README.md`에서 과정을 설명하면 충분합니다.
- preview는 대표 샘플만 업로드합니다.

### 3.2 RUGD

```text
RUGD/
├── README.md
├── scripts/
│   ├── 실제 최종 전처리 스크립트
│   └── 09_final_check.py
├── config/
│   └── label_mapping.json
├── splits/
│   ├── train.txt
│   ├── val.txt
│   └── test.txt
├── results/
│   ├── class_statistics.csv 또는 class_statistics.json
│   ├── final_check.txt
│   └── preprocessing_summary.md
└── previews/
    └── 대표 preview가 있을 경우만 추가
```

확인된 결과:

```text
train: 4,779 image-mask pairs
val: 733 image-mask pairs
valid mask IDs: 0, 1, 2, 3, 255
```

주의사항:

- `label_mapping.json`은 기존 변환 코드 내부에서 실제 사용한 매핑을 추출해 작성합니다.
- 빈 `source_mapping.csv`는 업로드하지 않습니다.
- `dataset_summary.json`은 실제로 생성하지 않았다면 업로드하지 않습니다.
- `class_distribution.csv` 대신 실제 `class_statistics` 파일을 사용합니다.

### 3.3 YCOR

```text
YCOR/
├── README.md
├── scripts/
│   ├── common.py
│   ├── 01_check_raw_structure.py
│   ├── 02_build_manifest.py
│   ├── 03_scan_source_labels.py
│   ├── 04_validate_raw_pairs.py
│   ├── 05_convert_dataset.py
│   ├── 06_qc_statistics.py
│   ├── 07_make_previews.py
│   ├── 08_write_training_info.py
│   └── 09_final_check.py
├── config/
│   └── label_mapping.json
├── splits/
│   ├── train.txt
│   └── val.txt
├── results/
│   ├── preprocessing_summary.md
│   ├── raw_structure_check.txt
│   ├── final_check.txt
│   └── class_statistics 관련 실제 결과 파일
└── previews/
    ├── train_preview_*.png
    ├── val_preview_*.png
    └── legend.png
```

확인된 원본 구조:

```text
train sample folders: 931
validation sample folders: 145
total samples: 1,076
RGB file: rgb.jpg
annotation file: labels.png
source mask: RGB palette PNG
resolution: 1024 × 544
```

주의사항:

- `label_mapping.json`은 `common.py` 또는 `05_convert_dataset.py`의 실제 palette mapping을 정리한 파일이어야 합니다.
- 공식 test split이 없으면 `test.txt`와 test 폴더를 만들지 않습니다.
- `raw_structure_check.txt`와 `final_check.txt`는 VSCode 터미널에서 스크립트 출력을 저장해 생성합니다.

---

## 4. What Is Uploaded to GitHub

GitHub 업로드 대상:

- 최종 전처리 Python 코드
- 실제 사용한 label mapping
- train/val/test split 목록
- class statistics
- QC report
- final check 로그
- preprocessing summary
- 대표 preview
- 데이터 접근 안내

GitHub 제외 대상:

- 원본 데이터셋 전체
- 전체 processed RGB 이미지
- 전체 processed mask
- 다운로드 압축 파일
- 가상환경
- 캐시
- 디버깅용 임시 코드
- 대량 QC 이미지
- 모델 checkpoint와 학습 결과

---

## 5. Training Data Package

모델 학습용 데이터는 GitHub 외부에서 다음 구조로 관리합니다.

```text
ADOM_Datasets/
├── RELLIS-3D/
│   └── v1.0/
│       └── RELLIS3D_ADOM_Cost4_v1.0/
├── RUGD/
│   └── v1.0/
│       └── RUGD_ADOM_Cost4_v1.0/
└── YCOR/
    └── v1.0/
        └── YCOR_ADOM_Cost4_v1.0/
```

각 데이터 패키지의 권장 구조:

```text
<DATASET>_ADOM_Cost4_v1.0/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── annotations/
│   ├── train/
│   ├── val/
│   └── test/
├── splits/
│   ├── train.txt
│   ├── val.txt
│   └── test.txt
├── config/
│   └── label_mapping.json 또는 class_mapping.yaml
├── results/
│   ├── class_statistics.csv
│   └── final_check.txt
├── README_DATASET.md
└── SHA256SUMS.txt
```

YCOR처럼 test split이 없는 데이터셋은 `test/`와 `test.txt`를 생략합니다.

### 5.1 Full Package

원본 데이터셋 라이선스가 팀 내부 재배포를 허용한다면 다음을 함께 공유합니다.

```text
images/
annotations/
splits/
config/
results/
```

이 경우 팀원이 다운로드 후 바로 학습할 수 있습니다.

### 5.2 Masks-Only Package

원본 RGB 재배포가 금지되거나 불명확하다면 다음만 공유합니다.

```text
annotations/
splits/
config/
results/
source_download_instructions.md
```

팀원은 원본 RGB를 공식 데이터셋 경로에서 직접 내려받아야 합니다.

---

## 6. Data Access

학습 데이터 접근 경로는 [`DATA_ACCESS.md`](DATA_ACCESS.md)에서 관리합니다.

권장 방식:

- 팀 전용 Google Shared Drive 사용
- 팀원 Google 계정에만 다운로드 권한 부여
- 공개 링크 사용 금지
- 데이터 버전별 폴더 사용
- 같은 ZIP 파일명을 덮어쓰지 않음

버전 예시:

```text
v1.0: 최초 전처리 완료본
v1.1: split 또는 문서 수정
v1.2: mapping 수정
v2.0: 클래스 체계 또는 변환 로직 변경
```

---

## 7. Clone and Push Workflow

### 7.1 Team Repository Clone

기존 `C:\Users\gahyu\ADOM` 폴더가 Git 저장소가 아니라면 먼저 다른 이름으로 백업합니다.

```powershell
cd C:\Users\gahyu
Rename-Item ADOM ADOM_local_backup
```

팀 저장소를 `ADOM` 이름으로 clone합니다.

```powershell
git clone <TEAM_GITHUB_REPOSITORY_URL> C:\Users\gahyu\ADOM
cd C:\Users\gahyu\ADOM
```

작업 브랜치를 만듭니다.

```powershell
git checkout -b feature/dataset-preprocessing
```

백업 폴더에서 필요한 파일을 복사합니다.

```text
C:\Users\gahyu\ADOM_local_backup
                    ↓
C:\Users\gahyu\ADOM
```

복사 후 상태 확인:

```powershell
git status
```

### 7.2 Commit

```powershell
git add README.md .gitignore DATA_ACCESS.md
git add RELLIS-3D RUGD YCOR

git commit -m "docs(data): add preprocessing results for RELLIS-3D RUGD and YCOR"
```

### 7.3 Push

```powershell
git push -u origin feature/dataset-preprocessing
```

그다음 GitHub에서 팀 저장소의 기본 브랜치로 Pull Request를 생성합니다.

직접 `main`에 push하기보다 작업 브랜치와 Pull Request를 사용하는 것을 권장합니다.

---

## 8. Final Check Before Push

```powershell
git status
git diff --cached --stat
git remote -v
git branch --show-current
```

반드시 확인할 항목:

- 원본 데이터가 포함되지 않았는가
- 전체 mask 폴더가 포함되지 않았는가
- `.venv`가 포함되지 않았는가
- 빈 mapping 템플릿이 포함되지 않았는가
- 로컬 절대경로가 README 외 코드에 하드코딩되지 않았는가
- 실제 전처리에 사용한 최종 스크립트만 포함되었는가
- Drive 접근 권한이 팀원으로 제한되었는가
