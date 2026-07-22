# YCOR Preprocessing Summary

## 1. Purpose

YCOR/Yamaha 원본 RGB 이미지와 palette 기반 semantic mask를
ADOM Cost4 single-channel indexed mask로 변환하여
SegFormer 학습용 데이터 구조로 구성하였다.

## 2. Raw Dataset

- Train sample folders: 931
- Validation sample folders: 145
- Total samples: 1,076
- RGB filename: rgb.jpg
- Source mask filename: labels.png
- Source mask format: RGB palette PNG
- Resolution: 1024 × 544

## 3. Final Scripts

- `01_check_raw_structure.py`
- `02_build_manifest.py`
- `03_scan_source_labels.py`
- `04_validate_raw_pairs.py`
- `05_convert_dataset.py`
- `06_qc_statistics.py`
- `07_make_previews.py`
- `08_write_training_info.py`
- `09_final_check.py`
- `common.py`

## 4. Preprocessing Process

1. 원본 train/valid 구조 검사
2. RGB와 labels.png pair 검사
3. source palette RGB 값 조사
4. palette RGB를 ADOM Cost4 ID로 변환
5. single-channel indexed PNG mask 생성
6. train/val 학습 구조 생성
7. class statistics 및 QC 결과 생성
8. 대표 preview 생성
9. 최종 image-mask pair와 mask ID 검사

## 5. Target Mask Format

- Format: single-channel indexed PNG
- Data type: uint8
- Valid IDs: 0, 1, 2, 3, 255
- Ignore index: 255

## 6. Raw Structure Check

~~~~text
[dataset root] C:\Users\gahyu\YCOR\raw\yamaha_v0
[train] source='train', sample folders=931
[val] source='valid', sample folders=145
[sample image] C:\Users\gahyu\YCOR\raw\yamaha_v0\train\iid000000\rgb.jpg
[sample mask]  C:\Users\gahyu\YCOR\raw\yamaha_v0\train\iid000000\labels.png
[mask encoding] rgb_palette
[mask shape]    (544, 1024, 3)
[all samples]   1,076

01_check_raw_structure.py: PASS

~~~~

## 7. Training Information

~~~~text
[saved] C:\Users\gahyu\YCOR\processed\YCOR_ADOM\dataset_info.json
[saved] C:\Users\gahyu\YCOR\processed\YCOR_ADOM\mmseg_dataset_snippet.py
08_write_training_info.py: PASS

~~~~

## 8. Final Validation

~~~~text

[train]
이미지 수: 931
마스크 수: 931
메타데이터 수: 931
공식 예상 수: 931
사용된 ID: [1, 2, 3, 255]
누락 마스크: 0
누락 이미지: 0
메타데이터 불일치: 0
크기 불일치: 0
잘못된 ID: []

[val]
이미지 수: 145
마스크 수: 145
메타데이터 수: 145
공식 예상 수: 145
사용된 ID: [1, 2, 3, 255]
누락 마스크: 0
누락 이미지: 0
메타데이터 불일치: 0
크기 불일치: 0
잘못된 ID: []

참고: YCOR에는 포장도로 클래스가 없어 ID 0이 없어도 정상입니다.
공식 test split은 없으며 train/val만 생성했습니다.
09_final_check.py: PASS
학습용 데이터 루트: C:\Users\gahyu\YCOR\processed\YCOR_ADOM

~~~~

## 9. Repository Scope

GitHub에는 최종 전처리 코드, palette mapping, 통계,
검증 로그와 representative preview를 업로드한다.

전체 RGB 이미지와 변환 mask는 팀 전용 데이터 저장소에서 별도 배포한다.
