# RUGD Preprocessing Summary

## 1. Purpose

RUGD 원본 RGB 이미지와 semantic ID mask를 ADOM Cost4 라벨 체계로 변환하여
SegFormer 기반 주행가능성 분할 모델 학습용 데이터로 구성하였다.

## 2. Final Scripts

- `01_inspect_dataset.py.py`
- `02_image_mask.py`
- `03_check_mask.py`
- `03_from_pathlib_import_Path.py`
- `03b_verify_index_color_mapping.py`
- `04_remap_rugd.py`
- `05_validate_processed.py`
- `06_make_splits.py`
- `07_generate_overlays.py`
- `08_generate_statistics.py`
- `09_final_check.py`
- `final_image_mask_check.py`
- `from pathlib import Path.py`

## 3. Target Mask Format

- Format: single-channel indexed PNG
- Data type: uint8
- Valid IDs: 0, 1, 2, 3, 255
- Ignore index: 255

## 4. Data Split

- Train: 4,779 image-mask pairs
- Validation: 733 image-mask pairs
- Test: final check 결과 참조

## 5. Class Statistics

클래스별 픽셀 수와 비율은 results/class_statistics 파일에 기록하였다.

## 6. Final Validation

~~~~text

[train]
이미지 수: 4779
마스크 수: 4779
사용된 ID: [0, 1, 2, 3, 255]

[val]
이미지 수: 733
마스크 수: 733
사용된 ID: [0, 1, 2, 3, 255]

[test]
이미지 수: 1924
마스크 수: 1924
사용된 ID: [0, 1, 2, 3, 255]

전체 파일 수: 7436

최종 검사 통과
RUGD 전처리 데이터가 준비되었습니다.

~~~~

## 7. Repository Scope

GitHub에는 최종 전처리 코드, label mapping, split, class statistics,
검증 로그를 업로드한다.

전체 RGB 이미지와 ADOM Cost4 mask는 GitHub에 포함하지 않고
팀 전용 데이터 저장소에서 별도로 배포한다.
