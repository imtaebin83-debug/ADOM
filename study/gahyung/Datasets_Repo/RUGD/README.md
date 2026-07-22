# RUGD Preprocessing

RUGD RGB와 semantic ID mask를 ADOM Cost4 indexed mask로 변환한 과정과 결과를 저장합니다.

## 확인된 결과

- train: 4,779 image-mask pairs
- validation: 733 image-mask pairs
- valid mask IDs: `0, 1, 2, 3, 255`

## 실제 업로드 항목

- 최종 전처리 스크립트
- 실제 코드에서 추출한 `config/label_mapping.json`
- train/val/test split
- 실제 `class_statistics` 파일
- `final_check.txt`
- `preprocessing_summary.md`
