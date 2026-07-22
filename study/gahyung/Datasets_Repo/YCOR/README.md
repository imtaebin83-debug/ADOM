# YCOR Preprocessing

YCOR/Yamaha RGB와 RGB palette annotation을 ADOM Cost4 indexed mask로 변환한 과정과 결과를 저장합니다.

## 확인된 원본 구조

- train: 931
- validation: 145
- total: 1,076
- RGB: `rgb.jpg`
- source mask: `labels.png`
- resolution: 1024 × 544

## 실제 업로드 항목

- `common.py`
- `01_check_raw_structure.py`부터 `09_final_check.py`
- 실제 palette mapping을 정리한 `config/label_mapping.json`
- train/val split
- `raw_structure_check.txt`
- `final_check.txt`
- `preprocessing_summary.md`
- 실제 class statistics
- 대표 preview
