# YCOR Preprocessing Summary

## 1. Purpose

YCOR/Yamaha ?? RGB ???? palette ?? semantic mask?
ADOM Cost4 single-channel indexed mask? ????
SegFormer ??? ??? ??? ?????.

## 2. Raw Dataset

- Train sample folders: 931
- Validation sample folders: 145
- Total samples: 1,076
- RGB filename: rgb.jpg
- Source mask filename: labels.png
- Source mask format: RGB palette PNG
- Resolution: 1024 ? 544
- Official test split: ??

## 3. Final Scripts

- 01_check_raw_structure.py
- 02_build_manifest.py
- 03_scan_source_labels.py
- 04_validate_raw_pairs.py
- 05_convert_dataset.py
- 06_qc_statistics.py
- 07_make_previews.py
- 08_write_training_info.py
- 09_final_check.py
- common.py

## 4. Preprocessing Process

1. ?? train/valid ??? ?? image-mask pair ??
2. dataset root ?? ???? manifest ??
3. source palette RGB ?? unknown label ??
4. ?? raw image-mask ?? ? ?? ?? ?? ??
5. palette RGB? ADOM Cost4 ID? ??
6. single-channel indexed PNG mask ??
7. train/val image-mask-metadata ?? ??
8. class statistics ? per-image QC ??
9. representative preview ??
10. ?? ??? ?? image-mask-metadata ??

## 5. Target Mask Format

- Format: single-channel indexed PNG
- Data type: uint8
- Valid IDs: 0, 1, 2, 3, 255
- Ignore index: 255
- reduce_zero_label: False
- Number of model classes: 4

YCOR?? ???? source class? ????
target ID 0? pixel ?? 0? ?? ????.

## 6. Reproducibility

- ?? dataset ??? CLI ??? ??? ? ??.
- manifest? metadata?? dataset root ?? ????? ????.
- processed output ??? CLI ??? ??? ? ??.
- mapping ?? ??? ??? ?????? ??? ? ??.
- ?? PC ????? ?? ??? ???? ???.
- ?? train 931?, val 145?? ??? ??? ?? ????.

## 7. Raw Structure Check

~~~~text
[dataset root] <YCOR_DATASET_ROOT>
[train] source='train', sample folders=931, pairs=931
[val] source='valid', sample folders=145, pairs=145
[sample image] train/iid000000/rgb.jpg
[sample mask]  train/iid000000/labels.png
[mask encoding] rgb_palette
[mask shape]    (544, 1024, 3)
[all samples]   1,076

01_check_raw_structure.py: PASS
~~~~

## 8. Training Information

~~~~text
[saved] dataset_info.json
[saved] mmseg_dataset_snippet.py
08_write_training_info.py: PASS
~~~~

## 9. Final Validation

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
메타데이터 중복 ID: 0
메타데이터 절대경로: 0
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
메타데이터 중복 ID: 0
메타데이터 절대경로: 0
크기 불일치: 0
잘못된 ID: []

참고: YCOR에는 포장도로 클래스가 없어 ID 0이 없어도 정상입니다.
공식 test split은 없으며 train/val만 생성했습니다.
09_final_check.py: PASS
학습용 데이터 루트: <YCOR_PROCESSED_ROOT>
~~~~

## 10. Repository Scope

GitHub?? ?? ??? ??, label mapping, split metadata,
??, ?? ??? representative preview? ?????.

?? ?? RGB ???, ?? ?? mask? ?? ???
processed dataset? GitHub? ????? ???.
