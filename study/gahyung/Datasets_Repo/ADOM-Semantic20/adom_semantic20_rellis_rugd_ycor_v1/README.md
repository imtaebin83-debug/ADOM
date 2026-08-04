# ADOM Semantic20 RELLIS-RUGD-YCOR v1

## 1. 목적

RELLIS-3D Semantic20을 기준 class 체계로 사용하고, RUGD와 YCOR에서 의미가 명확하게 대응되는 class만 추가한 통합 semantic segmentation 학습 package이다.

저장소에는 다음 항목만 포함한다.

- class mapping 설정
- 변환 및 검증 스크립트
- manifest
- train, validation, test split
- class 통계
- 최종 검증 결과

원본 RGB 이미지와 전체 mask는 Git 저장소에 포함하지 않는다.

## 2. Target label specification

- num_classes: 19
- ignore_index: 255
- reduce_zero_label: false

| ID | Class |
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

## 3. RUGD bridge mapping

다음 RUGD class를 Semantic20 target으로 사용한다.

| RUGD class | Target ID | Target class |
|---|---:|---|
| grass | 1 | grass |
| tree | 2 | tree |
| water | 4 | water |
| sky | 5 | sky |
| asphalt | 8 | asphalt |
| building | 9 | building |
| person | 11 | person |
| bush | 13 | bush |
| rock-bed | 18 | rubble |

다음 class는 의미 대응이 불명확하여 ignore로 처리한다.

- mulch
- gravel
- sign

## 4. YCOR bridge mapping

다음 YCOR class를 Semantic20 target으로 사용한다.

| YCOR class | Target ID | Target class |
|---|---:|---|
| traversable_grass | 1 | grass |
| puddle | 16 | puddle |

다음 class는 v1에서 ignore로 처리한다.

- background_or_unlabelled
- high_vegetation
- smooth_trail
- obstacle
- sky
- rough_trail
- non_traversable_low_vegetation

YCOR mask는 PNG palette index를 semantic ID로 직접 해석하지 않고, label_mapping.json의 RGB palette를 기준으로 source class를 판별한다.

Puddle이 존재하는 sample은 non-ignore pixel 비율이 1% 미만이어도 보존한다.

## 5. Dataset counts

### 전체 manifest

| Source | Count |
|---|---:|
| RELLIS-3D | 6,234 |
| RUGD | 7,436 |
| YCOR | 751 |
| Total | 14,421 |

### Main split

| Split | Count | 구성 |
|---|---:|---|
| train | 9,868 | RELLIS train + RUGD train + YCOR train |
| val | 900 | RELLIS val only |
| test | 899 | RELLIS test only |

### Main train source 구성

| Source | Count |
|---|---:|
| RELLIS-3D | 4,435 |
| RUGD | 4,779 |
| YCOR | 654 |

RUGD val/test와 YCOR valid는 main 평가 split에 포함하지 않고 source-specific diagnostic split으로 별도 보관한다.

## 6. 파일 구성

- config/bridge_mapping.yaml
  - RUGD와 YCOR source class를 Semantic20 target ID로 연결한다.

- scripts/01_convert_bridge_sources.py
  - RUGD와 YCOR mask를 Semantic20 ID로 변환한다.
  - image/mask pair, source ID, target ID를 검사한다.
  - manifest와 source split을 생성한다.

- scripts/02_audit_ycor_rgb_distribution.py
  - YCOR palette RGB를 기준으로 원본 class 분포를 계산한다.

- scripts/03_build_combined_package.py
  - 기존 RELLIS Semantic20 package와 RUGD/YCOR bridge 결과를 결합한다.
  - 최종 manifest와 main split 및 diagnostic split을 생성한다.

- scripts/04_validate_combined_package.py
  - 전체 image/mask pair와 mask ID, split 중복, 평가 정책을 검증한다.

- manifest.csv
  - 전체 14,421개 sample의 source, split, image 경로, mask 경로를 기록한다.

- splits/
  - main train, val, test split
  - RUGD 및 YCOR source-specific diagnostic split

- results/
  - source class 통계
  - 변환 결과 요약
  - 통합 package 요약
  - 최종 검증 결과

## 7. 실행 순서

1. RUGD 및 YCOR bridge mask 변환

python scripts/01_convert_bridge_sources.py --mapping config/bridge_mapping.yaml --rugd-image-root RUGD_IMAGE_ROOT --rugd-mask-root RUGD_MASK_ROOT --rugd-split-root RUGD_SPLIT_ROOT --ycor-root YCOR_ROOT --ycor-source-map YCOR_SOURCE_MAPPING --output-root OUTPUT_ROOT --min-non-ignore-ratio 0.01 --overwrite

2. YCOR RGB class 분포 검사

python scripts/02_audit_ycor_rgb_distribution.py --ycor-root YCOR_ROOT --source-map YCOR_SOURCE_MAPPING --output results/ycor_source_class_distribution_rgb.csv

3. RELLIS와 bridge source 결합

python scripts/03_build_combined_package.py --rellis-root RELLIS_SEMANTIC20_ROOT --output-root OUTPUT_ROOT --overwrite-rellis

4. 최종 package 검증

python scripts/04_validate_combined_package.py --output-root OUTPUT_ROOT

실제 경로는 명령행 인자로 전달하며, 스크립트와 설정 파일에는 개인 PC 절대경로를 저장하지 않는다.

## 8. 검증 결과

최종 package에서 다음 항목을 확인했다.

- manifest sample: 14,421개
- RELLIS-3D sample: 6,234개
- RUGD sample: 7,436개
- YCOR sample: 751개
- train: 9,868개
- val: 900개
- test: 899개
- val/test의 non-RELLIS sample: 0개
- RUGD unexpected target ID: 없음
- YCOR unexpected target ID: 없음
- YCOR puddle 포함 image: 44개 전부 보존
- YCOR puddle pixel: 1,064,853개 전부 보존
- 전체 mask: single-channel PNG
- 원본 RGB 및 전체 mask: Git 미포함
- 모델 artifact: Git 미포함

## 9. 주의사항

- label mapping은 bridge_mapping.yaml을 기준으로 한다.
- mapping을 임의로 변경하지 않는다.
- val/test의 RELLIS-only 정책을 유지한다.
- final_check.json과 통계 파일은 수동으로 수정하지 않는다.
- 결과 변경이 필요한 경우 수정된 스크립트를 다시 실행하여 재생성한다.
- 원본 RGB, 전체 mask, 압축 데이터셋과 모델 artifact는 Git에 추가하지 않는다.
