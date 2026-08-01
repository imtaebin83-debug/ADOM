# RUGD Preprocessing Summary

## 1. Purpose

RUGD 원본 RGB 이미지와 semantic color mask를 ADOM Cost4 라벨 체계로 변환하여
SegFormer 기반 주행가능성 분할 모델 학습용 데이터로 구성하였다.

## 2. Final Scripts

- `01_inspect_dataset.py`
- `02_image_mask.py`
- `03_verify_index_color_mapping.py`
- `04_remap_rugd.py`
- `05_validate_processed.py`
- `06_make_splits.py`
- `07_generate_overlays.py`
- `08_generate_statistics.py`
- `09_final_check.py`

## 3. Target Mask Format

- Format: single-channel indexed PNG
- Data type: uint8
- Valid IDs: 0, 1, 2, 3, 255
- Ignore index: 255

### ADOM Cost4 Labels

- `0`: paved_low_cost
- `1`: natural_low_cost
- `2`: medium_cost
- `3`: high_cost_or_obstacle
- `255`: ignore

## 4. Full Dataset Result

- Total image-mask pairs: 7,436
- Converted images: 7,436
- Converted masks: 7,436
- Conversion failures: 0
- Metadata rows: 7,436
- QC rows: 7,436
- QC status `ok`: 7,436

## 5. Data Split

- Train: 4,779 image-mask pairs
- Validation: 733 image-mask pairs
- Test: 1,924 image-mask pairs
- Total assigned samples: 7,436
- Unassigned samples: 0
- Split overlap: 0

The sequence-based split policy is implemented in `06_make_splits.py`.

The generated split manifests are stored in:

- `splits/train.txt`
- `splits/val.txt`
- `splits/test.txt`

## 6. Mapping Verification

`03_verify_index_color_mapping.py` validates:

- index and color mask filename pairing
- image readability and mask dimensions
- RGB coverage in `config/label_mapping.json`
- consistency among `RGB_TO_NAME`, `RUGD_TO_ADOM`, and `RGB_TO_ADOM`

The index values are not treated as global semantic class IDs.
The script records the observed index-RGB joint distribution.

## 7. Overlay QC

Overlay previews were generated with alpha `0.45`.

- Train overlays: 100
- Validation overlays: 100
- Test overlays: 100
- Total overlays: 300

Overlay images are visual QC artifacts and are not used directly for training.

## 8. Class Statistics

Class statistics were generated for train, validation, and test masks.

- `results/class_statistics.json`
- `results/class_statistics.csv`

The statistics contain:

- class pixel counts
- percentage of all pixels
- percentage excluding ignore pixels
- image count containing each class

## 9. Final Validation

```text
RUGD FINAL CHECK
status=PASS
total_samples=7436
expected_total=7436
metadata_rows=7436
qc_rows=7436
split_overlap=0
unassigned_samples=0
used_ids=0,1,2,3,255
statistics_status=PASS
overlay_status=PASS
train_samples=4779
train_images=4779
train_masks=4779
train_overlays=100
val_samples=733
val_images=733
val_masks=733
val_overlays=100
test_samples=1924
test_images=1924
test_masks=1924
test_overlays=100
final_status=PASS
```

Machine-readable validation results:

- `results/qc_report.csv`
- `results/final_check.txt`

## 10. Reproduction Commands

Run the following commands from the `RUGD` directory.

### Inspect the source structure

```powershell
python .\scripts\01_inspect_dataset.py `
    --input-root "<RUGD_INPUT_ROOT>"
```

### Verify source image-mask pairs

```powershell
python .\scripts\02_image_mask.py `
    --input-root "<RUGD_INPUT_ROOT>"
```

### Verify index and RGB mapping

```powershell
python .\scripts\03_verify_index_color_mapping.py `
    --input-root "<RUGD_INPUT_ROOT>" `
    --mapping ".\config\label_mapping.json"
```

### Convert masks to ADOM Cost4

```powershell
python .\scripts\04_remap_rugd.py `
    --input-root "<RUGD_INPUT_ROOT>" `
    --output-root "<RUGD_PROCESSED_ROOT>" `
    --mapping ".\config\label_mapping.json"
```

### Validate processed data

```powershell
python .\scripts\05_validate_processed.py `
    --processed-root "<RUGD_PROCESSED_ROOT>" `
    --metadata "<RUGD_PROCESSED_ROOT>\metadata.csv" `
    --results-dir "<RUGD_RESULTS_ROOT>"
```

### Generate train, validation, and test splits

```powershell
python .\scripts\06_make_splits.py `
    --processed-root "<RUGD_PROCESSED_ROOT>" `
    --split-output-root "<RUGD_SPLIT_ROOT>"
```

### Generate overlay previews

```powershell
python .\scripts\07_generate_overlays.py `
    --processed-root "<RUGD_SPLIT_ROOT>" `
    --output-root "<RUGD_OVERLAY_ROOT>"
```

### Generate class statistics

```powershell
python .\scripts\08_generate_statistics.py `
    --processed-root "<RUGD_SPLIT_ROOT>" `
    --output-dir "<RUGD_STATISTICS_ROOT>"
```

### Run final validation

```powershell
python .\scripts\09_final_check.py `
    --processed-root "<RUGD_PROCESSED_ROOT>" `
    --split-root "<RUGD_SPLIT_ROOT>" `
    --metadata "<RUGD_PROCESSED_ROOT>\metadata.csv" `
    --qc-report "<RUGD_QC_REPORT_PATH>" `
    --statistics-json "<RUGD_STATISTICS_JSON_PATH>" `
    --statistics-csv "<RUGD_STATISTICS_CSV_PATH>" `
    --overlay-manifest "<RUGD_OVERLAY_MANIFEST_PATH>" `
    --output "<RUGD_FINAL_CHECK_PATH>" `
    --expected-total 7436
```

## 11. Repository Scope

Included in GitHub:

- preprocessing and validation scripts
- `config/label_mapping.json`
- train, validation, and test split manifests
- class statistics
- QC report
- final-check result
- preprocessing summary

Excluded from GitHub:

- original RGB images and masks
- converted RGB images and ADOM Cost4 masks
- copied train, validation, and test image directories
- overlay PNG files and overlay manifest
- generated metadata containing local execution information
- temporary execution logs
- model checkpoints and deployment artifacts

Large dataset files are distributed separately through the team data storage.
