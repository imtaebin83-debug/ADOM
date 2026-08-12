# ADOM standalone Semantic20 preparation

This directory contains code and configuration only. Raw RGB images, CVAT
masks, normalized datasets, converted packages, and server upload packages
must not be committed to Git.

The package produced here remains independent from RELLIS, RUGD, and YCOR.
Dataset mixing is a later training-configuration decision; this workflow does
not modify any existing dataset preprocessing code or training config.

## Contracts

- Match PNG files by their complete path relative to the raw/mask roots. Never
  use only `Path.stem`, because frame names repeat between capture sessions.
- Use CVAT `SegmentationClass`, not `SegmentationObject`.
- Treat a mask PNG's existence as the labeling criterion during synchronization.
- Convert RGB masks through `config/label_mapping.json` into `L`/`uint8`
  Semantic20 train-ID masks.
- Map background and unlabelled pixels to `255` ignore.
- Fail on unknown RGB colors, unreadable images, size mismatches, missing pairs,
  duplicate keys, unassigned sequences, and sequence leakage across splits.
- Keep each continuous capture sequence wholly within one split.
- Use `reduce_zero_label=False`, 19 trainable classes (`0..18`), and ignore
  index `255`.
- Run `--dry-run` before material conversion.
- Use `shutil.copy2()` for RGB images and require a new or empty output root.
- Supply machine-specific paths through CLI arguments. Do not commit absolute
  paths, credentials, server mount locations, or generated data.

## Fixed sequence split

The source-of-truth split is `config/split_sequences.json`.

```text
train (133): 260810, 260811_2, 260811_4, 260811_5, 260811_6, 260811_7
val    (21): 260811_1
test   (61): 260811_3, 260811_8
```

`260811_3` and `260811_4` are each continuous captures with selected frames
removed. They must not be subdivided across splits. Person occurs only in
`260811_4`, so this version cannot report independent person validation/test
performance.

## 1. Date-level synchronization

```bash
python3 src/data/adom_data/scripts/sync_raw_masks.py \
  --raw <DATE_ROOT>/raw \
  --masks <DATE_ROOT>/SegmentationClass \
  --output <DATE_ROOT>/normalized \
  --dry-run

python3 src/data/adom_data/scripts/sync_raw_masks.py \
  --raw <DATE_ROOT>/raw \
  --masks <DATE_ROOT>/SegmentationClass \
  --output <DATE_ROOT>/normalized
```

## 2. Server upload package

```bash
python3 src/data/adom_data/scripts/build_upload_package.py \
  --source-root <ADOM_DATA_ROOT> \
  --output <ADOM_DATA_ROOT>/upload \
  --dry-run

python3 src/data/adom_data/scripts/build_upload_package.py \
  --source-root <ADOM_DATA_ROOT> \
  --output <ADOM_DATA_ROOT>/upload
```

The server package contains date-level `raw/`, `masks/`, `labelmap.txt`, and a
relative-path SHA-256 `manifest.json`. It remains an RGB-mask source package,
not a training package.

## 3. Semantic20 training package

```bash
python3 src/data/adom_data/scripts/convert_semantic20.py \
  --input-root <ADOM_UPLOAD_ROOT> \
  --output-root <NEW_SEMANTIC20_ROOT> \
  --dry-run

python3 src/data/adom_data/scripts/convert_semantic20.py \
  --input-root <ADOM_UPLOAD_ROOT> \
  --output-root <NEW_SEMANTIC20_ROOT>

python3 src/data/adom_data/scripts/validate_semantic20_package.py \
  --input-root <NEW_SEMANTIC20_ROOT>
```

The result is compatible with `AdomSemantic20Dataset` through its manifest
contract and is shared by SegFormer-B0 and SegFormer-B2:

```text
<NEW_SEMANTIC20_ROOT>/
|-- images/<date>/<session>/frame_XXXXXX.png
|-- masks/<date>/<session>/frame_XXXXXX.png
|-- splits/{train,val,test}.txt
|-- manifest.csv
`-- metadata/
    |-- label_mapping.json
    |-- split_sequences.json
    `-- conversion_summary.json
```

The manifest stores only package-root-relative POSIX paths. Future training
configs may combine this package with RELLIS/RUGD/YCOR, but that integration is
deliberately outside this standalone preprocessing scope.
