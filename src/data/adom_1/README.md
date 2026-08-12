# ADOM-1 dataset preparation

This directory contains code only. Raw RGB images, CVAT masks, normalized
datasets, and server upload packages must not be committed to Git.

## Required rules

- Match PNG files by the complete path relative to `--raw` and `--masks`.
  Never use only `Path.stem` because frame names repeat between sessions.
- Treat the existence of a mask PNG as the labeling criterion. Do not infer
  annotation presence from mask pixel values.
- Ignore non-PNG files during pairing.
- Reject unreadable pairs and raw/mask image-size mismatches.
- Run `--dry-run` before every material copy.
- Keep source data intact during normal preprocessing. Outputs use
  `shutil.copy2()` and must be new or empty directories.
- Pass machine-specific paths through CLI arguments. Never commit absolute
  paths, credentials, or server mount locations.
- Use `SegmentationClass` as the semantic training mask. Do not substitute
  `SegmentationObject`.

## Date-level synchronization

```bash
python3 src/data/adom_1/scripts/sync_raw_masks.py \
  --raw <DATE_ROOT>/raw \
  --masks <DATE_ROOT>/SegmentationClass \
  --output <DATE_ROOT>/normalized \
  --dry-run

python3 src/data/adom_1/scripts/sync_raw_masks.py \
  --raw <DATE_ROOT>/raw \
  --masks <DATE_ROOT>/SegmentationClass \
  --output <DATE_ROOT>/normalized
```

The expected result is:

```text
<DATE_ROOT>/normalized/
├── raw/<session>/frame_XXXXXX.png
└── masks/<session>/frame_XXXXXX.png
```

## Server upload package

The package builder copies verified normalized pairs into date folders,
preserves each date's `labelmap.txt`, and writes relative-path SHA-256 records
to `manifest.json`.

```bash
python3 src/data/adom_1/scripts/build_upload_package.py \
  --source-root <ADOM_DATA_ROOT> \
  --output <ADOM_DATA_ROOT>/upload \
  --dry-run

python3 src/data/adom_1/scripts/build_upload_package.py \
  --source-root <ADOM_DATA_ROOT> \
  --output <ADOM_DATA_ROOT>/upload
```

The output is data for the server, not Git content:

```text
upload/
├── manifest.json
├── 260810/{raw,masks,labelmap.txt}
└── 260811_1/{raw,masks,labelmap.txt}
```
