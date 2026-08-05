# ADOM Semantic24 bridge

This package does not download or unpack source datasets. It reads the outputs
or validated native layouts produced by each dataset-specific preprocessing
folder, converts masks into the shared ADOM Semantic24 ID space, and then
optionally combines the four converted bridge outputs.

## Target IDs

RELLIS trainable IDs `0..18` are preserved. ID `19` is reserved and must never
occur in a mask. The additions are `snow=20`, `animal=21`, `artifact=22`, and
`cobble=23`; `artifact` contains only GOOSE `traffic_light`, `traffic_sign`,
and `misc_sign`. Ignore remains `255`.

Classes unavailable in a source remain valid target columns with zero counts.

## Input contracts

- RELLIS: `rellis3d_semantic20_v1` output containing `images/`, `masks/`, and
  `splits/`. Its indexed IDs `0..18,255` are passed through.
- RUGD: validated original RGB and index-label masks plus `train/val/test` split
  files. Cost4 masks must not be used because they have lost the source semantic
  classes.
- YCOR: validated native `train/` and `valid/` sample folders containing
  `rgb.jpg` and original palette `labels.png`.
- GOOSE: output of `src/data/goose/scripts/01_materialize_native.py`.

## Convert each source

```bash
mapping=src/data/semantic_24/config/bridge_mapping.yaml
script=src/data/semantic_24/scripts/01_convert_dataset_bridge.py

python "$script" --dataset rellis \
  --input-root /workspace/adom/datasets/processed/phase1-20class-v1/rellis \
  --split-root /workspace/adom/src/data/rellis/splits \
  --output-root /workspace/adom/datasets/processed/phase1-24class-v1/bridge/rellis \
  --mapping "$mapping"

python "$script" --dataset rugd \
  --input-root "$RUGD_NATIVE_ROOT" \
  --image-root "$RUGD_IMAGE_ROOT" \
  --mask-root "$RUGD_INDEX_MASK_ROOT" \
  --split-root /workspace/adom/src/data/rugd/splits \
  --output-root /workspace/adom/datasets/processed/phase1-24class-v1/bridge/rugd \
  --mapping "$mapping"

python "$script" --dataset ycor \
  --input-root "$YCOR_NATIVE_ROOT" \
  --output-root /workspace/adom/datasets/processed/phase1-24class-v1/bridge/ycor \
  --mapping "$mapping"

python "$script" --dataset goose \
  --input-root /workspace/adom/datasets/raw/goose/dataset/goose-2d-visible-v1 \
  --output-root /workspace/adom/datasets/processed/phase1-24class-v1/bridge/goose \
  --mapping "$mapping"
```

Each output root must be absent or empty. Source files are never modified.

## GOOSE full-dataset policy

For every GOOSE native pair the converter:

1. maps the original 64-class mask to Semantic24;
2. calculates each target class pixel count and percentage using all image
   pixels, including ignored pixels, as the denominator;
3. materializes every RGB/mask pair without class-ratio or class-presence
   filtering.

`metadata/per_image_distribution.csv` remains an audit artifact, but it never
controls whether a GOOSE image is written. The GOOSE manifest count must equal
the GOOSE-native input manifest count.

GOOSE-specific mapping additions are `cobble -> cobble`, `crops -> bush`,
`rail_track -> asphalt`, `moss -> dirt`, and `road_marking -> asphalt`.
`scenery_vegetation` remains ignore.

## Combine and validate

```bash
root=/workspace/adom/datasets/processed/phase1-24class-v1

python src/data/semantic_24/scripts/02_build_combined_package.py \
  --rellis-root "$root/bridge/rellis" \
  --rugd-root "$root/bridge/rugd" \
  --ycor-root "$root/bridge/ycor" \
  --goose-root "$root/bridge/goose" \
  --output-root "$root/combined"

python src/data/semantic_24/scripts/03_validate_combined_package.py \
  --input-root "$root/combined"
```

Main train includes source train samples from all four datasets. Main val/test
remain RELLIS-only; other source val/test splits are written as diagnostics.
Raw RGB, masks, archives, and generated packages must not be committed to Git.
