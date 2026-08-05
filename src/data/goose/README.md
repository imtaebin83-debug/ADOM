# GOOSE native preprocessing

This directory materializes the original GOOSE 64-class source dataset. It
keeps visible windshield RGB images and original `*_labelids.png` masks. It
does not remap labels, select frames, or modify an existing output directory.

The resulting immutable source folder is the GOOSE input to
`src/data/semantic_24`; ADOM24 conversion is deliberately a separate stage.

## Server execution

Only the uploaded ZIP archives and this repository are required:

```bash
cd /workspace/adom
bash src/data/goose/run_server.sh \
  /workspace/adom/datasets/raw/goose/archives \
  /workspace/adom/datasets/raw/goose/dataset/goose-2d-visible-v1
```

The archive directory may be located elsewhere. Pass the actual server path;
the script recursively locates exactly one `goose_2d_train.zip` and one
`goose_2d_val.zip`.

## Output contract

```text
goose-2d-visible-v1/
|-- images/{train,val}/.../*_windshield_vis.*
|-- labels/{train,val}/.../*_labelids.png
`-- metadata/
    |-- goose64_classes.csv
    |-- pair_manifest.csv
    |-- per_image_goose64_distribution.csv
    |-- goose64_class_summary.csv
    `-- preprocess_summary.json
```

All manifest paths are relative to the output root. The known archive sizes and
SHA-256 values in `config/archive_checksums.csv` are verified first. Selected
ZIP members are then fully read, which validates their CRC. RGB/mask pairing,
dimensions, and mask IDs `0..63` are validated before the run is reported as
`PASS`.
