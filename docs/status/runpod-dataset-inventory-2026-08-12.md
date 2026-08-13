# RunPod dataset inventory — 2026-08-12 (updated 2026-08-13)

- Status: Verified snapshot plus standalone and TA superset releases
- Initial observation image: `imtaebeen/adom-mmseg:88b68f7af5e98326db4b76dd048a114173d23c9c`
- Standalone conversion image: `imtaebeen/adom-mmseg:6723095970c74e6ca35fb01d66955787fe76b971`
- Network Volume mount: `/workspace`
- Evidence source: read-only `ls`, `find`, manifest count/path audit, and `du`
- Scope: TA0/TA1/TA2 Semantic20 dataset readiness

이 문서는 2026-08-12 시점 RunPod Network Volume의 실제 구조 snapshot과
2026-08-13 standalone 및 target-adaptation superset materialization 결과를 함께
보존한다. 최초 관측과 후속 변경을 구분하며, package 상태는 validation artifact와
SHA를 기준으로 판단한다.

## Readiness verdict

| Asset | Path | State | Evidence |
| --- | --- | --- | --- |
| ADOM ZED2i raw upload | `/workspace/adom/datasets/raw/adomdata` | Preserved source | 9 capture groups, 215 pairs, upload `manifest.json` SHA-256 `5b51b560...580dfe5` |
| RELLIS-only E0 package | `/workspace/adom/datasets/processed/rellis3d_semantic20_v1` | Released | `_SUCCESS` present |
| RELLIS+RUGD+YCOR E1 package | `/workspace/adom/datasets/processed/adom_semantic20_rellis_rugd_ycor_v1` | Released and structurally verified | `_SUCCESS`, 14,421-row manifest, no missing image/mask paths |
| ADOM standalone Semantic20 | `/workspace/adom/datasets/processed/adom_zed2i_semantic20_v1` | **Released and validated** | 215 samples; 133/21/61 split; validator PASS; `_SUCCESS` present |
| TA0/TA1/TA2 superset | `/workspace/adom/datasets/processed/adom_semantic20_target_adaptation_v1` | **Released and validated** | 14,636 samples; expected splits; validator PASS; hardlink materialization |
| Old E1 staging | `/workspace/adom/datasets/.staging/semantic20/adom_semantic20_rellis_rugd_ycor_v1/20260804T155921Z` | Not a release | 18 GB; no manifest, `_SUCCESS`, `conversion_summary.json`, or `final_check.json` within the inspected depth |

따라서 자체 수집 데이터의 standalone Semantic20 전처리와 공통 target-adaptation
superset의 구조 검증까지 완료됐다. TA0/TA1/TA2가 공유할 dataset package는 release
상태이며, 이후 condition별 학습 code/image/checkpoint gate를 통과해야 한다.
임의의 RGB/mask 복사본이나 marker 없는 staging은 학습 입력으로 인정하지 않는다.

## Observed folder tree

```text
/workspace/adom/
├── code/
│   └── adom-preprocess-8ca6ab811b131889fd36f4788f21ab08c619500f/
│       └── study/gahyung/Datasets_Repo/ADOM-Semantic20/
│           └── adom_semantic20_rellis_rugd_ycor_v1/
│               └── results/conversion_summary.json  # code snapshot copy
├── datasets/
│   ├── .staging/
│   │   └── semantic20/adom_semantic20_rellis_rugd_ycor_v1/
│   │       └── 20260804T155921Z/
│   │           ├── raw/                              # 13 GB
│   │           ├── rugd-flat/                        # 5.5 GB
│   │           ├── **pycache**/                      # about 983 KB
│   │           └── adom_semantic20_rellis_rugd_ycor_v1/
│   │               # inspected staging total: 18 GB; no release markers
│   ├── goose/
│   ├── processed/
│   │   ├── rellis3d_semantic20_v1/
│   │   │   ├── images/{00000,00001,00002,00003,00004}/
│   │   │   ├── masks/{00000,00001,00002,00003,00004}/
│   │   │   ├── splits/
│   │   │   └── _SUCCESS
│   │   ├── adom_semantic20_rellis_rugd_ycor_v1/
│   │   │   ├── images/{rellis3d,rugd,ycor}/
│   │   │   ├── masks/{rellis3d,rugd,ycor}/
│   │   │   ├── splits/
│   │   │   ├── results/
│   │   │   │   ├── conversion_summary.json
│   │   │   │   └── final_check.json
│   │   │   ├── manifest.csv                         # 14,421 rows
│   │   │   └── _SUCCESS                             # legacy zero-byte marker
│   │   ├── adom_zed2i_semantic20_v1/
│   │   │   ├── images/<date>/<logical-session>/
│   │   │   ├── masks/<date>/<logical-session>/
│   │   │   ├── splits/{train,val,test}.txt
│   │   │   ├── metadata/
│   │   │   │   ├── conversion_summary.json          # 27 KB
│   │   │   │   ├── label_mapping.json
│   │   │   │   ├── source_upload_manifest.json
│   │   │   │   └── split_sequences.json
│   │   │   ├── results/validation_report.json
│   │   │   ├── manifest.csv                         # 36 KB, 215 rows
│   │   │   └── _SUCCESS                             # validator release marker
│   │   └── adom_semantic20_target_adaptation_v1/
│   │       ├── images/{rellis3d,rugd,ycor,adom_zed2i}/
│   │       ├── masks/{rellis3d,rugd,ycor,adom_zed2i}/
│   │       ├── splits/
│   │       │   ├── ta0_train.txt
│   │       │   ├── ta1_train.txt
│   │       │   ├── ta2_train.txt
│   │       │   ├── val.txt
│   │       │   ├── test.txt
│   │       │   ├── adom_val_diagnostic.txt
│   │       │   └── adom_test_diagnostic.txt
│   │       ├── metadata/package_summary.json
│   │       ├── results/validation_report.json
│   │       ├── manifest.csv                         # 14,636 rows
│   │       └── _SUCCESS                             # validator release marker
│   └── raw/
│       └── adomdata/
│           ├── 260810/
│           ├── 260811_1/
│           ├── 260811_2/
│           ├── 260811_3/
│           ├── 260811_4/
│           ├── 260811_5/
│           ├── 260811_6/
│           ├── 260811_7/
│           ├── 260811_8/
│           └── manifest.json                        # upload manifest, 59 KB
└── logs/
    └── preprocess/adom_semantic20_rellis_rugd_ycor_v1/
        └── 20260804T155921Z/release/conversion_summary.json
```

The raw package was owned by `root:root`, its directories were mode `750`, and
`manifest.json` was mode `600`. The released E1 package was owned by numeric UID/GID
`231072:231072` and was world-readable. Current training containers run as root, but a
future non-root Pod must resolve these raw-package permissions before conversion.

## Verified counts

### ADOM raw upload

| Capture group | Pairs |
| --- | ---: |
| `260810` | 28 |
| `260811_1` | 21 |
| `260811_2` | 10 |
| `260811_3` | 51 |
| `260811_4` | 60 |
| `260811_5` | 12 |
| `260811_6` | 15 |
| `260811_7` | 8 |
| `260811_8` | 10 |
| **Total** | **215** |

### Released E1 package

| Source/split | Count |
| --- | ---: |
| Manifest RELLIS | 6,234 |
| Manifest RUGD | 7,436 |
| Manifest YCOR | 751 |
| Main train | 9,868 = 4,435 RELLIS + 4,779 RUGD + 654 YCOR |
| Canonical RELLIS val | 900 |
| Canonical RELLIS test | 899 |
| Missing manifest image paths | 0 |
| Missing manifest mask paths | 0 |

Diagnostic splits were RUGD val 733/test 1,924 and YCOR val 97. The E1 package is ready
to serve as the immutable source package for the TA superset builder.

### Released ADOM standalone package (2026-08-13)

| Item | Verified value |
| --- | --- |
| Samples / sequences | 215 / 10 |
| Train / val / test | 133 / 21 / 61 |
| Observed train IDs | 10 log, 11 person, 18 rubble, 255 ignore |
| Observed val IDs | 10 log, 255 ignore |
| Observed test IDs | 10 log, 18 rubble, 255 ignore |
| All-ignore masks | 0 |
| Manifest SHA-256 | `f2b30e2c5fa30488e4955799a767cc727838898c96f4925cd35326baa065bdfd` |
| Conversion summary SHA-256 | `afc2d88ee88261e9b22f525add69c2f4993a255b04acea597818c9e0da639996` |
| Validator | `PASS`; `_SUCCESS` written |

Pixel totals are log 3,794,500, person 760,472, rubble 18,957,451 and ignore
174,631,577. Class coverage is intentionally partial: dirt and bush exist in the
mapping but are not present in this release. Standalone validation covers only log,
and standalone test covers log and rubble. These splits are domain diagnostics; they
must not replace canonical RELLIS validation/test for recipe selection or general
Semantic20 claims. Validator `PASS` proves the package, pair, split, ID, checksum and
release-marker contracts; it does not prove annotation-boundary correctness. Sampled
overlay review remains required before TA1/TA2 full training.

### Released target-adaptation superset (2026-08-13)

| Item | Verified value |
| --- | --- |
| Manifest sources | RELLIS 6,234; RUGD 7,436; YCOR 751; ADOM 215 |
| TA0 train | 4,435 RELLIS only |
| TA1 train | 4,435 RELLIS + 133 ADOM = 4,568 |
| TA2 train | 4,435 RELLIS + 4,779 RUGD + 654 YCOR + 133 ADOM = 10,001 |
| Canonical val / test | 900 / 899 RELLIS only |
| ADOM diagnostic val / test | 21 / 61 ADOM only |
| Observed IDs | `0..18`, `255` |
| All-ignore train masks | 0 |
| Manifest SHA-256 | `183dda705e76b451dc383a81f517d36df3d6032f00002ab225421b9ae316b9dd` |
| Package summary SHA-256 | `e29a3691472eb01767545dddebb877fbe609f1ec1d11596110d9c14d92968e07` |
| Dataset images SHA-256 | `f07e1ed3a463ade04834f6de8e5c80c531d2b67be2ca1df78f3de4d0fe57ef87` |
| Dataset masks SHA-256 | `975209f763326e5c86d9d54e55474997a76489ea6dcbf1ed51fe61f830c1bc69` |
| Storage | 14,636 image hardlinks and 14,636 mask hardlinks |
| Validator | `PASS`; `_SUCCESS` written |

Hardlinks avoid duplicating dataset blocks, but make the release roots share file
inodes. E1, standalone and target-adaptation package images/masks must all be treated as
immutable; editing a hardlinked file through any root invalidates the recorded digests.

## Next materialization paths

Do not write into E1, `.staging`, or raw roots. Use these versioned release targets:

```text
/workspace/adom/datasets/processed/adom_zed2i_semantic20_v1
/workspace/adom/datasets/processed/adom_semantic20_target_adaptation_v1
```

Materialization state and required order:

1. **Done:** standalone converter dry-run against the 215-pair raw root.
2. **Done:** materialize `adom_zed2i_semantic20_v1` into a new output root.
3. **Done:** validate it and write `_SUCCESS`.
4. **Done:** build the TA superset from released E1 plus released standalone.
5. **Done:** validate the TA superset with `PASS` and write `_SUCCESS`.
6. Treat all released roots as read-only during Pod training.

TA0 has RELLIS-only source exposure, but the current TA0 configs and
`TA0AblationContractHook` are fail-closed to the shared package's
`splits/ta0_train.txt` and `manifest.csv`. Therefore steps 4–5 are runtime prerequisites
for TA0 as well as TA1/TA2. This does not leak standalone samples into TA0: the validated
TA0 split must contain exactly 4,435 RELLIS samples and the sampler weight remains
`rellis3d=1.0`. Dataset preparation no longer blocks TA0; confirm the marker, E0
checkpoint SHA, code/image SHA and smoke gate before allocating a full run.

No staging cleanup is authorized by this inventory. The 18 GB staging tree may be
reviewed and removed only as a separate, explicitly approved storage-cleanup operation.

## Lookup commands

```bash
find /workspace/adom/datasets/processed -maxdepth 3 -type d -print | sort
```

```bash
find /workspace -maxdepth 10 -type f -name conversion_summary.json -print | sort
```

```bash
find /workspace/adom/datasets/processed -maxdepth 3 -type f \
  \( -name manifest.csv -o -name _SUCCESS \
     -o -name conversion_summary.json -o -name validation_report.json \
     -o -name final_check.json \) -print | sort
```

```bash
du -h --max-depth=4 /workspace/adom/datasets/.staging 2>/dev/null \
  | sort -h | tail -30
```
