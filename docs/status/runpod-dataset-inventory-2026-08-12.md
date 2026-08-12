# RunPod dataset inventory — 2026-08-12

- Status: Verified snapshot
- Observed from image: `imtaebeen/adom-mmseg:88b68f7af5e98326db4b76dd048a114173d23c9c`
- Network Volume mount: `/workspace`
- Evidence source: read-only `ls`, `find`, manifest count/path audit, and `du`
- Scope: TA0/TA1/TA2 Semantic20 dataset readiness

이 문서는 2026-08-12 시점 RunPod Network Volume의 실제 구조 snapshot이다. 이후
변환을 실행하면 새 package와 marker가 추가되므로, 현재 상태와 혼동하지 말고 새
validation artifact 및 SHA와 함께 후속 snapshot을 남긴다.

## Readiness verdict

| Asset | Path | State | Evidence |
| --- | --- | --- | --- |
| ADOM ZED2i raw upload | `/workspace/adom/datasets/raw/adomdata` | Ready for converter dry-run | 9 capture groups, 215 pairs, upload `manifest.json` present |
| RELLIS-only E0 package | `/workspace/adom/datasets/processed/rellis3d_semantic20_v1` | Released | `_SUCCESS` present |
| RELLIS+RUGD+YCOR E1 package | `/workspace/adom/datasets/processed/adom_semantic20_rellis_rugd_ycor_v1` | Released and structurally verified | `_SUCCESS`, 14,421-row manifest, no missing image/mask paths |
| ADOM standalone Semantic20 | `/workspace/adom/datasets/processed/adom_zed2i_semantic20_v1` | Not created | root absent; no standalone conversion summary or validated manifest anywhere under `/workspace` to depth 10 |
| TA0/TA1/TA2 superset | `/workspace/adom/datasets/processed/adom_semantic20_target_adaptation_v1` | Not created | root absent |
| Old E1 staging | `/workspace/adom/datasets/.staging/semantic20/adom_semantic20_rellis_rugd_ycor_v1/20260804T155921Z` | Not a release | 18 GB; no manifest, `_SUCCESS`, `conversion_summary.json`, or `final_check.json` within the inspected depth |

따라서 자체 수집 데이터는 **raw upload package까지 확보됐고 standalone Semantic20
전처리는 아직 실행되지 않은 상태**다. 유효한 standalone package는 converter가 만드는
manifest, split, metadata summary와 validator가 만드는 `_SUCCESS`를 모두 가져야 한다.
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
│   │   └── adom_semantic20_rellis_rugd_ycor_v1/
│   │       ├── images/{rellis3d,rugd,ycor}/
│   │       ├── masks/{rellis3d,rugd,ycor}/
│   │       ├── splits/
│   │       ├── results/
│   │       │   ├── conversion_summary.json
│   │       │   └── final_check.json
│   │       ├── manifest.csv                         # 14,421 rows
│   │       └── _SUCCESS                             # legacy zero-byte marker
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

## Next materialization paths

Do not write into E1, `.staging`, or raw roots. Use these versioned release targets:

```text
/workspace/adom/datasets/processed/adom_zed2i_semantic20_v1
/workspace/adom/datasets/processed/adom_semantic20_target_adaptation_v1
```

Required order:

1. Run standalone converter `--dry-run` against the 215-pair raw root.
2. Materialize the new standalone root only if the output root is absent/new.
3. Run the standalone validator with `--write-success-marker`.
4. Build the TA superset from released E1 plus released standalone.
5. Validate the TA superset and write its `_SUCCESS`.
6. Treat both released roots as read-only during parallel Pod training.

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
