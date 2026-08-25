# B2-E-ADOM capacity × domain seed 42 run record

> 상태: **protocol/config frozen; pre-image static gate passed; GPU gates blocked on immutable image**
> 실행 계약: [B2-E-ADOM capacity × domain study](b2-eadom-capacity-domain-study.md)
> branch: `codex/b2-eadom-capacity-domain`
> base commit: `5426f297c05568c11445c0b86510195ebf6f4646`
> primary split: legacy-matched `ta1_train.txt`, 4,568 rows

이 문서는 사전등록 문서를 수정하지 않고 실행 중 확인된 identity, gate와 결과를
누적하는 run record다. 아래 값은 2026-08-25에 RunPod Network Volume의 원시
artifact에서 다시 계산하거나 직접 읽었다. 아직 실행하지 않은 항목은 결과로
간주하지 않는다.

## Existing-condition provenance audit

| Condition | Training/config identity | Selected checkpoint | Evaluation state |
| --- | --- | --- | --- |
| B0-E0 | image Git SHA `5c50bfdf2900596bcd447ed6c44ce7924bf10453`; legacy Stage 2 selection by raw validation mIoU | iter 6,000; SHA-256 `d76229ff623eb382fd48011decf54c342d88a113bcbe650fb58cc20e42cabe73` | fresh RELLIS 899 and Korean 61 evaluation exists under `paper_eval_outputs/20260824T152720Z` |
| B0-E-ADOM | image Git SHA `9d4f08e4d12af58eae96a99ea8a75eccfc5f6e90`; resolved Stage 1/2 config snapshots SHA-256 `aab0cc92d1a5c839d6d73ac4045d4a52dee98e1fca409d42fefc0d64ff1eaee9` / `e328ac0abbdb0143968907471eb2a15e4f679bb928ac5c0ce122e009a3473b9b` | constrained RELLIS-val selection iter 26,000; SHA-256 `f4cc41fd91e9df8e7aa3f726498e80636b736dfadf0e1baf338fe7c82a83399c` | fresh RELLIS 899 and Korean 61 evaluation exists under the same ordered manifests as B0-E0 |
| B2-E0 | image Git SHA `5c50bfdf2900596bcd447ed6c44ce7924bf10453`; Stage 1/2 config snapshots SHA-256 `b2c2ca278d38ed4b397b0c4c50708d79a4402a0545a53f8ec93af2404f59f75a` / `bb44efd9626c9d5ef62f440481dd2cfa9f24faab83bdda092bf7d44dba1793db` | legacy raw-validation selection iter 14,000; SHA-256 `c47288019185e18fffdb856d2f47f56936adb06db7579416271ab468b3849f4f` | legacy canonical RELLIS test exists; no Korean fresh evaluation and no B2 entry in the 2026-08-24 paper-eval output |

The 2026-08-24 fresh B0 comparison used evaluation-contract SHA-256
`096467321246732da9d2f4a31ad8f75626b1aba0500e00680ba4ddd778241635e`.
The ordered manifest SHA-256 values were:

- RELLIS test 899: `2e078a3ac89d870b4dfb5838f8cc2772e788ecdd7cb011c3309d59b4ca6a66918`
- Korean held-out 61: `1eb86ff65620fb5c0afc1d58c572c517cacc937468ebd8655375aaa26d81eb782`

The fresh B2-E0 and B2-E-ADOM evaluations must reproduce those ordered-manifest
and evaluation-contract digests. A mismatch is a comparison stop condition.

## Frozen primary dataset identity

The existing B0-E-ADOM run recorded the following contract. B2-E-ADOM must match
it before any optimizer update:

| Field | Frozen value |
| --- | --- |
| train / val / test | 4,568 / 900 / 899 |
| verified manifest pairs | 14,636 |
| ontology | Semantic20 19 trainable classes; ignore 255 |
| split contract SHA-256 | `fab9c136c81081464d9db099656680dac3bf2921a4ae2bbd76055c383b309ab93` |
| manifest SHA-256 | `183dda705e76b451dc383a81f517d36df3d6032f00002ab225421b9ae3316b9dd` |
| image content SHA-256 | `ce06265e6146bcd37692938786386cbd9b844e9742f831284ee55d26aedd15305` |
| mask content SHA-256 | `5ae15ab1eff69921168b15811683edab41472456a439b58aa63844c6d472c377e` |
| combined content SHA-256 | `a70c6b9467b692a4797976659c6dcd501c80938626226000a6cc214efcdec5e42` |
| canonical mapping SHA-256 | `ecfa61662ddbf16c801bcac22db11b0e7ee2408d635e3018a21dd389933a6bc55` |

This is the legacy-matched primary run, including the known 12 train-RGB
conflicts with the diagnostic validation export. The conflict-free 4,556-row
split remains a separate sensitivity condition and is not materialized or used
by this primary run.

## RunPod pre-training observation

Observed before any B2-E-ADOM gate on 2026-08-25 UTC:

- GPU: NVIDIA GeForce RTX 4090, 24,564 MiB physical, 24,080 MiB free
- driver: 580.126.20; training image CUDA: 12.2.2; PyTorch: 2.1.0a0
- Network Volume: `/workspace`, approximately 1,001 TiB reported available
- image source root: `/opt/adom`
- current image Git SHA: `9d4f08e4d12af58eae96a99ea8a75eccfc5f6e90`

The current image predates this branch and therefore cannot run the new config
under a truthful immutable-image SHA. Static and GPU gates stay closed until
`/opt/adom` contains a verifiable build of the protocol/config commit and
`ADOM_GIT_SHA` matches that commit.

## Gate ledger

| Gate | Status | Evidence |
| --- | --- | --- |
| local source/config tests | PASS with MMEngine import deferred to training image | `test_b2_eadom_contract`, `test_runtime_contracts`, `test_semantic20_training` |
| resolved architecture-only diff | PASS on isolated commit replay; immutable-image replay pending | Stage 1/2 contain only the six allowlisted B0→B2 architecture paths |
| split/mapping/manifest/content digest | PASS on isolated commit replay; immutable-image replay pending | all values equal the frozen table above |
| RTX 4090 memory probe 16/1 → 8/2 → 4/4 | PENDING | `gates/probe/b2/batch_plan.json` and probe log |
| 50-update smoke | PENDING | isolated `gates/smoke` output |
| 500-update mini + RELLIS validation | PENDING | isolated `gates/mini` output |
| optimizer/scheduler resume | PENDING | isolated `gates/resume` output |
| Stage 1 4k + Stage 2 40k | PENDING | `full/b2` output |
| checkpoint freeze from RELLIS val only | PENDING | checkpoint SHA-256 and selection JSON |
| fresh RELLIS 899 evaluation | PENDING | raw prediction/metric provenance under `/workspace` |
| fresh Korean held-out 61 evaluation | LOCKED until checkpoint freeze | test-only; never recipe, threshold, or checkpoint input |

### Pre-image static replay artifact

Because the active `/opt/adom` image still reports Git SHA `9d4f08e4...`, the
frozen commit was replayed only in an isolated `/tmp/adom-acdb0c75-runtime`
tree. No optimizer update or checkpoint was produced. The replay used the
training image's MMEngine 0.10.7 and decoded all dataset pairs.

- source commit: `acdb0c75c156bef8f42744f1d3fb3ac9ae869678`
- runtime archive SHA-256:
  `293330ac0ee32c1a45433705e58c3ab7b9ec6f704e6b8236fbc5015723aced3d`
- report:
  `/workspace/adom/runs/semantic20/eadom/seed42/protocol/static_contract_preimage_acdb0c75.json`
- report SHA-256:
  `88fd5efbe907b8d42acb7ca4302af97f40297b837b0562d9261214064da00f59`
- log SHA-256:
  `5c771f6d370712f347e8ca0d4b81f877563533ed7c4f52a3bed9de434b79fccf`
- Stage 1 non-architecture resolved-config SHA-256:
  `a4134e45a57ea4ea7b6ccf3c7e83d5ee178d4391bebcde1fb6d3f70d12e9f0a0`
- Stage 2 non-architecture resolved-config SHA-256:
  `b1d4ca6a9d8c2b3ffae4c741089e75b04b92a4b083cdf258257abcc0ec65f1f4`

This replay proves that the committed contract resolves and matches the frozen
dataset. It does not authorize a GPU probe from `/tmp` and does not replace the
required `/opt/adom` immutable-image SHA check.

## Reporting limits frozen before results

Korean held-out contains one independent positive sequence per class (log 10
images; rubble 51 images), no negative sequence and no co-occurrence sequence.
Its masks are target-only partial labels. Results cannot be described as 61
independent samples, full-scene precision/false-stop evidence, or a deployment
safety guarantee. B5 is not part of this run and may only be proposed after the
B2 result is interpreted against the preregistered uncertainty rule.
