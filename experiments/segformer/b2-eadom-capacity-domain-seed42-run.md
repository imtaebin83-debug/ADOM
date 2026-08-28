# B2-E-ADOM capacity × domain seed 42 run record

> 상태: **primary matched-legacy seed 42 training and canonical evaluation completed**
> 실행 계약: [B2-E-ADOM capacity × domain study](b2-eadom-capacity-domain-study.md)
> branch: `codex/b2-eadom-capacity-domain`
> base commit: `5426f297c05568c11445c0b86510195ebf6f4646`
> primary split: legacy-matched `ta1_train.txt`, 4,568 rows

이 문서는 사전등록 문서를 수정하지 않고 실행 중 확인된 identity, gate와 결과를
누적하는 run record다. 아래 값은 2026-08-25부터 2026-08-28까지 RunPod
Network Volume의 원시 artifact에서 다시 계산하거나 직접 읽었다. 결과 표는
Korean held-out을 checkpoint/recipe 선택에 사용하지 않고 checkpoint를 동결한 뒤
수행한 direct inference만 사용한다.

## Existing-condition provenance audit

| Condition | Training/config identity | Selected checkpoint | Evaluation state |
| --- | --- | --- | --- |
| B0-E0 | image Git SHA `5c50bfdf2900596bcd447ed6c44ce7924bf10453`; legacy Stage 2 selection by raw validation mIoU | iter 6,000; SHA-256 `d76229ff623eb382fd48011decf54c342d88a113bcbe650fb58cc20e42cabe73` | fresh RELLIS 899 and Korean 61 evaluation exists under `paper_eval_outputs/20260824T152720Z` |
| B0-E-ADOM | image Git SHA `9d4f08e4d12af58eae96a99ea8a75eccfc5f6e90`; resolved Stage 1/2 config snapshots SHA-256 `aab0cc92d1a5c839d6d73ac4045d4a52dee98e1fca409d42fefc0d64ff1eaee9` / `e328ac0abbdb0143968907471eb2a15e4f679bb928ac5c0ce122e009a3473b9b` | constrained RELLIS-val selection iter 26,000; SHA-256 `f4cc41fd91e9df8e7aa3f726498e80636b736dfadf0e1baf338fe7c82a83399c` | fresh RELLIS 899 and Korean 61 evaluation exists under the same ordered manifests as B0-E0 |
| B2-E0 | image Git SHA `5c50bfdf2900596bcd447ed6c44ce7924bf10453`; Stage 1/2 config snapshots SHA-256 `b2c2ca278d38ed4b397b0c4c50708d79a4402a0545a53f8ec93af2404f59f75a` / `bb44efd9626c9d5ef62f440481dd2cfa9f24faab83bdda092bf7d44dba1793db` | legacy raw-validation selection iter 14,000; SHA-256 `c47288019185e18fffdb856d2f47f56936adb06db7579416271ab468b3849f4f` | fresh RELLIS 899 and Korean 61 evaluation completed under the B2 shared inference contract |
| B2-E-ADOM | immutable training image Git SHA `7cafc31683934d6b8b224f0ebd1592d5d5c1c72c`; matched-legacy 4,568-row train; RELLIS-only validation selection | constrained RELLIS-val selection iter 9,000; SHA-256 `b1b9cded88fa091d503fb48c0fd1f9fafd3df938030bb767a04d7a9aab96707b` | fresh RELLIS 899 and Korean 61 evaluation completed under the same B2 inference contract as B2-E0 |

The 2026-08-24 fresh B0 comparison used evaluation-contract SHA-256
`096467321246732da9d2f4a31ad8f75626b1aba0500e0680ba4ddd778241635e`.
The ordered manifest SHA-256 values were:

- RELLIS test 899: `2e078a3ac89d870b4dfb5838f8cc2772e788ecdd7cb011c309d59b4ca6a66918`
- Korean held-out 61: `1eb86ff65620fb5c0afc1d58c572c517cacc937468ebd865375aaa26d81eb782`

The raw evaluation-contract digest includes the resolved architecture, so it is
expected to differ across capacities. The B0 pair shares contract SHA-256
`096467321246732da9d2f4a31ad8f75626b1aba0500e0680ba4ddd778241635e`; the B2
pair shares `4adfcb3ae550274ed3436c695c872e030c804bb8c16c09025958797312d8d592`.
Both pairs use the ordered manifests above, and the B2 pair uses one shared B2
inference config. Cross-capacity comparison is therefore based on matching all
non-architecture evaluation fields, not on incorrectly requiring the raw digest
to be equal across B0 and B2.

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
| resolved architecture-only diff | PASS on immutable training image | Stage 1/2 contain only the six allowlisted B0→B2 architecture paths |
| split/mapping/manifest/content digest | PASS on immutable training image | all values equal the frozen table above before optimizer update |
| RTX 4090 memory probe 16/1 → 8/2 → 4/4 | PASS at first candidate | effective batch 16 frozen at micro-batch 16 / accumulation 1 |
| 50-update smoke | PASS | finite-loss smoke artifact under the immutable-image run root |
| 500-update mini + RELLIS validation | PASS | isolated mini-run completed before full training |
| optimizer/scheduler resume | PASS | optimizer, scheduler and update counter resumed consistently |
| Stage 1 4k + Stage 2 40k | PASS | `/workspace/adom/runs/semantic20/eadom/b2-capacity-domain/seed42/full/b2` |
| checkpoint freeze from RELLIS val only | PASS | iter 9,000; SHA-256 `b1b9cded88fa091d503fb48c0fd1f9fafd3df938030bb767a04d7a9aab96707b` |
| fresh RELLIS 899 evaluation | PASS | 899 masks per condition and metric provenance under the paper-eval root |
| fresh Korean held-out 61 evaluation | PASS after checkpoint freeze | 61 masks per condition; test-only and never recipe, threshold, early-stop, or checkpoint input |

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

## Full training and checkpoint freeze

The immutable training image and `/opt/adom` source reported Git SHA
`7cafc31683934d6b8b224f0ebd1592d5d5c1c72c`. Data, checkpoints, logs and W&B
state stayed under `/workspace`. The full run completed at
`2026-08-27T02:53:38.962098+00:00`.

- run root:
  `/workspace/adom/runs/semantic20/eadom/b2-capacity-domain/seed42/full`
- full log:
  `/workspace/adom/logs/b2-eadom-capacity-domain/7cafc316-seed42/full.log`
- Stage 1 selected iter 1,500 on RELLIS validation:
  ValSupported13 mIoU `57.9788`, RareRisk4 mIoU `43.3272`, checkpoint SHA-256
  `d2568180484dbbfb6b7f76b219f1449f46c889b0c908484569b5c9fb5dc31180`
- Stage 1 frozen-backbone identity: PASS; before/after SHA-256
  `26c048e4e7b6814f5bbe3d3814dc66a50859748748aa95bf1970b3f3dd21b3ad`
- Stage 2 selected iter 9,000 using RELLIS validation only:
  ValSupported13 mIoU `61.4633`, RareRisk4 mIoU `46.3990`
- selected checkpoint:
  `/workspace/adom/runs/semantic20/eadom/b2-capacity-domain/seed42/full/b2/stage2/best_clean_selection_iter_9000.pth`
- selected checkpoint SHA-256:
  `b1b9cded88fa091d503fb48c0fd1f9fafd3df938030bb767a04d7a9aab96707b`
- final iter 40,000 validation mIoU was `60.3551`; it was not selected.
- W&B Stage 1: <https://wandb.ai/imtaebin83-seoul-national-university/adom/runs/full-b2-eadom-stage1-full>
- W&B Stage 2: <https://wandb.ai/imtaebin83-seoul-national-university/adom/runs/full-b2-eadom-stage2-full>

Training kept canonical test inference disabled. The Korean held-out was not
read for recipe, threshold, early stopping or checkpoint selection.

## Fresh canonical evaluation provenance

The raw evaluation root is:

`/workspace/adom/paper_eval_outputs/20260827T043544Z-b2-capacity-domain`

Both conditions used the B2 shared inference config
`segformer_b2_stage2_e0_rellis.py`, seed 42, TTA off, and direct MMSeg inference.
The evaluator saved one indexed prediction mask and per-image confusion for every
manifest row: RELLIS `899 + 899`, Korean `61 + 61`. The checkpoint audit,
per-file image/mask audit, manifest digest and ordered-manifest digest remained
fail-closed.

The original evaluation Pod stopped after both RELLIS conditions completed. The
Korean test-only evaluations resumed on Pod `yf7hvjgxyfzeu4`. Its image metadata
reported Git SHA `9d4f08e4d12af58eae96a99ea8a75eccfc5f6e90`, while the original evaluation
environment recorded `7cafc31683934d6b8b224f0ebd1592d5d5c1c72c`. This is recorded rather
than hidden. The resumed evaluation was accepted because all executable inputs
to this inference path were identical: resolved B2 evaluation-contract SHA-256
`4adfcb3ae550274ed3436c695c872e030c804bb8c16c09025958797312d8d592`,
config SHA-256 `bd73ef864eddd9fbd0ac7f5b6174dbc718a438cf3d3bc939b41eb3a79769deca`,
frozen evaluator SHA-256
`38541321f4691cf0f11d62b200d9ea71c74a211a14dc62b3864c9e08edaf4044`,
checkpoint hashes, manifest hashes and package versions. The Git diff between
those image metadata SHAs does not modify the shared inference config or
`adom.mmseg` implementation.

Key provenance SHA-256 values:

| Artifact | SHA-256 |
| --- | --- |
| original `environment.json` | `09955030138c0ff9e48b4c4d2e1df3045ab6eb4235a76eb13a9317ed5a58d65d` |
| resume `resume_environment.json` | `1abf84637689f5bed344b362fc7c981094fab88b90a03c7269f7686dc29c731a` |
| `checkpoint_manifest.json` | `866bcdc47630ffd184b8b899fd051d7cc18f14bcfa9ec414fe36fa59842e3b02` |
| `dataset_manifest_summary.json` | `be1f56557ad07cb5ab17c09f37fc2a98e996178d8ee366caa4b4c11f3bdba566` |
| B2-E0 RELLIS summary | `032d7cc078a911944d23862f8075373cc6862d5e53930cd0417d94813902784b` |
| B2-E-ADOM RELLIS summary | `9617e7dd0126caaff3c08daa0f35d12cc8eff635f9230ef3223d521e0f9ef93d` |
| B2-E0 Korean summary | `3dbcab19828dc9a44d8864cc403b280734bfd5974d903ef4a6b1deef7286735f` |
| B2-E-ADOM Korean summary | `1dcd7500463bdbd1f4d945e909692317130e909c2e2a868e222b4031694e5de8` |
| RELLIS paired-bootstrap JSON | `3b78cf7dc0a4bcd3ad9c4df49aa5863d308298a374631144da82c2215de2aa9e` |
| Korean paired-bootstrap JSON | `2d93b4337fba5d9c1dcd5768863f2a0e462db461b1da7557e3fe54b6c144d557` |

## Results

### Korean held-out target-only diagnostic

All values are percent. `common mIoU` is the mean over log and rubble only.

| Capacity | Condition | common mIoU | log IoU | log recall | rubble IoU | rubble recall |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| B0 | E0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| B0 | E-ADOM | 56.9586 | 71.9334 | 100.0000 | 41.9838 | 41.9838 |
| B2 | E0 | 0.1156 | 0.2090 | 0.2090 | 0.0222 | 0.0222 |
| B2 | E-ADOM | 95.4852 | 96.7669 | 100.0000 | 94.2035 | 94.2035 |

For B2, E-ADOM minus E0 common mIoU is `+95.3696 pp`. The sequence-aware
10,000-sample paired bootstrap gives 95% CI `[94.1813, 99.7910]` over the two
available sequence units. This interval must not be mistaken for broad field
coverage: each class is positive in only one unit. Consequently the class-level
log/rubble intervals remain `INSUFFICIENT_SUPPORT`.

### RELLIS canonical test source guardrail

| Capacity | Condition | native-supported mIoU | common mIoU | aAcc |
| --- | --- | ---: | ---: | ---: |
| B0 | E0 | 59.1118 | 46.8318 | 89.7847 |
| B0 | E-ADOM | 58.0352 | 51.8470 | 89.4871 |
| B2 | E0 | 58.4762 | 53.5174 | 89.5352 |
| B2 | E-ADOM | 61.4450 | 56.0096 | 89.5953 |

B2 adaptation changes native-supported mIoU by `+2.9688 pp`, so the preregistered
no-more-than-2 pp decline guardrail passes. Mandatory class IoU deltas are all
positive: log `+2.5742`, rubble `+2.4102`, barrier `+17.3964`, mud `+4.6280`,
puddle `+4.9986`, concrete `+2.8638 pp`. Across all 11 supported classes, the
worst IoU delta is bush `-1.9217 pp` (tree is `-0.5589 pp`). The RELLIS manifest
contains only one independent sequence unit, so its paired-bootstrap intervals
remain `INSUFFICIENT_SUPPORT` rather than using frames as independent samples.

### Difference-in-differences

`Interaction = (B2 E-ADOM - B2 E0) - (B0 E-ADOM - B0 E0)`.

| Korean metric | B0 adaptation delta | B2 adaptation delta | Interaction |
| --- | ---: | ---: | ---: |
| common mIoU | +56.9586 | +95.3696 | +38.4111 |
| log IoU | +71.9334 | +96.5579 | +24.6245 |
| log recall | +100.0000 | +99.7910 | -0.2090 |
| rubble IoU | +41.9838 | +94.1813 | +52.1976 |
| rubble recall | +41.9838 | +94.1813 | +52.1975 |

## Preregistered H1-H5 interpretation

- **H1 supported strongly on this diagnostic.** B2 target gain is `+95.3696 pp`,
  well above the 20 pp operational threshold, and both class recalls are nonzero.
  Increasing capacity alone did not remove the observed Korean failure.
- **H2 shows only a negligible numerical capacity-only contribution.** B2-E0
  exceeds B0-E0 by `+0.1156 pp` common mIoU and produces a few true-positive
  pixels, but log/rubble recall remains approximately zero. This is not an
  operational zero-shot recovery.
- **H3 yields a positive interaction candidate.** Common mIoU, log IoU and rubble
  IoU interactions exceed the preregistered `+10 pp` exploratory threshold.
  Rubble recall also does; log recall is saturated for both adapted models and
  shows no interaction. Single-seed and sequence-support limits prohibit a
  population-level synergy claim.
- **H4 supported.** Target recovery did not trade away average RELLIS performance;
  native-supported mIoU improved. Small bush/tree regressions remain visible and
  are not hidden by the mean.
- **H5 supported descriptively.** On RELLIS E0, B2 versus B0 changes log IoU by
  `-0.1600 pp` but rubble by `+13.5312 pp`. After adaptation, B2 versus B0 changes
  Korean log IoU by `+24.8336 pp` and rubble by `+52.2197 pp`. The rare-class
  response is class dependent, with the larger observed capacity/adaptation gain
  on rubble.

The conservative conclusion is that target-domain supervision is the dominant
factor for this failure case, while capacity modifies the magnitude in a
class-dependent way. The positive interaction passes the preregistered threshold
for considering a separate B5 protocol, but this run does not execute B5. More
independent Korean positive, negative and co-occurrence sequences remain more
important for a strong claim than another single-seed backbone result.

## Verification on handoff

- remote fail-closed final evaluation assertion: PASS for four summaries,
  checkpoint SHA-256, evaluation contract, ordered manifests, split, image count,
  prediction count and locked-test training status
- focused local unittest: 24 passed, 1 training-image-only skip
- full CI unittest baseline: 125 passed, 12 expected environment/hardware skips
- `python scripts/check_git_artifacts.py`: PASS
- `git diff --check`: PASS

The first host-Python test attempt lacked project/NumPy dependencies, and the
first bundled-Python full-suite attempt lacked PyYAML. These were environment
setup failures before relevant assertions, not test regressions. Final results
above use the bundled workspace Python plus a temporary PyYAML 6.0.3 target; no
test dependency or generated output was added to Git.
