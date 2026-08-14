# E-ADOM-B0-seed42

## Identity

| Field | Value |
| --- | --- |
| Status | training, canonical test, export image, metadata and parity complete |
| Model | SegFormer-B0 |
| Dataset condition | E-ADOM, RELLIS anchors + newly labeled standalone data |
| Ontology | Semantic20 IDs `0..18`, ignore `255` |
| Seed | 42 |
| Training recipe | legacy B0-E0 two-stage recipe, data-only emergency comparison |
| Selected checkpoint | Stage 2 `iter_26000.pth` |
| Checkpoint SHA256 | `f4cc41fd91e9df8e7aa3f726498e80636b736dfadf0e1baf338fe7c82a83399c` |
| Immutable image Git SHA | `9d4f08e4d12af58eae96a99ea8a75eccfc5f6e90` |
| W&B | [full-b0-eadom-stage2-full](https://wandb.ai/imtaebin83-seoul-national-university/adom/runs/full-b0-eadom-stage2-full/overview) |

E-ADOM was an emergency data-only experiment. It does not establish a general
Clean-v1 multi-seed claim and must not silently replace E0.

## Checkpoint selection

Selection used validation only. Stage 2 iter 26,000 was both the highest observed
`ValSupported13-mIoU` candidate among 40 evaluated checkpoints and the constrained
selection result, so canonical-test output does not justify reselecting a checkpoint.

| Validation metric | E-ADOM iter 26k | B0-E0 legacy | Delta |
| --- | ---: | ---: | ---: |
| ValSupported13 mIoU | 60.44 | 58.92 | +1.52 pp |
| RareRisk4 mIoU | 44.78 | - | - |
| log IoU | 43.13 | 37.91 | +5.22 pp |

## Canonical RELLIS test

The final selected model was evaluated once on the locked canonical test. The frozen
B0-E0 and E-ADOM checkpoints were then reproduced with the same config, metric and
899-image test split in audit `audit-20260814T013811Z`. Values are percent. The
supported panels use GT support fixed independently of predictions.

| Metric | E-ADOM | B0-E0 | Delta |
| --- | ---: | ---: | ---: |
| aAcc | 89.4871 | 89.7847 | -0.2976 pp |
| TestSupported11/Core11 mIoU | 58.0352 | 59.1118 | -1.0766 pp |
| RareRisk4 mIoU | 36.4164 | 37.5852 | -1.1688 pp |
| AugmentedRisk2 mIoU | 31.5624 | 26.6671 | +4.8953 pp |
| TerrainHazard mIoU | 53.8464 | 57.7222 | -3.8758 pp |

### Supported class metrics

The audit directly confirmed the complete class table, including `puddle`.

| ID | Class | B0-E0 IoU | E-ADOM IoU | Delta | GT supported |
| ---: | --- | ---: | ---: | ---: | --- |
| 0 | dirt | N/A | N/A | N/A | false |
| 1 | grass | 83.9225 | 83.0634 | -0.8592 pp | true |
| 2 | tree | 76.0919 | 76.9464 | +0.8545 pp | true |
| 3 | pole | 0.0000 | 0.0000 | +0.0000 pp | true |
| 4 | water | N/A | N/A | N/A | false |
| 5 | sky | 95.2903 | 95.4594 | +0.1690 pp | true |
| 6 | vehicle | N/A | N/A | N/A | false |
| 7 | object | N/A | N/A | N/A | false |
| 8 | asphalt | N/A | N/A | N/A | false |
| 9 | building | N/A | N/A | N/A | false |
| 10 | log | 40.3294 | 40.5692 | +0.2398 pp | true |
| 11 | person | N/A | N/A | N/A | false |
| 12 | fence | N/A | N/A | N/A | false |
| 13 | bush | 69.9722 | 69.2929 | -0.6793 pp | true |
| 14 | concrete | 59.1676 | 60.2665 | +1.0989 pp | true |
| 15 | barrier | 56.6771 | 41.9717 | -14.7054 pp | true |
| 16 | puddle | 70.9335 | 69.7079 | -1.2256 pp | true |
| 17 | mud | 44.5108 | 37.9848 | -6.5260 pp | true |
| 18 | rubble | 53.3341 | 63.1248 | +9.7907 pp | true |

Absent-class false positives totaled 1,259 pixels: building 436, fence 772,
object 42 and person 9; dirt, water, vehicle and asphalt were zero.

## Export hand-off

| Check | Result |
| --- | --- |
| Export image | PASS |
| Metadata validation | PASS, exit code 0 |
| PyTorch-to-ONNX parity | PASS, exit code 0 |
| All finite logits | true |
| Maximum absolute error | `4.1961669921875e-05` (limit `0.001`) |
| Overall pixel argmax agreement | `0.9999996609157986` (minimum `0.999`) |
| Minimum per-image argmax agreement | `0.9999959309895833` |
| Reference images | at least 10, all Semantic20 IDs `0..18` reported |
| ROI parity | not evaluated |
| Archive | `eadom-b0-seed42-iter26000.tar.gz` |
| Archive SHA256 | `2fb6b6a37e994b8ad4d79235f1de26c7a71f132906655ac5b96227bf64d4c94e` |

The RunPod source package was
`/workspace/adom/artifacts/eadom-b0-seed42-iter26000`, and the frozen transfer
archive was `/workspace/adom/exports/eadom-b0-seed42-iter26000.tar.gz`. These are
provenance records, not repository-relative runtime paths; the binary artifacts stay
outside Git.

## Interpretation and deployment decision

- E-ADOM improved validation and rubble test IoU, but did not improve the locked
  canonical-test overall, RareRisk4, barrier, terrain, or pole results.
- The canonical-test log gain is only about 0.24 pp, far smaller than the validation
  gain. It is not evidence of a robust log improvement.
- B0-E0 remains the default/fallback profile. E-ADOM is a separately named field A/B
  candidate for the newly collected domain, not an automatic replacement.
- Do not choose between profiles using the canonical test again. Use fixed, labeled
  field evidence or a predeclared deployment validation set for the remaining A/B.

Raw checkpoints, ONNX files, logs, `test_metrics.json`, parity images and archives are
not committed. This reviewed run record preserves their immutable identities and the
decision-relevant metrics.
