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

The final selected model was evaluated once on the locked canonical test. Values are
percent. The supported panels use GT support fixed independently of predictions.

| Metric | E-ADOM | B0-E0 | Approx. delta |
| --- | ---: | ---: | ---: |
| aAcc | 89.49 | 89.78 | -0.29 pp |
| TestSupported11/Core11 mIoU | 58.04 | 59.11 | -1.07 pp |
| RareRisk4 mIoU | 36.42 | 37.59 | -1.17 pp |
| AugmentedRisk2 mIoU | 31.56 | 26.67 | +4.89 pp |
| TerrainHazard mIoU | 53.85 | 57.72 | -3.87 pp |

### Supported class metrics

`puddle` IoU is reconstructed from the stored two-class TerrainHazard mean and the
stored `mud` IoU because the pasted console JSON omitted its individual IoU line.

| Class | IoU | Recall | B0-E0 IoU | Approx. IoU delta |
| --- | ---: | ---: | ---: | ---: |
| grass | 83.06 | 97.79 | 83.92 | -0.86 pp |
| tree | 76.95 | 94.40 | 76.09 | +0.86 pp |
| pole | 0.00 | 0.00 | 0.00 | 0.00 pp |
| sky | 95.46 | 97.43 | 95.29 | +0.17 pp |
| log | 40.57 | 63.37 | 40.33 | +0.24 pp |
| bush | 69.29 | 74.33 | 69.97 | -0.68 pp |
| concrete | 60.27 | 62.88 | 59.17 | +1.10 pp |
| barrier | 41.97 | 42.46 | 56.68 | -14.71 pp |
| puddle | 69.71 | 90.29 | 70.93 | -1.22 pp |
| mud | 37.98 | 39.21 | 44.51 | -6.53 pp |
| rubble | 63.12 | 65.47 | 53.34 | +9.78 pp |

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
