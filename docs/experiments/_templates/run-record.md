# <Model>-<Condition>-seed<Seed>

## Identity

| Field | Value |
| --- | --- |
| Status | planned / running / complete / invalid |
| Experiment version | |
| Model | |
| Dataset condition | |
| Ontology | |
| Mapping version | |
| Seed | |
| Git commit | |
| Container revision | |
| Dataset/split digest | |
| W&B group | |

## Purpose and hypothesis

- Purpose:
- Controlled hypothesis:
- Primary comparison:

## Controlled variables

| Variable | Value |
| --- | --- |
| Input/crop resolution | |
| Augmentation | |
| Loss | |
| Optimizer | |
| Stage 1 schedule | |
| Stage 2 schedule | |
| Checkpoint selection | |
| Test policy | final selected model only |

## Dataset and coverage

- Train sources and counts:
- Validation sources and counts:
- Test sources and counts:
- Fixed supported-class sets:
- Rare-risk set:
- Absent-class set:

## Runs and artifacts

| Stage | W&B run | Selected checkpoint | Artifact |
| --- | --- | --- | --- |
| Stage 1 | | | |
| Stage 2 | | | |
| Final test | | | |

## Results

### Aggregate

| Split/checkpoint | aAcc | raw MMSeg mIoU | fixed mIoU | mAcc | RareRisk mIoU |
| --- | ---: | ---: | ---: | ---: | ---: |

### Class results

| Class | GT pixels | GT images | IoU | Recall | Precision | F1/Dice |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |

### Absent-class false positives

| Class | FP pixels | Predicted area share | FP images | FP image rate |
| --- | ---: | ---: | ---: | ---: |

## Interpretation

- Supported conclusion:
- Unsupported conclusion:
- Failure modes:
- Seed uncertainty:

## Decision

- Decision:
- Follow-up:
- Known caveats:
