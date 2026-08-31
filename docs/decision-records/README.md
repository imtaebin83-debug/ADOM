# Decision Records

되돌리기 비용이 크거나 여러 팀원의 작업 계약을 바꾸는 결정을 기록한다. 현재
유효한 상태는 항상 루트 [ADOM_CONTEXT.md](../../ADOM_CONTEXT.md)가 우선하며,
decision record는 결정 당시의 맥락과 근거를 보존한다.

## When to write one

- 프로젝트 범위, 성공 기준, 모델/데이터 ontology 변경
- 외부 interface, 배포 stack, 안전 정책 변경
- 실험 비교의 split, metric, checkpoint 선택 규칙 변경
- 기존 결정을 폐기하거나 대체하는 피봇

단순 구현 세부사항, 일회성 TODO, 회의 전체 내용은 기록 대상이 아니다.

## Status vocabulary

- `Proposed`: 검토 중이며 구현 계약이 아님
- `Accepted`: 승인돼 Source of Truth에 반영됨
- `Superseded`: 새 record가 대체함
- `Rejected`: 검토했으나 채택하지 않음

## Index

| ID | Decision | Status | Current note |
| --- | --- | --- | --- |
| [0001](0001-repo-structure.md) | Repository structure | Accepted | 폴더 책임 유지 |
| [0002](0002-initial-benchmark-scope.md) | Initial benchmark scope | Superseded in part | D-5 범위는 0006이 대체 |
| [0003](0003-5th-meeting-decision-logs.md) | 5차 회의 결정 | Superseded in part | 데모/Cost5 범위는 후속 결정 우선 |
| [0004](0004-runpod-mlops-policy.md) | RunPod MLOps policy | Accepted | 학습 인프라 기준 |
| [0005](0005-semantic20-segformer.md) | Semantic20 SegFormer baseline | Accepted | 연구 baseline; D-5에는 B0 E0 사용 |
| [0006](0006-d5-poc-pivot.md) | D-5 live stop PoC pivot | Accepted | 현재 발표 범위 |
| [0007](0007-camera-only-data-collection.md) | Camera-only data collection | Accepted | rosbag은 ZED RGB 토픽만 기록 |
| [0008](0008-defer-target-selection.md) | Defer target selection to field visualization | Accepted | 0006의 log 기본 target 부분 대체 |
| [0009](0009-target-adaptation-comparison.md) | B0-E0 target-adaptation comparison | Accepted | TA0/TA1/TA2 병렬 비교 계약 |
| [0010](0010-ta0-common-recipe-selection.md) | TA0 common fine-tuning recipe selection | Accepted | 장기 공통 recipe discovery 계약 |
| [0011](0011-emergency-eadom-e0-recipe.md) | Emergency E-ADOM using the E0 recipe | Accepted | 내일 Jetson 배포용 data-only 단일-seed 경로 |
| [0012](0012-emergency-eadom-rtx4090-runtime.md) | Emergency E-ADOM on RTX 4090 | Accepted | A100 재고 소진 시 Ada-compatible 긴급 runtime 계약 |
| [0013](0013-b5-capacity-domain-preregistration.md) | B5 capacity × domain preregistration | Accepted | B2 uncertainty GO artifact와 exact GPU profile 없이는 실행 금지 |
| [0014](0014-b5-blackwell-32gb-runtime-profiles.md) | B5 Blackwell 32GB runtime profiles | Accepted | RTX PRO 4500/RTX 5090 exact profile, PTX-JIT warning과 probe 의무화 |

`decision_logs.md`는 번호형 record 도입 전의 historical changelog다. 새 결정은 개별
번호 파일로 만든다.

## Template

```markdown
# NNNN. Short decision title

- Status: Proposed / Accepted / Superseded / Rejected
- Date: YYYY-MM-DD
- Owners: names or roles
- Supersedes: record IDs or none

## Context
## Decision
## Rationale and evidence
## Alternatives considered
## Consequences
## Validation and rollback
```

파일명은 `NNNN-short-title.md`를 사용한다. 결정이 바뀌면 기존 문서를 수정해 새
결론처럼 만들지 않고, 새 record에서 변경 이유와 `Supersedes` 관계를 남긴다.
