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
| [0008b](0008-semantic20-perception-foundation.md) | Semantic20 autonomous perception foundation | Accepted | ontology별 runtime topic 분리 |
| [0009](0009-semantic20-local-gap-planning.md) | Semantic20 local gap planning | Superseded in part | 방향 선택/control은 0010 우선 |
| [0010](0010-low-level-tree-autonomy-logging.md) | Low-level direction-tree autonomy and logging | Superseded in part | tree 방향 선택은 0019 우선; logging 계약 유지 |
| [0011](0011-live-bag-bandwidth-policy.md) | Live autonomy bag bandwidth policy | Superseded in part | 0016이 live bag topic 범위를 더 축소 |
| [0012](0012-planner-source-age-tolerance.md) | Planner source-age tolerance | Accepted | camera source age 폐기 기준 0.80초 |
| [0013](0013-lethal-only-costmap-inflation.md) | Lethal-only semantic costmap inflation | Accepted | cost 90 이상만 inflation seed로 사용 |
| [0014](0014-planned-speed-handoff.md) | Preserve planner speed through local control | Superseded in part | speed handoff 유지; 6 m/s mapping은 0015가 대체 |
| [0015](0015-imu-aided-12mps-pwm-calibration.md) | IMU-aided nominal 12 m/s PWM calibration | Accepted | 1500..2000 us = 0..12 m/s, IMU bias/ZUPT/P-feedback |
| [0016](0016-lightweight-autonomy-evidence-bag.md) | Lightweight autonomy evidence bag | Accepted | t2는 수치·상태·raw GPS만 기록; rec는 RGB 전용 유지 |
| [0017](0017-zed-depth-ground-filter.md) | ZED depth and ground-frame filtering | Accepted | NEURAL 0.30--8.0 m, verified 0.21 m mount height |
| [0018](0018-autonomy-speed-cap-from-bag-latency.md) | Autonomous speed cap from bag latency | Accepted | planner/control hard ceiling 0.75 m/s |
| [0019](0019-gap-guided-direction-tree.md) | Gap-guided direction tree | Accepted | 중앙 장애물에서 gap 첫 조향 + 25개 tree |
| [0020](0020-blocked-release-debounce.md) | BLOCKED release debounce | Accepted | 3개 연속 clear costmap 뒤 주행 재개 |
| [0020](0020-sampled-semantic-autonomy-evidence.md) | Sampled semantic autonomy evidence | Accepted | 2 Hz mask sample + costmap grid + per-class status |

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
