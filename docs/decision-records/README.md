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
| [0008b](0008-semantic20-perception-foundation.md) | Semantic20 autonomous perception foundation | Accepted | ontology별 runtime topic 분리 |
| [0009](0009-semantic20-local-gap-planning.md) | Semantic20 local gap planning | Superseded in part | 방향 선택/control은 0010 우선 |
| [0010](0010-low-level-tree-autonomy-logging.md) | Low-level direction-tree autonomy and logging | Superseded in part | tree 방향 선택은 0019 우선; logging 계약 유지 |
| [0011](0011-live-bag-bandwidth-policy.md) | Live autonomy bag bandwidth policy | Superseded in part | 0016이 live bag topic 범위를 더 축소 |
| [0012](0012-planner-source-age-tolerance.md) | Planner source-age tolerance | Superseded in part | 0024 이후 costmap output age 기준 |
| [0013](0013-lethal-only-costmap-inflation.md) | Lethal-only semantic costmap inflation | Accepted | cost 90 이상만 inflation seed로 사용 |
| [0014](0014-planned-speed-handoff.md) | Preserve planner speed through local control | Superseded in part | speed handoff 유지; profile은 0029가 대체 |
| [0015](0015-imu-aided-12mps-pwm-calibration.md) | IMU-aided nominal 12 m/s PWM calibration | Accepted | 1500..2000 us = 0..12 m/s, IMU bias/ZUPT/P-feedback |
| [0016](0016-lightweight-autonomy-evidence-bag.md) | Lightweight autonomy evidence bag | Accepted | t2는 수치·상태·raw GPS만 기록; rec는 RGB 전용 유지 |
| [0017](0017-zed-depth-ground-filter.md) | ZED depth and ground-frame filtering | Accepted | NEURAL 0.30--8.0 m, verified 0.21 m mount height |
| [0018](0018-autonomy-speed-cap-from-bag-latency.md) | Autonomous speed cap from bag latency | Superseded | 0029가 autonomous profile을 대체; latency evidence는 유효 |
| [0019](0019-gap-guided-direction-tree.md) | Gap-guided direction tree | Superseded | 0021이 gap 폭/거리와 no-gap BLOCKED를 제거 |
| [0020](0020-blocked-release-debounce.md) | BLOCKED release debounce | Superseded in part | release 유지; no-gap trigger는 0021이 제거 |
| [0021](0021-side-cost-assisted-tree.md) | Side-cost-assisted direction tree | Superseded in part | 좌우 cost 선택 유지; activation은 0022 우선 |
| [0022](0022-straight-avoid-blocked-modes.md) | Straight/avoid/blocked modes | Accepted | 직진 장애물 거리로 mode 전환 |
| [0023](0023-avoid-trigger-distance-tuning.md) | Avoid trigger distance tuning | Accepted | AVOID 진입 거리 3.50 m → 1.50 m |
| [0024](0024-costmap-clock-domain-boundary.md) | Costmap clock-domain boundary | Accepted | 센서 stamp는 sync까지 유지, costmap 출력은 ROS clock |
| [0020](0020-sampled-semantic-autonomy-evidence.md) | Sampled semantic autonomy evidence | Superseded in part | status/costmap 유지; sampled mask 기본 활성화는 0025가 대체 |
| [0025](0025-optional-sampled-mask-recording.md) | Optional sampled mask recording | Accepted | `t2`는 mask 제외, `t2 mask`만 2 Hz evidence mask 추가 |
| [0026](0026-paired-manual-perception-evidence.md) | Paired manual perception evidence | Superseded in part | sampled RGB 설계는 0027이 full-rate source RGB로 대체 |
| [0027](0027-full-rate-manual-perception-evidence.md) | Full-rate manual perception evidence | Accepted | `t2 evidence`는 full-rate source RGB + 2 Hz mask 기록 |
| [0028](0028-semantic20-mask-color-monitor.md) | Semantic20 mask color monitor | Accepted | mono8 ID mask를 canonical BGR image와 JSON legend로 변환 |
| [0029](0029-autonomous-speed-profile.md) | Autonomous 0.30..3.00 m/s speed profile | Accepted | planner/local controller 범위 통일 |
| [0030](0030-sampled-semantic-preview.md) | Sampled Semantic20 preview evidence | Accepted | `t2 preview`는 2 Hz mask + 동일-frame 45% overlay 기록 |
| [0031](0031-jetson-dual-model-profiles.md) | Jetson dual Semantic20 model profiles | Accepted | `t4`는 SHA-locked `b0-e0`/`eadom`; TensorRT ROS 연결은 후속 |
| [0032](0032-empty-costmap-diagnostics.md) | Empty costmap projection diagnostics | Accepted | depth/label/height/grid 단계별 count와 `empty_reason` 상태 추가 |

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
