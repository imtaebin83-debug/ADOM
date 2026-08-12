# Project Status Dashboard

이 폴더는 프로젝트별 실제 진행 상태를 공유한다. 목표, interface, 책임의 정의는
[ADOM_CONTEXT.md](../../ADOM_CONTEXT.md)에만 두고 여기에는 완료 증거, blocker,
다음 hand-off를 기록한다.

## Portfolio

| Project | Priority | State | Owner(s) | Next gate | Detail |
| --- | --- | --- | --- | --- | --- |
| D-5 RC Car live stop PoC | P0 | Active | 전원 | Jetson file inference + control hardware 독립 통과 | [status](d5-poc.md) |
| Phase 1 Semantic20 Clean v1 | P1 | Paused for PoC | 태빈 | E2 package 확정 후 paired seeds | [TODO](../TODO.md), [registry](../experiments/phase1-semantic20/README.md) |
| TA0/TA1/TA2 dataset readiness | P1 | Active | 태빈 | ADOM standalone conversion + TA superset validation | [RunPod inventory](runpod-dataset-inventory-2026-08-12.md), [runbook](../experiments/phase1-semantic20/target-adaptation-runbook.md) |
| Semantic23 unified dataset | P2 | Backlog | 미지정 | ontology/provenance/license audit | [Source of Truth](../../ADOM_CONTEXT.md#15-발표-이후-1015일-목표) |
| Deployment/MLOps web UI | P2 | Backlog | 태빈 | PoC artifact contract 동결 | [Source of Truth](../../ADOM_CONTEXT.md#15-발표-이후-1015일-목표) |

## Status rules

- 상태는 `Planned`, `Active`, `Blocked`, `Paused`, `Complete` 중 하나만 쓴다.
- `Complete`에는 재현 명령, PR, 로그, artifact SHA, 영상 등 evidence가 필요하다.
- `Blocked`에는 증상, 재현 방법, 해제 조건, 담당자와 다음 확인 시점을 기록한다.
- 숫자와 interface는 실측 전 `Unverified`로 표시한다.
- 상세 문서는 최소 하루 두 번 또는 gate 통과 직후 갱신한다.
- 범위나 성공 기준이 변하면 status만 고치지 않고 Source of Truth와 decision record를
  함께 수정한다.

새 프로젝트는 [_template.md](_template.md)를 복사해 작성한다.
