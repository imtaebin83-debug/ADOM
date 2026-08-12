# ADOM Docs Hub

이 폴더는 ADOM 협업 문서의 진입점이다. 현재 범위와 계약의 유일한 기준은 저장소
루트의 [ADOM_CONTEXT.md](../ADOM_CONTEXT.md)이며, `docs/`는 진행 증거와 결정 근거,
재현 정보를 역할별로 분리해 보관한다.

## Start here

1. [Source of Truth](../ADOM_CONTEXT.md) — 지금 무엇을 왜 만드는가
2. [Project status](status/README.md) — 무엇이 끝났고 무엇이 막혔는가
3. [Decision records](decision-records/README.md) — 중요한 결정의 근거와 대안
4. [Experiment registry](experiments/README.md) — 어떤 조건으로 무엇을 측정했는가
5. [Final presentation](presentation/README.md) — 최종 발표의 주장, 구성, 필요 evidence

## Document map

| 위치 | 용도 | 쓰지 않는 내용 |
| --- | --- | --- |
| `../ADOM_CONTEXT.md` | 현재 범위, 계약, R&R, gates, 현재 결정 | 일별 작업 로그, 원시 회의록 |
| `status/` | 프로젝트별 완료/진행/blocker/evidence/다음 hand-off | 장기 설계 설명, 결정 근거 복제 |
| `decision-records/` | 되돌리기 비싼 결정의 배경, 대안, 결과 | 단순 TODO, 회의 전체 내용 |
| `ai-collaboration/` | AI 검토에서 나온 근거·결론 후보·미검증 사항 | 원문 transcript, 비밀, 승인되지 않은 확정 표현 |
| `meeting-notes/` | 회의 논의, 결정 후보, 담당 액션 | 현재 계약의 유일한 사본 |
| `experiments/` | protocol, run identity, 결과, 해석 | 프로젝트 전체 일정 |
| `metrics/` | benchmark와 metric 정의 | 개별 run 결과 |
| `system-architecture/` | 안정화된 시스템 설계와 interface | 실측 전 topic/버전의 확정 표현 |
| `setup-guides/` | 재현 가능한 환경 구축·운영 절차 | 특정 개인 PC의 임시 설정 |
| `presentation/` | 발표 스토리보드, 결과 준비표, claim/evidence 경계 | 프로젝트 현재 상태의 원본, 생성된 대용량 미디어 |

## Minimum update workflow

### 작업을 시작할 때

- `ADOM_CONTEXT.md`에서 현재 목표와 non-goal을 확인한다.
- `status/README.md`에서 프로젝트와 담당자를 찾고 상세 상태 문서를 연다.
- 관련 decision record와 experiment protocol을 확인한다.

### 작업 중

- PR 또는 실험 링크, 로그, artifact SHA처럼 다른 사람이 확인할 수 있는 evidence를 남긴다.
- 실측하지 않은 버전, ROS topic, latency는 `미검증`으로 표시한다.
- blocker가 생기면 증상, 재현 명령, 담당자, 다음 판정 시점을 함께 기록한다.

### 범위나 계약을 바꿀 때

하나의 변경에서 다음 세 곳을 함께 갱신한다.

1. `ADOM_CONTEXT.md`: 새 현재 상태
2. 새 decision record: 배경, 대안, 결정 이유, 후속 검증
3. `status/`: 영향받은 milestone과 다음 hand-off

기존 decision record는 지우거나 다시 쓰지 않는다. 새 record에서 `Supersedes` 관계를
남긴다.

### 회의 또는 AI 대화가 끝났을 때

- 단순 설명이나 아이디어 나열은 기록하지 않아도 된다.
- 일정, 모델, 데이터, interface, 안전 기준을 바꿀 수 있는 결론만 요약한다.
- 승인되지 않은 결론은 `Proposed`, 외부 검증이 필요한 사실은 `Unverified`로 둔다.
- 승인되면 Source of Truth와 decision record에 반영하고 노트에서 링크한다.

## Review checklist

- 동일한 값이 여러 문서에서 서로 다르게 확정돼 있지 않은가?
- 현재 상태와 과거 결정이 구분되는가?
- 담당자, due/gate, evidence가 있는가?
- 실패·한계·미검증 사항이 보이는가?
- 대용량 artifact, credential, 개인 로컬 설정이 포함되지 않았는가?
