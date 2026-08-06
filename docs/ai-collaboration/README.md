# AI Collaboration Notes

AI와의 대화에서 프로젝트에 장기 영향을 줄 검토 결과만 짧게 보존한다. 이 폴더는
대화 transcript 저장소가 아니며, 여기의 내용만으로 프로젝트 결정이 승인되지 않는다.

## Record when

- 모델, 데이터, 평가, 배포, 일정 또는 안전 계약의 변경 후보가 생겼다.
- 최신 문헌·공식 문서 검증이 기존 가정을 바꿨다.
- 팀원이 재검증해야 하는 기술적 위험이나 반례가 발견됐다.
- 여러 대화에서 반복 사용할 hand-off 기준이나 decision rationale가 만들어졌다.

일반 설명, brainstorming 전부, 단순 코드 질의는 기록하지 않는다.

## Required fields

- 질문과 당시 context
- 확인된 사실과 evidence link
- AI의 제안 또는 비판
- 팀이 승인한 결정과 아직 승인되지 않은 제안의 분리
- 미검증 사항, owner, 다음 action
- 반영된 Source of Truth/decision record/PR 링크

## Promotion rule

```text
AI/meeting note (source)
    └─ 팀 승인 필요
        ├─ 현재 계약 변경 → ADOM_CONTEXT.md
        ├─ 중요한 이유 보존 → numbered decision record
        ├─ 실제 진행 변화 → docs/status
        └─ 실험 조건/결과 → docs/experiments
```

AI가 제안한 수치, 환경 버전, ROS interface는 담당자가 실측하기 전 `Unverified`다.
credential, 내부 개인정보, 전체 대화 원문은 커밋하지 않는다.

파일명은 `YYYY-MM-DD-topic.md`이며 [_template.md](_template.md)를 사용한다.

## Notes

- [2026-08-06 D-5 PoC technical review](2026-08-06-d5-poc-technical-review.md)
