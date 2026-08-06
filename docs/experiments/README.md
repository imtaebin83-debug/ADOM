# Experiment records

모델 버전별 목적, 변경점, 학습 결과, 해석과 후속 결정을 보관한다.
W&B는 원시 history와 artifact의 보관소이고, 이 폴더는 연구 판단의
재현 가능한 요약을 담당한다.

## Structure

```text
docs/experiments/
├── README.md
├── _templates/
│   └── run-record.md
└── phase1-semantic20/
    ├── README.md
    ├── protocols/
    │   └── clean-baseline-v1.md
    └── versions/
        ├── v0-legacy/
        │   ├── README.md
        │   └── b0-e0-seed42.md
        └── v1-clean-baseline/
            └── README.md
```

## Record boundaries

- `protocols/`: run 전에 고정하는 metric, split, model selection 규칙
- `versions/`: 같은 연구 계약을 공유하는 model/data recipe 묶음
- 개별 run 문서: 실제 실행 조건, 결과, 해석, artifact 링크
- architecture, dataset, loss 또는 split 정책이 바뀌면 같은 run을 덮어쓰지
  않고 새 version 또는 새 experiment condition을 만든다.

## Required identifiers

모든 run 문서에는 다음을 기록한다.

- experiment version과 condition (`E0`, `E1`, `E2`)
- model (`B0`, `B2`)과 seed
- ontology와 mapping version
- dataset/split digest
- loss, optimizer, schedule, input resolution
- git commit, container/image revision, W&B run URL
- validation으로 선택한 checkpoint와 선택 metric
- test가 실행되었는지와 실행 사유
- raw metric, fixed-support metric, class metric, confusion matrix 위치
- 결과 해석, 한계, 다음 결정

## Naming

권장 파일명은 `<model>-<condition>-seed<seed>.md`이다. 예:

- `b0-e0-seed42.md`
- `b2-e2-seed44.md`

`E1` 같은 기존 identifier의 의미를 바꾸지 않는다. 데이터 소스가 추가되면
새 condition을 만든다.
