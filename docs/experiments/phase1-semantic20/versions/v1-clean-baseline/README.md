# v1 Clean baseline

- Status: protocol accepted; runtime/config implementation complete
- Ontology: Semantic20
- Primary goal: multi-source train data가 target-domain 전체 성능을 보존하면서
  pole과 rubble을 개선하는지 검증

## Planned conditions

| Condition | Train data | Primary causal question |
| --- | --- | --- |
| E0 | RELLIS | clean model baseline |
| E1 | RELLIS + RUGD + YCOR | 기존 bridge data의 효과 |
| E2 | E1 + GOOSE direct-only | GOOSE의 추가 pole 효과 |

## Required improvements from v0

- GT-supported fixed class averaging과 denominator logging
- RareRisk-4, AugmentedRisk-2, TerrainHazard, AbsentClassFP panel
- approved constrained validation checkpoint selection
- validation-only recipe/model selection과 final-model-only test
- mapping/dataset/split digest 및 per-source coverage artifact
- confusion matrix, Precision, Recall, F1/Dice의 영구 저장
- paired 3-seed 비교와 mean/std 보고
- model, data, loss, schedule 변경을 분리한 experiment ID

## Implemented contract

- E0와 E1의 의미를 유지하고 E2 config/runtime contract를 추가했다.
- E2는 RELLIS/RUGD/YCOR 수량을 그대로 유지하면서 GOOSE train sample이
  추가된 manifest만 허용한다.
- GOOSE는 `goose_direct_mapping.yaml`의 Semantic20 direct mapping만 허용한다.
- `ValSupported13`, `TestSupported11`, `Core11`, `RareRisk-4`,
  `AugmentedRisk-2`, `TerrainHazard`, `AbsentClassFP`를 JSON과 W&B에 기록한다.
- Precision, Recall, IoU, F1/Dice, confusion matrix를 evaluation마다 보존한다.
- approved constrained checkpoint hook이 선택 후보 weight와 결정 JSON을
  보존한다.
- canonical test는 CLI와 MMEngine `before_test` hook에서 이중 잠금되며,
  `--run-test --final-test-model`을 함께 준 final run만 unlock token을 받는다.
- dataset validation에서 class별 pixel/image support와 mapping/manifest/split,
  decoded image/mask/content digest를 생성한다.
- seed는 CLI에서 주입하며 deterministic mode를 사용한다.
- paired seed aggregator가 각 조건 내부의 dataset content digest와 B0/B2 존재를
  검사한 뒤 42/43/44의 mean, sample std, paired delta를 만든다.
- B0/B2는 같은 dataset base, CE, optimizer-update schedule을 상속한다.

## Run order

각 seed는 별도 output root를 사용한다.

```powershell
python -m adom.runtime.semantic20_cycle --experiment e1 --models b0,b2 --seed 42 --dataset <E1_ROOT> --output <OUT>/e1-seed42
python -m adom.runtime.semantic20_cycle --experiment e2 --models b0,b2 --seed 42 --dataset <E2_ROOT> --output <OUT>/e2-seed42
```

같은 명령을 seed 43, 44에 반복한다. 이 단계에서는 test가 실행되지 않는다.

```powershell
python -m adom.runtime.semantic20_aggregate `
  --baseline <OUT>/e1-seed42/summary.json <OUT>/e1-seed43/summary.json <OUT>/e1-seed44/summary.json `
  --candidate <OUT>/e2-seed42/summary.json <OUT>/e2-seed43/summary.json <OUT>/e2-seed44/summary.json `
  --output <OUT>/e1-vs-e2-paired.json
```

validation과 paired 결과로 model/seed를 최종 확정한 뒤 해당 output에서만 test를
명시적으로 해제한다.

```powershell
python -m adom.runtime.semantic20_cycle --experiment e2 --models b0 --seed 42 --dataset <E2_ROOT> --output <OUT>/e2-seed42 --resume --run-test --final-test-model b0
```

## Required E2 dataset artifact

runtime은 E2 package를 생성하지 않는다. 학습 전에 다음을 포함한 검증 완료
package가 필요하다.

- `_SUCCESS`
- `manifest.csv` with `rellis3d`, `rugd`, `ycor`, `goose`
- canonical RELLIS-only `splits/val.txt`, `splits/test.txt`
- four-source `splits/train.txt`
- `results/final_check.json` with `status=PASS`

E2 package 생성기는 GOOSE 실제 저장 구조와 확정 train/diagnostic split을 입력으로
별도 연결해야 한다.

세부 제안과 미결정 항목은
[Clean Baseline v1 protocol](../../protocols/clean-baseline-v1.md)에 기록한다.
