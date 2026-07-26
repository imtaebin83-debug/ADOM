# Scripts

반복 작업 자동화 스크립트를 보관합니다.

예상 범위:

- dataset preparation
- training launch helpers
- evaluation and metric aggregation
- ONNX/TensorRT export
- Jetson benchmark helpers

스크립트가 특정 실험에만 쓰이면 먼저 `experiments/<name>/`에 두고, 여러 실험에서 재사용되면 이곳으로 옮깁니다.

팀 공용 데이터 전처리 코드는 `scripts/data_preprocessing/<dataset>/`에 둡니다.
실제 raw/processed 데이터는 이 디렉토리에 두지 않으며, 실행 시 다음 경로 계약을
사용합니다.

```text
input:   /workspace/adom/datasets/<dataset>/raw
output:  /workspace/adom/outputs/preprocessing/<dataset>
mapping: /workspace/adom/repo/configs/datasets/<dataset>/
```

각 스크립트는 개인 PC 절대경로 대신 `--input-root`, `--output-root`, `--mapping`
인자를 받아야 합니다. 자세한 이동 계획은
[`docs/dataset-preprocessing-migration-plan.md`](../docs/dataset-preprocessing-migration-plan.md)를
참고합니다.
