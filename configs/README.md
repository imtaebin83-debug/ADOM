# Configs

ADOM Phase 1의 학습 대상은 Cost4의 `0..3`이며 `255`는 loss/metric에서 제외한다.

```text
configs/
├── adom/
│   ├── _base_/                 # model, dataset, schedule, runtime
│   ├── export/                 # profile별 keep-ratio + pad model config
│   └── segformer_*_stage*.py   # B0/B2 × Stage 1/2
├── datasets/rellis3d/
│   └── label_mapping.yaml      # source ID → Cost4의 유일한 mapping source
└── deployment/
    └── mmseg_onnxruntime_*.py  # static ONNX 640×384, 384×384
```

- Stage 1: MiT backbone freeze/eval, head-only, 4k iter, LR `6e-4`
- Stage 2: Stage 1 weight만 load하고 optimizer reset, end-to-end 40k iter,
  LR `6e-5`, early stopping
- 공통: `num_classes=4`, `ignore_index=255`, `reduce_zero_label=False`
- export: keep-ratio resize 후 오른쪽/아래쪽 padding, static batch 1

직접 `train.py`를 호출하기보다 `scripts/run_training_cycle.sh`를 사용한다. 이
실행기가 dataset checksum, freeze audit, checkpoint 연결, test, ONNX parity를
하나의 상태 파일로 묶는다.
