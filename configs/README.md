# Configs

기존 Cost4 (`0..3`, ignore `255`) 설정은 Phase 2/reference 용도로 보존한다.
Phase 1 Semantic20은 `configs/adom/phase1_semantic20/` 아래에 분리되어 있으며
원 ontology 20개 중 void를 제외한 train ID `0..18`, ignore `255`를 사용한다.
RunPod gate와 optimizer-update 환산 규약은
[`docs/semantic20-runpod-gates.md`](../docs/semantic20-runpod-gates.md)를 따른다.

```text
configs/
├── adom/
│   ├── _base_/                 # model, dataset, schedule, runtime
│   ├── export/                 # Semantic20 E0 profile별 export config
│   │   └── cost4/              # 기존 Cost4 reference export config
│   ├── phase1_semantic20/      # Semantic20 B0/B2 E0/E1/E2 training config
│   └── segformer_*_stage*.py   # B0/B2 × Stage 1/2
├── datasets/rellis3d/
│   └── label_mapping.yaml      # source ID → Cost4의 유일한 mapping source
└── deployment/
    └── mmseg_onnxruntime_*.py  # static ONNX 640×384, 384×384
```

- Stage 1: MiT backbone freeze/eval, head-only, 4k iter, LR `6e-4`
- Stage 2: Stage 1 weight만 load하고 optimizer reset, end-to-end 40k iter,
  LR `6e-5`, early stopping
- Cost4 reference: `num_classes=4`, `ignore_index=255`
- D-5 export: Semantic20 E0, `num_classes=19`, `ignore_index=255`, raw logits
- export preprocessing: keep-ratio resize 후 오른쪽/아래쪽 padding, static batch 1

TA0 method discovery config는 같은 디렉터리의 `segformer_b0_ta0_*`로 분리한다.
`C0/I/B/L`의 LP-FT config는 Stage 1+2 합계가 6,000 optimizer update이고,
`O0 direct-FT`와 `O2 discriminative-LR`도 각각 6,000 update다. 모든 config는
`TA0AblationContractHook`으로 E0 SHA, `ta0_train`, seed, effective batch 16과 phase
update를 검사한다. 500 update를 넘는 full phase는
`ADOM_TA0_FULL_TRAINING_APPROVED=user-approved` 없이는 시작되지 않는다.
`segformer_b0_ta0_r_combined.py`는 개별 결과 검토 전 import 자체가 잠긴 provisional
interaction-check config다.

직접 `train.py`를 호출하기보다 `scripts/run_training_cycle.sh`를 사용한다. 이
실행기가 dataset checksum, freeze audit, checkpoint 연결, test, ONNX parity를
하나의 상태 파일로 묶는다.
