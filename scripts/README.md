# Scripts

Phase 1 Semantic20 B0/B2 gate는 `run_semantic20_cycle.sh`로 실행한다. 데이터
root는 `--dataset` 또는 `ADOM_DATA_ROOT`로 전달하며 ONNX export는 학습과
분리되어 있다. 자세한 명령은 `docs/semantic20-runpod-gates.md`에 있다.

반복 가능한 운영 진입점만 둔다. 데이터 로직은 `src/adom/data`, MMSeg 확장은
`src/adom/mmseg`, 실행 상태 관리는 `src/adom/runtime`에 있다.

## RunPod 학습 1-cycle

```bash
bash scripts/run_training_cycle.sh \
  --dataset /workspace/adom/datasets/processed/rellis3d \
  --models b0,b2 \
  --output /workspace/adom/runs/<run-id>
```

- strict dataset QC와 GPU/runtime doctor를 먼저 수행한다.
- 실제 GPU 메모리 probe로 micro-batch를 정하고 effective batch를 16 이상으로
  유지한다.
- B0의 Stage 1, Stage 2, test, 두 ONNX parity가 모두 성공해야 B2를 시작한다.
- 중단 후 같은 output에 `--resume`을 추가하면 완료된 phase는 건너뛰고 진행
  중이던 train phase는 `last_checkpoint`에서 optimizer/scheduler와 함께 재개한다.
- 임의 glob으로 checkpoint를 선택하지 않는다. 각 stage에 best mIoU checkpoint가
  정확히 하나 있어야 다음 단계로 진행한다.

## Dataset cache

`init_workspace.sh`는 Network Volume의 canonical tar를 `/tmp/data`로 복사하고
안전하게 staging 추출한 뒤 컨테이너의 `/workspace/adom/datasets`에 노출한다.

```bash
bash scripts/init_workspace.sh \
  --dataset rellis3d \
  --archive /workspace/adom/network-volume/rellis3d-cost4-v2.tar
```

## Git artifact guard

```bash
python scripts/check_git_artifacts.py
```

학습 데이터, checkpoint, ONNX, TensorRT engine, 로그와 새 개인 절대경로가 Git에
들어오는 것을 차단한다. `study/gahyung/Datasets_Repo`는 검증 근거용 legacy
snapshot이라 검사 예외지만 canonical 코드에서는 import하지 않는다.
