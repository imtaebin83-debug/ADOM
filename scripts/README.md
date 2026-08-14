# Scripts

Phase 1 Semantic20 B0/B2 gate는 `run_semantic20_cycle.sh`로 실행한다. 데이터
root는 `--dataset` 또는 `ADOM_DATA_ROOT`로 전달하며 ONNX export는 학습과
분리되어 있다. 자세한 명령은 `docs/semantic20-runpod-gates.md`에 있다.

반복 가능한 운영 진입점만 둔다. 데이터 로직은 `src/adom/data`, MMSeg 확장은
`src/adom/mmseg`, 실행 상태 관리는 `src/adom/runtime`에 있다.

## Jetson B0-E0 / E-ADOM perception

```bash
scripts/run_jetson_t4.sh b0-e0
scripts/run_jetson_t4.sh eadom
```

명시한 Semantic20 SegFormer-B0 profile의 config, checkpoint 수와 SHA256을 검증한 뒤
`adom_perception_ros`를 빌드·실행한다. 기본 checkpoint 위치는 각각
`models/checkpoints/b0-e0/`, `models/checkpoints/eadom/`이며 각 디렉터리에 정확히 한
개의 `.pth`만 허용한다. 다른 위치를 사용할 때는 `ADOM_CHECKPOINT`에 절대 경로를
지정한다. SHA override가 필요한 의도적 신규 artifact는
`ADOM_EXPECTED_CHECKPOINT_SHA256`도 함께 명시해야 한다. 자세한 설치와 `t4` wrapper는
[`SHORTCUT.md`](../SHORTCUT.md)를 따른다. RunPod, 로컬 컴퓨터와 Jetson 사이 checkpoint
전달 및 실기 검증은
[`jetson-model-checkpoint-handoff.md`](../docs/setup-guides/jetson-model-checkpoint-handoff.md)를
따른다. 현재 두 profile 모두 PyTorch/MMSeg CUDA backend이며 TensorRT ROS backend
연결은 [`docs/TODO.md`](../docs/TODO.md)에 남긴다.

## Jetson autonomy logging

```bash
scripts/run_jetson_t2.sh       # raster mask 제외
scripts/run_jetson_t2.sh mask  # 2 Hz Semantic20 evidence mask 추가
scripts/run_jetson_t2.sh evidence  # full-rate source RGB + 2 Hz mask 추가
```

모든 모드가 inference-frame별 class pixel 통계를 기록한다. Jetson의
`t2` 함수는 이 wrapper에 `"$@"`를 전달하도록 정의한다.

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
