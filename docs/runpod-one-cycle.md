# RunPod 학습·평가·ONNX 1-cycle

> 이 문서는 Cost4 reference cycle을 설명한다. Cost4의 selection/export 규칙을
> Semantic20에 그대로 적용하지 않는다.

## 사전 조건

- A100 80GB Secure Cloud RunPod
- ADOM training image
- Git SHA image의 `/opt/adom`에 코드/config/script 포함
- `/workspace/adom/datasets/processed/rellis3d`에 strict QC를 통과한 canonical package
- `/workspace/adom/runs`에 영속 Network Volume mount

Docker image는 Python 3.10, PyTorch 2.1/CUDA 12.2, NumPy `1.24.4`, OpenCV
`4.8.0.76`, MMCV `2.1.0`, MMEngine `0.10.7`, MMSegmentation `1.2.2`,
MMDeploy `1.3.1`을 assert한다. RunPod 시작 시 doctor가 `mmcv.ops`, 실제 CUDA
GPU, package checksum까지 다시 검사한다.

## 실행

```bash
cd /opt/adom

bash scripts/run_training_cycle.sh \
  --dataset /workspace/adom/datasets/processed/rellis3d \
  --models b0,b2 \
  --output /workspace/adom/runs/$(date -u +%Y%m%dT%H%M%SZ)
```

순서는 다음과 같다.

1. runtime doctor, strict QC, committed mapping/split hash 비교
2. B0 GPU micro-batch probe
3. B0 Stage 1 head-only와 backbone hash 불변 audit
4. Stage 1 best weight를 load한 B0 Stage 2 end-to-end 학습
5. test metric JSON과 backbone 변경 audit
6. static ONNX `1×3×384×640`, `1×3×384×384`
7. PyTorch/ONNX logits 최대 오차와 pixel argmax parity
8. B0 전체 gate 통과 후 B2에서 같은 순서

Stage 2는 `load_from`을 사용하고 `resume=False`이므로 Stage 1 optimizer state를
이어받지 않는다. 단, 중단된 Stage 2를 `--resume`하면 Stage 2 자체의 optimizer와
scheduler를 이어받는다. 각 model의 micro-batch는 probe 결과를 사용하며 gradient
accumulation으로 effective batch 16 이상을 유지한다.

## 재개

```bash
bash scripts/run_training_cycle.sh \
  --dataset /workspace/adom/datasets/processed/rellis3d \
  --models b0,b2 \
  --output /workspace/adom/runs/<existing-run-id> \
  --resume
```

`status.json`에서 `completed`이고 필수 artifact가 실제 존재하는 phase만
건너뛴다. 진행 중인 학습 phase는 `last_checkpoint`가 가리키는 iteration
checkpoint에서 optimizer/scheduler까지 복구한다. checkpoint glob은 쓰지 않으며
work directory에 best mIoU checkpoint가 정확히 하나가 아니면 중단한다.

기본 checkpoint 주기는 500 iteration이며 `ADOM_CHECKPOINT_INTERVAL`로 변경할
수 있다. W&B run ID는 `<logical-run>-<model>-<phase>`로 고정되어 같은 phase의
재실행이 기존 W&B run을 이어간다. TensorBoard event는 동일 work directory에
남는다.

## 결과

```text
<run-id>/
├── doctor.json
├── status.json
├── summary.json
├── summary.csv
├── b0/
│   ├── batch_plan.json
│   ├── stage1/
│   ├── stage2/
│   ├── test/test_metrics.json
│   └── onnx/{640x384,384x384}/
└── b2/
    └── ...
```

각 ONNX 디렉터리에는 `end2end.onnx`, MMDeploy `deploy.json`,
`parity.json`, `metadata.json`이 있다. metadata는 class/normalization/padding
규약, dataset/config/checkpoint/ONNX checksum, Git SHA를 기록한다.

RunPod 산출물에는 TensorRT engine이 없어야 한다. `.engine`은 JetPack,
TensorRT, CUDA, GPU 아키텍처가 실제 target과 결합되므로 NVIDIA Jetson Orin
Nano 8GB에서 별도 생성한다.
