# RunPod 학습·평가·ONNX 1-cycle

## 사전 조건

- RTX 4090 또는 A6000 RunPod
- ADOM training image
- `/workspace/adom/repo`에 저장소 mount
- `/workspace/adom/datasets/rellis3d`에 strict QC와 checksum-bound 수동 preview
  승인을 통과한 canonical v2 package
- `/workspace/adom/outputs`에 영속 storage mount

Docker image는 Python 3.10, PyTorch 2.1/CUDA 12.2, NumPy `1.24.4`, OpenCV
`4.8.0.76`, MMCV `2.1.0`, MMEngine `0.10.7`, MMSegmentation `1.2.2`,
MMDeploy `1.3.1`을 assert한다. RunPod 시작 시 doctor가 `mmcv.ops`, 실제 CUDA
GPU, package checksum까지 다시 검사한다.

## 실행

```bash
cd /workspace/adom/repo

bash scripts/run_training_cycle.sh \
  --dataset /workspace/adom/datasets/rellis3d \
  --models b0,b2 \
  --output /workspace/adom/outputs/runs/$(date -u +%Y%m%dT%H%M%SZ)
```

ONNX parity sample 수는 기본 16개이며 필요하면 `--parity-samples N`으로
조정한다.

순서는 다음과 같다.

1. runtime doctor, strict QC, committed mapping/split hash 비교
2. B0 Stage 1/Stage 2 각각의 GPU micro-batch probe
3. B0 Stage 1 head-only와 backbone hash 불변 audit
4. Stage 1 best weight를 load한 B0 Stage 2 end-to-end 학습
5. test metric JSON과 backbone 변경 audit
6. static ONNX `1×3×384×640`, `1×3×384×384`
7. class/ignore/sequence를 대표하는 기본 16개 test image에서 PyTorch/ONNX
   logits 최대 오차와 pixel argmax parity
8. B0 전체 gate 통과 후 B2에서 같은 순서

Stage 2는 `load_from`을 사용하고 `resume=False`이므로 Stage 1 optimizer state를
이어받지 않는다. Stage 1의 frozen-backbone probe 결과를 Stage 2에 재사용하지
않으며, 단계별 probe 결과에 따라 gradient accumulation으로 effective batch
16 이상을 유지한다.

## 재개

```bash
bash scripts/run_training_cycle.sh \
  --dataset /workspace/adom/datasets/rellis3d \
  --models b0,b2 \
  --output /workspace/adom/outputs/runs/<existing-run-id> \
  --resume
```

runtime doctor와 dataset QC는 resume에서도 다시 수행한다. Git SHA, source/config
tree hash, dataset checksum, mapping, model 목록, parity sample 수, GPU/VRAM,
package version이 이전 실행과 같고 artifact checksum까지 일치하는 phase만
건너뛴다. 하나라도 달라지면 별도 output에서 새 run을 시작해야 한다.
checkpoint glob은 쓰지 않으며 work directory에 best mIoU checkpoint가 정확히
하나가 아니면 중단한다.

## 결과

```text
<run-id>/
├── doctor.json
├── status.json
├── summary.json
├── summary.csv
├── parity_inputs.json
├── b0/
│   ├── batch_plan.json
│   ├── stage1/
│   ├── stage2/
│   ├── test/test_metrics.json
│   └── onnx/{640x384,384x384}/
└── b2/
    └── ...
```

각 ONNX 디렉터리에는 `end2end.onnx`, MMDeploy `deploy.json`, `detail.json`,
`pipeline.json`, `parity.json`, `metadata.json`이 있다. metadata는 B0/B2 variant,
class/normalization/padding 규약과 dataset/config/checkpoint/test metric/ONNX 및
MMDeploy JSON checksum, Git SHA를 기록한다.

RunPod 산출물에는 TensorRT engine이 없어야 한다. `.engine`은 JetPack,
TensorRT, CUDA, GPU 아키텍처가 실제 target과 결합되므로 NVIDIA Jetson Orin
Nano 8GB에서 별도 생성한다.
