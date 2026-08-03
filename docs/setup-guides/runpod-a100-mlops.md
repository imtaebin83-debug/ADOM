# RunPod A100 80GB MLOps 설정

## 비용과 GPU

- Cloud: Secure Cloud
- GPU: `NVIDIA A100 80GB PCIe` 우선, 없으면 `NVIDIA A100-SXM4-80GB`
- GPU count: 1
- Rental: 첫 학습은 On-Demand/non-interruptible
- Container disk: 80GB
- Network Volume: 300GB로 시작하고 전처리 결과 크기 확인 후 확장
- Network Volume mount: `/workspace`
- Image: `imtaebeen/adom-mmseg:<full-git-sha>`

원본 학습 데이터는 RELLIS 31GB, RUGD 6GB, YCOR 1GB, GOOSE 62GB로 약 100GB다. 전처리본과 checkpoint/run 결과를 합치면 250GB를 넘을 수 있으므로 300GB에서 시작하고 부족할 때 늘린다.

공개 가격의 A100 PCIe 80GB 약 `$1.39/hour`를 기준으로 20만원은 환율과 storage를 제외하면 약 90 GPU-hour 전후다. 10일 24시간 연속 실행은 예산을 초과하므로 실제 학습 시간에만 Pod를 실행한다.

한국/아시아의 고정 datacenter를 가정하지 않는다. Network Volume 생성 화면에서 다음 조건을 동시에 만족하는 가장 가까운 지역을 선택한다.

1. Secure Cloud
2. A100 80GB 재고
3. Network Volume 생성 가능

현재 공개 목록상 한국 datacenter가 고정적으로 보장되지 않으므로 `OC-AU-1`, `US-WA-1`, `US-CA-2` 순으로 재고를 확인하되, 학습 중 데이터와 GPU는 같은 datacenter에 있으므로 지연시간보다 A100 재고와 가격을 우선한다.

## 공유 Network Volume

여러 Pod가 같은 Network Volume을 동시에 attach할 수 있다. 모든 Pod는 같은 datacenter에서 생성하고 Pod 생성 시 해당 volume을 선택한다.

```text
/workspace/adom/
├── datasets/
│   ├── raw/
│   │   ├── rellis/
│   │   ├── rugd/
│   │   ├── ycor/
│   │   └── goose/
│   └── processed/
│       └── phase1-20class-v1/
├── runs/
│   └── <run-id>/
├── checkpoints/
└── tensorboard/
```

원본과 전처리본은 공유 읽기한다. 여러 Pod가 같은 파일이나 같은 run directory에 동시에 쓰지 않는다. 각 학습 명령에 dataset root와 고유 output root를 명시한다.

## Pod template

| 항목 | 값 |
|---|---|
| Name | `adom-a100-cu122-v1` |
| Image | `imtaebeen/adom-mmseg:<full-git-sha>` |
| Registry credential | 불필요 — public image |
| Container disk | 80GB |
| Volume mount | `/workspace` |
| Start command | Docker image 기본값 `sleep infinity` |
| HTTP port | label `tensorboard`, port `6006` (TensorBoard가 필요할 때만) |

재사용하는 template에는 고정값만 둔다.

```text
PYTHONUNBUFFERED=1
PYTHONPATH=/opt/adom/src
ADOM_CHECKPOINT_INTERVAL=500
ADOM_NUM_WORKERS=8
WANDB_PROJECT=adom
WANDB_ENTITY=<wandb-user-or-team-slug>
WANDB_MODE=online
WANDB_RESUME=allow
WANDB_API_KEY={{ RUNPOD_SECRET_wandb_api_key }}
WANDB_CACHE_DIR=/workspace/adom/cache/wandb
TORCH_HOME=/workspace/adom/cache/torch
```

`WANDB_API_KEY`는 RunPod Secret에 저장하고 template에는 secret reference만 둔다.
`WANDB_ENTITY`는 W&B URL의 `wandb.ai/<entity>/<project>`에 표시되는 사용자 또는
team slug다.

다음 값은 Pod를 배포하거나 학습을 시작할 때 실험별로 지정한다. 현재 Cost4
quick test 예시는 다음과 같다.

```text
ADOM_RUN_ID=phase1-cost4-quick-a100-20260803-01
WANDB_RUN_GROUP=phase1-cost4-quick
WANDB_TAGS=phase1,cost4,a100,quick-test
```

20-class adapter/config가 준비된 뒤에는 `cost4` tag와 group을 `20class`로 바꾼다.
`WANDB_RUN_ID`, `WANDB_NAME`, `WANDB_JOB_TYPE`, `WANDB_DIR`은 template에 고정하지
않는다. 실행기가 logical run, model, phase를 조합해 W&B run ID/name/job type을
만들고, `WANDB_DIR`은 전달된 output root 아래의 `wandb` directory로 설정한다.

## 첫 실행

Git SHA 이미지에는 코드가 `/opt/adom`에 포함되어 있으므로 Git clone은 필요 없다.

```bash
cd /opt/adom
nvidia-smi

python -m adom.runtime.doctor \
  --dataset-root /workspace/adom/datasets/processed/phase1-20class-v1 \
  --require-gpu \
  --output /workspace/adom/runs/<run-id>/doctor.json
```

20-class adapter가 준비되면 dataset 경로를 학습 실행기에 직접 전달한다.

```bash
bash scripts/run_training_cycle.sh \
  --dataset /workspace/adom/datasets/processed/phase1-20class-v1 \
  --models b0,b2 \
  --output /workspace/adom/runs/<run-id>
```

같은 run을 재개한다.

```bash
bash scripts/run_training_cycle.sh \
  --dataset /workspace/adom/datasets/processed/phase1-20class-v1 \
  --models b0,b2 \
  --output /workspace/adom/runs/<same-run-id> \
  --resume
```

현재 추적된 학습 config는 Cost4 전용이다. 위 20-class 경로는 팀의 adapter/config 수정이 합쳐진 뒤 활성화한다.
