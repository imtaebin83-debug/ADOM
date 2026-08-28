# Emergency E-ADOM RunPod runbook

E-ADOM은 내일 Jetson 배포를 위한 단일-seed data-only 후보이다. ImageNet MiT-B0에서
시작해 B0-E0의 학습 recipe를 그대로 사용하고, train split만 RELLIS 4,435 + ADOM
standalone 133의 `ta1_train.txt`로 바꾼다. TA0 method ablation이나 canonical test를
이 실행에 섞지 않는다.

긴급 실행 hardware는 RTX 4090 24GB를 허용한다. wrapper는 runtime doctor에
`RTX 4090`과 최소 22GiB를 명시하되 micro-batch/effective batch와 optimizer update
계약은 변경하지 않는다. 다른 cycle의 기본 A100 75GiB 계약은 그대로 유지한다.

## RunPod

GitHub Actions가 E-ADOM commit의 immutable Git-SHA image를 Docker Hub에 push한 뒤
그 SHA로 A100 Pod를 재생성한다. PR branch image도 commit SHA가 일치하면 사용할 수
있다. Network Volume은 기존 `/workspace`를 그대로 연결한다.

```bash
tmux new -s eadom
```

```bash
export EADOM_IMAGE_SHA=<full-eadom-git-sha>
```

W&B를 쓰면 API key를 shell history에 남기지 않는다.

```bash
read -s WANDB_API_KEY
```

```bash
export WANDB_API_KEY
```

Full 4k+40k training은 dataset contract를 한 번만 검사하고, 이미 검증된 B0
micro-batch 16을 고정한다.

```bash
bash scripts/run_emergency_eadom.sh full
```

터미널 분리는 `Ctrl-b`, `d`이고 재접속은 다음과 같다.

```bash
tmux attach -t eadom
```

중단 후 같은 output에서 optimizer/scheduler를 복구한다.

```bash
bash scripts/run_emergency_eadom.sh resume
```

상태와 선택 checkpoint를 확인한다.

```bash
R=/workspace/adom/runs/semantic20/eadom/seed42/full
```

```bash
grep -E '"status"|"error"' "$R/status.json"
```

```bash
find "$R/b0/stage2" -maxdepth 1 -name 'best_clean_selection_iter_*.pth'
```

## Selection and deployment

canonical RELLIS validation의 `ValSupported13`, `RareRisk4`, absent-class FP를 B0-E0와
비교한다. ADOM diagnostic은 선택 후에만 실행한다. 통과한 하나의 checkpoint만
640x384 raw-logits ONNX로 export하고 PyTorch parity를 검증한다. TensorRT engine은
RunPod에서 만들지 않고 target Jetson에서 생성한다. 현장 실패 시 기존 B0-E0 handoff
package로 rollback한다.
