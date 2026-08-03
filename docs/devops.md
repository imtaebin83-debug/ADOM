# ADOM RunPod 학습 환경 및 CI/CD

이 문서는 학습 컨테이너와 storage 운영 경계를 정의한다. 실제 데이터 생성은
[`datasets/rellis3d-cost4.md`](datasets/rellis3d-cost4.md), 한 명령 학습은
[`runpod-one-cycle.md`](runpod-one-cycle.md)를 따른다.

## 책임 경계

- RunPod A100 80GB Secure Cloud: 전처리, 학습, 평가, checkpoint, static ONNX
- Jetson Orin Nano 8GB: TensorRT `.engine` 직접 build와 runtime benchmark
- Git: 코드, config, Docker/Compose, workflow, script, 문서
- Network Volume: canonical dataset tar/checksum, checkpoint, ONNX, run 결과
- `/tmp/data`: Pod 생존 기간의 빠른 dataset cache

RunPod에서 TensorRT engine을 만들거나 Git에 model/data artifact를 commit하지
않는다.

## Training image

Base image는 `nvcr.io/nvidia/pytorch:23.10-py3`다. Dockerfile은 다음 확정
버전을 직접 설치하고 마지막에 `pip check`, 정확한 version assert, 실제 import,
`mmcv.ops` import를 수행한다.

| package | version |
|---|---:|
| NumPy | 1.24.4 |
| setuptools | 69.5.1 |
| OpenCV Python | 4.8.0.76 |
| MMCV | 2.1.0 |
| MMEngine | 0.10.7 |
| MMSegmentation | 1.2.2 |
| MMDeploy | 1.3.1 |
| Weights & Biases | 0.22.3 |

OpenMIM의 자동 dependency 결정을 사용하지 않는다. ABI 민감 package는
Dockerfile의 `--no-deps`/`--ignore-installed` 정책을 유지한다.
Albumentations가 요구하는 `opencv-python-headless` 이름은 module이 없는
metadata shim으로만 충족한다. 실제 `cv2` 파일 제공자는
`opencv-python==4.8.0.76` 하나이며 Docker sanity check가 이를 assert한다.

## CI

`code-smoke.yml`은 pull request와 main push에서 다음을 수행한다.

1. Git artifact/personal-path guard
2. synthetic RELLIS preprocessing과 실패 조건
3. deterministic checksum
4. config와 metric contract

`docker-build.yml`은 dependency image를 build한 뒤 저장소를 mount하여 전체
test suite를 image 안에서 실행한다. 이때 MMSeg registry, B0/B2 config parse,
freeze/unfreeze hook, CPU B0 1-batch forward도 실행된다. 이 smoke test가
성공한 동일 image만 Docker Hub의 `latest`와 Git SHA tag로 push한다. Git SHA
image는 `/opt/adom`에 코드/config/script를 포함하므로 RunPod에서 별도 Git
clone이 필요 없다. Network Volume은 `/workspace`에 mount되어 코드와 분리된다.

필요한 GitHub Actions secret:

- `DOCKERHUB_USERNAME`
- `DOCKERHUB_TOKEN`

## RunPod workspace

```text
/opt/adom/                       # immutable code from Git SHA image
/workspace/adom/                 # shared Network Volume
├── datasets/{raw,processed}/
└── runs/<run-id>/
```

RunPod에서는 학습 명령의 `--dataset`에 Network Volume의 versioned processed
dataset 경로를 직접 넘긴다. Compose는 로컬 개발 호환용이며 RunPod 실행 경로의
기준이 아니다.

```bash
bash /opt/adom/scripts/init_workspace.sh \
  --dataset rellis3d \
  --archive /workspace/adom/datasets/archives/rellis3d-cost4-v2.tar \
  --cache-root /tmp/data \
  --no-link

cd /opt/adom
bash scripts/run_training_cycle.sh \
  --dataset /workspace/adom/datasets/processed/rellis3d-cost4-v2 \
  --models b0,b2 \
  --output /workspace/adom/runs/<run-id>
```

중단된 동일 run은 `--resume`을 붙인다. 상태 파일에 완료로 기록됐고 필수
artifact가 실제 존재하는 phase는 건너뛴다. 진행 중이던 Stage 1/2는 work
directory의 `last_checkpoint`가 가리키는 iteration checkpoint에서 model,
optimizer, parameter scheduler, iteration과 randomness 상태를 복구한다.

W&B를 주 실험 추적기로 사용하고 TensorBoard/Local backend를 Network Volume의
run directory에 백업한다. batch probe와 ONNX export에는 W&B logging을 끈다.

## Artifact 보존

각 run의 `status.json`, `summary.json/csv`, test metric, backbone audit, best
checkpoint, ONNX, parity, metadata를 함께 보존한다. metadata는 dataset/config/
checkpoint/Git/artifact checksum과 입력 resize/padding/normalization 규약을
포함한다.

Git artifact guard:

```bash
python scripts/check_git_artifacts.py
```
