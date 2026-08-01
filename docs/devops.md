# ADOM RunPod 학습 환경 및 CI/CD

이 문서는 학습 컨테이너와 storage 운영 경계를 정의한다. 실제 데이터 생성은
[`datasets/rellis3d-cost4.md`](datasets/rellis3d-cost4.md), 한 명령 학습은
[`runpod-one-cycle.md`](runpod-one-cycle.md)를 따른다.

## 책임 경계

- RunPod RTX 4090/A6000: 전처리, 학습, 평가, checkpoint, static ONNX
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
성공한 image만 Docker Hub의 `latest`와 Git SHA tag로 push한다.

필요한 GitHub Actions secret:

- `DOCKERHUB_USERNAME`
- `DOCKERHUB_TOKEN`

## RunPod workspace

```text
/workspace/adom/
├── repo/
├── outputs/
└── network-volume/
    ├── rellis3d-cost4-v2.tar
    └── rellis3d-cost4-v2.tar.sha256

/tmp/data/
└── rellis3d/
```

Compose는 저장소를 `/workspace/adom/repo`, cache를
`/workspace/adom/datasets`, output을 `/workspace/adom/outputs`에 mount하고
`PYTHONPATH=/workspace/adom/repo/src`를 설정한다.

```bash
bash /workspace/adom/repo/scripts/init_workspace.sh \
  --dataset rellis3d \
  --archive /workspace/adom/network-volume/rellis3d-cost4-v2.tar

cd /workspace/adom/repo
bash scripts/run_training_cycle.sh \
  --dataset /workspace/adom/datasets/rellis3d \
  --models b0,b2 \
  --output /workspace/adom/outputs/runs/<run-id>
```

중단된 동일 run은 `--resume`을 붙인다. runtime doctor와 dataset QC는 다시
실행하며, 코드/config/dataset/GPU fingerprint와 필수 artifact checksum이 모두
일치하는 완료 phase만 건너뛴다.

## Artifact 보존

각 run의 `status.json`, `summary.json/csv`, `parity_inputs.json`, test metric,
backbone audit, best checkpoint, ONNX, MMDeploy JSON 세트, parity, metadata를
함께 보존한다. metadata는 model variant, dataset/config/checkpoint/test metric/
Git/artifact checksum과 입력 resize/padding/normalization 규약을 포함한다.

Git artifact guard:

```bash
python scripts/check_git_artifacts.py
```
