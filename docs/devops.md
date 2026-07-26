# ADOM RunPod 학습 환경 및 CI/CD 운영 가이드

이 문서는 SegFormer-B0/B2 파인튜닝용 RunPod 환경을 다룬다. 학습 컨테이너는
OpenMMLab 기반 학습, 평가, ONNX export까지만 담당한다. NVIDIA Jetson Orin Nano
8GB의 TensorRT engine(`.engine`/`.plan`)은 Jetson 장치에서 직접 생성한다.

## 1. 학습 이미지와 CI/CD

학습 이미지는 `nvcr.io/nvidia/pytorch:23.10-py3`를 기반으로 하며 MMCV,
MMEngine, MMSegmentation, MMDeploy 및 ONNX 도구를 포함한다. 데이터셋,
checkpoint, ONNX, TensorRT engine은 이미지에 포함하지 않고 volume으로 연결한다.

`.github/workflows/docker-build.yml`은 다음 조건을 모두 만족할 때 실행된다.

1. `main` 브랜치에 push한다.
2. 해당 push에 `Dockerfile` 변경이 포함된다.

워크플로는 Buildx 캐시를 사용해 이미지를 빌드하고 아래 두 태그를 Docker Hub에
push한다.

- `<DOCKERHUB_USERNAME>/adom-mmseg:latest`
- `<DOCKERHUB_USERNAME>/adom-mmseg:<git-commit-sha>`

저장소의 GitHub **Settings > Secrets and variables > Actions**에 다음 repository
secrets를 등록해야 한다.

- `DOCKERHUB_USERNAME`: Docker Hub 사용자명
- `DOCKERHUB_TOKEN`: Docker Hub에서 발급한 access token

의존성 파일만 바꾼 push는 의도적으로 자동 빌드를 시작하지 않는다. 새 의존성을
배포할 때는 호환성 관련 Dockerfile 변경도 같은 커밋에 포함한다.

## 2. 권장 RunPod workspace 구조

Compose 파일의 host volume 경로는 `/workspace/adom`을 기준으로 설계했다.
RunPod의 Network Volume에는 원본 압축 파일과 보존할 산출물만 두고, 반복해서
읽는 데이터셋은 Pod의 `/tmp/data`로 캐싱한다.

```text
/workspace/adom/
├── docker-compose.yaml          # repo/docker-compose.yaml을 복사한 실행용 파일
├── repo/                        # 이 Git 저장소
│   ├── configs/
│   ├── scripts/
│   └── src/
├── outputs/                     # checkpoint, log, ONNX (Git 미커밋)
└── network-volume/              # RunPod Network Volume mount
    ├── rellis3d.tar             # archive 이름이 cache dataset 이름이 됨
    ├── rugd.tar
    └── ycor.tar

/tmp/data/                       # 휘발성 Pod Volume/NVMe 캐시
├── rellis3d/
│   └── raw/
├── rugd/
│   └── raw/
└── ycor/
    └── raw/
```

초기 디렉토리는 다음처럼 준비한다. 실제 저장소 URL로 바꿔 실행한다.

```bash
mkdir -p /workspace/adom/outputs
git clone <ADOM_REPOSITORY_URL> /workspace/adom/repo
cp /workspace/adom/repo/docker-compose.yaml /workspace/adom/docker-compose.yaml
cd /workspace/adom
export DOCKERHUB_USERNAME=<your-dockerhub-username>
docker compose pull adom-train
```

## 3. Network Volume 데이터셋을 Pod Volume에 캐싱

`scripts/init_workspace.sh`는 각 Network Volume archive를 독립된
`/tmp/data/<dataset>`에 캐싱한다. Compose의
`/tmp/data:/workspace/adom/datasets` mount와 결합하면 컨테이너에서는
`/workspace/adom/datasets/<dataset>`로 바로 접근한다. 데이터셋별 source
signature와 staging 디렉토리를 사용하므로 한 데이터셋을 갱신해도 다른 cache를
교체하지 않는다.

archive 하나를 명시적으로 캐싱하는 방법은 다음과 같다.

```bash
bash /workspace/adom/repo/scripts/init_workspace.sh \
  --dataset rellis3d \
  --archive /workspace/adom/network-volume/rellis3d.tar
```

RUGD와 YCOR도 동일하게 추가할 수 있다.

```bash
bash /workspace/adom/repo/scripts/init_workspace.sh \
  --dataset rugd \
  --archive /workspace/adom/network-volume/rugd.tar

bash /workspace/adom/repo/scripts/init_workspace.sh \
  --dataset ycor \
  --archive /workspace/adom/network-volume/ycor.tar
```

Network Volume 바로 아래의 모든 `.tar`를 한 번에 캐싱하려면 canonical archive
이름(`rellis3d.tar`, `rugd.tar`, `ycor.tar`)을 사용하고 다음을 실행한다.

```bash
bash /workspace/adom/repo/scripts/init_workspace.sh \
  --all \
  --network-volume /workspace/adom/network-volume
```

archive 내부는 한 데이터셋 디렉토리의 내용이다. 전처리 전 source archive는
`raw/`에서 시작하는 구조를 권장하고, 학습용 package는
`images/`, `annotations/`, `splits/` 등을 포함할 수 있다. archive가
`rellis3d/`처럼 dataset 이름과 같은 단일 최상위 디렉토리를 포함하면 기본
`auto` 모드가 한 계층을 제거해 중복 nesting을 방지한다. 그 외 형식은
`--strip-components N`으로 명시한다.

기본적으로 `repo/data/external/<dataset>`에는 컨테이너 경로
`/workspace/adom/datasets/<dataset>`을 가리키는 symlink가 생성된다. 링크가
필요하지 않으면 `--no-link`를 사용한다. 이 링크는 container 경로를 대상으로
하므로 host에서는 dangling link처럼 보일 수 있지만 Compose container 안에서는
정상이다. 다른 mount 구조에서는
`--cache-root`, `--link-root` 또는 `ADOM_CACHE_ROOT`, `ADOM_LINK_ROOT`,
`ADOM_CONTAINER_DATA_ROOT` 환경변수를 사용한다.

`/tmp/data`는 Pod 종료 시 사라질 수 있으므로 전처리 결과, checkpoint와 ONNX는
항상 `/workspace/adom/outputs` 또는 별도 Network Volume에 보존한다.

## 4. 학습 컨테이너 실행 및 점검

Compose v2의 권장 실행 명령은 다음과 같다.

```bash
cd /workspace/adom
docker compose run --rm adom-train bash
```

`docker-compose` compatibility command가 설치된 환경에서는 동일하게 실행할 수
있다.

```bash
docker-compose run --rm adom-train bash
```

컨테이너 안에서 GPU와 핵심 라이브러리를 확인한다.

```bash
nvidia-smi
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
python -c "import mmcv, mmengine, mmseg, mmdeploy, onnx; print('OpenMMLab/ONNX OK')"
```

`torch.cuda.is_available()`은 RunPod 컨테이너에서 `True`여야 한다. 이미지 build
단계의 sanity check에서는 GPU가 전달되지 않으므로 `False`가 출력될 수 있으며,
이는 정상이다.

## 5. SegFormer-B0/B2 파인튜닝과 평가

아래 예시는 프로젝트별 config가
`configs/segformer/segformer_b0_adom.py`와
`configs/segformer/segformer_b2_adom.py`에 준비되어 있다고 가정한다. 두 실험은
서로 다른 `work-dir`를 사용해 checkpoint와 로그가 섞이지 않게 한다.

pip 배포판에 포함된 MMSegmentation `train.py`와 `test.py` 위치를 먼저 구한다.

```bash
cd /workspace/adom/repo
MMSEG_ROOT="$(python -c 'from pathlib import Path; import mmseg; print(Path(mmseg.__file__).resolve().parent)')"

python "${MMSEG_ROOT}/.mim/tools/train.py" \
  configs/segformer/segformer_b0_adom.py \
  --work-dir /workspace/adom/outputs/segformer-b0

python "${MMSEG_ROOT}/.mim/tools/train.py" \
  configs/segformer/segformer_b2_adom.py \
  --work-dir /workspace/adom/outputs/segformer-b2
```

학습이 끝나면 각 config와 checkpoint 조합을 `test.py`로 평가한다.

```bash
python "${MMSEG_ROOT}/.mim/tools/test.py" \
  configs/segformer/segformer_b0_adom.py \
  /workspace/adom/outputs/segformer-b0/best_mIoU_iter_*.pth \
  --work-dir /workspace/adom/outputs/segformer-b0/test

python "${MMSEG_ROOT}/.mim/tools/test.py" \
  configs/segformer/segformer_b2_adom.py \
  /workspace/adom/outputs/segformer-b2/best_mIoU_iter_*.pth \
  --work-dir /workspace/adom/outputs/segformer-b2/test
```

shell glob이 checkpoint를 둘 이상 선택하지 않도록 실제 파일명을 확인한 뒤 실행한다.
config의 `data_root`는 실제 package 종류에 맞춘다. 캐싱된 학습용 package는
`/workspace/adom/datasets/<dataset>` 아래를 사용하고, RunPod에서 전처리한
결과는 `/workspace/adom/outputs/preprocessing/<dataset>`을 사용한다.

## 6. ONNX export와 Jetson 경계

ONNX export는 RunPod에서 학습에 사용한 config와 checkpoint를 기준으로 수행하고,
결과는 `/workspace/adom/outputs/.../*.onnx`에 저장한다. export 전후에 입력 크기,
dynamic/static shape, opset, preprocessing 정규화, class 순서를 기록하고 ONNX
Runtime으로 출력의 shape와 수치 오차를 검증한다.

MMDeploy를 사용할 때는 **ONNX Runtime backend용 deploy config**를 선택한다.
TensorRT backend deploy config로 RunPod에서 `.engine`/`.plan`을 만들지 않는다.
ONNX와 함께 model config, label/class metadata, 입력 shape를 Jetson으로 전달한다.

Jetson Orin Nano 8GB에서는 JetPack에 포함된 TensorRT 버전으로 ONNX를 받아 engine을
직접 컴파일한다. TensorRT engine은 GPU 아키텍처, TensorRT/CUDA/JetPack 버전과
밀접하게 결합되므로 RunPod RTX 4090/A6000에서 생성한 engine을 재사용하지 않는다.
1차 데모는 파인튜닝 정확도 향상 입증을 우선하며 FP16/INT8, 양자화 및 INT8
calibration dataset 정책은 후속 실험에서 결정한다.

**Jetson용 ROS2/TensorRT 배포 컨테이너는 추후 별도 작성한다.**

## 7. Git 및 artifact 보존 원칙

Git에는 코드, config, Dockerfile, Compose, CI/CD 및 문서만 커밋한다. 다음 파일은
`.gitignore`와 `.dockerignore` 대상이며 Docker build context에도 포함하지 않는다.

- dataset 및 dataset archive
- `*.pth`, `*.pt` checkpoint
- `*.onnx`
- `*.engine`, `*.plan` TensorRT artifact
- `outputs/`, log, W&B cache

중요한 checkpoint와 ONNX는 Git 대신 RunPod Network Volume 또는 별도 artifact
storage에 보관한다.
