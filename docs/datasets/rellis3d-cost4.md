# RELLIS-3D → ADOM Cost4 데이터 계약

## 승인된 입력

공식 RELLIS-3D raw의 다음 구조만 승인한다.

```text
<input-root>/
└── Rellis-3D/
    ├── 00000/
    │   ├── pylon_camera_node/
    │   └── pylon_camera_node_label_id/
    └── ...
```

`Rellis-3D`, `RELLIS-3D`, 바로 아래 sequence가 있는 root를 지원한다. RGB는
`.jpg`, `.jpeg`, `.png`가 가능하며 source mask는 indexed PNG여야 한다. 팀
Drive의 기존 v1은 `images/images`에 Matplotlib UI asset이 섞여 있고 annotation
구조가 계약과 달라 `rejected` 상태다. 삭제하지는 않지만 학습 입력으로 사용하지
않는다.

## 라벨

| ID | 이름 | 의미 |
|---:|---|---|
| 0 | `paved_low_cost` | 포장·인공 안정 지면 |
| 1 | `natural_low_cost` | 일반 흙길·짧은 풀 |
| 2 | `medium_cost` | 진흙·덤불·물웅덩이 등 주의 구간 |
| 3 | `high_cost_or_obstacle` | 물·구조물·사람·차량·통나무 등 |
| 255 | `ignore` | sky, void, 불확실 영역 |

source mapping의 유일한 관리본은
`configs/datasets/rellis3d/label_mapping.yaml`이다. 코드에 mapping을 복제하지
않는다.

## 생성

```bash
export PYTHONPATH="$PWD/src"

python -m adom.data inspect \
  --dataset rellis3d \
  --input-root /workspace/adom/datasets/rellis3d-raw \
  --mapping configs/datasets/rellis3d/label_mapping.yaml \
  --report /workspace/adom/outputs/rellis-inspection.json

python -m adom.data prepare \
  --dataset rellis3d \
  --input-root /workspace/adom/datasets/rellis3d-raw \
  --output-root /workspace/adom/outputs/preprocessing/rellis3d \
  --mapping configs/datasets/rellis3d/label_mapping.yaml \
  --split-root data/splits/rellis3d/official \
  --version v2.0

python -m adom.data validate \
  --dataset-root /workspace/adom/outputs/preprocessing/rellis3d \
  --strict
```

`prepare`는 staging에서 변환과 최종 checksum 검증을 끝낸 후에만 output을
승격한다. 기존 output은 기본적으로 거부하며 의도적인 재생성만 `--overwrite`를
사용한다.

## 수동 QC와 보존

`reports/class_statistics.csv`에서 class/ignore 비율을 확인하고
`reports/previews/`의 class별 대표 sample을 사람이 검토한다. 승인 후:

```bash
python -m adom.data package \
  --dataset-root /workspace/adom/outputs/preprocessing/rellis3d \
  --archive /workspace/adom/network-volume/rellis3d-cost4-v2.tar
```

tar와 `.sha256`을 Network Volume에 함께 보존한다. manifest는 상대경로만
포함하고 모든 파일은 `SHA256SUMS.txt`에 연결된다.

## 알려진 평가 한계

공식 train/val/test의 sample ID는 겹치지 않지만 동일 sequence가 여러 split에
등장한다. 가까운 frame의 시간적 상관 때문에 일반화 성능이 낙관적으로 보일 수
있다. 공식 split을 유지하되 모든 dataset/model report에 이 한계를 적는다.
