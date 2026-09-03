# Data

데이터셋 원본과 전처리 산출물은 git에 커밋하지 않습니다.

저장소 내부에는 split, 상대경로 manifest, 배치 규칙만 관리합니다. RunPod에서
실제 데이터는 Compose volume을 통해 `/workspace/adom/datasets`에 마운트합니다.

권장 저장소 구조:

```text
data/
├── splits/          # Git 추적, sample ID 또는 상대경로만 포함
├── registry/        # Git 추적, 데이터셋 배포본 메타데이터
├── captures/        # 실차 rosbag 세션 (Git 제외)
├── autonomy_bags/   # autonomy 세션 rosbag (Git 제외)
└── external/        # runtime dataset symlink (Git 제외)
```

현재 Git이 추적하는 것은 `splits/`와 `registry/`뿐이다. 나머지는
`scripts/init_workspace.sh`가 실행 시점에 만들며 `scripts/check_git_artifacts.py`가
커밋 유입을 차단한다.

RunPod container 구조:

```text
/workspace/adom/datasets/
├── rellis3d/raw/
├── rugd/raw/
└── ycor/raw/
```

데이터를 추가할 때는 이 파일이나 별도 메타데이터 문서에 다음 정보를 남깁니다.

- dataset name and source
- download/access date
- license or usage restriction
- class ontology
- train/validation/test split
- preprocessing command

전처리 결과는 `/workspace/adom/outputs/preprocessing/<dataset>`에 저장하며 Git에
커밋하지 않습니다. 데이터셋 계약과 클래스 매핑 문서는
[`docs/datasets/`](../docs/datasets/), 변환 스크립트는
[`src/data/`](../src/README.md)에 있습니다.
