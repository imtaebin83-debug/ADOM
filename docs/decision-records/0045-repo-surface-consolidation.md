# 0045. Consolidate the repository surface into three lanes

- Status: Accepted
- Date: 2026-09-07
- Owners: repo maintainer
- Supersedes: none; amends 0001

## Context

0001은 `study/`, `experiments/`, `external/`, `src/`, `ros2_ws/`를 전제로 초기 구조를
정했다. 이후 `study/`와 `experiments/`는 실제로 만들어지지 않았고, `external/`은
README 한 개 외에 어떤 코드도 참조하지 않는 빈 디렉터리로 남았다. `models/` 역시
추적 파일은 README 한 개뿐이며 실제 `checkpoints/`, `exports/`는 gitignore 대상이다.

그 결과 저장소 루트는 14개 항목 중 3개가 껍데기이고, 루트 마크다운이 5개까지 늘어
GitHub 첫 화면에서 데이터·학습·주행 세 갈래가 보이지 않는다.

## Decision

디렉터리를 물리적으로 3개로 합치지 않는다. 대신 껍데기를 제거하고 문서를 재배치해
루트 표면만 줄인다.

- `external/` 디렉터리를 삭제하고 외부 오픈소스 사용 규칙을 `CONTRIBUTING.md`로 옮긴다.
- `CONTRIBUTING.md`를 `.github/CONTRIBUTING.md`로 옮긴다. GitHub은 루트, `docs/`,
  `.github/` 세 위치를 모두 인식해 PR·이슈 화면에 자동 링크하므로 기능 손실 없이
  루트 목록에서만 사라진다.
- `models/README.md`를 삭제하고 배치 규칙과 model card 항목을
  `docs/setup-guides/jetson-model-checkpoint-handoff.md`로 옮긴다.
  런타임 경로 `models/checkpoints/<profile>/`은 그대로 유지한다.
- `requirements/`를 `docker/requirements/`로 옮기고 `Dockerfile`과
  `.github/workflows/docker-build.yml`의 경로를 갱신한다.
- `SHORTCUT.md` → `docs/jetson-shortcuts.md`,
  `RC_SETTING.md` → `docs/setup-guides/rc-vehicle-esc.md`로 옮긴다.
- 두 README의 구조 섹션을 데이터 / 모델 학습 / 인지·제어 세 개 표로 다시 쓴다.

## Rationale and evidence

`src/`, `configs/`, `data/`를 세 레인 디렉터리로 물리 이동하는 안은 다음 지점을
동시에 깨뜨린다.

- `PYTHONPATH=/opt/adom/src` (`Dockerfile`)와 `${REPO_ROOT}/src` (셸 스크립트 6개,
  `scripts/run_jetson_t4.sh`)
- `scripts/run_b2_eadom_capacity_study.sh`의 `REPO_ROOT = /opt/adom` fail-closed 단언
- `pyproject.toml`의 `where = ["src"]`와 `data.rellis` / `data.rugd` /
  `data.semantic_20` / `data.ycor` package-data
- `ros2_ws/`의 `adom.autonomy`, `adom.perception` import (단일 설치 패키지 전제)
- 경로 문자열을 직접 단언하는 `tests/test_runtime_contracts.py`,
  `tests/test_b2_eadom_contract.py`
- Jetson 실물에 이미 배치된 `~/ADOM/models/checkpoints/<profile>/`과 이를 증거로
  기록한 0012 및 hand-off 문서

즉 체감되는 난해함의 원인은 디렉터리 개수가 아니라 빈 껍데기와 루트 마크다운이며,
이번 변경은 실행 경로를 하나도 건드리지 않고 그 원인만 제거한다.

## Alternatives considered

- **`data/` · `training/` · `robot/` 물리 재편**: 위 6개 결합점 전부를 수정해야 하고
  Docker 이미지 재빌드와 Jetson·RunPod 재검증이 선행되어야 한다. 진행 중인 실차
  실험 대비 이득이 작아 기각한다.
- **`tools/`를 `scripts/`로 흡수**: `tests/test_paper_eval.py`와
  `tests/test_rc_eval.py`가 `tools/` 경로를 `sys.path`에 넣어 모듈을 직접 import한다.
  표면 축소 효과 대비 계약 변경이 있어 이번 범위에서 제외한다.
- **`Dockerfile`과 `docker-compose.yaml`을 `docker/`로 이동**: 빌드 컨텍스트는 루트로
  유지해야 하므로 `docker build -f docker/Dockerfile .` 형태가 되고, compose의
  `volumes` 상대경로가 compose 파일 위치 기준이라 `../`로 바꿔야 한다. 루트 항목
  2개를 줄이는 대가로 일상 실행 명령이 길어져 기각한다.
- **`LICENSE`, `pyproject.toml` 이동**: 불가능하다. GitHub의 license 자동 감지는
  루트를 읽고, `pyproject.toml`은 `where = ["src"]` 기준의 빌드 진입점이며 둘 다
  `Dockerfile`의 `COPY pyproject.toml README.md LICENSE ./`에 묶여 있다.

## Consequences

- 루트 항목이 20개에서 15개로, 루트 마크다운이 5개에서 2개(`README.md`,
  `README_en.md`)로 줄어든다.
- `external/`은 더 이상 존재하지 않으므로 0001의 "`external/`에 submodule로 연결한다"는
  조항은 "필요해지는 시점에 디렉터리를 다시 만든다"로 읽는다. 규칙 자체는
  `.github/CONTRIBUTING.md`에서 유지된다.
- Docker 빌드 컨텍스트가 바뀌므로 다음 이미지 빌드에서 `docker/requirements/` 경로가
  검증된다.
- Python import 경로, 설치 패키지 이름, 런타임 체크포인트 경로는 변경되지 않는다.

## Validation and rollback

- `python -m unittest discover -s tests`로 계약 테스트 전체를 실행한다.
- 저장소 전체 마크다운 상대 링크가 모두 존재하는 파일을 가리키는지 확인한다.
- `docker-build` 워크플로가 `docker/requirements/openmmlab.txt`를 COPY해 이미지가
  빌드되는지 확인한다.
- 롤백은 이 커밋 하나를 revert하면 된다. 코드 경로 변경이 없어 부분 롤백이 필요 없다.
