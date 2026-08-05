# Folder refactor backlog

이 문서는 ADOM 저장소에서 발견된 폴더·패키지 구조 개선안을 누적 기록한다.
현재 기능 안정화와 RunPod gate 통과가 우선이며, 이 문서의 항목은 별도 정리 작업에서
검증 후 적용한다. 여기에 기록됐다는 이유만으로 파일을 즉시 이동하거나 삭제하지 않는다.

## 운영 원칙

- 기존 사용자 변경과 미추적 파일은 소유권과 용도를 확인하기 전까지 보존한다.
- 이동과 로직 변경을 같은 커밋에 섞지 않는다. 가능하면 먼저 `git mv`만 수행한다.
- legacy 파일은 canonical 경로의 동작·테스트·산출물 parity가 확인된 뒤 제거한다.
- raw/processed dataset, checkpoint, 로그는 Git 저장소 구조에 포함하지 않는다.
- Docker의 `COPY study`처럼 legacy 경로를 배포 구조에 다시 연결하지 않는다.
- 경로 변경 시 Docker image, Python wheel, CLI, runtime, test가 동일한 canonical
  resource를 읽는지 함께 검증한다.
- 각 제안은 `제안 → 승인 → 이동 → 참조 수정 → 테스트 → legacy 제거` 순서로 진행한다.

## 제안 현황

| ID | 상태 | 우선순위 | 제안 |
|---|---|---:|---|
| FR-001 | 제안 | P1 | `src/data`와 `src/adom/data`를 하나의 ADOM package namespace로 통합 |
| FR-002 | 제안 | P1 | 숫자 접두사 전처리 스크립트를 import 가능한 command module로 전환 |
| FR-003 | 대기 | P1 | `study/gahyung/Datasets_Repo` legacy 전처리 코드의 단계적 제거 |
| FR-004 | 제안 | P2 | RELLIS Cost4 official split과 Semantic20 split의 경로 명시성 강화 |
| FR-005 | 검토 필요 | P2 | `src/adom_mmseg`와 `src/adom/mmseg`의 역할·소유권 정리 |
| FR-006 | 검토 필요 | P2 | dataset/config 이름에서 Semantic20과 Phase 2 Cost 계열 구분 강화 |

## FR-001: 전처리 package namespace 통합

### 현재 상태

- 재사용 runtime/data package는 `src/adom/data`에 있다.
- 팀원이 승격한 Semantic20 전처리 코드와 resource는 별도 top-level
  `src/data/{rellis,rugd,semantic_20,ycor}`에 있다.
- Docker에는 `COPY src`로 두 경로가 모두 들어가지만, Python packaging을 위해
  top-level `data` package를 별도로 취급해야 한다.
- `data`라는 범용 top-level package 이름은 외부 package와 충돌할 가능성이 있고,
  ADOM 소유 코드라는 사실도 명확하지 않다.

### 제안 구조

```text
src/adom/data/
├── preprocessing/
│   └── semantic20/
│       ├── cli.py
│       ├── bridge.py
│       ├── combined.py
│       ├── validation.py
│       └── sources/
│           ├── rellis3d.py
│           ├── rugd.py
│           └── ycor.py
└── resources/
    └── semantic20/
        ├── bridge_mapping.yaml
        ├── rellis3d/
        │   ├── class_mapping.yaml
        │   └── splits/{train,val,test}.txt
        ├── rugd/
        │   ├── label_mapping.json
        │   └── splits/{train,val,test}.txt
        └── ycor/
            └── label_mapping.json
```

### 기대 효과

- `adom` 하나의 namespace와 wheel로 코드·mapping·split을 배포할 수 있다.
- runtime과 preprocessing이 `importlib.resources` 기반의 같은 resource를 사용한다.
- Docker source tree와 설치된 wheel 사이의 경로 차이를 줄인다.

### 선행조건

- 현재 RunPod readiness branch의 Docker image tests가 통과해야 한다.
- E0 `4435/900/899`, RUGD `4779/733/1924` split checksum을 이동 전후 비교한다.
- 기존 canonical CLI와 synthetic preprocessing tests를 새 모듈에 그대로 적용한다.

## FR-002: 숫자 접두사 스크립트의 command module화

### 현재 상태

`01_convert_bridge_sources.py` 같은 파일명은 실행 순서는 잘 드러나지만 일반 Python
import 문으로 다루기 어렵다. 현재 canonical CLI가 동적 import를 사용하는 이유이기도
하다.

### 제안

- 숫자 파일은 보존 기간 동안 compatibility wrapper로 유지한다.
- 실제 구현은 `adom.data.preprocessing.semantic20` 아래 의미 기반 모듈로 이동한다.
- 하나의 CLI에서 명시적 subcommand를 제공한다.

```text
adom-semantic20-preprocess audit-rugd ...
adom-semantic20-preprocess convert-bridges ...
adom-semantic20-preprocess build-combined ...
adom-semantic20-preprocess validate-combined ...
```

- wrapper와 새 CLI의 synthetic output 및 failure behavior가 같은지 테스트한 뒤 wrapper를
  제거한다.

## FR-003: legacy `study/gahyung/Datasets_Repo` 제거

### 현재 상태

- 학습 runtime의 실행 의존성은 canonical package resource로 제거하는 중이다.
- legacy tree에는 과거 mapping, 결과 보고서, 테스트, 전처리 스크립트가 남아 있다.
- 일부 자료는 migration 근거와 결과 재현 기록으로 아직 가치가 있다.

### 제거 조건

- `rg` 기준 실행 코드·config·test·Dockerfile에 `study` 의존성이 없어야 한다.
- canonical 경로가 RUGD RGB/indexed, YCOR, RELLIS 변환 회귀 테스트를 모두 보유한다.
- mapping과 split의 checksum 또는 의미적 snapshot이 canonical 경로에 기록돼야 한다.
- 필요한 결과 보고서는 `docs/datasets` 또는 decision record로 옮겨야 한다.
- 담당 팀원이 legacy tree 삭제 범위를 승인해야 한다.

### 주의

현재 `scripts/check_git_artifacts.py`의 legacy 경로 상수는 실행 의존성이 아니라 해당
경로에 대형 artifact가 다시 들어오는 것을 막는 guard다. legacy 삭제 시에도 동일한
보호 정책을 다른 방식으로 유지할지 결정해야 한다.

## FR-004: RELLIS split 경로 구분 강화

### 현재 상태

- `data/splits/rellis3d/official`은 기존 Cost 계열 계약에 사용된다.
- Semantic20 canonical split은 별도이며 개수가 `4435/900/899`다.
- 이름만 보고 기존 official split을 Semantic20에서 재사용할 위험이 있다.

### 제안

```text
data/splits/rellis3d/
├── cost4_official/...
└── semantic20_v1/...
```

최종적으로 package resource만 runtime source of truth로 정한다면 Git root의 split은
문서·검증 snapshot 역할로 제한하고 README에 ontology와 expected count를 명시한다.

## FR-005: MMSeg extension 중복 후보 정리

### 관찰

- 추적 경로 `src/adom/mmseg`가 현재 Semantic20 runtime에서 사용된다.
- worktree에 미추적 `src/adom_mmseg`가 존재한다.

### 제안 전 확인사항

- `src/adom_mmseg`의 작성자, 용도, import name, 기존 실험 의존성을 먼저 확인한다.
- 두 경로의 dataset registration과 class metadata 차이를 비교한다.
- 사용자 소유 미추적 파일이므로 확인 전 이동·병합·삭제하지 않는다.

정식 package는 가능하면 `adom.mmseg` 하나로 통일하고, 외부에서 사용된 import가 있다면
한시적 compatibility import를 둔다.

## FR-006: config ontology 경계 명시

### 관찰

- Phase 1은 19 trainable class의 Semantic20 계열이다.
- Phase 2/reference에는 Cost4와 5-class 후보 config가 공존한다.
- `official`, `rellis3d`, `semantic`, `cost`만으로는 잘못된 mapping/split을 선택하기 쉽다.

### 제안

```text
configs/adom/
├── phase1_semantic20/
└── phase2_costmap/

configs/datasets/
├── semantic20/
│   ├── rellis3d_v1/
│   ├── rugd_bridge_v1/
│   └── ycor_bridge_v1/
└── costmap/
    ├── cost4/
    └── cost5/
```

이 변경은 Phase 2 ontology가 확정된 뒤 수행한다. 현재 미추적 5-class config는 사용자
소유이므로 이 문서에서는 이동 대상으로 확정하지 않는다.

## 권장 실행 순서

1. 현재 Semantic20 B0 probe/smoke/mini/resume gate를 먼저 통과시킨다.
2. FR-001과 FR-002를 하나의 별도 refactor branch에서 수행한다.
3. Docker image와 wheel에서 canonical resource parity를 검증한다.
4. FR-003의 legacy 제거 후보를 목록화하고 팀원 승인을 받는다.
5. Cost ontology 결정 후 FR-004와 FR-006을 함께 정리한다.
6. FR-005의 미추적 코드 소유권 확인 후 MMSeg package를 통합한다.

## 변경 기록

### 2026-08-05

- Semantic20 RunPod readiness 감사에서 발견된 FR-001~FR-006을 최초 기록했다.
- 현 학습 blocker 해결에는 대규모 폴더 이동이 필요하지 않다고 판단해 실제 이동은
  보류했다.
- top-level `src/data`의 packaging은 현재 Docker readiness를 위한 호환 단계이며,
  장기적으로 FR-001 구조로 통합하는 방향을 제안했다.
