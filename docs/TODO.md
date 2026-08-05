# TODO: Phase 1 Semantic20 전처리 안정화

## 범위와 담당 구분

- 대상은 공식 RUGD RGB annotation을 RELLIS 기반 Semantic20 공간으로 변환하는
  Phase 1 전처리 경로다.
- `study/gahyung/Datasets_Repo`의 코드를 정식 `src` 경로로 승격하는 작업은 팀원이
  담당한다. 이 문서는 승격 위치를 다시 설계하지 않고, 승격되는 원본 코드에 반드시
  반영해야 할 결함과 완료 조건을 기록한다.
- 현재 RunPod에서 사용하는 임시 RGB 변환 어댑터는 복구용일 뿐 canonical
  pipeline으로 간주하지 않는다.

## P0: 공식 RUGD RGB mask 입력 지원

### 재현된 증상

`RUGD_annotations.zip`을 풀어 만든 mask는 `(H, W, 3)` RGB color mask다. 그러나
`01_convert_bridge_sources.py`의 `load_index_mask()`는 `(H, W)` 단일 채널 또는
모든 채널 값이 같은 mask만 허용하여 첫 샘플에서 다음 오류가 발생한다.

```text
ValueError: Expected indexed mask, got shape (550, 688, 3):
.../rugd-flat/indexLabel/park-2_00001.png
```

### 근본 원인

- flat index 생성 과정은 파일 이름과 pair만 검증하고 mask encoding은 검증하지
  않았다.
- 디렉터리 이름을 `indexLabel`로 만들었지만 그 안에는 RGB mask가 들어갔다.
- Semantic20 bridge는 RUGD source ID를 기대하지만, 공식 다운로드 경로는 RGB
  palette mask를 제공한다.
- 기존 RUGD `label_mapping.json`의 `RGB_TO_ADOM`은 Cost4 target을 위한 mapping이므로
  Semantic20 bridge에 그대로 사용할 수 없다.

### 필수 수정

- RUGD mask 입력 형식을 `auto`, `rgb`, `indexed` 중 하나로 명시하거나 자동
  판별한다.
- RGB 입력은 공식 RGB palette를 class name으로 해석한 뒤 Semantic20 bridge의
  `source_class -> target_id`와 합성한다.
- bridge에 없는 RUGD class는 명시적으로 `ignore_index=255`로 처리하고, 알려지지
  않은 RGB 값은 조용히 무시하지 말고 즉시 실패시킨다.
- 변환 전에 전체 palette coverage, image/mask pair, 크기, split 중복을 검사한다.
- RGB lookup은 per-pixel Python loop 대신 packed RGB LUT 또는 동등한 vectorized
  구현을 사용한다.
- 결과 mask는 `uint8`, 단일 채널이며 값의 집합이 Semantic20 `0..18, 255`의
  부분집합임을 보장한다.

## P0: 중단 후 재개와 원자적 publish

- 원격 연결 종료와 관계없이 tmux에서 실행할 수 있는 canonical entrypoint를 둔다.
- sample 단위 완료 manifest 또는 검증된 기존 출력 skip 기능을 제공해 RUGD
  7,436개 변환 중 중단되어도 처음부터 다시 시작하지 않게 한다.
- 진행 중 출력은 `<output>.partial` 또는 고유 staging run 아래에 기록한다.
- 모든 검증이 통과하기 전에는 `datasets/processed` 경로로 이동하지 않는다.
- 실패 시 stage, sample ID, 입력 경로, exception을 상태 JSON과 로그에 기록한다.
- 재실행 시 완성된 RELLIS 6,234개를 다시 변환하지 않고 실패한 stage부터 재개한다.
- `--overwrite`는 명시적으로 요청한 경우에만 허용하며 기본 동작은 fail-closed로
  유지한다.

## P1: Docker image 전처리 자산 패키징

- 현재 Dockerfile은 `src`, `configs`, `scripts`, `data`만 `/opt/adom`에 복사하고
  legacy `study`는 복사하지 않는다. 따라서 현 이미지에는 Semantic20 전처리
  스크립트와 RUGD split이 없다.
- `src/adom/runtime/semantic20_cycle.py`의 `REFERENCE_SPLITS`도 이미지에 없는
  `study/.../rellis3d_semantic20_v1/splits`를 참조한다. 현재 이미지에서는 E0/E1
  모두 GPU 학습 시작 전 dataset contract 검사에서 실패하므로, canonical split을
  `data/splits` 아래로 승격하고 runtime과 테스트가 같은 경로를 사용하게 수정한다.
- `COPY study`로 우회하지 않는다. 팀원이 승격한 canonical 코드, palette mapping,
  bridge mapping, RELLIS/RUGD split만 이미지에 포함한다.
- 이미지 빌드 후 다음 파일을 확인하는 smoke test를 추가한다.
  - Semantic20 preprocess CLI
  - RUGD RGB palette mapping
  - Semantic20 bridge mapping
  - RELLIS 및 RUGD train/val/test split
- 실행 로그에 image/source Git SHA와 dataset mapping version을 남긴다.

## P2: split 및 사전 검증 개선

- RUGD split 파일 세 개는 마지막 EOF newline이 없어 `wc -l`이 각각 1개씩 적게
  표시된다. 실제 레코드는 train 4,779, val 733, test 1,924로 총 7,436개다.
- split 검사 코드는 newline 개수가 아니라 비어 있지 않은 `splitlines()` 레코드
  수를 사용한다.
- 저장소의 text file은 가능하면 EOF newline을 추가하되, 입력 parser는 EOF
  newline 유무와 CRLF/LF 모두 처리한다.
- flat directory의 단순 파일 수뿐 아니라 expected split과 actual basename을
  양방향 비교하여 missing, unexpected, duplicate를 모두 검출한다.
- preflight에서 mask mode, shape, dtype, observed palette/ID를 표본이 아닌 전체
  또는 충분히 강한 deterministic audit으로 확인한다.

## 테스트 요구사항

- 합성 RGB RUGD mask가 기대한 Semantic20 ID와 255로 변환되는 단위 테스트
- indexed mask 입력의 기존 동작을 보존하는 회귀 테스트
- unknown RGB, 중복 파일명, 누락 pair, image/mask 크기 불일치 실패 테스트
- 중간 중단 후 재개 시 완료 파일을 재작성하지 않는 테스트
- 빈 staging, 부분 staging, 이미 publish된 output에 대한 idempotency 테스트
- Docker image 안에서 preprocess `--help`와 필요한 mapping/split 존재 여부 검사
- Docker image와 동일한 파일 집합에서 E0/E1 `validate_semantic20_dataset()`을
  실행하여 legacy `study` 경로 의존성이 없음을 확인하는 테스트

## 실제 데이터 완료 조건

- RELLIS 변환: 6,234개
- RUGD 입력/출력: 7,436개
- YCOR 최종 포함: 751개
- combined manifest: 14,421개
- main split: train 9,868 / val 900 / test 899
- validation/test: RELLIS-only 정책 유지
- missing image/mask, size mismatch, non-single-channel mask: 모두 0
- 최종 `results/final_check.json`의 `status`가 `PASS`
