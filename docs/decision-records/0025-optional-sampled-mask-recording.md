# 0025. Optional sampled mask recording

- Status: Accepted
- Date: 2026-08-13
- Owners: ADOM team
- Supersedes: 0020 sampled semantic autonomy evidence의 기본 mask 활성화 범위

## Context

0020은 live autonomy bag에 2 Hz Semantic20 evidence mask를 기본 추가했다. 발표용 공간
근거에는 유용하지만, 모든 장시간 주행에서 raster를 기록할 필요는 없으며 Jetson의
serialization과 disk I/O 영향은 target 실측 전까지 미검증이다. 클래스 발생 시간과 면적은
inference-rate `/adom/perception/status`에 이미 남고 작은 semantic costmap도 기록된다.

## Decision

`t2`를 두 가지 명시적 모드로 운영한다.

- `t2`: 2 Hz raster mask를 제외한다.
- `t2 mask`: 기본 topic에 `/adom/perception/semantic20_mask_evidence`를 추가한다.

두 모드 모두 perception status의 Semantic20 class pixel count/ratio, semantic costmap,
planner/controller/drive/PWM/watchdog와 GPS evidence를 기록한다. Full-rate Semantic20 mask,
RGB, confidence와 overlay는 어느 모드에서도 기록하지 않는다. Bash 인자는 `t2 mask`처럼
공백으로 전달하며, 지원하지 않는 인자는 녹화를 시작하지 않고 오류로 종료한다.

## Rationale

기본 세션은 수치·상태와 작은 grid로 부하를 줄이고, 고정 장면 A/B 및 판단 공간 근거가
필요한 trial만 sampled raster 비용을 지불한다. 클래스 통계를 기본 모드에 유지하므로 mask를
끄더라도 class 출현과 pixel share 시간축은 분석할 수 있다.

## Consequences

기본 `t2` bag만으로 pixel 위치를 재구성할 수 없다. 공간 mask가 필요한 trial은 시작 전에
`t2 mask`를 선택해야 하며, 두 모드는 session metadata의 `record_mask`와 `topic_regex`, 종료
후 `ros2 bag info`로 구분한다. 2 Hz sampling 사이의 짧은 공간 변화는 mask 모드에서도 놓칠
수 있다.

## Validation

두 모드에서 `ros2 bag info`를 확인한다. 기본 모드에는 perception status와 semantic
costmap이 있고 evidence mask가 없어야 한다. mask 모드에는 같은 topic에 evidence mask가
추가돼야 한다. 동일 장면 60초 A/B로 perception/controller tail latency와 watchdog 발생을
비교한다.
