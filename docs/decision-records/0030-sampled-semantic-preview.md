# 0030. Sampled Semantic20 preview evidence

- Status: Accepted
- Date: 2026-08-14
- Owners: ADOM team
- Supersedes: none

## Context

`t2 evidence`는 manual perception trial의 전체 장면을 보존하기 위해 t4 입력 RGB를
full-rate로 기록하고 2 Hz Semantic20 mask를 같은 timestamp로 결합한다. 이 계약은 사후
분석에는 유용하지만 짧은 현장 확인에서도 raw BGR 기록 대역폭이 크며, bag을 열자마자
Semantic20 결과가 장면의 어디에 놓였는지 보려면 RGB-mask 결합 후처리가 필요하다.

기존 `/adom/perception/overlay`를 bag에서 구독하면 t4가 추론 주기마다 BGR 합성을 수행하므로
sampled preview 용도로는 부하가 불필요하게 크다.

## Decision

`t4` perception node는 `/adom/perception/semantic20_overlay_evidence`를 추가한다. Subscriber가
있을 때만 2 Hz evidence cadence에서 canonical Semantic20 palette를 해당 추론 입력 RGB에
`overlay_alpha=0.45`로 합성한다. Overlay와
`/adom/perception/semantic20_mask_evidence`는 동일한 source image header를 보존한다.

`t2 preview`는 2 Hz evidence mask와 evidence overlay를 기록한다. Full-rate source RGB,
full-rate mask, confidence와 `/adom/perception/overlay`는 기록하지 않는다. 기존 `t2`,
`t2 mask`, `t2 evidence`의 의미는 변경하지 않는다.

## Rationale and evidence

Perception worker가 이미 해당 frame의 RGB와 mask를 함께 가지고 있으므로 별도 colorizer나
timestamp join 없이 정확히 대응하는 합성 영상을 만들 수 있다. 전용 sampled topic은
recorder가 full-rate `/overlay`를 활성화하는 것을 방지한다.

640x360 `bgr8` overlay를 2 Hz로 기록할 때 raw payload 추정치는 약 1.38 MB/s 또는
83 MB/min으로, 30 FPS source RGB 약 20.7 MB/s보다 작다. 실제 MCAP/DDS overhead와 Jetson
latency 영향은 아직 실기 검증 전이다.

## Alternatives considered

- 기존 `/overlay`를 직접 기록: 구독 시 추론 주기 전체의 합성과 기록이 활성화돼 제외했다.
- `semantic20_colorizer` 출력을 기록: mask 색상만 보여 원본 장면과의 위치 관계를 한 화면에서
  확인할 수 없으며 별도 node가 필요해 제외했다.
- `t2 evidence`를 preview로 대체: 연속 원본 영상과 재처리 가능한 evidence가 사라지므로 기존
  모드는 유지한다.

## Consequences

Preview는 사람이 즉시 확인하기 위한 손실형 파생 영상이며 원본 RGB나 class ID mask를
대체하지 않는다. 따라서 `t2 preview`도 mono8 mask를 함께 보존한다. Subscriber가 없으면
evidence overlay 합성 비용은 발생하지 않는다.

## Validation and rollback

Wheels-off 상태에서 `t2 preview`, `t4`를 실행하고 `ros2 bag info`로 mask와 overlay가 각각
약 2 Hz인지 확인한다. 모든 overlay header stamp가 같은 bag의 mask stamp와 일치하고,
overlay가 45% alpha의 canonical palette인지 확인한다. Perception processing p95/p99,
overwritten frames와 bag MB/s가 안전 운용에 영향을 주면 `t2 mask`로 되돌리고 preview
publisher 구독을 중단한다.
