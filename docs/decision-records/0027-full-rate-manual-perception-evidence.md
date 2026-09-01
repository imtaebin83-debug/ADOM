# 0027. Full-rate manual perception evidence

- Status: Accepted
- Date: 2026-08-13
- Owners: ADOM team
- Supersedes: 0026의 sampled RGB 기록 방식

## Context

0026의 `t2 evidence`는 t4가 추론한 RGB와 mask를 약 2 Hz로 pair해 기록했다. 이 방식은
정적 공간 비교에는 충분하지만 manual 주행 중 물체 접근, 시야 변화와 실패 전후의 연속
영상을 보존하지 못한다. 기존 `rec`는 full-rate ZED RGB를 기록할 수 있으므로 manual
perception trial에서는 같은 원본 stream과 2 Hz mask를 함께 보존할 필요가 있다.

## Decision

`t2 evidence`는 t4 입력인 `/zed/zed_node/rgb/color/rect/image` full-rate source RGB와
`/adom/perception/semantic20_mask_evidence` 2 Hz mask를 기록한다. Perception node의 별도
2 Hz RGB 복제 publisher는 제거한다. Mask는 추론 입력 image message의 header를 그대로
보존하므로 analysis는 header timestamp로 full-rate RGB 중 정확한 source frame을 결합한다.

`t2`와 `t2 mask`는 변경하지 않는다. Full-rate Semantic20 mask, confidence와 overlay는
계속 제외한다. `t2 evidence`는 t5 없이 t0, t1, t3 manual, t4를 사용하는 짧은 manual
perception trial 전용이다.

## Rationale

원본 stream을 한 번 기록하면 전체 주행 영상과 정확한 RGB-mask pair를 모두 얻으며 sampled
RGB를 중복 저장하지 않는다. t4의 inference와 mask 내용은 recording profile에 따라 바뀌지
않는다.

## Consequences

Raw BGR 대역폭은 크다. 640x360 30 FPS는 약 20.7 MB/s, HD720 30 FPS는 약 82.9 MB/s이며
MCAP/DDS overhead와 mask가 추가된다. 따라서 full autonomy나 장시간 기본 기록으로 사용하지
않고 짧은 manual trial로 제한한다. Recorder가 별도 camera subscriber가 되므로 target
Jetson의 DDS copy, disk I/O, perception tail latency와 dropped frame을 실측해야 한다.

## Validation

Wheels-off에서 10초 warm-up 후 60초 trial을 실행한다. `ros2 bag info`에서 full-rate RGB와
약 2 Hz mask를 확인하고, 모든 mask header stamp에 정확히 일치하는 RGB가 존재하는지 검사한다.
RGB actual FPS, bag MB/s, perception processing p50/p95/p99, received/overwritten frames,
temperature, RAM, power와 throttling을 기록한다. Pair가 누락되거나 control 안정성에 영향이
있으면 0026 sampled profile로 되돌리거나 camera-side compressed recording을 별도 설계한다.
