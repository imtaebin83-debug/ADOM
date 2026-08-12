# 0015. IMU-aided nominal 12 m/s PWM calibration

- Status: Accepted
- Date: 2026-08-12
- Owners: ADOM control integration
- Supersedes: 6.0 m/s downstream mapping portions of 0014

## Context

Control calibration은 forward ESC pulse 2000 us를 12 m/s로 정의하도록 변경 요청됐다.
Planner profile 0.25..3.0 m/s는 유지한다. 차량에는 wheel encoder나 VESC velocity
telemetry가 없지만 ZED IMU topic은 주행 중 수신할 수 있다.

## Decision

PCA9685와 gamepad forward ceiling을 nominal 12.0 m/s로 둔다. Forward pulse width는
1500 us = 0 m/s, 2000 us = 12 m/s 사이를 선형 보간한다. PWM carrier는 계속 50 Hz이며
속도 대응은 frequency가 아니라 pulse width로 정의한다.

Local controller는 최종 `/drive.speed`가 zero 부근에서 0.5초 유지될 때만 IMU x축 bias를
EMA로 학습하고 velocity를 zero로 갱신한다. Motion command 중에는 bias를 제거한 가속도를
제한 적분하고, planner target과 추정 속도의 오차에 `speed_kp: 0.15`를 적용한다. 최종
command는 planner의 3.0 m/s maximum을 넘지 않는다.

## Rationale and evidence

정지 구간 bias 학습은 sensor offset과 고정 mounting/gravity projection을 줄이고,
zero-velocity update는 주행 사이의 누적 drift를 제거한다. 주행 중 적분은 짧은 가감속
구간의 오차 방향을 feedback에 반영한다. IMU acceleration만으로 등속 absolute velocity는
관측할 수 없으므로 이 보정은 bounded short-horizon feedback이다.

## Consequences

- Nominal forward mapping은 `v = (pulse_us - 1500) * 12 / 500`이다.
- Planner 0.25 m/s는 약 1510.4 us, 3.0 m/s는 1625 us다.
- Status에는 estimated speed, learned bias, speed error, stationary-update 여부가 기록된다.
- 12 m/s는 명령 calibration이지 실측 최고속도 주장이 아니다.
- 절대 속도 정확도에는 향후 encoder, VESC telemetry 또는 외부 속도 기준이 필요하다.

## Validation and rollback

단위 테스트에서 12 m/s config consistency, stationary bias 학습, ZUPT, bias-corrected
integration과 feedback 방향을 확인한다. 실차에서는 먼저 바퀴를 띄우고 1500 us neutral,
planner 범위 1510.4..1625 us, watchdog neutral을 확인한다. 이후 폐쇄된 공간에서 낮은
planner speed부터 실제 속도를 별도 측정한다. IMU estimate가 발산하거나 command가
oscillate하면 `speed_kp`를 0으로 내려 open-loop로 즉시 rollback한다.
