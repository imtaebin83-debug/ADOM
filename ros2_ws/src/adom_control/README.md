# adom_control

Nav2의 `/cmd_vel`을 bicycle model로 조향각에 변환하고 PCA9685 CH0/CH1 PWM으로
출력한다. 기본값은 `dry_run: true`다.

실차 투입 전 순서:

1. LiPo를 분리하고 I2C 인식 확인
2. 바퀴를 지면에서 띄움
3. ESC neutral/arming 범위 측정
4. servo center/left/right 기계 한계 측정
5. `/emergency_stop`과 0.25 s watchdog 검증
6. 저속 제한으로 시작

하드웨어 모드에는 `adafruit-circuitpython-pca9685`가 필요하다. 현재 속도 제어는
엔코더가 없는 open-loop throttle 제어이며 실제 m/s를 보장하지 않는다.

