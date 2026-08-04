# adom_safety

안전 계약과 제한값을 보관한다. `/emergency_stop: std_msgs/Bool`이 true이면 control은
즉시 neutral PWM을 출력한다. 소프트웨어 E-stop은 물리적인 LiPo 차단 스위치나 RC
override를 대체하지 않는다. 실제 supervisor/collision monitor launch는 센서 시험 후 추가한다.

