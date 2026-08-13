# adom_logging

자율주행 세션의 perception/planning/control 상태, 속도·조향 명령, PWM과 GPS fix를
하나의 bounded rosbag으로 기록한다. 기본 모드는 raster mask를 제외하고, 선택적으로 2 Hz
evidence mask를 추가할 수 있다. 작은 semantic costmap grid와 perception status의 클래스별
픽셀 통계는 두 모드 모두 기록한다.
카메라, full-rate mask, confidence/overlay, path, IMU와 TF는 실시간 처리에 recorder 부하를
더하지 않도록 기록하지 않는다. GPS logger는 최초 valid fix를 원점으로 한 local ENU 근사 경로를
`/adom/logging/gps_path`로 발행하지만, bag에는 작은 raw `/fix` 연속값만 저장해 경로를
재구성한다. GPS는 planning/control에는 연결하지 않는다.

```bash
export ADOM_REPO_ROOT="$(git rev-parse --show-toplevel)"
ros2 launch adom_logging autonomy_logging.launch.py \
  capture_root:=data/autonomy_bags

# 2 Hz Semantic20 evidence mask까지 필요할 때만 사용
ros2 launch adom_logging autonomy_logging.launch.py \
  capture_root:=data/autonomy_bags record_mask:=true
```

recorder는 launch와 함께 자동 시작하고 종료 시 SIGINT로 rosbag metadata를 정상적으로
닫는다. `rule_status`에는 blocked/driving, 선택 조향 시퀀스, 속도와 clearance가 있고
`local_path_status`에는 명령/추정 속도와 watchdog 상태가 있어 회피 과정과 속도 변화를
분석할 수 있다. 결과 디렉터리는 Git에서 제외되며 기본 20 GB 제한과 1 GB split을
사용한다.

`record_mask:=true`에서 640x384 mono8 evidence mask 2 Hz는 약 0.49 MB/s, 80x60 int8
costmap 10 Hz는 payload 기준 약 0.05 MB/s다. DDS/MCAP overhead를 제외하면 mask 모드의
두 raster 합계는 약 33 MB/min이며, target Jetson에서 기본/mask 모드의 camera→command
p95와 watchdog 발생을 A/B 비교해야 한다.
