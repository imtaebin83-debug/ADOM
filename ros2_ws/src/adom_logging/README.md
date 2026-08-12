# adom_logging

자율주행 세션의 perception 판단·mask, costmap, tree planning, control, PWM, GPS를
하나의 bounded rosbag으로 기록한다. 고대역폭 진단 영상인 confidence와 BGR overlay는
live bag에서 제외한다. GPS는 최초 valid fix를 원점으로 한 짧은 local ENU 근사 경로를
`/adom/logging/gps_path`로 발행하지만 planning/control에는 연결하지 않는다.

```bash
export ADOM_REPO_ROOT="$(git rev-parse --show-toplevel)"
ros2 launch adom_logging autonomy_logging.launch.py \
  capture_root:=data/autonomy_bags
```

recorder는 launch와 함께 자동 시작하고 종료 시 SIGINT로 rosbag metadata를 정상적으로
닫는다. 결과 디렉터리는 Git에서 제외되며 기본 20 GB 제한과 1 GB split을 사용한다.
