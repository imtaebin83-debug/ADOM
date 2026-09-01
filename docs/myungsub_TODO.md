# 명섭 Jetson ROS 통합 TODO

- 업데이트: 2026-08-14 KST
- 점검 대상: Jetson `ztr@192.168.0.201`, ROS 2 Jazzy, `ROS_DOMAIN_ID=11`
- 범위: `t0`∼`t5` 실행 이후 ROS graph, 토픽 발행, 제어 중립 상태
- 상위 문서: [ADOM_CONTEXT.md](../ADOM_CONTEXT.md)
- 명령어 안내: [SHORTCUT.md](../SHORTCUT.md)

## 현재까지의 진행 상황

### 요약

`t0`∼`t3`의 센서, TF/localization, gamepad, PCA9685 기본 경로는 동작한다.
`t4`는 B0-E0/E-ADOM profile 분리와 checkpoint SHA 검증까지 코드로 준비됐지만
Jetson ROS graph 실측은 아직 필요하다. `t5` local planning도 현재 보존된 실측에서는
발견되지 않아 전체 Semantic20 자율주행 경로는 연결 검증 전이다.

| 구간 | 상태 | 확인 결과 |
| --- | --- | --- |
| `t0` 센서 | 대체로 정상 | ZED RGB/depth 프레임 수신, IMU 약 90 Hz, `/fix` 약 1 Hz |
| `t1` URDF/TF | 정상 | `/robot_state_publisher`, `/tf`, `/tf_static` 확인 |
| `t2` localization | 동작/경고 | local/global odometry 약 12∼14 Hz, EKF diagnostic error 발생 |
| `t3` 제어 | 정상/안전 중립 | `/drive`·PWM 약 50 Hz, ESC 1500 µs, steering 1556 µs |
| `t4` perception | 전달 준비/실측 필요 | `t4 b0-e0`/`t4 eadom` SHA-locked wrapper 준비 |
| `t5` planning/control | 중단 | costmap, planner, local path controller 노드·토픽 없음 |

### 센서와 TF

- `/zed/zed_node/rgb/color/rect/image`에서 현재 timestamp와
  `zed_left_camera_frame_optical` frame ID를 가진 메시지를 수신했다.
- `/zed/zed_node/depth/depth_registered`에서도 동일 optical frame 기준의
  depth 메시지를 수신했다.
- `/zed/zed_node/imu/data`는 약 90 Hz로 수신되었다.
- raw RGB/depth는 노트북↔Jetson Wi-Fi 구독에서 안정적인 FPS를 측정하지
  못했다. Jetson 로컬 구독으로 다시 측정해야 한다.
- `/robot_state_publisher`, `/tf`, `/tf_static`과 ZED state publisher가 발견되었다.

### GNSS와 localization

- `/nmea_serial_driver`, `/ekf_local_filter`, `/ekf_global_filter`,
  `/navsat_transform` 노드가 발견되었다.
- `/fix`는 약 1 Hz로 발행되지만 `status: -1`(위치 fix 없음)이다.
  실내 점검이므로 현재는 예상 가능한 결과다.
- `/odometry/local`과 `/odometry/global`은 외부 측정 기준 약 12∼14 Hz로
  메시지가 발행되었다.
- 다만 `/diagnostics`에서 `ekf_local_filter: odometry/filtered topic status`가
  `ERROR / No events recorded / Actual frequency 0 Hz`를 보고했다. 실제 odometry
  발행과 diagnostic 결과가 다르므로 추가 조사가 필요하다.

### 제어와 안전 상태

- `/gamepad_control`, `/joy_node`, `/pca9685_control` 노드가 발견되었다.
- 제어 경로는 `/cmd_vel` 구독 → `/drive` 발행 → PCA9685 출력으로
  설정되어 있다.
- 현재 mode는 `autonomous`이지만 `/cmd_vel` publisher는 0개이다.
- gamepad watchdog이 발행한 `/drive`는 `speed: 0.0`, `steering_angle: 0.0`이었다.
- `/adom/control/pwm_us`는 `[1500.0, 1556.0]`이었다. 이는
  `vehicle.yaml`의 ESC neutral/steering center 설정과 일치한다.
- 현재 관찰 시점에 구동·조향 출력은 중립이지만, 통합 점검은 계속
  바퀴를 띄운 상태에서 수행한다.

### `t4` Semantic20 perception blocker

- `ADOM_CHECKPOINT` 값이 실제 파일이 아닌
  `<CHECKPOINT_ABSOLUTE_PATH>` 자리표시자로 남아 있어 checkpoint 가독성
  검사가 실패했다.
- `t4` 함수의 실패 경로가 `exit 1`을 사용해 함수만 종료되지 않고
  SSH shell까지 logout됐다.
- ROS graph에 `/adom_perception`, `/adom/perception/semantic20_mask`,
  `/adom/perception/status`가 없다.
- 이 증상은 2026-08-07 historical observation이다. 현재 wrapper는 profile을
  명시하고 실패 시 launch 전에 종료한다.
- B0-E0는 `best_mIoU_iter_6000.pth`와
  `configs/adom/runtime/segformer_b0_640x384_rellis3d.py`, E-ADOM은 frozen
  `checkpoint.pth`와 `configs/adom/runtime/segformer_b0_640x384_eadom.py`를 쓴다.
  Cost4/B2/E1/다른 ontology checkpoint를 섞어 사용하지 않는다.

### `t5` Semantic20 local planning blocker

- `/semantic_costmap`, `/rule_planner`, `/local_path_control` 노드가 없다.
- `/adom/navigation/semantic_costmap`, `/adom/navigation/local_path`,
  `/adom/navigation/costmap_status`, `/adom/navigation/rule_status`,
  `/adom/control/local_path_status` 토픽이 없다.
- `t5` launch가 종료된 정확한 이유는 해당 SSH 터미널의 첫 오류 로그를
  확인해야 한다.
- `t4`가 정상화되기 전에는 Semantic20 mask가 없으므로 perception→costmap
  end-to-end 검증을 진행할 수 없다.

## 앞으로 해야 할 것

### P0 — `t4` dual-profile perception 전달

- [ ] B0-E0와 E-ADOM archive를 Jetson으로 전달하고 archive/checkpoint SHA256을
  각각 검증한다.
- [x] legacy B0-E0 run의 `best_mIoU_iter_6000.pth`를 선택했다. 이 run에는 후대의
  `checkpoint_selection.json`이 없으므로 해당 계약을 소급 적용하지 않는다.
- [x] E-ADOM Stage 2 iter 26,000을 validation으로 선택하고 canonical test/export
  parity를 완료했다. canonical test 전체는 B0-E0보다 낮아 대체 모델이 아닌 A/B
  후보로 둔다.
- [ ] checkpoint를 `~/ADOM/models/checkpoints/b0-e0/`와
  `~/ADOM/models/checkpoints/eadom/`에 각각 정확히 하나씩 둔다.
- [ ] Jetson `~/.bashrc`의 `t4` 함수가 `"$@"`를 wrapper로 전달하도록 교체한다.
- [ ] `t4 b0-e0`, `t4 eadom`을 각각 실행하고 출력된 checkpoint SHA가 계약과
  일치하는지 확인한다.
- [ ] `/adom_perception`, `/adom/perception/semantic20_mask`,
  `/adom/perception/status`를 확인한다.
- [ ] status JSON에서 inference error, overwritten frame, queue/model/end-to-end latency를
  기록한다.

### P0 — `t5` local planning 복구

- [ ] 기존 `t5` 터미널에 남은 최초 build/launch error를 보존하고 확인한다.
- [ ] `t4` mask 발행을 확인한 뒤 `t5`를 다시 실행한다.
- [ ] `/semantic_costmap`, `/rule_planner`, `/local_path_control` 노드를 확인한다.
- [ ] costmap/path/controller status 토픽에서 stale/empty input이 zero command로 이어지는지
  확인한다.
- [ ] Semantic20 mask가 Cost4 설정이 아닌
  `semantic20_costmap.launch.py`와 `semantic20_costs.yaml`을 사용하는지 확인한다.
- [ ] `/cmd_vel` publisher가 `local_path_control` 1개인지 확인한다.

### P1 — localization diagnostic 조사

- [ ] Jetson 로컬에서 `/odometry/local`, `/odometry/global`, ZED IMU 주기를 다시
  측정한다. Wi-Fi 구독 측정과 구분해 기록한다.
- [ ] EKF 설정의 목표 `frequency`와 실제 발행 주기를 비교한다.
- [ ] `No events recorded` diagnostic이 실제 출력 정지인지, diagnostic updater
  계측 문제인지 구분한다.
- [ ] 야외에서 GNSS fix 후 `/odometry/gps`, `/gps/filtered`, global EKF를 다시
  검증한다.

### P1 — wheels-off 통합 안전 검증

- [ ] STOPPED 또는 manual reset 상태에서 시작한다.
- [ ] perception/path/IMU/GPS 각 timeout에서 `/cmd_vel` zero와 neutral PWM을 확인한다.
- [ ] `t4` 또는 `t5` 프로세스를 종료했을 때 0.25초 이내 neutral 복귀를
  실측한다.
- [ ] STOP 후 자동 재출발하지 않고 수동 reset이 필요한지 확인한다.
- [ ] wheels-off 검증이 모두 통과한 뒤에만 0.3 m/s 이하 저속 시험으로
  이동한다.

## 완료 기준

- `t0`∼`t5` 모든 핵심 노드가 동시에 ROS graph에 존재한다.
- RGB → Semantic20 mask → semantic costmap → local path → `/cmd_vel` 경로의
  timestamp와 status를 확인한다.
- 정상 입력에서 최신 frame 기반 명령이 발행되고, stale/empty/timeout에서
  zero command와 neutral PWM으로 fail-safe한다.
- STOP 후 수동 reset 계약을 지킨다.
- GPS 미수신 실내 시험과 야외 GNSS fix 시험 결과를 구분해 기록한다.

## 2026-08-07 점검 메모

- 노트북에서 Jetson으로 연결된 일반 SSH 세션 5개와 VS Code Remote SSH
  세션 1개를 확인했다.
- 터미널 탭 하나는 SSH 자식 프로세스가 없었으며, `t4` 실패 후
  `exit 1`로 logout된 현상과 일치했다.
- 새 SSH 진단 세션은 key/password 인증이 없어 실패했다. 대신 동일
  LAN과 `ROS_DOMAIN_ID=11`에서 ROS 2 DDS graph와 실제 메시지를 읽기 전용으로
  측정했다.
