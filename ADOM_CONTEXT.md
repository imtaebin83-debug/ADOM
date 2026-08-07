# ADOM Project Context — Autonomous Driving Foundation Single Source of Truth

> 상태: **ACTIVE / Semantic20 자율주행 기반 구축**
> 기준일: **2026-08-07**
> 권위: 이 파일이 ADOM의 유일한 프로젝트 Source of Truth다. 기존 설계 문서나
> 회의 기록과 충돌하면 이 문서를 우선한다.
> 변경 규칙: 범위, 인터페이스, 담당자, 성공 기준을 변경할 때는 이 파일의
> `Decision Log`에 날짜·결정·이유를 함께 기록한다.

## 0. 문서 운영 계약

문서는 아래 순서로 해석한다. 하위 문서가 상위 문서와 충돌하면 상위 문서를 따른다.

1. **이 파일:** 지금 유효한 범위, 시스템 계약, 책임, 성공 기준
2. **Accepted decision record:** 결정 당시의 근거와 대안; 현재 상태는 이 파일 기준
3. **Project status:** 실제 완료율, blocker, evidence, 다음 hand-off
4. **Experiment protocol/run record:** 재현 가능한 연구 조건과 결과
5. **Meeting/AI collaboration note:** 논의 출처이며 그 자체로 승인된 결정은 아님

운영 링크:

- [Docs hub](docs/README.md)
- [Project status dashboard](docs/status/README.md)
- [D-5 PoC live status](docs/status/d5-poc.md)
- [Decision record index](docs/decision-records/README.md)
- [AI collaboration notes](docs/ai-collaboration/README.md)

상태 문서에는 계획을 복제하지 않고 `완료/진행/막힘`, evidence와 다음 행동만 쓴다.
회의나 AI 대화가 범위·계약을 바꾸면 대화 노트만 남겨서는 안 되며, 같은 변경에서
이 파일과 decision record를 함께 갱신한다. 추측값은 `제안/미검증`으로 표시하고
담당자가 실측한 뒤에만 `검증됨`으로 바꾼다.

## 1. 프로젝트 개요

ADOM은 산악·오프로드 환경의 1/10 RC Car에서 Semantic20 perception을 기반으로
실제 자율주행 stack을 단계적으로 구축한다. 현재 increment는 최신 카메라 frame을
우선하는 Semantic20 perception과 camera→software-action latency 계측이다. 클래스별
주행 비용, depth 결합, localization/Nav2 및 closed-loop 주행 계약은 검증·결정 기록을
거쳐 후속 increment에서 연결한다.

2026-08-06의 D-5 Go/Stop PoC 범위와 결과는 안전 baseline 및 historical milestone로
보존한다. 이 문서의 D-5 상세 절은 당시 계약을 설명하며, 현재 범위와 충돌할 때는
이 절과 decision record 0008을 우선한다. watchdog, command timeout neutral, STOP 후
수동 reset, wheels-off→저속 순서의 안전 정책은 계속 유효하다.

### 현재 Semantic20 perception 계약

- Canonical ontology: `src/data/semantic_20/config/bridge_mapping.yaml`
- Train IDs: `0..18`; ignore: `255`; Cost4/Cost5와 topic·config·artifact 분리
- Output: `/adom/perception/semantic20_mask` (`sensor_msgs/Image`, `mono8`)
- Input QoS: Best Effort, Keep Last 1; callback은 one-slot mailbox의 최신 frame만 보존
- Scheduling: 현재 추론 완료 후 그 시점의 최신 frame을 선택; 시작률 상한 30 FPS
- Latency: capture→receive, queue, inference, capture→perception output을 status로 발행
- Software action latency: 호환되는 costmap 연결 후 camera stamp→각 costmap의 첫
  `/cmd_vel` publish 및 rolling p50/p95를 `/adom/navigation/action_latency`로 발행
- Physical action latency: PCA9685/ESC/servo 응답은 software metric에 포함되지 않으며
  target hardware에서 외부 계측 필요
- Semantic20→주행 비용 mapping: **미결정**; 기존 Cost4 costmap에 자동 연결 금지

30 FPS는 설정된 추론 시작률 상한이며 Jetson 실측 처리량이 아니다. camera timestamp와
ROS clock이 같은 time domain인지 target 장치에서 확인하기 전 end-to-end latency 값은
검증됨으로 간주하지 않는다.

### 현재 확보 자산

- MMSegmentation 1.2.2 기반 Semantic20 학습·평가 파이프라인
- RELLIS 기반 SegFormer-B0/B2 E0 checkpoint
- RunPod 학습·resume·W&B·고정 split 및 metric contract
- Jetson Orin Nano 8GB, ZED 2i, PCA9685 PWM, 배터리가 장착된 RC Car
- ROS 2 control node, gamepad manual/autonomous/stop mode, command watchdog
- 640x384 ONNX export 설정과 PyTorch↔ONNX logits parity 검사

### 아직 구현되지 않았거나 target hardware에서 검증되지 않은 핵심 구간

- TensorRT engine 및 Jetson standalone inference
- `adom_perception_ros` Semantic20 inference 코드는 구현됨; Jetson/ZED 실측 미검증
- target mask에서 Go/Stop을 결정하는 safety-reflex node
- perception→`/drive/autonomous`→PCA9685 end-to-end 검증
- 신규 target-class 촬영·CVAT 라벨·short fine-tuning

## 2. Historical D-5 Definition of Done

### 필수 성공 조건

1. ZED 2i RGB frame이 Jetson의 SegFormer-B0 FP16 TensorRT engine으로 입력된다.
2. target-class mask와 ROI 판정 값이 ROS topic으로 발행된다.
3. target이 중앙 safety corridor에 진입하면 차량이 0.3 m/s 이하에서 정지한다.
4. perception/control message가 끊기면 0.25초 이내 neutral PWM으로 복귀한다.
5. E0의 실패와 신규 fine-tuned 모델의 성공을 같은 고정 장면에서 촬영한다.
6. 신규 학습이 실패해도 아래 fallback 중 하나로 발표 가능한 PoC를 보존한다.

### 비목표

- 완전 자율 경로계획 또는 Nav2 navigation
- 조향 회피; D-5 성공 조건은 **정지**까지다.
- ZED depth, point cloud, VIO, 3D/BEV costmap
- LoRA, adapter, 새 backbone/decoder
- INT8 calibration 또는 dynamic-shape engine
- uncertainty/OOD 보장, 물리적 traversability 예측
- 야간·경사·완전 가림 문제 해결 주장
- 웹 대시보드와 배포 자동화

## 3. D-5 시연 시나리오와 타겟 클래스

### 3.1 기본 타겟: `log`

`log`를 발표 PoC의 기본 target class로 사용한다.

- B0 IoU 40.33, B2 IoU 40.17로 backbone 용량 증가가 문제를 해결하지 못했다.
- RELLIS E0 train 노출은 94 images로 매우 희소하다.
- 산과 도시에서 통나무·굵은 나뭇가지를 안전하게 배치해 반복 촬영할 수 있다.
- pole보다 영상에서 차지하는 면적이 넓어 resize 후에도 label이 보존된다.
- 차량 진행 경로를 가로지르는 collision hazard라는 시연 서사가 명확하다.

### 3.2 Stretch: `pole`

pole은 B0/B2 모두 IoU·Recall 0으로 연구적 실패 사례는 가장 강하다. 그러나 얇은
구조가 resize/crop에서 소실되고 전체 화면 면적 기준과 맞지 않아 D-5 기본 타겟으로는
위험하다. log pipeline이 Day 2 안에 통과한 경우에만 추가 수집·정성 시연한다.

### 3.3 Hard fallback: `rubble`

신규 fine-tuning이 개선되지 않으면 기존 B0→B2의 rubble IoU 개선
`53.34 → 66.87 (+13.53%p)`을 사용한다. 동일 TensorRT/ROS pipeline에서 B0와 B2
engine만 교체해 recorded/live 비교를 시도한다. B2 latency가 라이브 요구를 충족하지
못하면 recorded ZED input에서 결과를 비교하고, 라이브 정지는 B0 pipeline으로 보인다.

### 3.4 제외 클래스

- water: canonical test GT가 없고 RGB만으로 깊이·견인 위험을 알 수 없다.
- puddle: B0 IoU가 이미 70.93이며 개선 서사가 약하다.
- mud: 환경 재현성이 낮고 정지 장애물보다 주행 비용 문제에 가깝다.
- barrier: B0는 56.68이나 B2는 39.27로 악화되어 post-model 성공 보장이 없다.

## 4. 정지 판정 계약

전체 화면의 단순 5% threshold를 사용하지 않는다. 영상 하단 중앙의 고정
`safety corridor`에서 target connected component를 평가한다.

초기 파라미터는 validation 영상에서만 조정하고 최종 촬영 전에 동결한다.

| 항목 | 초기값 | 규칙 |
| --- | ---: | --- |
| 최대 자율 속도 | 0.30 m/s | 실제 속도는 open-loop이므로 저속 실측 필요 |
| target | log, class ID 10 | Semantic20 ID 고정 |
| ROI | 하단 중앙 trapezoid | 카메라 장착 후 고정 |
| target area ratio | ROI의 1.0% | log validation에서 0.5–3% 사이 조정 |
| stop debounce | 3 frames | 연속 충족 시 STOP |
| release debounce | 5 frames | 발표 시 자동 release 대신 수동 reset 권장 |
| command timeout | 0.25 s | timeout 시 neutral |

STOP 이후 자동 재출발은 기본적으로 금지한다. 운영자가 게임패드로 scene을 확인하고
manual 또는 autonomous mode를 다시 선택해야 한다.

## 5. D-5 파이프라인 아키텍처

```mermaid
flowchart LR
    A["ZED 2i RGB"] --> B["Verified Preprocess / Candidate 640x384"]
    B --> C["SegFormer-B0 TensorRT FP16"]
    C --> D["Semantic20 Argmax Mask"]
    D --> E["Log ROI / Temporal Debounce"]
    E --> F["Go/Stop Safety Reflex"]
    F --> G["/drive/autonomous"]
    G --> H["Existing Gamepad Mode Mux"]
    H --> I["/drive"]
    I --> J["PCA9685 PWM"]
```

### ROS 최소 인터페이스

새 custom message는 D-5 이후로 연기하고 표준 message만 사용한다.
아래 topic은 구현을 시작하기 위한 **제안 계약**이며, 실제 설치된 ZED wrapper와
control node에서 담당자가 검증하기 전까지 확정값으로 간주하지 않는다.

| 데이터 | 제안 topic | type | 검증 담당 | 상태 |
| --- | --- | --- | --- | --- |
| ZED rectified RGB | `ros2 topic list -t` 결과로 확정 | `sensor_msgs/Image` | 명섭 | 미검증 |
| semantic mask | `/adom/perception/semantic_mask` | `sensor_msgs/Image` (`mono8`) | 가형 | 제안 |
| target area ratio | `/adom/perception/target_area_ratio` | `std_msgs/Float32` | 가형 | 제안 |
| target detected | `/adom/perception/target_detected` | `std_msgs/Bool` | 가형·명섭 | 제안 |
| autonomous command | `/drive/autonomous` | `ackermann_msgs/AckermannDriveStamped` | 명섭 | 저장소 코드 기준, 실기 검증 필요 |
| final actuator command | `/drive` | `ackermann_msgs/AckermannDriveStamped` | 명섭 | 저장소 코드 기준, 실기 검증 필요 |
| emergency stop | `/emergency_stop` | `std_msgs/Bool` | 명섭 | 저장소 코드 기준, 실기 검증 필요 |

ZED image topic 이름은 wrapper 버전에 따라 다를 수 있으므로 문자열을 추정하지 않는다.
Day 1에 `ros2 topic list -t`로 실제 topic을 확정하고 launch parameter에 기록한다.
QoS, frame ID, timestamp, message frequency도 `ros2 topic info --verbose`,
`ros2 topic hz`, 실제 callback log로 검증한다. 검증 결과가 이 표와 다르면 담당자가
실제 동작값으로 이 문서와 launch/config를 함께 수정한다.

## 6. Jetson 하드웨어·소프트웨어 기준

### 하드웨어

- NVIDIA Jetson Orin Nano 8GB
- ZED 2i with polarizer, 2.1 mm lens
- USB-C dual-screw 0.3 m cable; USB 3 링크로 직접 연결하고 hub를 사용하지 않는다.
- PCA9685 PWM, ESC/servo, 별도 안정화된 Jetson 전원

### 설치 보고와 검증 상태

팀은 NVIDIA SDK가 권장한 방식으로 JetPack 7.2를 설치했다고 보고했다. 이 경우
예상되는 공식 조합은 L4T 39.2, Ubuntu 24.04, CUDA 13.2.1, TensorRT 10.16.2다.
이 보고는 합리적이지만 실제 package와 PATH는 아직 audit 전이므로 검증 완료 상태가
아니다. 이전에 확인한 `JetPack 6.x` 또는 `CUDA 13.5` 표기는 오인·복수 toolkit·PATH
문제일 수 있다.

- JetPack 6.2.2: L4T 36.5, Ubuntu 22.04, CUDA 12.6, TensorRT 10.3
- JetPack 7.2: L4T 39.2, Ubuntu 24.04, CUDA 13.2.1, TensorRT 10.16.2

CUDA 13.5는 공식 JetPack 7.2 bundle로 확인되지 않았다. `nvcc` 하나만 보지 말고
BSP, runtime package, symlink를 함께 확인해 아래 audit 결과를 이 문서에 기록한다.

### Day 1 read-only stack audit

```bash
cat /etc/nv_tegra_release
cat /etc/os-release
uname -a
dpkg-query -W nvidia-l4t-core nvidia-jetpack 2>/dev/null
nvcc --version
readlink -f /usr/local/cuda
ls -ld /usr/local/cuda*
dpkg-query -W 'libnvinfer*' 2>/dev/null
/usr/src/tensorrt/bin/trtexec --version || trtexec --version
ros2 doctor --report
lsusb -t
```

`lsusb -t`에서 ZED 2i가 USB 3 속도로 연결됐는지 확인한다.

### D-5 stack 결정 규칙

1. L4T가 `R39.2`이고 Ubuntu 24.04/Jazzy, CUDA 13.2.x, TensorRT 10.16.x가
   일관되면 재설치하지 않고 현재 JetPack 7.2 stack을 사용한다.
2. L4T가 `R36.x`인데 rootfs가 Ubuntu 24.04이거나 CUDA 13.x라면 unsupported hybrid로
   간주한다. ZED·TensorRT·ROS control이 모두 실측 통과하지 않았다면 Day 1에만
   JetPack 7.2 clean flash를 허용한다. Day 2 이후 reflash는 금지한다.
3. L4T R36.5 / Ubuntu 22.04가 일관되고 전체 hardware가 이미 동작하면
   JetPack 6.2.2를 유지하되 ROS는 Humble 경로로 전환해야 한다. Jazzy와 혼합하지 않는다.

Ubuntu 24.04/Jazzy를 유지하는 프로젝트 결정 때문에 장기 표준은 JetPack 7.2로 둔다.
ZED 2i USB는 JetPack 7.2/Orin용 ZED SDK 5.4 package를 사용한다. GMSL driver 제약은
USB ZED 2i에는 해당하지 않는다.

### 성능·안전 설정

- Day 1에는 bare-metal TensorRT runtime을 먼저 통과시킨다. Jetson Dockerfile은
  라이브 PoC 이후 만든다.
- `nvpmodel -q --verbose`, `tegrastats`로 전력·온도·RAM을 기록한다.
- cooling fan과 안정적인 전원을 사용한다.
- 실제 power mode 변경은 현재 mode ID를 확인한 뒤 수행한다.

## 7. ZED 2i D-5 설정

첫 라이브 PoC는 RGB-only다.

| 모듈 | D-5 설정 | 이유 |
| --- | --- | --- |
| grab resolution | HD720 | 충분한 target detail |
| RGB publish | 640x360, 10–15 Hz 후보 | 담당자가 실제 wrapper parameter·topic hz 검증 |
| model input | 640x384 static 후보 | export config 재사용, accuracy/latency 검증 후 동결 |
| depth | NONE | TensorRT와 GPU 경합 방지 |
| point cloud | OFF | 불필요한 GPU/CPU/DDS 부하 제거 |
| positional tracking/VIO | OFF | Go/Stop에 불필요 |
| object/body detection | OFF | 불필요 |
| RViz | 최종 live에서 OFF | subscriber·rendering 부하 방지 |
| recording | 수집 단계만 ON | live inference와 분리 |

ROS image subscriber는 Best Effort, Keep Last 1을 사용하고 inference 중 도착한 과거
frame은 쌓지 않는 구성을 우선 검토한다. 실제 ZED publisher QoS와 호환되는지는
담당자가 `ros2 topic info --verbose`로 확인한다. 최종 시연 영상은 외부 카메라로
촬영한다.

## 8. 모델·배포 계약

### D-5 모델

- Architecture: SegFormer-B0
- Baseline: E0 RELLIS checkpoint
- Ontology: Semantic20, 19 trainable classes, void/unknown 255
- Demo precision: TensorRT FP16
- Shape candidate: static batch 1, `1x3x384x640`; Day 1 parity·latency 후 동결
- INT8, dynamic shape, batch inference: 금지

### 전처리 후보와 동결 조건

```text
ZED RGB 1280x720
→ resize 640x360
→ pad to 640x384
→ MMSeg training과 동일한 RGB order, mean, std
→ NCHW static tensor 1x3x384x640
```

이 선택은 SegFormer의 필수 입력 크기가 아니다. 640x360은 16:9 영상을 보존하고,
640x384는 기존 export config를 재사용하면서 각 축을 32 배수로 맞춰 static TensorRT
shape를 단순화하는 후보다. 직접 640x384 resize하면 세로 왜곡이 생기므로 padding을
우선 검토한다.

단, 현재 MMSeg 학습은 512x512 random crop과 test keep-ratio resize를 사용한다.
또한 MMDeploy/MMSeg의 실제 pad 방향은 제안한 대칭 padding과 다를 수 있다. 따라서
`top/bottom 12 px`를 확정 계약으로 두지 않는다. 태빈과 가형은 같은 reference image에
대해 MMSeg `task_processor.create_input()`의 실제 tensor shape·pixel·padding 위치를
dump하고 Jetson preprocessing을 동일하게 맞춘다. 후보는 다음 세 개만 비교한다.

1. 640x360 → 640x384 padding
2. 640x384 direct resize
3. 기존 학습 계약과 가까운 512x512

Day 1에 PyTorch/ONNX parity, target mask 보존, Jetson latency를 확인해 하나를 동결하고
`preprocess.json`에 resize, interpolation, padding 방향·값, RGB/BGR, mean/std를 기록한다.

### 태빈→가형 hand-off package

```text
adom-b0-<target>-<version>/
├── model_static_1x3x384x640.onnx
├── checkpoint.pth
├── resolved_mmseg_config.py
├── labels.json
├── palette.json
├── preprocess.json
├── export_report.json
├── pytorch_onnx_parity.json
├── reference_images/
├── reference_masks/
├── build_engine.sh
└── SHA256SUMS
```

태빈은 RunPod/A100에서 TensorRT engine을 만들지 않는다. TensorRT engine은 target
Jetson의 실제 TensorRT 버전과 GPU에서 가형이 생성한다. 태빈이 원격으로 빌드를
지원하더라도 생성 위치와 최종 검증 책임은 target Jetson이다.

### 배포 acceptance gate

- PyTorch↔ONNX pixel argmax agreement ≥ 99.9%
- ONNX↔TensorRT pixel argmax agreement ≥ 99.0%
- target ROI area ratio 차이 ≤ 0.2%p
- reference image 10장 이상 parity 통과
- camera→command p95 latency 기록
- 10 Hz control update 목표; 최소 5 Hz 미만이면 live GO 금지
- 0.25초 command loss 시 neutral

MMDeploy는 ONNX export/graph rewrite까지만 사용한다. Jetson에는 training Docker나
전체 MMSeg stack을 설치하지 않고 native TensorRT runtime과 최소 ROS node를 우선한다.

## 9. 신규 데이터 계약

### CVAT와 라벨

- CVAT Docker 설치·project 생성: 태빈
- Annotation: Semantic20 `log` ID 10만 라벨
- 나머지 픽셀: `255 ignore`; 임의 background class를 만들지 않는다.
- 원본 영상과 annotation export를 모두 versioned archive로 보관한다.
- target-only partial label은 RELLIS full-label anchor data와 섞어서만 학습한다.

### 수집·분할

- 장소: 산과 도시 모두 가능
- 같은 영상의 인접 frame을 train/val/test에 나누지 않는다.
- 최소 3개의 독립 촬영 sequence를 사용한다.
- 권장 목표: train 120, val 30, untouched test 30 masks
- 정면·좌우 위치, 거리, 조명, 배경을 변화시킨다.
- target이 없는 negative video도 별도 보존해 false-stop test에 사용한다.

### 라벨 QC

- log 경계가 실제 물체를 포함하는지 overlay 확인
- class ID 10과 ignore 255 외 값이 없는지 자동 검사
- mask/image 크기와 pair 검증
- train/val/test source sequence 중복 검사
- resize 후 log pixel이 소실되지 않는지 640x384 preview 확인

## 10. Short fine-tuning recipe

1. B0-E0 selected checkpoint에서 시작한다.
2. RELLIS anchor와 custom-log partial data를 초기 1:1 exposure로 구성한다.
3. Head 중심 500–1,000 optimizer updates를 실행한다.
4. Backbone LR을 head의 0.1배로 두고 full model을 2,000–5,000 updates fine-tune한다.
5. 500 updates마다 custom validation target IoU/Recall과 RELLIS validation을 평가한다.
6. 신규 target이 학습되지 않을 때만 target class weight를 최대 3배로 적용한다.
7. test와 최종 시연 영상을 보고 threshold·checkpoint를 반복 선택하지 않는다.

### 선택 기준

- Primary: custom validation `log Recall`
- Secondary: custom validation `log IoU/Precision`
- Safety: negative clips의 false-stop rate
- Non-degradation: RELLIS supported mIoU와 주요 클래스가 크게 붕괴하지 않음

시연 GO 기준은 held-out custom test에서 E0 대비 target Recall이 명확히 증가하고,
고정 negative 장면에서 과도한 정지가 발생하지 않는 것이다. 단일 seed·소규모 데이터
결과는 PoC evidence로만 보고 일반화 연구 결과로 표현하지 않는다.

## 11. 팀 R&R

### 태빈 — Model/Data Lead

- CVAT Docker/project와 label export contract 설정
- 신규 partial-mask dataset validation·통합
- B0 short fine-tuning과 checkpoint 선택
- ONNX export, PyTorch↔ONNX parity, hand-off package
- W&B 결과와 전·후 confusion/target metric 정리
- D-5 중 웹 제작과 Jetson Dockerfile 고도화는 수행하지 않는다.

### 가형 — Edge Deployment Lead

- Day 1 Jetson stack audit와 버전 동결
- target Jetson에서 TensorRT FP16 engine 생성
- standalone file/camera inference와 ONNX↔TensorRT parity
- `adom_perception_ros` 최소 inference node 구현
- 실제 ZED/ROS topic·type·QoS·frequency를 검증하고 Source of Truth 갱신
- p50/p95 latency, RAM, 온도, frame-drop 측정
- CVAT 작업은 담당하지 않는다.

### 명섭 — Sensor/Control Lead

- ZED 2i RGB 설정과 원본 영상/SVO 수집
- `/drive/autonomous` Go/Stop safety-reflex node
- 실제 control topic·watchdog·E-stop 계약을 실기에서 검증하고 Source of Truth 갱신
- ROI, temporal debounce, manual reset 구현
- PCA9685 steering/ESC neutral 실측, watchdog, gamepad, E-stop 검증
- wheels-off→저속 직진→target stop 순으로 실차 검증

### 용준 — Annotation/Presentation Support, 3 days

- target-class CVAT annotation
- mask overlay와 sequence split QC
- 시연 영상 정리와 발표 자료 작성
- 실패 사례와 제한사항 표 정리

### 공동 책임

- 물리 emergency stop 담당자는 매 실차 시험 전에 지정한다.
- model/threshold가 실패하면 숨기지 않고 fallback gate를 즉시 적용한다.
- 최종 촬영 전에 model, engine, ROI, threshold, speed를 동결한다.

## 12. D-5 일정과 Gates

### Day 1 — Baseline pipeline first

- Jetson stack audit 및 최종 stack 결정
- PWM neutral/steering/gamepad/watchdog wheels-off 검증
- B0-E0 ONNX export와 target Jetson FP16 engine
- file inference 성공
- log/pole 후보 각 20개 frame의 E0 miss pattern 확인
- 기본 target은 log; log 촬영이 불가능할 때만 pole로 전환

**Gate 1:** file→TensorRT mask와 control hardware가 각각 독립 통과하지 않으면
데이터 수집 외 신규 기능 개발을 중단하고 해당 blocker를 먼저 해결한다.

### Day 2 — Live E0 and data

- ZED RGB→TensorRT→mask live
- mask→Go/Stop→`/drive/autonomous` shadow mode
- E0 failure scene 촬영
- 산/도시 target video 수집, CVAT annotation 시작

**Gate 2:** Day 2 종료까지 live E0 mask가 없으면 최종 시연은 recorded input으로
전환하고, 모델 fine-tuning과 ROS integration을 병렬 유지한다.

### Day 3 — Fine-tune and swap

- label QC와 sequence split 동결
- B0 short fine-tuning
- validation checkpoint 선택
- ONNX export와 parity
- 새 Jetson engine 생성·교체

### Day 4 — End-to-end rehearsal

- E0/new model 동일 고정 장면 A/B 평가
- wheels-off full pipeline
- 0.3 m/s 이하 직선 주행과 정지 반복
- p95 latency·정지거리·false stop 기록
- fallback 필요 여부 결정

### Day 5 — Freeze and film

- 코드·model·engine·threshold 동결
- 전체 trial을 보존하며 최종 영상 촬영
- live 실패에 대비한 recorded replay 영상 확보
- 결과표, 한계, 재현 명령과 artifact SHA 정리

## 13. Fallback Ladder

1. **Primary:** B0-E0 log 실패 → custom-log B0 성공, live RC stop
2. **Fallback A:** log improvement가 약하면 pole로 늦게 갈아타지 않는다. 가장 좋은
   custom checkpoint와 E0의 정량/정성 차이를 recorded input에서 보이고, live에서는
   안전한 고정 target으로 pipeline만 증명한다.
3. **Fallback B:** 신규 학습이 붕괴하면 기존 B0/B2 rubble 차이를 사용한다.
4. **Fallback C:** live camera가 불안정하면 고정 ZED SVO/rosbag replay로
   perception→control topic을 재현하고 RC는 wheels-off로 검증한다.
5. **최소 산출물:** E0 failure 영상, TensorRT 실시간 mask, ROS stop command,
   PWM watchdog 증거를 남긴다.

## 14. 안전 시험 순서

1. Jetson·ZED·TensorRT file inference
2. ZED live inference, actuator 미연결
3. `/drive/autonomous` topic echo, actuator 미연결
4. LiPo 분리·바퀴를 띄우고 PWM test
5. 게임패드 B stop, command timeout, process kill 시험
6. 물리 전원 차단 담당자 배치
7. 0.3 m/s 이하 직선 저속 시험
8. soft target을 사용한 자동 정지 시험

소프트웨어 E-stop은 물리적 LiPo/ESC 차단을 대체하지 않는다.

## 15. 발표 이후 10–15일 목표

### 모델·데이터

- pole/log/water 등 target 확장과 3-seed evaluation
- Semantic23 ontology와 source mapping audit
- RELLIS/RUGD/YCOR/Freiburg/GOOSE 통합 package의 라이선스·split·provenance 정리
- 공개 가능한 변환 script, mapping, manifest, dataset card 작성

ORATOR-ATLAS는 ontology와 변환 코드가 공개돼 있고 converted unified dataset은
요청 또는 직접 생성하는 형태다. ADOM은 “아무도 만들지 않았다”가 아니라
**재현 가능하고 공개 가능한 Semantic23 materialization과 검증 protocol을 제공한다**고
주장해야 한다.

### 배포·시스템

- Jetson runtime Dockerfile와 on-device engine cache
- depth `NEURAL_LIGHT` 재도입과 semantic-depth projection
- uncertainty·cost distribution 및 semantic costmap
- Nav2와 closed-loop obstacle avoidance
- 모델 export·TensorRT build·CVAT·W&B를 연결한 웹 UI
- ONNX/TensorRT regression과 target hardware benchmark 자동화

## 16. Decision Log

- **[2026-08-07] D-5 PoC에서 Semantic20 자율주행 기반 구축으로 전환**
  이유: 실제 자율주행 개발의 첫 단계로 canonical Semantic20 perception, 최신 frame
  scheduling 및 camera→action 지연 계측을 확립해야 한다. Cost4 주행 비용과는 명시적으로
  분리하며 세부 계약은 decision record 0008을 따른다.

- **[2026-08-06] 연구 중심에서 D-5 라이브 PoC 중심으로 전환**
  이유: 제한된 기간에 모델 novelty보다 RGB→인지→정지 end-to-end 신뢰성을 먼저
  증명해야 한다.
- **[2026-08-06] SegFormer-B0 E0를 baseline deployment 모델로 동결**
  이유: 학습·평가가 완료됐고 B2보다 Orin Nano 배포 위험이 낮다.
- **[2026-08-06] 기본 시연 target을 log로 선정**
  이유: B0/B2 모두 약 40 IoU로 데이터 병목이 명확하고, pole보다 수집·라벨·ROI
  검출·실차 재현이 쉽다. Pole은 stretch, rubble은 fallback으로 둔다.
- **[2026-08-06] target-only label + 나머지 ignore 255 승인**
  이유: Semantic20 contract를 유지하면서 3일 annotation 범위에 맞춘다.
- **[2026-08-06] D-5 autonomy를 직진 Go/Stop까지로 제한**
  이유: Nav2/localization/depth costmap은 현재 구현·검증 비용이 너무 크다.
- **[2026-08-06] TensorRT engine은 target Jetson에서 생성**
  이유: serialized engine은 TensorRT·플랫폼·GPU compatibility 영향을 받는다.
- **[2026-08-06] RGB-only, depth/VIO/point cloud 비활성화**
  이유: Orin Nano 8GB에서 ZED compute와 TensorRT의 GPU/RAM 경합을 줄인다.
- **[2026-08-06] 설치 보고는 JetPack 7.2이며 Day 1 package-level audit 후 동결**
  이유: Ubuntu 24.04/Jazzy는 JetPack 7.2와 일치하지만 이전 JetPack 6/CUDA 13.5
  표기가 있어 L4T·CUDA symlink·TensorRT package를 함께 검증해야 한다.
- **[2026-08-06] 640x384와 ROS topic은 제안 계약으로 두고 담당자 실측 후 동결**
  이유: 학습은 512x512 crop이고 ZED wrapper별 topic/QoS가 달라 문서 추정값을 실제
  인터페이스보다 우선하면 안 된다.
- **[2026-08-06] 문서 권위와 진행 기록을 분리**
  이유: 이 파일은 현재 계약만 유지하고, 실제 진행률은 `docs/status`, 결정 근거는
  decision record, AI 대화의 검토 흔적은 `docs/ai-collaboration`에 분리해야
  중복된 진실과 오래된 지시를 피할 수 있다.
- **[2026-08-07] 실차 데이터 수집 rosbag을 ZED RGB 토픽으로 제한**
  이유: D-5의 신규 데이터 목적은 RGB 기반 인지 학습이며, GNSS와 제어 입력까지
  동기 수집하는 범위는 현재 PoC에 과하다. `data_collection.launch.py`는 GNSS를
  시작하지 않고 recorder는 ZED의 `/rgb` 하위 토픽만 기록한다. capture 경로는
  저장소 기준 상대경로를 기본으로 한다. 상세 근거는 decision record 0007을 따른다.

## 17. Primary References

- NVIDIA JetPack 6.2.2: <https://developer.nvidia.com/embedded/jetpack-sdk-622>
- NVIDIA JetPack 7.2 release matrix: <https://developer.nvidia.com/embedded/jetpack/downloads>
- TensorRT engine compatibility:
  <https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/engine-compatibility.html>
- ZED SDK 5.4 downloads: <https://www.stereolabs.com/developers/release>
- ZED ROS 2 wrapper: <https://github.com/stereolabs/zed-ros2-wrapper>
- MMSeg/MMDeploy deployment:
  <https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/5_deployment.md>
