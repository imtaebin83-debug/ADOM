# D-5 RC Car Live Stop PoC Status

- State: Active
- Updated: 2026-08-07 KST
- Coordinator: 태빈
- Source of Truth: [ADOM_CONTEXT.md](../../ADOM_CONTEXT.md)
- Decision: [0006 D-5 live stop PoC pivot](../decision-records/0006-d5-poc-pivot.md)
- Target decision: [0008 Defer target selection](../decision-records/0008-defer-target-selection.md)

## Current gate

**Gate 1:** Jetson에서 B0-E0 file→TensorRT mask와 control hardware가 각각 독립적으로
동작해야 한다. 둘 중 하나가 실패하면 신규 기능을 늘리지 않고 blocker를 먼저 해결한다.

## Progress snapshot

| Workstream | Owner | State | Evidence | Next action |
| --- | --- | --- | --- | --- |
| B0/B2 E0 training | 태빈 | Complete | canonical test와 class table 확보 | B0 checkpoint/export 입력 동결 |
| RC Car hardware assembly | 명섭 | Complete | Jetson, ZED 2i, PWM, battery 장착 보고 | wheels-off neutral/watchdog 실측 |
| Jetson software stack | 가형 | Active | JetPack 7.2, CUDA 13.2.78, TRT 10.16.2 실측 | ZED/ROS audit 결과 기록 |
| B0 ONNX hand-off | 태빈 | Complete | 12장 CPU parity 100%, max error 0.0001035, package SHA 검증 | engine parity 지원 |
| TensorRT FP16 engine | 가형 | Active | target Jetson `trtexec` build PASS | engine SHA 후 file inference/parity |
| ZED live perception ROS | 가형·명섭 | Planned | 없음 | 실제 image topic/QoS 확인 후 live mask |
| Go/Stop safety reflex | 명섭 | Planned | control node 기반 존재 보고 | shadow mode→wheels-off 순서 검증 |
| target discovery | 태빈·명섭 | Active | E0 class table과 target 미동결 결정 | 19-class overlay/per-class ROI로 후보 장면 확인 |
| custom target dataset | 태빈·용준 | Blocked on target | CVAT Docker 설치, project 미생성 | target 동결 후 pilot 20 masks와 QC |
| short fine-tuning | 태빈 | Planned | 없음 | label gate 뒤 B0-E0에서 시작 |
| final rehearsal/video | 전원 | Planned | 없음 | 고정 장면 A/B와 replay fallback 확보 |

## Immediate handoffs

| From | To | Artifact/contract | State | Acceptance evidence |
| --- | --- | --- | --- | --- |
| 태빈 | 가형 | ONNX, resolved config, labels/palette, preprocess, reference I/O, SHA | Active | PyTorch↔ONNX argmax parity와 파일 checksum |
| 명섭 | 가형 | 실제 ZED image topic/type/QoS/frame rate | Planned | `ros2 topic info --verbose`, `ros2 topic hz` 결과 |
| 가형 | 명섭 | semantic mask, target ratio/detected topic 계약 | Planned | rosbag/replay에서 timestamp와 drop 정책 확인 |
| 태빈·용준 | 태빈 | selected-target masks와 sequence split | Blocked on target | 선택 ID/255, pair/shape/leakage QC 통과 |
| 명섭 | 전원 | control safety evidence | Planned | neutral, watchdog, E-stop, process-kill 결과 |

## Blockers and risks

| Severity | Risk | Current evidence | Owner | Unblock condition |
| --- | --- | --- | --- | --- |
| P0 | JetPack/CUDA/TRT 조합 불일치 가능성 | JetPack 7.2/CUDA 13.2/TRT 10.16 실측 | 가형 | L4T package audit 기록 |
| P0 | ONNX preprocessing과 Jetson preprocessing 불일치 | MMSeg right/bottom padding 동결, ONNX parity 통과 | 태빈·가형 | Jetson 구현과 reference tensor 비교 |
| P0 | ROS topic/QoS를 문서가 잘못 단정할 위험 | 제안 topic만 존재 | 가형·명섭 | 실제 graph와 callback 측정 후 문서 갱신 |
| P1 | target을 현장 evidence 없이 고를 위험 | canonical split의 class support가 불균형 | 태빈 | 전체 overlay와 실패·시나리오·재현성 근거로 동결 |
| P1 | 얇은 target은 resize 후 소실될 수 있음 | pole 등 작은 구조 후보 존재 | 태빈 | 20-frame preview로 보존 여부 판정 |

## Next 24 hours

- 태빈: B0-E0 checkpoint와 입력 후보의 export/parity, 전체 class overlay를 hand-off
- 가형: Jetson stack audit, `trtexec` FP16 build, reference file inference
- 명섭: ZED USB 3/RGB topic 확인, depth 계열 비활성화, wheels-off PWM/watchdog 시험
- 용준: annotation guide 확인과 pilot mask QC, 발표 전/후 비교 틀 준비

## Update log

- 2026-08-06 — D-5 PoC를 P0로 등록하고 알려진 완료 상태와 미검증 항목을 분리함.
- 2026-08-06 — 640x384, JetPack stack, ROS topic을 담당자 실측 전까지 Unverified로 유지함.
- 2026-08-07 — target을 미동결로 전환하고 Semantic20 전체 현장 시각화 뒤 선택하도록 변경함.
- 2026-08-07 — B0 E0 raw-logits ONNX opset 13을 12장 CPU parity로 검증하고 hand-off함.
- 2026-08-07 — Jetson TensorRT 10.16.2에서 FP16 engine build가 통과함. engine parity와 latency는 전원 복구 후 수행함.
