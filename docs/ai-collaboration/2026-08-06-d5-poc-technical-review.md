# 2026-08-06 D-5 PoC Technical Review

- Participants: ADOM 팀장, AI technical reviewer
- Status: Promoted
- Source of Truth: [ADOM_CONTEXT.md](../../ADOM_CONTEXT.md)
- Decision: [0006 D-5 live stop PoC pivot](../decision-records/0006-d5-poc-pivot.md)
- Status page: [D-5 PoC](../status/d5-poc.md)

## Question and context

남은 5일 동안 연구 novelty보다 밀리테크 발표용 end-to-end RC Car PoC를 우선할 때,
fine-tuning, Jetson 배포, ZED/ROS 부하, target class와 팀 hand-off를 어떻게 제한해야
실패 확률을 낮출 수 있는지 검토했다.

## Verified facts and evidence

- SegFormer-B0/B2 E0 학습은 완료됐고 B0 canonical test는 aAcc 89.78,
  raw mIoU 43.35, mAcc 67.22다.
- pole IoU는 B0/B2 모두 0이며, log IoU는 40.33/40.17로 모델 용량 증가의 이득이 없다.
- rubble은 B0 53.34에서 B2 66.87로 증가했지만 단일 seed·동일 latency 조건의
  배포 비교는 아직 아니다.
- 저장소 학습 전처리는 512x512 crop이고 640x384는 export 후보이므로 동일한
  전처리라고 가정할 수 없다.
- NVIDIA 공식 JetPack 7.2 release 조합은 L4T 39.2, Ubuntu 24.04,
  CUDA 13.2.1, TensorRT 10.16.2다. 설치 장비의 실제 package는 아직 미검증이다.
- TensorRT serialized engine은 platform/version compatibility 영향을 받으므로
  실제 target Jetson에서 생성·검증하는 것이 hand-off 기준이다.

공식 근거는 Source of Truth의 Primary References에 링크돼 있다.

## Critical review

- LoRA는 MMSegmentation을 버려야 하는 기술은 아니지만 adapter 주입, optimizer,
  export/merge와 회귀 검증 비용이 생긴다. D-5에는 기존 two-stage 또는 짧은 full
  fine-tuning이 더 낮은 위험이다.
- 전체 화면 5% threshold는 작은 장애물과 원근 변화에 취약하다. 하단 중앙 safety
  corridor, connected component, temporal debounce가 시연 조건에 더 적합하다.
- depth, point cloud, VIO를 동시에 켜면 Orin Nano 8GB에서 TensorRT와 GPU/RAM/DDS
  자원을 경쟁한다. 정지 PoC는 RGB-only로 시작해야 한다.
- water는 GT 부재와 RGB-only 물성 모호성 때문에 5일 실험의 주 target으로 부적합하다.
- pole은 강한 실패 사례지만 thin-object downsampling 때문에 유일한 success path로
  선택하면 위험하다. log를 기본으로 하고 실제 20-frame failure check로 확정해야 한다.

## Decisions promoted

- B0-E0를 배포 baseline으로 사용하고 target Jetson에서 FP16 engine을 만든다.
- 기본 target은 log, pole은 stretch, rubble은 hard fallback으로 둔다.
- target-only ID 10과 ignore 255 annotation을 RELLIS anchor와 함께 학습한다.
- D-5 성공을 Go/Stop까지로 제한하고 depth/Nav2/INT8/LoRA/web은 non-goal로 둔다.
- 640x384 input과 ROS topic은 담당자 parity/graph 실측 전 확정하지 않는다.

## Unverified proposals

- 640x360 resize 후 640x384 padding이 512x512/direct-resize보다 최선인지 여부
- ZED wrapper의 실제 RGB topic, QoS, frame rate
- 설치된 JetPack/CUDA/TensorRT/ZED SDK의 package-level 일관성
- log ROI 1%, 3-frame debounce, 0.3 m/s가 실제 stopping distance에 충분한지 여부
- 자체 log 데이터가 held-out Recall과 false-stop을 실제로 개선하는지 여부

## Actions

| Owner | Action | Due/gate | Evidence expected |
| --- | --- | --- | --- |
| 태빈 | 세 input 후보의 target 보존/parity 비교 | Gate 1 | tensor dump, argmax agreement, export report |
| 가형 | Jetson stack audit와 target engine build | Gate 1 | package versions, `trtexec`, latency log |
| 명섭 | 실제 ZED/ROS/control graph와 safety path 검증 | Gate 1–2 | topic/QoS/hz, watchdog, wheels-off video |
| 태빈·용준 | pilot target label과 split/QC | Gate 2 | mask overlay, ID/shape/leakage report |

## Risks and limitations

- 소규모 단일 환경 fine-tuning은 오프로드 일반화나 군용 작전 성능을 증명하지 않는다.
- closed-loop 정지 성공은 semantic 정확도뿐 아니라 threshold와 연출 장면의 영향을 받는다.
- 따라서 발표에는 모델 성능, 시스템 성공률, latency, false-stop과 제한사항을 분리해 제시한다.
