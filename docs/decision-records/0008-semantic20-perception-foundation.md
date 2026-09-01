# 0008. Semantic20 autonomous perception foundation

- Status: Accepted
- Date: 2026-08-07
- Owners: ADOM team
- Supersedes: 0006 as the current project scope; 0006 remains the historical D-5 record

## Context

D-5 Go/Stop PoC 이후 실제 자율주행 프로젝트를 시작한다. 기존 `ros2_ws`의 perception과
semantic costmap은 Cost4 ID를 전제로 했지만, 현재 모델·데이터의 canonical ontology는
RELLIS + RUGD + YCOR + 신규 라벨 데이터로 구성한 Semantic20이다. 추론이 카메라보다
느릴 때 DDS queue의 과거 frame을 순서대로 처리하면 액션이 오래된 장면을 기반으로
결정되는 문제가 있다. 기존 status의 `latency_ms`는 model callback 처리시간만 나타내어
카메라 입력에서 액션까지의 지연을 답하지 못한다.

## Decision

자율주행 기반의 첫 increment는 Semantic20 perception으로 한다. canonical
`src/data/semantic_20/config/bridge_mapping.yaml`의 ID `0..18`과 ignore `255`를 runtime에서
검증하고, Cost4와 구분되는 `/adom/perception/semantic20_mask`를 발행한다. subscriber
callback과 inference worker를 분리하고 한 칸짜리 latest-frame mailbox를 사용한다.
추론 시작률 상한은 30 FPS로 둔다.

카메라 timestamp에서 perception 출력까지의 queue/model/end-to-end 시간을 구조화해
발행한다. downstream이 원본 timestamp를 보존하고, planner가 해당 costmap에서 처음
발행하는 `/cmd_vel` 시점에 camera→software-action latency와 rolling p50/p95를 별도
topic으로 발행한다.

Semantic20→주행 비용 매핑은 이 결정에 포함하지 않는다. 기존 Cost4 costmap에
Semantic20 mask를 자동 연결하거나 클래스 ID를 암묵 변환하지 않는다. watchdog,
timeout neutral, 수동 reset, 저속 단계 검증은 자율주행 개발에서도 유지한다.

## Rationale and evidence

한 칸짜리 mailbox는 처리 중 도착한 frame을 누적하지 않고 가장 최근 frame으로
덮어쓰므로 scene age를 제한한다. sensor timestamp를 mask와 costmap에 보존하면 단순
model 시간과 실제 software decision 지연을 구분할 수 있다. ontology별 topic과 config를
분리하면 같은 숫자 ID가 Cost4와 Semantic20에서 전혀 다른 의미를 갖는 사고를 막는다.

## Alternatives considered

- ROS subscription depth만 1로 설정: middleware backlog는 줄지만 긴 callback 안에서
  수신·교체 시점을 명시적으로 관찰하기 어렵다.
- 모든 frame을 FIFO 처리: frame 손실은 적지만 자율주행 액션의 scene age가 계속 늘 수 있다.
- Cost4 costmap에 Semantic20 ID를 바로 입력: ontology 의미가 달라 안전하지 않다.

## Consequences

입력 FPS가 처리량보다 높으면 `overwritten_frames`가 증가하는 것이 정상이다. 30 FPS는
목표 시작률의 상한이며 hardware가 30 FPS를 보장한다는 실측 결과가 아니다. camera와
ROS clock domain이 다르거나 header stamp가 0이면 end-to-end 값은 무효다. software
action latency는 `/cmd_vel` publish까지이며 물리 PWM/조향/구동 응답은 GPIO 또는 외부
계측으로 별도 측정해야 한다. Semantic20 cost 정책이 승인되기 전까지 기존 Cost4
costmap/planner와 end-to-end autonomous drive를 연결하지 않는다.

## Validation and rollback

unit test에서 canonical mapping, 출력 ID 검증, latest-frame overwrite를 확인한다.
target Jetson에서는 `ros2 topic hz`, status JSON, action latency p50/p95와 camera clock을
검증한다. rollback은 Semantic20 topic을 중단하는 방식으로 하며 Cost4 config/artifact를
Semantic20 checkpoint에 대체 사용하지 않는다.
