# 0046. Retire the Cost4 ROS perception path

- Status: Accepted
- Date: 2026-09-07
- Owners: autonomy integration 담당자
- Supersedes: none; retires the ROS-side surface introduced before 0033

## Context

0033이 Semantic20을 인지 기반으로 확정한 뒤에도 `ros2_ws/`에는 Cost4 시절의 실행
경로가 그대로 남아 있었다. 활성 자율주행 스택인 `low_level_autonomy.launch.py`는
`perception.launch.py` → `semantic20_local_planning.launch.py` →
`semantic20_costmap.launch.py`만 호출하며, Cost4 쪽 진입점은 어디에서도 참조되지
않았다.

저장소를 연구 공개용으로 정리하는 시점에서, 실행되지 않는 두 번째 인지 경로는
독자가 어느 쪽이 실제 시스템인지 판단하기 어렵게 만든다.

## Decision

ROS 쪽 Cost4 경로만 제거한다.

- `adom_perception_ros`: `scripts/perception_cost4_node.py`,
  `launch/perception_cost4.launch.py`, `config/perception.yaml`
- `adom_costmap_ros`: `launch/semantic_costmap.launch.py`,
  `config/semantic_costs.yaml`
- `adom_bringup`: `launch/rule_autonomy.launch.py`, `config/rule_autonomy.rviz`
- `docs/setup-guides/rule-autonomy-demo.md`

`adom_bringup`의 `config/`가 비므로 해당 `CMakeLists.txt`의
`install(DIRECTORY config launch ...)`에서 `config`를 제거한다.
`adom_perception_ros`의 `adom_cost4_perception_node` install 항목도 제거한다.

학습 쪽 Cost4는 **제거하지 않는다.** `src/adom/data/`, `src/adom/mmseg/`,
`src/adom/evaluation.py`, `src/adom/runtime/cycle.py`, `configs/adom/`,
`src/data/cost_4/`와 테스트 7개에 걸쳐 있어 파일 삭제가 아니라 대규모 리팩터가 되며,
Cost4는 Semantic20 이전 baseline의 재현 근거로 논문에 남는다.

## Rationale and evidence

`adom_bringup/launch/*.py`의 include 그래프를 전수 확인한 결과, Cost4 진입점은
`rule_autonomy.launch.py` 하나이며 이 파일 자체가 Cost4 데모 전용이다.
`rule_autonomy.rviz`도 이 launch와 삭제한 데모 문서에서만 참조된다.

`semantic_costmap_node.py`는 Semantic20 경로가 공유하므로 **삭제하지 않는다.**
`rule_planning.launch.py`와 `rule_planner` 노드도 Semantic20 스택이 계속 사용한다.

## Alternatives considered

- **`semantic_costmap_node.py`의 `cost4` ontology 분기까지 제거**: 이 노드는 활성
  자율주행 경로에 있고, 기본값(`ontology: "cost4"`)을 바꾸는 것은 실차 동작 변경이다.
  차량 없이 검증할 수 없어 이번 범위에서 제외한다. 아래 후속 작업 참조.
- **학습 쪽 Cost4 동시 제거**: 위에 적은 이유로 기각.

## Consequences

- ROS 인지 경로가 Semantic20 하나로 단일화되어, 독자가 실제 시스템을 오해할 여지가 없다.
- `rule_autonomy.launch.py`로 실행하던 Cost4 데모는 더 이상 재현할 수 없다. 해당
  절차가 필요하면 이 커밋 이전 이력에서 복원한다.
- `tests/test_perception_runtime.py`의 Cost4/Semantic20 config 분리 테스트는
  "Semantic20 config가 유일한 인지 파라미터 집합"을 단언하도록 다시 썼다.

## Follow-up

`semantic_costmap_node.py`의 `ontology` 기본값은 여전히 `"cost4"`다. Cost4 config를
모두 지운 지금, 파라미터 파일 없이 이 노드를 띄우면 도달 불가능한 기본값으로 떨어진다.
`semantic20_costmap.launch.py`는 항상 파라미터 파일을 넘기므로 현재 동작에는 영향이
없으나, 다음 실차 검증 기회에 기본값을 `"semantic20"`으로 바꾸는 것을 권한다.

## Validation and rollback

- 삭제한 파일을 참조하는 잔여 문자열이 저장소에 없는지 전수 확인한다
  (결정 기록은 불변이므로 제외).
- `python -m unittest discover -s tests` 전체 통과를 확인한다.
- colcon 빌드는 Jetson에서 검증한다. 이 변경은 install 대상 축소와 launch 삭제뿐이며
  활성 스택의 노드·설정·토픽은 바뀌지 않는다.
- 롤백은 이 커밋 하나를 revert하면 된다.
