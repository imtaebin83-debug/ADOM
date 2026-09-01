# 0043. Autonomous 0.10..1.00 m/s speed profile

- Status: Accepted
- Date: 2026-09-01
- Owners: ADOM control integration
- Supersedes: speed-profile portions of 0029

## Context

0029는 autonomous command 범위를 0.30..3.00 m/s로 고정하고, YAML과 ROS node fallback,
reusable planner/controller defaults가 모두 같은 값을 쓰도록 요구했다. 이후 jetson
브랜치의 `b2b3a4c3604dfd7faede4b68f0f10e01041ed1f7`("속도조절")가 현장 조정으로 배포
YAML만 0.10..1.00 m/s로 낮추고 나머지 계층과 0029를 갱신하지 않았다.

두 CI workflow는 `main` push와 PR에서만 실행되므로 jetson 브랜치에 직접 쌓인 62개
커밋은 검증되지 않았고, 이 불일치는 jetson을 main에 병합한 뒤에야
`test_ros_speed_limits_are_consistent_across_pipeline` 실패로 드러났다. 0042는 이
불일치를 숨기지 않고 autonomy 담당자의 별도 결정으로 남겨뒀다. 이 record가 그
결정이다.

## Decision

배포 YAML의 0.10..1.00 m/s를 현재 승인된 autonomous command 범위로 확정하고, 0029가
요구한 "모든 계층이 같은 값을 쓴다" 불변식을 이 값 기준으로 복원한다.

- `ros2_ws/src/adom_planning/config/rule_planner.yaml` (변경 없음, 이미 0.10/1.00)
- `ros2_ws/src/adom_control/config/local_path_control.yaml` (변경 없음, 이미 0.10/1.00)
- `ros2_ws/src/adom_planning/scripts/rule_planner.py`의 node fallback
- `ros2_ws/src/adom_control/adom_control/local_path_control.py`의 node fallback
- `src/adom/autonomy/rule_planner.py`의 `PlannerConfig` defaults
- `src/adom/autonomy/path_control.py`의 `PathControlConfig` defaults
- `tests/test_autonomy.py`의 pipeline 일관성 기대값

STOP, command watchdog, 수동 reset, manual gamepad ceiling과 PCA9685의 nominal
12.0 m/s mapping은 변경하지 않는다. 안전 판정으로 정지할 때의 0 명령은 최소 속도
적용 대상이 아니다.

또한 재발 방지를 위해 `code-smoke.yml`이 `jetson` 브랜치 push에서도 실행되도록 한다.

## Rationale and evidence

`b2b3a4c`는 이 profile에 대한 가장 최근의 의도적 현장 조정이고 이후 되돌린 적이 없다.
0018에서 측정한 약 0.33--0.43초 camera source-to-command 지연은 여전히 해소됐다고
검증되지 않았으므로, 낮은 command profile이 현재 확보된 latency evidence와도 부합한다.

node fallback을 0.30/3.00으로 남겨두면 파라미터가 누락된 degraded 실행에서 배포 값의
3배 속도를 명령하게 된다. YAML이 정상 로드되는 경로에서는 노드가 항상 명시적 값을
넘기므로 fallback 정렬은 차량 동작을 바꾸지 않고 degraded 경로만 안전해진다.

`PlannerConfig`/`PathControlConfig` defaults는 ROS 노드가 사용하지 않고 offline 도구와
테스트에서만 쓰이지만, 0029의 불변식 대상이므로 같이 맞춘다. 정렬 후 저장소 전체
193개 테스트가 통과한다.

## Alternatives considered

- YAML을 0.30..3.00으로 복원: 0029 문면에는 맞지만 실차 최고 속도가 3배가 되고 현장
  조정 의도를 되돌린다. wheels-off와 폐쇄 공간 재검증 없이는 채택할 수 없다.
- 테스트 기대값만 수정: CI는 녹색이 되지만 node fallback의 3배 속도 위험이 남는다.
- 테스트에서 절대값 단정을 제거하고 계층 간 일관성만 검사: 값이 다시 조용히 바뀌어도
  감지하지 못한다.

## Consequences

- 자율주행 양수 주행 명령은 0.10..1.00 m/s로 제한된다. BLOCKED, timeout, E-stop과
  mode 전환은 계속 0/neutral을 명령한다.
- 파라미터 누락 시 node fallback도 같은 범위를 사용한다.
- 0.30..3.00 profile을 다시 쓰려면 이 record를 supersede하는 새 record가 필요하다.
- jetson 브랜치 push가 이제 repository-wide smoke를 거치므로, 같은 종류의 drift가
  main 병합 전에 드러난다.

## Validation and rollback

`test_ros_speed_limits_are_consistent_across_pipeline`이 planner YAML, local controller
YAML, vehicle YAML의 속도·조향·downstream ceiling 일관성을 검사한다. 저장소 전체
smoke(`python -m unittest discover -s tests`)로 회귀를 확인한다.

실차에서 이 profile이 부적절하다고 판정되면 위 7개 지점을 함께 되돌리고 그 결정을 새
record로 남긴다. 배포 YAML만 바꾸고 나머지를 두는 방식은 이 불일치를 재발시키므로
사용하지 않는다.
