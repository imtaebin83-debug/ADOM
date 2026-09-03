# adom_description

차체와 센서의 TF를 정의한다. `wheelbase`, 차체 크기, ZED/GNSS 장착 위치는 실측 후
반드시 수정한다. ZED Wrapper가 자체 카메라 URDF를 발행할 때 프레임 이름이 충돌하지
않는지 `ros2 run tf2_tools view_frames`로 확인한다.

ZED optical center의 지면 기준 높이는 2026-08-12 정밀 재측정한 0.21 m를 기본값으로
반영한다. `base_link -> zed_camera_link` TF의 pitch는 별도로 검증한다. Costmap의 높이
필터는 optical-frame Y를 직접 해석하지 않고 이 TF로 변환된 `base_link` Z를 사용한다.

## 구조

```text
adom_description/
├── launch/     # description.launch.py (robot_state_publisher)
├── meshes/     # 차체 CAD 메시
└── urdf/       # adom_vehicle.urdf.xacro
```

## Meshes

`meshes/middle_chassis.stl` — 1/10 RC 플랫폼의 미들 섀시 CAD 메시다.

| 항목 | 값 |
| --- | --- |
| 출처 | CATIA STL export |
| 형식 | binary STL (1.2 MB) |
| 삼각형 수 | 24,318 |
| 단위 | **밀리미터** |
| Bounding box | X `-95 .. 155`, Y `-95 .. 95`, Z `2 .. 115` (250 x 190 x 113 mm) |

원본은 6.8 MB ASCII STL이었고 형상 변화 없이 binary로 변환했다. facet 수와 bounding box가
동일하며 정점 최대 편차는 `7e-06 mm`(float32 반올림)다. `.gitattributes`에서 `*.stl binary`로
지정해 EOL 변환이 적용되지 않도록 막는다.

> [!IMPORTANT]
> **현재 URDF는 이 메시를 사용하지 않는다.** `base_link`의 visual과 collision은 여전히
> `box` primitive(`body_length` x `body_width` x `body_height`)다. 메시는 형상 참조용으로만
> 저장돼 있다.

URDF에 연결할 때는 두 가지를 반드시 처리한다.

1. **단위 변환.** STL이 mm이고 URDF는 m이므로 `scale="0.001 0.001 0.001"`이 필요하다.
   빠뜨리면 차량이 1000배 크기로 렌더링된다.
2. **원점 정렬.** CAD 원점이 `base_link` 원점과 일치하지 않는다. bounding box가 X축으로
   `-95 .. 155`로 비대칭이므로 `<origin>` 오프셋을 실측해 맞춰야 한다.

```xml
<visual>
  <origin xyz="0 0 0" rpy="0 0 0"/>
  <geometry>
    <mesh filename="package://adom_description/meshes/middle_chassis.stl"
          scale="0.001 0.001 0.001"/>
  </geometry>
</visual>
```

메시를 시각화용으로 쓰더라도 **collision geometry는 primitive를 유지한다.** 24k 삼각형
메시를 collision에 쓰면 planner와 costmap의 부하가 불필요하게 커진다.

`meshes/`는 `CMakeLists.txt`에서 `share/adom_description`으로 설치되므로 빌드 후
`package://adom_description/meshes/...` 경로로 참조할 수 있다.
