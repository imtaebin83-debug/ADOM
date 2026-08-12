# adom_description

차체와 센서의 TF를 정의한다. `wheelbase`, 차체 크기, ZED/GNSS 장착 위치는 실측 후
반드시 수정한다. ZED Wrapper가 자체 카메라 URDF를 발행할 때 프레임 이름이 충돌하지
않는지 `ros2 run tf2_tools view_frames`로 확인한다.

ZED optical center의 지면 기준 높이는 2026-08-12 정밀 재측정한 0.21 m를 기본값으로
반영한다. `base_link -> zed_camera_link` TF의 pitch는 별도로 검증한다. Costmap의 높이
필터는 optical-frame Y를 직접 해석하지 않고 이 TF로 변환된 `base_link` Z를 사용한다.
