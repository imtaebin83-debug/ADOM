# adom_description

차체와 센서의 TF를 정의한다. `wheelbase`, 차체 크기, ZED/GNSS 장착 위치는 실측 후
반드시 수정한다. ZED Wrapper가 자체 카메라 URDF를 발행할 때 프레임 이름이 충돌하지
않는지 `ros2 run tf2_tools view_frames`로 확인한다.

