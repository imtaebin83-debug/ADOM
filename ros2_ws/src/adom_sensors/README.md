# adom_sensors

ZED 2i와 RTK GNSS 드라이버의 ADOM용 launch/config adapter다. GNSS 드라이버는
반드시 `/fix`에 `sensor_msgs/msg/NavSatFix`를 발행해야 한다.

- NMEA 수신기: `nmea_navsat_driver`
- UBX 수신기: 호환 UBX 드라이버로 교체
- RTK correction: 수신기 내부 또는 별도 NTRIP client가 처리

`zed2i.yaml`의 파라미터 이름은 설치한 ZED Wrapper 버전에서 `ros2 launch ... -s`로
검증한다. ZED GNSS fusion과 `adom_localization`을 동시에 활성화하지 않는다.

