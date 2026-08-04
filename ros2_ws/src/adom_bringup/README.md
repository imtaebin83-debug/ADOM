# adom_bringup

기본 실행은 planning을 의도적으로 비활성화하고 control을 dry-run으로 시작한다.

```bash
ros2 launch adom_bringup vehicle.launch.py
```

센서/TF/localization 검증 후에만 `start_planning:=true`를 사용한다.

