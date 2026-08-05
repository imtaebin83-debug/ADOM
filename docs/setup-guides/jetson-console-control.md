# Jetson console control setup

This setup follows the F1TENTH command boundary (`/drive`,
`ackermann_msgs/AckermannDriveStamped`) but replaces its VESC output path with
the PCA9685 required by this vehicle.

## Assumptions

- Jetson runs JetPack 7.2 based on Ubuntu 24.04 and ROS 2 Jazzy.
- PCA9685 logic `VCC` is connected to Jetson 3.3 V. Do not connect 6 V to VCC.
- XL-5 BEC powers only the PCA9685 `V+` servo rail.
- Jetson, PCA9685, ESC, and servo have a common ground.
- The vehicle has a physical, reachable LiPo/ESC disconnect. Software E-stop
  is not a substitute for it.

## Remote installation

From the operator computer (replace the host):

```bash
ssh -t jetson@JETSON_IP
```

On the Jetson:

```bash
git clone https://github.com/imtaebin83-debug/ADOM.git ~/ADOM
cd ~/ADOM
chmod +x scripts/install_jetson_control.sh
./scripts/install_jetson_control.sh
exit
```

Reconnect after group membership is changed:

```bash
ssh -t jetson@JETSON_IP
source /opt/ros/jazzy/setup.bash
source ~/ADOM/ros2_ws/install/setup.bash
i2cdetect -l
```

Find the header I2C bus from that list, then scan that explicit bus number. For
example, if it is bus 7:

```bash
i2cdetect -y -r 7
```

Address `40` must appear. Do not scan guessed buses while other sensitive I2C
devices are connected.

## Hardware-output test

Keep the LiPo disconnected and lift the wheels before starting. The control node
always initializes the PCA9685 and writes PWM; there is no software-only mode.
Terminal 1 on the Jetson:

```bash
source /opt/ros/jazzy/setup.bash
source ~/ADOM/ros2_ws/install/setup.bash
ros2 launch adom_control control.launch.py
```

Terminal 2 (a second `ssh -t` session):

```bash
source /opt/ros/jazzy/setup.bash
source ~/ADOM/ros2_ws/install/setup.bash
ros2 run adom_control keyboard_teleop --ros-args \
  --params-file ~/ADOM/ros2_ws/src/adom_control/config/vehicle.yaml
```

Terminal 3 can inspect the PWM values being written to the hardware:

```bash
source /opt/ros/jazzy/setup.bash
source ~/ADOM/ros2_ws/install/setup.bash
ros2 topic echo /adom/control/pwm_us
```

Keys are `w/s` for throttle, `a/d` for steering, `c` to center, Space or `x`
for E-stop, `r` to release E-stop, and `q` to E-stop and quit. Throttle returns
to neutral unless `w` or `s` is pressed repeatedly. The output node also has a
0.25 s command watchdog, so loss of SSH or the teleop process commands neutral.

## Hardware calibration

Lift all driven wheels clear of the ground and arrange a second person at the
physical disconnect. Confirm steering first with the ESC signal disconnected.
Edit `ros2_ws/src/adom_control/config/vehicle.yaml` and determine:

1. `steering_center_us`, then conservative left/right limits that do not bind.
2. The XL-5 neutral/arming pulse required by its configured mode.
3. The smallest forward pulse that turns the wheels.

The supplied 1300--1700 us steering and 1500--1600 us throttle numbers are
conservative placeholders, not calibration data. Reverse is disabled because
XL-5 reverse behavior may require a brake-neutral-reverse sequence.

After every configuration edit, rebuild (the package installs a copy of YAML):

```bash
cd ~/ADOM/ros2_ws
colcon build --symlink-install --packages-select adom_control
source install/setup.bash
```

Only after the E-stop and watchdog pass, reconnect the signal and start the
control node before powering the ESC. If `board.I2C()` cannot open the bus,
verify Jetson header pinmux with Jetson-IO rather than hard-coding a different
Linux bus in the node.

## Why the full F1TENTH repository is not installed

`f1tenth_system` currently targets the F1TENTH VESC, joystick, and Hokuyo stack.
This car has an XL-5 PWM ESC and no VESC telemetry/odometry. Installing its VESC
driver would therefore add the wrong actuator path. The compatible parts used
here are the Ackermann `/drive` API, command timeout, and deadman behavior.
