# M12 physical-camera CPU evidence

Date: 2026-07-28

## Transport path

The Gazebo world now loads `gz::sim::systems::Sensors` and publishes the
680×520 RGB8 top camera on `/vsss/camera/image_raw`. The container smoke:

1. receives a native `gz.msgs.Image`;
2. starts `ros_gz_image image_bridge`;
3. observes `sensor_msgs/msg/Image` on the same ROS topic; and
4. checks the bridged image width is 680 pixels.

This follows the current Gazebo
[Sensors system](https://github.com/gazebosim/docs/blob/master/rotary/sensors.md)
and
[ROS image bridge](https://github.com/gazebosim/docs/blob/master/common/migrating_gazebo_classic_ros2_packages.md)
guidance. `decode_ros_image` uses structural typing, so importing the core
estimator never requires a ROS installation.

## Recorded frame

`just ros-gazebo-smoke` captured a real overhead frame with SHA-256
`2a523f8ae32c472924432adc7109d3ecc1c648333268dabbe5611850d1107485`.
The decoded image was 680×520 RGB8 and the CPU pipeline detected the orange
ball at `(-0.001553, +0.001543)` metres versus its `(0, 0)` initial truth:
2.19 mm radial error. The measurement was accepted by the ball Kalman filter.

One warm pass on this host measured:

| Stage | Latency |
|---|---:|
| Gazebo text-fixture decode | 23.95 ms |
| NumPy RGB segmentation | 1.97–2.14 ms |
| Ball association | <0.001 ms |
| Kalman update | 0.36 ms |

The 23.95 ms decode is specific to Gazebo's escaped text recording. Live ROS
uses the packed byte buffer directly. Segmentation plus filtering is below
2.6 ms and is not a measured bottleneck at the 30 Hz camera rate, so M12 does
not add a CUDA vision implementation. CPU/CUDA agreement is therefore not
applicable; CUDA remains dedicated to batched policy optimization.

## Known limitation

The captured ball is identifiable, but the six M10 Gazebo robots have identical
gray visuals and no top markers. Robot segmentation and association cannot be
validated honestly from this frame. M12 keeps the transport, measurement
contract, robot EKF, and association inputs separate; a marked-robot Gazebo
fixture is still required before closing the full stage profile and physical
accuracy thresholds.

## Reproduction

```text
just ros-gazebo-smoke
.venv/bin/pytest -q tests/test_camera_image.py tests/test_camera_bridge.py
```

The container exited successfully and `docker compose ps --all` showed no
remaining containers.
