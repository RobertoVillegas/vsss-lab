## 1. Runtime and assets

- [x] 1.1 Add pinned ROS 2 Lyrical/Gazebo Jetty container
- [x] 1.2 Migrate field, six robots, ball, wheels, and top-camera geometry to SDF
- [x] 1.3 Add headless world parse/step smoke

## 2. Canonical bridge

- [x] 2.1 Define backend-neutral reset/step process contract
- [x] 2.2 Implement native and subprocess adapters with validation
- [x] 2.3 Emit canonical replay-compatible state only

## 3. Sim-to-sim gate

- [x] 3.1 Run one unchanged policy through native, bridged, and Gazebo adapters
- [x] 3.2 Record runtime versions, evidence, limitations, and rollback
- [x] 3.3 Run full gates, archive OpenSpec, sign commit, and push
