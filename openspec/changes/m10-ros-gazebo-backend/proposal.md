## Why

Fast simulation needs an independent, higher-fidelity integration target without
leaking ROS or Gazebo types into policies, training, or canonical state.

## What Changes

- Add a pinned ROS 2 Lyrical/Gazebo Jetty headless container.
- Migrate canonical field, robot, wheel, ball, and camera geometry to SDF.
- Add a canonical request/reply backend bridge and replay-compatible snapshots.
- Gate one unchanged policy through native and bridged backend adapters.

## Capabilities

### New Capabilities

- `canonical-backend-bridge`: backend-neutral reset/step transport.
- `ros-gazebo-validation-backend`: isolated ROS/Gazebo runtime and migrated world.

## Impact

Adds a profile-gated large container, SDF assets, Python bridge adapters, tests,
commands, and M10 evidence. Default training remains native and headless.
