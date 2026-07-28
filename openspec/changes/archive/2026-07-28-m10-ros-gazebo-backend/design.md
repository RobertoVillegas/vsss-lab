## Context

Gazebo is a validation backend, not a renderer for Rapier and not the rollout
hot loop. The canonical policy boundary is a 77-value state and six wheel-action
pairs; ROS topics and Gazebo entities stay behind a sidecar.

## Decisions

1. Pin the official `ros:lyrical-ros-base-resolute` multi-arch image and install
   the matching `ros_gz` integration only in `containers/ros`.
2. Keep the SDF world self-contained, headless, and in SI units.
3. Use newline-delimited JSON reset/step messages for the process boundary.
   The adapter validates shape, sequence, finite values, and child exit status.
4. Prove policy API portability with an out-of-process conformance sidecar;
   separately prove Gazebo Jetty can parse and advance the migrated world.
5. Replay uses the canonical state returned by the bridge, never simulator-
   specific pose messages.

## Risks

The ROS/Gazebo image is large and release repositories can move. The base digest
is pinned, the package version is recorded by the smoke, and the profile is
opt-in. Full camera fidelity belongs to M12.

## Rollback

Remove the opt-in profile and bridge adapter. Native backend and policy APIs are
unchanged.
