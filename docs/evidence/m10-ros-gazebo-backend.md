# M10 ROS 2/Gazebo validation backend

## Outcome

The opt-in validation stack runs headlessly with ROS 2 Lyrical and Gazebo Jetty
(`gz sim 10.4.0`) in a pinned official ROS image. Gazebo validates the expanded
SDF, finds exactly nine models (field, ball, top camera, and six robots),
advances a fixed-step world, discovers the differential-drive command endpoint,
and moves `blue_0` with the canonical constant-wheel policy.

```bash
just backend-bridge-smoke
just ros-gazebo-smoke
```

Both commands pass. The container is isolated from the host install, publishes
no ports, drops all Linux capabilities, uses no-new-privileges and a read-only
root filesystem, and writes only to temporary mounts.

## Unchanged policy boundary

`CanonicalBackend` exposes only `reset() -> state[77]` and
`step(actions[6,2]) -> state[77]`. `NativeBackend` and `JsonLineBackend` run the
same `[10, 10]` rad/s robot-0 wheel policy for eight ticks with no policy branch;
the conformance sidecar returns byte-equal canonical trajectories.

The Gazebo smoke applies that same wheel pair through the canonical
differential-drive adapter:

```text
v     = wheel_radius * (left + right) / 2 = 0.25 m/s
omega = wheel_radius * (right - left) / axle_track = 0 rad/s
```

After transport discovery, Gazebo receives the resulting Twist and the smoke
requires the robot pose to move from its initial x position. ROS/Gazebo entity
and message types remain behind the adapter.

## Migrated assets

The self-contained SDF carries canonical 1.5×1.3 m field geometry, 75 mm robot
bodies, 25 mm wheel radius, 60 mm axle track, 21.5 mm ball radius, six
differential-drive models, and a 680×520 top camera at 2 m. It does not depend
on Julio's ROS 1/Catkin package or copy assets with unclear licensing.

## Replay and limitations

Bridge responses are validated for request sequence, finite action shape,
finite 77-value state shape, and child lifecycle. Returned canonical states can
feed the existing JSONL replay/viewer path without ROS-specific fields.

M10 proves runtime, world, actuator translation, backend process conformance,
and policy API portability. It is not yet a calibrated camera/vision pipeline,
does not claim contact equivalence beyond M9's current scenarios, and does not
use Gazebo for training throughput. The process conformance sidecar is the
deterministic test double; the container smoke independently exercises the real
Gazebo Jetty engine and differential-drive system.

Rollback removes the `gazebo` Compose profile, SDF, and bridge adapters. Native
training, policies, checkpoints, and replays are unchanged.
