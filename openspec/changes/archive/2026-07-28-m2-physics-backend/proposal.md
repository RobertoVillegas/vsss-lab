## Why

M1 defines canonical state but cannot advance it. M2 adds the deterministic CPU
reference implementation required before bindings, environments, or baselines.

## What Changes

- Add a backend-neutral `PhysicsBackend` API with reset, step, snapshot, restore,
  checksum, and batch contracts.
- Add a headless fixed-step Rapier2D backend for six differential-drive robots,
  one ball, walls, and goals.
- Add actuator saturation, goal events, individual-world reset, deterministic
  replay tests, and a throughput benchmark.
- Do not add Python bindings, RL APIs, replay files/viewer, ROS, networking,
  rendering, parallel execution, or optimized SIMD mode.

## Capabilities

### New Capabilities

- `physics-backend-api`: Backend lifecycle, snapshots, checksums, and batch semantics.
- `rapier-reference-physics`: Fixed-step VSSS bodies, differential drive, collisions,
  walls, goals, and deterministic stepping.

### Modified Capabilities

None.

## Impact

Adds `vsss-physics-api`, `vsss-physics-rapier`, and `vsss-batch` workspace crates,
with Rapier 0.34 locked using enhanced determinism.
