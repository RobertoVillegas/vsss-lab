## Context

M2 must prove correctness and replay determinism on one platform before optimizing.
The canonical crate remains dependency-free from physics.

## Goals / Non-Goals

**Goals:** fixed-step reference physics, six differential-drive robots, ball/wall
collisions, goals, snapshots, checksums, sequential batch, and benchmarks.

**Non-Goals:** Python, parallelism, high-fidelity motors, ROS, rendering, and a
persistent replay format.

## Decisions

1. `vsss-physics-api` owns an object-safe backend trait using canonical types.
2. Rapier 0.34 uses `enhanced-determinism`; gravity is zero and timestep is config.
3. Robots are dynamic cuboids and the ball is a CCD-enabled dynamic circle.
4. Differential drive converts wheel angular speeds to body `v/omega`; commands
   are saturated and applied as velocity targets in the M2 reference model.
5. Snapshots contain canonical state, not opaque Rapier internals. Restore rebuilds
   the world, making compatibility explicit and checksums portable within schema.
6. Batch is sequential `Vec` storage in M2; parallelism is deferred until measured.

## Risks / Trade-offs

- Velocity targets are idealized motors → calibrate acceleration/lag in later work.
- Rebuilding snapshots costs allocations → correctness precedes optimized snapshots.
- Rapier determinism is platform/version scoped → pin crate and test replay checksum.

## Migration Plan

Add crates without changing M1 JSON. Rollback removes the three crates and ADR.
