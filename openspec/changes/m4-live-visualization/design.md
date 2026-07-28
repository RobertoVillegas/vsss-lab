## Context

M4 writes deterministic JSONL records containing actions, snapshots, event
flags, and checksums, but its viewer only validates and summarizes text. The
Rapier2D backend is intentionally standalone and headless. The project needs a
graphical debugging path now without moving M10 ROS 2/Gazebo work forward or
allowing visualization latency into the physics hot loop.

The contract choice is recorded in
`docs/decisions/0006-decoupled-fast-sim-visualization.md`.
The comparison evidence is recorded in
`docs/architecture/simulator-landscape.md`.

## Goals / Non-Goals

**Goals:**

- Give live and replay consumers one backend-neutral visual-frame model.
- Preserve deterministic headless execution when no observer is configured.
- Bound live visualization cost and prevent viewer backpressure.
- Support interactive 2D inspection and deterministic headless render tests.
- Leave room for Bevy, Rerun, and ROS/Gazebo adapters without coupling them to
  `vsss-spec`.

**Non-Goals:**

- Changing Rapier physics, observations, actions, rewards, or policy APIs.
- Building the M10 Gazebo backend, ROS bridges, or sensor simulation.
- Defining the M8 remote-controller protocol.
- Replacing the current replay storage format with FlatBuffers or MCAP.
- Building a general-purpose web administration product.

## Decisions

### Frames adapt canonical state rather than extending it

The authoritative state remains `vsss_spec::MatchState`. A visual frame
envelope adds presentation metadata and references/copies canonical snapshots,
actions, events, and optional rewards. This prevents renderer concepts from
entering `vsss-spec`.

Alternative: add renderer fields to `MatchState`. Rejected because state must
remain backend- and presentation-independent.

### Observer sinks have explicit delivery semantics

Match execution can fan out to `Null`, lossless replay/metrics, and bounded live
sinks. The live sink retains the newest frame and counts dropped frames.
Sampling is based on simulation ticks, not wall-clock timing. Observer failures
are reported out of band and do not mutate simulation state.

Alternative: publish every physics step over a mandatory IPC bus. Rejected
because serialization and backpressure would enter every rollout.

### Live and replay paths converge before rendering

Both sources decode into the same visual-frame type. The viewer owns playback
rate, interpolation, pause, seek, overlays, and skipped-frame display; it never
advances physics implicitly.

Alternative: implement separate live and replay viewers. Rejected because they
would drift and make debugging recorded behavior less trustworthy.

### Build the viewer in layers

First provide deterministic 2D scene projection and headless artifacts. Then
add a Bevy native/WASM shell for interaction. An optional Rerun adapter can
offer immediate timelines and diagnostic plots; it is not the canonical file
format or required runtime. Foxglove is deferred to the ROS/MCAP milestones.

## Risks / Trade-offs

- [Live frames are intentionally incomplete] → Display drop counts and use the
  lossless replay for authoritative analysis.
- [Copying snapshots costs CPU] → Sample only watched worlds and benchmark with
  observation disabled and enabled.
- [Bevy adds compile weight] → Keep it in a leaf viewer crate and preserve a
  dependency-light headless renderer.
- [JSONL is verbose] → Keep adapters versioned and migrate storage only when the
protocol milestone provides measured justification.

### Follow the proven standalone-simulator/viewer split

VSS-SDK separates its Bullet simulator from a viewer that consumes states from
simulation or vision. RocketSim/RLGym makes rendering optional, and RLViser is
a Rust/Bevy process that listens for state packets. These independent VSSS and
high-throughput RL precedents support a leaf Bevy viewer behind a replaceable
transport rather than rendering inside Rapier.
- [Interpolation can show non-physical intermediate poses] → Label interpolated
  display state and retain exact-tick stepping.

## Migration Plan

1. Add frame adapters and sink contract tests without changing replay bytes.
2. Adapt existing JSONL records to frames and add deterministic headless
   rendering.
3. Add bounded live delivery and prove simulation checksum equivalence.
4. Add the Bevy viewer shell and native/WASM playback controls.
5. Add optional observability adapters only after benchmark evidence.

Rollback removes the observer/viewer leaf components and match-runner hooks.
Existing replay JSONL, canonical snapshots, and physics APIs remain readable.

## Open Questions

- Choose the long-term replay container (FlatBuffers versus MCAP) at M8.
- Benchmark the acceptable watched-world count and default sampling interval.
- Decide whether the first remote browser transport is WebSocket or Rerun gRPC
  after the local viewer contract is stable.
