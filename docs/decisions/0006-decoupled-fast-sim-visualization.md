# ADR-0006: Decoupled fast-simulator visualization

- Status: accepted
- Date: 2026-07-27
- Owners: Roberto Villegas

## Context

The fast Rapier2D backend must maximize deterministic rollout throughput, while
developers also need to watch live matches and inspect recorded behavior. Using
Gazebo only to render the fast simulator would add unnecessary runtime and
would conflate visualization with the later high-fidelity validation backend.

## Decision

Treat rendering as a consumer of canonical simulation data, not as a physics
backend. Match execution emits a versioned visual frame containing the
canonical `MatchState`, actions, event flags, rewards, tick, and simulation
time through optional observer sinks.

The no-op path remains the default for training. Replay sinks are lossless and
part of explicit evaluation runs. Live sinks are sampled, bounded, and lossy:
they replace or drop stale frames rather than applying backpressure to physics.
Consumers use the same frame semantics whether reading a live stream or replay.

Use a lightweight 2D viewer for the fast simulator, with Bevy as the preferred
native/WASM implementation after the frame boundary is proven. Rerun may be
added as an observability adapter. ROS 2/Gazebo remains an independent M10
physics and sensor validation backend connected through canonical contracts.

## Consequences

Simulation can remain fully headless while live, offline, native, browser, and
headless image consumers evolve independently. Live visualization can skip
intermediate frames and therefore is not an authoritative replay. Lossless
recording must be requested separately.

## Alternatives considered

Embedding rendering in Rapier execution makes the hot loop depend on graphics
and window lifecycle. Treating Gazebo as the fast simulator's renderer creates
two coupled worlds and synchronization ambiguity. Emitting only discrete
events cannot animate continuous robot and ball motion. Making every observer
lossless lets slow viewers reduce rollout throughput.

## Validation and rollback

Contract tests verify equivalent frames from live observation and replay,
monotonic ticks, bounded live queues, and unchanged simulation checksums with
observation enabled or disabled. Headless render tests compare deterministic
artifacts. Rollback removes observer adapters and viewer packages; canonical
physics state, replay source data, and backend APIs remain valid.
