## Why

M4 can record and inspect deterministic matches, but developers cannot yet
observe a running match or replay it visually. Visualization must consume the
same canonical state without coupling rendering, networking, or backpressure to
the deterministic physics hot loop.

## What Changes

- Define backend-neutral visual frames composed from canonical match snapshots,
  actions, event flags, rewards, and monotonic tick metadata.
- Add optional observer sinks for no-op training, deterministic replay,
  diagnostics, and lossy live visualization.
- Require live visualization to sample independently and drop stale frames
  instead of slowing simulation.
- Add a lightweight 2D viewer that can consume both a live frame stream and the
  existing replay records with pause, seek, speed, overlays, and headless
  rendering support.
- Keep ROS 2/Gazebo as the later M10 validation backend rather than using it to
  render the fast simulator.
- Document Rerun as an optional observability adapter and Bevy as the preferred
  product viewer implementation; neither enters `vsss-spec` or the physics hot
  loop.

## Capabilities

### New Capabilities

- `simulation-observer-stream`: Backend-neutral frame envelopes and
  non-blocking observer sink behavior for live and recorded consumers.
- `fast-sim-2d-viewer`: A viewer that renders the same canonical state from
  live streams and deterministic replays.

### Modified Capabilities

- `scripted-match-replay`: Replays become a supported visual-frame source in
  addition to headless validation and summaries.

## Impact

This is a post-M4 visualization increment and does not change physics,
controller observations, or policy APIs. It affects replay tooling, match
execution, a new observer boundary, local commands, tests, ADRs, and the PRD.
The first implementation may use the existing JSONL replay format; binary
protocols, ROS, Gazebo, remote match control, and production web UI remain
explicit non-goals.
