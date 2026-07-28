## Why

Captured training iterations are inspectable, but the native viewer must be
launched once per replay and `league-run` gives no visual feedback. M7 needs a
single local surface where a developer can compare and scrub every captured
iteration without coupling rendering to training.

## What Changes

- Add a loopback-only web application that discovers replays in one run.
- Render canonical replay snapshots in a responsive 2D field.
- Add iteration selection, play/pause, speed, exact stepping, skipping, and
  timeline seeking.
- Add a local command and operator documentation; keep the native and headless
  SVG viewers available.

## Capabilities

### New Capabilities

- `training-replay-web`: Local discovery and interactive playback of captured
  training iterations.

### Modified Capabilities

- `fast-sim-2d-viewer`: Extend the existing interactive replay controls to a
  browser-based projection without changing replay contracts.

## Impact

Adds a small React/Vite frontend under `web/replay-viewer`, a standard-library
Python static/API server under `tools/replay_web`, tests, and `just` commands.
The JSONL replay format, simulator hot loop, and training runtime do not change.
