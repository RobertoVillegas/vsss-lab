## Why

Long self-play runs need physically contained robots, bounded actuator response,
observable action smoothing, honest accelerator selection, and terminal
progress that remains readable for hours.

## What Changes

- Close goal side/back walls for robots while retaining goal detection.
- Slew applied wheel speed from force, mass, radius, and fixed timestep.
- Penalize abrupt policy action changes and replay applied wheel telemetry.
- Add CUDA/CPU/auto selection and vectorized network inference.
- Add a Rich metrics table above a fixed progress bar with plain-text fallback.

## Capabilities

### Modified Capabilities

- `rapier-reference-physics`: physical goal enclosure and actuator response.
- `ippo-mappo-training`: device selection, vector worlds, action regularization,
  and terminal observability.

## Impact

Existing checkpoints are intentionally incompatible because the MARL
configuration fingerprint includes device-independent behavior changes. Existing
replays remain viewable but do not contain applied-wheel telemetry.
