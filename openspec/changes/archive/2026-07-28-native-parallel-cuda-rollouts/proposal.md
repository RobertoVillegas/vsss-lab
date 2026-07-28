## Why

CUDA was underfed because Python owned and stepped each small Rapier world
separately. Rapid proportional-number changes also caused visible viewer text
movement.

## What Changes

- Own vector worlds in one native batch and release the GIL while stepping.
- Step fixed repeated actions inside Rust and use adaptive Rayon parallelism.
- Increase the balanced CUDA default from 16 to 64 worlds.
- Use monospace tabular text throughout the replay viewer.

## Capabilities

### Modified Capabilities

- `python-batch-bindings`: repeated and parallel native stepping.
- `ippo-mappo-training`: one vector environment and a 64-world CUDA default.
- `training-replay-web`: stable-width global viewer typography.

## Impact

Changing the default world count changes the configuration fingerprint.
Sixteen-world checkpoints remain resumable only with an explicit
`num_envs=16`; new high-throughput runs use 64.
