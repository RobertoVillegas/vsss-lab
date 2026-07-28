## Why

Long headless training runs need observable progress without coupling rendering
to the learner. The existing viewer discovers captures only once, which forces
manual reloads and obscures the difference between simulated real time and
inspection speed.

## What Changes

- Poll immutable run metadata while the viewer is open.
- Follow and loop the latest completed replay by default.
- Expose the latest checkpoint and training metric alongside replay progress.
- Permit historical inspection without fighting the live-follow behavior.
- Label 1× as the recorded simulation clock while defaulting inspection to 4×.
- Show per-frame wheel commands, intensity, linear speed, turn rate, and direction
  for all six actors.
- Classify captures by score, win/loss/draw, and goal presence for filtering.

## Capabilities

### Modified Capabilities

- `training-replay-web`: Add live run discovery and checkpoint visibility.

## Impact

The loopback API gains read-only checkpoint and metric metadata. The React
client uses TanStack Query for cached polling. Training and replay contracts
gain resumable sparse checkpointing and 60-second evaluation captures.
