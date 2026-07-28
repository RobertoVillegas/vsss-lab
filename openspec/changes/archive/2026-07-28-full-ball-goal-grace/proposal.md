## Why

Recorded goals fired when only the ball center crossed the line and matches cut
immediately, making valid play look like false goals and abrupt edits.

## What Changes

- Require the complete ball to cross the goal line.
- Emit the goal only on the crossing edge, including fused physics repeats.
- Continue simulation for the configured one-second grace before termination.
- Tighten contact stiffness after replay-based overlap measurement.

## Capabilities

### Modified Capabilities

- `rapier-reference-physics`: full-ball, single-event scoring.
- `ippo-mappo-training`: delayed post-goal match termination.
