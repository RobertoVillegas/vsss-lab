# Evaluation foundation evidence — 2026-07-28

## Sources reviewed

- Ray RLlib 2.55.1 `Algorithm.evaluate`: evaluation remains a distinct policy
  operation rather than a training-return shortcut.
- RLGym v2 configuration objects: terminal and truncated conditions, reward,
  state mutation, observation, action parsing, and rendering are independent
  contracts.
- Optuna current documentation: durable storage is required for persistent
  studies; in-memory studies do not provide restart evidence.

## Decisions

- **Adopt:** paired seeds and both team colors as the indivisible evaluation
  unit.
- **Adapt:** use a deterministic bootstrap interval over paired terminal match
  scores; persist every raw outcome beside its estimate.
- **Adopt:** gate each opponent category by the lower confidence bound and gate
  the aggregate by its point margin.
- **Adapt:** write evaluation and promotion JSON atomically so interrupted
  evaluation cannot masquerade as complete evidence.
- **Reject:** shaped return, progress, newest checkpoint, or Elo alone as a
  promotion criterion.
- **Defer:** distributed evaluation orchestration. Local paired evaluation is
  the correctness baseline and can later be scheduled externally.

## Falsifiable gate

An uncertain fixture with a positive point estimate but a lower confidence
bound below its regression floor must reject promotion. Repeating the same
ordered evidence must produce byte-identical estimates and decision JSON.

## Replay analytics completion

The viewer now exposes a seekable filtered event timeline, a normalized
ball-position heatmap, side-by-side team metrics, and a combined team/event CSV
export. Golden replay tests preserve goal attribution and bound possession-time
drift under temporal downsampling. Derived defense, congestion, and
double-commit descriptors are SHA-256 deduplicated and routed into matching
curriculum rehearsal buckets after captured replays; no descriptor is added to
the reward function.
