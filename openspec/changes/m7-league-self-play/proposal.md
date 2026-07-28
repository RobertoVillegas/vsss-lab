## Why

M6 can optimize and evaluate one shared policy but has no durable population,
historical opponents, promotion decision, or learned-policy replay. M7 turns
isolated checkpoints into reproducible runs and inspectable competition.

## What Changes

- Add a versioned policy registry for candidate, main, heuristic, and historical entries.
- Add deterministic weighted matchmaking and synchronous self-play iterations.
- Add fixed-seed evaluation workers, Elo updates, and tournament JSON reports.
- Add reproducible non-regression promotion gates.
- Preserve iteration and promoted checkpoints outside Git with hashes in registry.
- Record learned-policy evaluation replays compatible with the existing 2D viewer.
- Add commands to run training iterations, inspect league state, evaluate, and view snapshots.
- Do not add remote match servers, ZeroMQ, FlatBuffers, Ray, TrueSkill, distributed
  workers, population-based training, or ROS/Gazebo evaluation.

## Capabilities

### New Capabilities

- `policy-league-registry`: Versioned population, historical checkpoints, and deterministic matchmaking.
- `league-evaluation-promotion`: Elo tournaments and reproducible non-regression promotion.
- `self-play-run-capture`: Real rollout/optimize iterations with checkpoint and visual replay capture.

### Modified Capabilities

None.

## Impact

Activates `python/vsss_league`, extends the M6 environment with checkpoint
opponents, and writes active artifacts under `/home/rob/runs`,
`/home/rob/checkpoints`, and `/home/rob/replays`.
