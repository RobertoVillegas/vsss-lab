## Context

M6 has policy-versioned trajectories, IPPO/MAPPO learners, and a viewer-compatible
simulation stack. M7 must preserve every promotion decision and make intermediate
training behavior observable without coupling visualization to the learner.

## Goals / Non-Goals

**Goals:** local registry, deterministic matchmaking, real self-play
rollout/optimize iterations, historical checkpoints, Elo, tournaments,
non-regression promotion, and learned-policy JSONL replays.

**Non-Goals:** remote workers, async learning, large population algorithms,
TrueSkill, external controller protocols, or statistically definitive research
claims from smoke-sized tournaments.

## Decisions

1. Registry and reports are schema-versioned canonical JSON. Checkpoints remain
   trusted local PyTorch files and registry entries store SHA-256.
2. Registry mutations use write-to-temporary plus atomic replace and reject
   duplicate policy IDs/versions.
3. Matchmaking is a seeded weighted draw over eligible categories; the selected
   opponent and RNG seed are recorded before rollout.
4. A training iteration collects a fresh native C8 trajectory against a frozen
   opponent, rejects stale policy versions, optimizes once, and writes an
   immutable checkpoint.
5. Evaluation workers are synchronous local functions in M7. They side-switch
   fixed seeds and produce both match records and viewer-compatible JSONL replay.
6. Elo uses the standard logistic expectation and configurable K-factor. Draws
   are explicit; infrastructure failures never update rating.
7. Promotion compares candidate versus main, historical, and heuristic fixtures.
   It requires a positive configured aggregate margin and zero fixture
   regressions. Re-running the same manifest must yield the same decision/report.
8. Training capture is decoupled: the learner emits immutable checkpoints;
   evaluation loads selected checkpoints and emits replays. The viewer never
   runs inside the rollout hot loop.

## Risks / Trade-offs

- **Short matches rarely score** → record progress outcome alongside score;
  promotion defaults remain conservative and later league scale adds confidence.
- **Checkpoint code execution risk** → accept trusted local files only and use
  weights-only loading.
- **A candidate overfits one opponent** → require main, historical, heuristic,
  unseen seed, and side-switch fixtures.
- **Frequent capture slows training** → capture interval is configurable and
  evaluation remains out of band.

## Migration Plan

Add standalone Python league modules and commands. Rollback removes registry/run
artifacts and M7 code while M6 checkpoints remain directly evaluable.

## Open Questions

TrueSkill, parallel evaluation workers, adaptive matchmaking, and statistical
confidence rules are deferred until league volume justifies them.
