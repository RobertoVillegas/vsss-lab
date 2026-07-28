# ADR-0009: Local league, promotion, and training capture

- Status: accepted
- Date: 2026-07-28
- Owners: Roberto Villegas

## Context

M7 must compare current, candidate, historical, and heuristic policies while
allowing developers to inspect how behavior changes between training iterations.

## Decision

Use an atomic versioned JSON registry pointing to immutable trusted-local
checkpoints. Run seeded synchronous matchmaking and evaluation locally. Use
standard Elo for an initial rating and a deterministic fixture manifest for
promotion. Keep training and capture separate: rollout/optimize writes a
checkpoint, then evaluation loads it and writes the existing replay schema with
policy metadata for the 2D viewer.

## Consequences

Runs are inspectable and reproducible without a service or database. Local JSON
does not support concurrent writers, and Elo is descriptive rather than a sole
promotion criterion. Viewer capture costs no rollout hot-loop allocations.

## Alternatives considered

SQLite offers stronger concurrency but is unnecessary for one synchronous M7
writer. Rendering during rollout would simplify live viewing but contaminates
throughput. Promoting only by Elo hides fixture regressions.

## Validation and rollback

Atomicity, duplicate protection, deterministic matchmaking/tournaments, Elo
conservation, non-regression, real policy-version updates, replay checksums, and
viewer compatibility gate M7. Rollback removes league metadata without changing
checkpoint contents.
