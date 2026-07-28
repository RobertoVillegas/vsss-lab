# Search and lineage evidence — 2026-07-28

## Current implementation reviewed

Optuna 4.9 provides durable SQL-backed studies, NSGA-II multiobjective
sampling, typed suggestion APIs, and trial metadata. The project uses those
maintained primitives instead of implementing an optimizer.

## Decisions

- **Adopt:** SQLite-backed `load_if_exists` studies and deterministic seeded
  NSGA-II sampling.
- **Adapt:** three objectives: maximize terminal evidence while minimizing
  coordination failure and compute.
- **Adopt:** one, three, and five independent seeds for smoke, screen, and
  confirmation.
- **Adapt:** a small typed reward/PPO parameter surface with hard bounds and a
  canonical configuration hash.
- **Adopt:** append-only fidelity lineage containing commit, parent, exact seed
  set, objectives, prune reason, and compute.
- **Reject:** arbitrary generated reward code and mutation within an active
  checkpoint lineage.
- **Defer:** large studies until the fixed-reward curriculum ablation establishes
  that allocation itself improves terminal holdout outcomes.

## Falsifiable gate

Restarting a named study must retain completed trial numbers. Re-recording
identical lineage is idempotent, while conflicting evidence for the same
study/trial/fidelity identity must fail.
