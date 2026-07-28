## Why

The completed 50 million-step M12 run shows genuine learning and physically
valid contacts, but its later policy becomes nearly deterministic and often
stalls or clusters. Proximity and raw x-progress shaping can reward possession
without producing a terminal goal, so the latest checkpoint is not necessarily
the strongest policy.

## What Changes

- Replace proximity/x-progress shaping with ball-to-goal direction and dynamic
  attacker-to-ball alignment signals adapted from Julio De La Torre's thesis.
- Add bounded time and wheel-effort costs while retaining small congestion and
  defensive-coverage regularizers.
- Clamp policy exploration to a configurable minimum standard deviation.
- Train first against the dynamic heuristic, then transition to self-play.
- Rank selected checkpoints by terminal W/D/L and goals against a fixed
  heuristic rather than choosing the latest version.
- Preserve permutation-safe dynamic roles and shared-policy symmetry.
- Start a fresh checkpoint lineage; do not resume the old reward fingerprint.
- Preserve strict loading of historical checkpoints with neutral defaults for
  fields that did not exist when they were written.

## PRD Milestone

M13 — directional, coordinated self-play after the M12 physical/training
diagnostic.
