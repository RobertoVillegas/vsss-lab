# Native Environment Hot Loop

## Why

`AGENTS.md` states that Python dictionaries never enter the simulation hot loop. The physics
has honoured that since M2; everything above it has not. Observations, role assignment, every
reward term, contact and deadlock detection, idle-spin flags and the free-ball restart are
computed in a Python loop over worlds on every decision.

Measured over one iteration of the live configuration, 64 worlds and 256 decisions: the native
physics costs 0.06 seconds and the Python layer above it costs 4.95, so the simulation is 1.2
per cent of `env.step`. A profile puts ninety-seven per cent of an iteration in the rollout
and three in the PPO update, with the GPU at seventeen per cent. Pure-Python fixes already
returned 2.8x; what is left above the physics is per-world Python.

Moving that layer down leaves a floor of physics plus policy forward plus update, about 0.68
seconds against today's 8.76. Fifty million steps would take half an hour rather than a
working day.

See ADR 0019 and `docs/evidence/m24-3-rollout-throughput.md`.

## Milestone and non-goals

Platform work supporting the active milestone. Non-goals:

- the learner does not move: PPO and the network are three per cent of the time and porting
  them would trade PyTorch's ecosystem for that three per cent;
- no change to what any quantity means; every slice must agree with the Python reference
  before it replaces it;
- `vsss-spec` acquires no training concerns.

## What changes

- a new crate holds the ported per-world computation, exposed through the existing PyO3 batch
  surface as one call per decision rather than one call per world;
- observations move first, then role assignment, then reward terms, then contact and
  deadlock detection, then the free-ball restart;
- each slice ships with a golden-equivalence test against the Python implementation, which
  stays in the tree as the reference until its slice is retired.

## Success criteria

- a slice is accepted only when the native and Python paths agree on recorded states within a
  stated tolerance, and the tolerance is chosen so no decision branch can differ;
- the free-ball restart no longer edits a JSON dictionary inside the hot loop;
- throughput is reported per slice, so the migration's value is visible rather than asserted;
- the tree remains working and reversible after every slice.
