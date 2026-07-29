# M17: Coordination and contact curriculum

## Why

Run `vsss-semantic-run-0006` mastered isolated approach and shot tasks while
pass/receive regressed, rotation recovery remained at zero, defensive coverage
degraded, and sustained robot contacts grew after the best checkpoint. More
steps with the same objective would reinforce pushing and deadlocks rather than
team play.

## What changes

- Introduce deterministic roster ladders from 1v0 through 3v3 inside semantic
  scenarios, retaining rehearsal of simpler rosters.
- Penalize sustained, avoidable contact rather than every physical collision.
- Reward disengagement from deadlocks and preserve legitimate ball challenges,
  saves, and opponent blocks.
- Measure ally/opponent contact duration, deadlocks, escapes, and completed
  rotations.
- Make league-opponent sampling visible and begin diversity before the previous
  1,000-iteration heuristic-only warmup.
- Protect pass and rotation competence from regression during checkpoint
  selection.

## Non-goals

- Static obstacles are not a primary training opponent.
- Physical contact is not forbidden.
- Fixed robot identities do not own tactical roles.
- This change does not address sim-to-real hardware integration.
