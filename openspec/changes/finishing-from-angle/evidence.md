# Evidence — M24.6 run

## Run identity

- Config: `experiments/configs/m24-6-mappo-role-formation.toml` (seed 246, policy id
  `m24.6-clearing-angle-mappo-shared-v1`, fingerprint `ebc5fb8b33c7499b`; M24.5 was
  `3d2f73d3e06d5c44`). Generator revision `m24.6-clearing-angle`, so the fresh curriculum
  state cannot mix the old `shot` drill distribution.
- Smoke run: `/home/rob/runs/vsss-m24-6-run-0001/` — 25 iterations, 409,600 steps, clean
  exit, curriculum `allocation_valid`, first eval success_rate 0.393.
- Full run: `/home/rob/runs/vsss-m24-6-run-0002/` — 2,900 iterations, 47,513,600 steps,
  90,630 matches. Stopped by the built-in early-stop at iteration 2,900 ("Semantic holdouts
  regressed for 12 evaluations; stopping at the last completed checkpoint"), not by budget
  or error. Checkpoints through `iteration-002900.pt`; `best-semantic.json` selected.

## Curriculum phase progression (the headline difference)

| | M24.5 (2,012 it) | M24.6 (2,900 it) |
| --- | --- | --- |
| final phase | `defense` (idx 1) | `integration` (idx 4) |
| phase at iter 500 | defense | defense |
| phase at iter 1,000 | defense | rotation (idx 3) |
| phase at iter 2,500 | defense | integration (idx 4) |

M24.5 never left the `defense` phase. M24.6 passed through `rotation` (iter ~1,000) and
entered `integration` (iter ~2,500). The phase machinery that stalled in M24.5 advances
with the M24.6 curriculum.

## Semantic evaluations (success_rate, every 250 iterations)

| iter | M24.6 | M24.5 | iter | M24.6 | M24.5 |
| --- | --- | --- | --- | --- | --- |
| 250 | 0.637 | 0.568 | 1,500 | 0.667 | 0.575 |
| 500 | 0.637 | 0.489 | 1,750 | 0.708 | 0.564 |
| 750 | 0.631 | 0.632 | 2,000 | 0.619 | 0.611 |
| 1,000 | 0.560 | 0.671 | 2,250 | 0.726 | — |
| 1,250 | 0.685 | 0.621 | 2,500 | **0.768** | — |

- M24.6 peaks at **0.768** (iter 2,500) and holds ≥0.70 through the last quarter; M24.5
  peaked at 0.671 (iter 1,000) and regressed to 0.56–0.61 thereafter.
- The iter-1,000 dip (0.560) is real and coincides with the phase transition into
  `rotation` and the worst full-match stretch (draw 0.883); it recovers by iter 1,250.

## Full-match gate (win / draw / loss, cumulative)

| iter | M24.6 | M24.5 |
| --- | --- | --- |
| 1,000 | 0.052 / 0.883 / 0.066 | 0.077 / 0.853 / 0.071 |
| 1,500 | 0.068 / 0.865 / 0.067 | 0.076 / 0.852 / 0.072 |
| 2,000 | 0.077 / 0.852 / 0.072 | 0.079 / 0.845 / 0.076 |
| 2,500 | 0.082 / 0.842 / 0.076 | — |
| 2,900 | **0.089 / 0.834 / 0.077** | — |

- M24.6 starts worse than M24.5 (draw 0.883 at iter 1,000 vs 0.853) but improves
  monotonically in the last third (0.883 → 0.834 draw, 0.052 → 0.089 win); M24.5 plateaued
  at ~0.85 draw from iter 1,000 to 2,012.
- **The gate (draw ≤ 0.70) is not crossed.** Final gap to gate: draw 0.834.
- The early-stop fired while the match trend was still improving — the run was mid-
  `integration` phase with the gate still moving.

## Goal events (cumulative)

| | M24.6 | M24.5 |
| --- | --- | --- |
| full_match for / against | 2,633 / 2,262 (+1.16) | 1,674 / 1,602 (+1.04) |
| skill for / against | 4,415 / 6,129 (0.72) | 6,055 / 9,720 (0.62) |

Match-level scoring ratio improved modestly; the skill-scenario ratio is negative in both
runs but M24.6's is less negative.

## Per-family outcome (final)

| family | M24.6 success | M24.6 levels (angle / speed / spawn) | note |
| --- | --- | --- | --- |
| approach | 0.667 | 0.05 / 0.05 / 1.00 (maxed) | recovered from 0.542 at iter 1,000 |
| shot | 0.479 | 0.375 / 0.45 / 0.275 | +15pp vs M24.5 at iter 1,000 (0.625 vs 0.479), eroded by iter 2,900 |
| interception | 0.583 | 0.05 / 0.775 / 0.05 | climbing |
| clearance | 0.521 | 0.05 / 0.275 / 0.725 | new family (ADR 0027), active |

`shot ball_angle` ramped to level 0.575 by iter 1,000 (M24.5 kept it at the floor), pulled
back to 0.375 as failures accumulated — the angle ladder is learnable but not yet
conquered.

## Behavior signals

- Rotation: M24.6 attempts rotations (6 attempts, 3 completed, 50%) where M24.5 recorded
  zero attempts in 2,012 iterations.
- Idle spin ratio 0.021 vs 0.031 (M24.5); uncovered ratio ~0.146 in both.
- Contact: ally deadlocks 14 / opponent 23; opponent_seconds 139.7 vs 153.9 (M24.5).

## Verdict

The ADR 0027 mechanism is effective and the curriculum responds: semantic rate up
(peak 0.768 vs 0.671), phase progression defense → integration, rotation attempts appear,
match scoring ratio +1.16. The match-level gate did not move enough (draw 0.834 vs 0.70).
The stall M24.5 showed at the *phase* level (stuck in defense) is fixed; the stall at the
*match outcome* level persists and is the target of the next run.
