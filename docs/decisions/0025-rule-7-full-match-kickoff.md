# ADR 0025: Legal full-match kickoff distribution

- Status: accepted
- Date: 2026-08-03
- Owners: Roberto Villegas

## Context

VSSS Rule 7 (Começo do jogo) requires the ball to be placed inside the center
circle (20 cm radius) at the start and at every restart, the attacking team free
to place robots anywhere in its own half including inside the circle, and the
defending team in its own half but outside the circle.

The full-match reset in `_seeded_snapshot`
(`python/vsss_train/marl_env.py`) instead samples the ball uniformly over
`x in [-0.15, 0.25]`, `y in [-0.35, 0.35]` — a roughly 0.7 x 0.7 m rectangle.
This distribution is illegal and not representative. Replay analysis of run
`vsss-m24-3-role-run-0002` showed the policy learned to score from the center
kickoff by striking forward while a ball placed farther out was often not
tracked at all: the same unrepresentative-reset shortcut that inflates the
kickoff-goal metric.

## Decision

Sample the full-match kickoff ball uniformly in area inside the legal center
circle, via `radius = R * sqrt(u)`, `angle = 2 pi * u`. Add a validated
`full_match_kickoff_radius` field to the versioned training config
(`MarlConfig`) with default `0.20` m and a hard ceiling of `0.20` m so the knob
can only narrow the legal distribution. `0` reproduces the exact-center skill
kickoff.

The knob lives in the training config, not in `vsss-spec` `ResetRules`. The
kickoff sampling is a training-distribution concern; the simulator restores any
validated state and Rule 7 governs tournament placement, not the physics
contract. Robot placements remain legal: the defending-half starts already sit
outside the circle.

## Consequences

- Full-match kickoffs become rule-faithful, so kickoff goals and ball tracking
  are measured under the real placement condition.
- Returns and behavior are not comparable with runs trained on the rectangular
  distribution; the next run must start fresh.
- Checkpoints, replays, and the spec contract are unchanged; rollback is a
  two-line revert plus any config value.
