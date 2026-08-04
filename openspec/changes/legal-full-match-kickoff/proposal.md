# Legal full-match kickoff

## Why

VSSS Rule 7 (Começo do jogo) requires the ball to start inside the center circle
(20 cm radius) on every kickoff and restart, with the defending team outside that
circle. The full-match reset instead samples the ball across a ~0.7 x 0.7 m
rectangle centered on the field (`_seeded_snapshot` in
`python/vsss_train/marl_env.py` samples `x in [-0.15, 0.25]`,
`y in [-0.35, 0.35]`). That distribution is illegal and teaches the policy to
locate the ball anywhere and push it forward, rather than executing a legal
kickoff. Replay diagnosis of run `vsss-m24-3-role-run-0002` showed every
kickoff goal from the center area and poor ball tracking when the ball started
farther out, consistent with a shortcut learned on an unrepresentative reset
distribution.

## Milestone and non-goals

This is an M24.3 evidence-driven correction to the full-match training
distribution. Non-goals:

- no change to observations, actions, physical identity, or policy architecture;
- no referee whistle / false-start gating or STOP phase before kickoff;
- no modelling of after-goal possession rotation (conceding team restarts);
- no new reward term and no change to promotion or behavior gates;
- no change to skill scenario kickoff (`routine-kickoff-center` already places
  the ball exactly at center);
- the strike-everything / poor ball-tracking collapse is tracked separately and
  is out of scope here.

## What changes

- Sample the full-match kickoff ball uniformly within the legal center circle.
- Add a validated `full_match_kickoff_radius` training knob, capped at the legal
  20 cm radius so the knob cannot reintroduce illegal kickoff positions.
- Keep robot kickoff placements legal: attackers may enter the center circle;
  defenders must remain outside it.
- Add executable tests for the kickoff legality contract.

## Success criteria

- a fresh full-match episode always starts with the ball inside the 20 cm circle;
- every non-attacking robot starts outside the center circle;
- `full_match_kickoff_radius = 0` reproduces the exact center kickoff;
- lint, build, and the full test suite remain green;
- the rollback is config-and-commit reversible and leaves run artifacts intact.
