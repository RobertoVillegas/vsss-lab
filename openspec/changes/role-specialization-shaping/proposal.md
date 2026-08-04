# Role-specialization shaping and stable assignment

## Why

Run `vsss-m24-3-role-run-0002` collapsed toward "everyone is a striker": support and coverage
selected `strike` on 60 and 62 per cent of decisions against attacker 72 per cent, while
`role_formation` contributed about -0.0002 per decision against `goal_scored` at 0.0018-0.0043
and the assignment churned 500-800 times per iteration. 2025-26 cooperative-MARL results
(ICLR 2026 reward curvature, behavioural-diversity soccer, role-based MAPPO) agree that role
specialization must be rewarded by the reward structure and that the role signal must stay
stable enough to learn against.

## Milestone and non-goals

This is an M24.4 evidence-driven correction. Non-goals:

- no new observation, action, or policy architecture;
- no robot identity owning a role and no dedicated per-identity policies;
- no new role discovered by the learner (RODE-style role clustering is out of scope);
- no change to goals, terminal sharing, or promotion gates;
- no change to the kickoff distribution (ADR 0025) or the physics contract.

## What changes

- Split the combined support/coverage formation potential into per-role potentials, each the
  geometric-mean bottleneck over the robots currently assigned that role, with independent
  coefficients `support_formation_coefficient` and `coverage_formation_coefficient`.
- Add configurable role hysteresis `role_switch_penalty` and `role_emergency_margin`, threaded
  through the native assigner and the Python reference, so a run can stabilize the one-hot role
  signal and cut churn.
- Keep `role_formation_coefficient` as the legacy combined term so older checkpoints stay
  loadable at neutral defaults.
- Give the fresh run a new policy fingerprint and start it from a fresh run directory.

## Success criteria

- support and coverage strike fractions separate materially from attacker without collapsing to
  stop, and pass/receive, rotation, and defensive metrics do not regress;
- role-switch rate per iteration falls materially below the M24.3 run's 500-800;
- `support_formation` and `coverage_formation` each contribute audibly (non-noise) to the reward
  ledger and never exceed a small fraction of the goal coefficient;
- full-match goals per minute, draw rate, and the semantic gates remain green;
- lint, build, and the full test suite remain green; rollback is configuration-only.
