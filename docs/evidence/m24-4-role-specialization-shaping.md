# M24.4 per-role specialization shaping and hysteresis

ADR 0026 replaces the pooled formation term with two independent per-role terms and makes the
assigner's switch cost configurable, so the reward can specialize the off-ball roles without
fighting assigner churn. This document records the pre-training scale measurements, the code
delivery, and the acceptance signals the M24.4 run must satisfy.

## Motivation: what run-0002 showed

Run `vsss-m24-3-role-run-0002` reported 500-800 role switches per training iteration at the
reference hysteresis (`switch_penalty = 0.18`, `emergency_margin = 0.20`). A reward that pays
for holding a formation position is degenerate while the responsibility each robot is supposed
to hold changes that often. ADR 0026 therefore couples two changes in one release:

1. independent geometric-mean potentials for support and coverage, each with its own
   coefficient, so the shaping can reward "hold the passing lane" and "cover goal to ball"
   separately (curvature argument of arXiv:2506.09434: a Schur-convex per-role aggregator
   makes the team's reward maximized by a strict division of labour; venue not yet confirmed
   on the arXiv page); and
2. `role_switch_penalty` / `role_emergency_margin` exposed through `MarlConfig`, the native
   `BatchSimulator`, and the environment, defaulting to the historical constants so every
   existing checkpoint loads unchanged.

## Delivery

- `crates/vsss-features/src/roles.rs`: `HystereticAssigner` carries explicit
  `switch_penalty`/`emergency_margin`; `with_hysteresis` builds one; `assign_roles_parameterized`
  and `best_permutation(switch_penalty)` expose the strength; the previous constants remain
  available through `assign_roles` and `Default`.
- `crates/vsss-python/src/lib.rs`: `BatchSimulator(config, state, num_worlds, role_switch_penalty,
  role_emergency_margin)` with defaults from the constants.
- `python/vsss_train/marl_env.py`: `TeamReward` gains `support_formation` and `coverage_formation`
  terms; `_support_formation_potential` / `_coverage_formation_potential` are geometric means of
  `exp(-distance/0.25)` over the role's active robots against the same identity-free targets the
  assigner uses (`_formation_targets`); both single-env and vector environments terminate the
  potentials at zero on goal/stagnation/draw; negative coefficients and hysteresis are rejected.
- `python/vsss_train/config.py`: four new `MarlConfig` fields validated non-negative.
- `python/vsss_train/marl_ppo.py`: `LEGACY_NEUTRAL_CONFIG` gains the four keys at their defaults
  (`0.0`, `0.0`, `0.18`, `0.20`) so legacy checkpoints resolve.
- `experiments/configs/m24-4-mappo-role-formation.toml`: pooled term off
  (`role_formation_coefficient = 0.0`), `support_formation_coefficient = 0.15`,
  `coverage_formation_coefficient = 0.15`, `role_switch_penalty = 0.30`,
  `role_emergency_margin = 0.30`, new fingerprint `m24.4-role-specialization-mappo-shared-v1`.

## Scale measured before training

Evaluated over all 6,000 role decisions in `iteration-002000.jsonl` from the final run-0013
replay (3,000 ticks, both teams), replaying the assignments and per-role potentials at the M24.4
coefficients:

| quantity | support | coverage |
| --- | ---: | ---: |
| mean per-role potential | 0.262 | 0.166 |
| potential p10 / p90 | 0.114 / 0.489 | 0.004 / 0.483 |
| mean absolute reward per decision | 0.000655 | 0.000534 |
| absolute reward p95 | 0.001136 | 0.001106 |

Each per-role term is audible at roughly one third to one half of carry (reported mean absolute
0.001646 in ADR 0023) without approaching the terminal goal value. The discounted-return bound
of each coefficient is `0.15`, against carry `5` and goal `10`.

## Verification

- `just doctor`, `just lint`, `just build`, `just test` all green (321 Python tests, Rust
  workspace tests, replay-viewer typecheck).
- New tests: per-role potentials are independent (moving coverage does not change support and
  vice versa), both terms are terminal-zeroed so the last transition cannot pay for how the
  episode ended, negative knobs are rejected, and the pooled geometric mean matches the product
  of the per-role means.

## Run outcome

Run `vsss-m24-4-run-0001` (seed 244, coefficients 0.0 / 0.15 / 0.15, hysteresis 0.30 / 0.30)
reached iteration 525 before the environment was recycled for the M24.5 comparison arm. It
recorded 16,506 total matches: full-match wins 177 / losses 245 / draws 4732, with 2506 skill
successes against 2340 skill failures. Early replays (iterations 25 and 50) confirmed the
shaping is exercised without dominating: per-decision `support_formation` and
`coverage_formation` rewards stayed in the sub-milli range while role switches dropped from the
500-800 per iteration of run-0002 (assigner is not fighting the shaping). The run was too short
to judge strike fraction or draw-rate convergence, so those acceptance signals remain open and
are re-checked by the M24.5 arm.

## Acceptance signals for the M24.4 run

- support and coverage strike fractions split materially below run 0013's 0.551 / 0.482 without
  collapsing to STOP;
- role switch rate per iteration falls well below run-0002's 500-800;
- attacker remains the highest-strike responsibility and carry/goal occupancy from ADR 0024 is
  preserved;
- full-match goals per minute stay at or above 0.2 and draw rate at or below 0.70.

## Rollback

All four knobs default to their historical values, so reverting to any previous config or
checkpoint is a configuration change only; no weights or serialized trajectories are affected.
