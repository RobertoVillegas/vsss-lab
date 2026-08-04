# Design

## Per-role formation potentials

Replace the single `_role_formation_potential` (a geometric mean pooled over support and
coverage) with two additive potentials sharing the same target geometry and the same
smooth-Schur-concave bottleneck:

- `_support_formation_potential(state, team, assignment)` = geometric mean of
  `exp(-distance / 0.25)` over the robots currently assigned `support`, at the support target
  (the pass lane behind the ball).
- `_coverage_formation_potential(state, team, assignment)` = geometric mean of the same
  exponential over the robots currently assigned `coverage`, at the coverage target (between
  ball and own goal).

Both are rewarded as terminal-zeroed discounted potential shaping, exactly like the combined
term today:

```text
support_formation  = support_coefficient  * (discount * Phi_support(next) - Phi_support(current))
coverage_formation = coverage_coefficient * (discount * Phi_coverage(next) - Phi_coverage(current))
```

`role_formation_coefficient` keeps its combined meaning: when it is non-zero the pooled
geometric mean is used instead. `0` disables a per-role term. Terminal states carry zero
potential so shaping stays policy-invariant (ADR 0015 / ADR 0023).

The two additive terms are the task-allocation curvature ADR 0026 adopts: because the outer sum
does not let a strong coverage contribution hide an abandoned support lane, the team must divide
effort. Identity stays out of the reward: the assignment moves the responsibility and its
shaping together.

## Stable assignment

The native `HystereticAssigner` and the Python `DynamicRoleAssigner` already implement the same
hysteresis; only the native side hard-codes the two constants. Add `role_switch_penalty` and
`role_emergency_margin` to `MarlConfig` (defaults `0.18`/`0.20`), thread them into the native
`BatchSimulator` constructor so every world's assigner is built with them, and into the Python
`DynamicRoleAssigner` in `MarlMatchEnv`. `assign_roles(state, team, previous)` and the stateless
path keep the module constants, so existing callers and the native/Python equivalence tests are
unchanged at the defaults.

## Configuration

Add to `MarlConfig` (`python/vsss_train/config.py`):

- `role_switch_penalty: float = 0.18` — per-robot switch cost, validated non-negative;
- `role_emergency_margin: float = 0.20` — gap that abandons hysteresis, validated non-negative;
- `support_formation_coefficient: float = 0.0` — validated non-negative;
- `coverage_formation_coefficient: float = 0.0` — validated non-negative.

The fresh M24.4 configuration raises the hysteresis to `0.30`/`0.30` and enables both per-role
terms at `0.15` while leaving `role_formation_coefficient` at `0`. All four keys are added to
`LEGACY_NEUTRAL_CONFIG` at their defaults so checkpoints written before this change stay
loadable when the knobs sit at those defaults.

## Compatibility and rollback

- The native constructor signature keeps its old parameters as defaults, so every existing
  `BatchSimulator(config, state, n)` call is untouched.
- Checkpoints written before ADR 0026 load only when every new knob equals its neutral default;
  the M24.4 fingerprint changes, so the run starts fresh and is not return-comparable.
- Rollback is configuration-only: per-role coefficients to `0`, hysteresis to `0.18`/`0.20`.
  No checkpoint or artifact is deleted.
