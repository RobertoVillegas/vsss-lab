## ADDED Requirements

### Requirement: per-role formation shaping

Each non-attacking responsibility SHALL have its own bounded formation potential, equal to the
geometric mean of `exp(-distance / 0.25)` over the robots currently assigned that role at that
role's existing target, rewarded as terminal-zeroed discounted potential shaping.

#### Scenario: support credit is measured independently

- **GIVEN** a configuration with `support_formation_coefficient > 0` and
  `coverage_formation_coefficient = 0`
- **WHEN** the reward ledger is accumulated over a rollout
- **THEN** `support_formation` SHALL be present in `reward_terms` and `coverage_formation`
  SHALL be exactly zero.

#### Scenario: coverage credit is measured independently

- **GIVEN** a configuration with `coverage_formation_coefficient > 0` and
  `support_formation_coefficient = 0`
- **WHEN** the reward ledger is accumulated over a rollout
- **THEN** `coverage_formation` SHALL be present in `reward_terms` and `support_formation`
  SHALL be exactly zero.

#### Scenario: terminal state carries no potential

- **GIVEN** a terminal transition
- **WHEN** the shaping is computed
- **THEN** both per-role potentials SHALL be zero on the terminal state.

### Requirement: configurable role hysteresis

The assignment SHALL apply a per-robot `role_switch_penalty` and an `role_emergency_margin`
gap that are explicit training knobs and SHALL be threaded through both the native and Python
assigners.

#### Scenario: a stronger penalty blocks marginal churn

- **GIVEN** an assigner with a high `role_switch_penalty`
- **WHEN** the state changes marginally
- **THEN** the roles SHALL remain unchanged.

#### Scenario: emergency override still rotates

- **GIVEN** an assigner with a high `role_switch_penalty`
- **WHEN** staying put costs at least `role_emergency_margin` more than the unpenalized best
- **THEN** the assignment SHALL rotate to the unpenalized best.

#### Scenario: invalid hysteresis values are rejected

- **WHEN** a configuration sets `role_switch_penalty` or `role_emergency_margin` negative
- **THEN** configuration loading SHALL reject it with a validation error.

### Requirement: legacy checkpoint compatibility

Checkpoints written before the per-role terms and hysteresis knobs existed SHALL remain loadable
when every new knob sits at its neutral default.

#### Scenario: neutral defaults load legacy checkpoints

- **GIVEN** a checkpoint whose stored configuration lacks the four new keys
- **WHEN** it is loaded with the four knobs at their neutral defaults
- **THEN** loading SHALL succeed.

#### Scenario: non-default knobs reject legacy checkpoints

- **GIVEN** a checkpoint whose stored configuration lacks the four new keys
- **WHEN** it is loaded with a non-default `role_switch_penalty`
- **THEN** loading SHALL fail with a fingerprint mismatch.
