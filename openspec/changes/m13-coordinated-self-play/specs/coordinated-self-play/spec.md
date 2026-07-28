## ADDED Requirements

### Requirement: Dense coordination without fixed identities

Training SHALL expose configurable teammate-congestion and defensive-coverage
signals that remain invariant to teammate identity and storage ordering.

#### Scenario: Teammates make sustained body contact

- **WHEN** two controlled robots remain closer than the configured spacing
- **THEN** the team receives a bounded congestion cost
- **AND** no permanent role is assigned to either robot

#### Scenario: The ball threatens the own goal

- **WHEN** the ball moves into the configured defensive activation region
- **THEN** progress by the nearest teammate toward goal-mouth coverage is rewarded
- **AND** moving away from coverage receives the inverse signal

### Requirement: Reward changes start fresh lineage

Reward coefficients SHALL participate in the training configuration fingerprint.

#### Scenario: Start coordinated training after the baseline

- **WHEN** the coordinated reward differs from the baseline
- **THEN** training starts a new run and checkpoint lineage
- **AND** the baseline remains available for paired evaluation
