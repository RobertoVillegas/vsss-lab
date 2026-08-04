## ADDED Requirements

### Requirement: legal kickoff ball placement

Every full-match kickoff SHALL start the ball uniformly inside the center circle
of radius `full_match_kickoff_radius` metres, and that radius SHALL be validated
to at most the VSSS Rule 7 circle of 0.20 m.

#### Scenario: ball inside the circle

- **GIVEN** a seeded full-match reset
- **WHEN** the snapshot is drawn
- **THEN** the ball SHALL satisfy `hypot(x, y) <= full_match_kickoff_radius`.

#### Scenario: radius ceiling

- **WHEN** a configuration sets `full_match_kickoff_radius` above 0.20
- **THEN** configuration loading SHALL reject it with a validation error.

#### Scenario: exact center

- **GIVEN** `full_match_kickoff_radius = 0`
- **WHEN** the snapshot is drawn
- **THEN** the ball SHALL rest at `(0, 0)`.

### Requirement: legal defender placement

Every non-attacking robot SHALL start outside the center circle at kickoff.

#### Scenario: defenders outside the circle

- **GIVEN** a seeded full-match reset with the default radius
- **WHEN** the snapshot is drawn
- **THEN** every robot in the defending half SHALL satisfy
  `hypot(x, y) > 0.20` (outside the circle).

### Requirement: deterministic seeded resets

Two full-match resets with the same seed SHALL produce identical ball and robot
positions, so paired evaluation and distillation remain reproducible.

#### Scenario: seeded determinism

- **GIVEN** two environments reset with the same seed and radius
- **WHEN** both snapshots are drawn
- **THEN** the ball and robot poses SHALL be identical.
