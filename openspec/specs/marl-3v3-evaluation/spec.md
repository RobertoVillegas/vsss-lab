# marl-3v3-evaluation Specification

## Purpose
TBD - created by archiving change m6-marl-baselines. Update Purpose after archive.
## Requirements
### Requirement: Three-agent curriculum
The system SHALL define C7 coordinated 3v0 and C8 3v3 against the deterministic
M4 heuristic with explicit seeds, horizons, rewards, and promotion thresholds.

#### Scenario: Run C7
- **WHEN** C7 resets
- **THEN** three shared-policy agents act independently and opponents remain inactive

#### Scenario: Run C8
- **WHEN** C8 resets
- **THEN** three shared-policy agents face the deterministic heuristic team

### Requirement: Blocking identity gate
The M6 gate SHALL reject policies or critics whose outputs are not equivariant
to teammate identity and slot permutations within configured tolerance.

#### Scenario: Run identity evaluation
- **WHEN** team agent order is permuted over fixed canonical states
- **THEN** actor actions and critic values permute equivalently

### Requirement: Better-than-random gate
The M6 evaluator SHALL compare fixed-seed shared-policy team progress against
random wheel actions under identical initial states.

#### Scenario: Evaluate a competent policy
- **WHEN** mean shared-policy progress exceeds the random baseline by the configured margin
- **THEN** the executable M6 gate passes and records both measurements

