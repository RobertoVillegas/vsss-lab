## ADDED Requirements

### Requirement: Contextual idle-spin control

Training MUST distinguish sustained remote turn-in-place behavior from useful
orientation and ball control.

#### Scenario: Brief orientation correction

- **WHEN** a robot turns in place for less than the configured grace period
- **THEN** no idle-spin reward penalty is applied

#### Scenario: Ball-control turn

- **WHEN** a robot turns within the configured ball-control envelope
- **THEN** no idle-spin reward penalty is applied

#### Scenario: Sustained remote spin

- **WHEN** opposite wheel commands keep a robot slow and remote from the ball
  beyond the grace period
- **THEN** a bounded penalty proportional to turn intensity is applied
- **AND** idle-spin telemetry increases

### Requirement: Behavior-safe semantic selection

Checkpoint selection and phase promotion MUST require deterministic behavioral
eligibility in addition to semantic skill gates.

#### Scenario: Skill score rises while idle spin collapses behavior

- **WHEN** idle-spin ratio exceeds the configured ceiling
- **THEN** the checkpoint cannot replace a behavior-eligible semantic best
- **AND** the phase-promotion streak resets
