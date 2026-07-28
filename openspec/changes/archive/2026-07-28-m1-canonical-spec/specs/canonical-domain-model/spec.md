## ADDED Requirements

### Requirement: Explicit canonical units
Every physical scalar in the canonical domain model SHALL use a named SI unit
type, and each unit SHALL serialize as its SI scalar.

#### Scenario: Serialize a distance
- **WHEN** a distance of 1.5 metres is serialized
- **THEN** the resulting scalar is `1.5` and its Rust type is `Distance`

### Requirement: Complete match state
The system SHALL represent a versioned match tick containing time, score, one
ball, exactly six uniquely identified robots, and event flags.

#### Scenario: Validate a complete state
- **WHEN** a state contains schema version 1, finite values, and robot IDs 0–5
- **THEN** canonical validation succeeds

#### Scenario: Reject duplicate identities
- **WHEN** two robots in a state have the same physical robot ID
- **THEN** canonical validation returns a stable robot identity error

### Requirement: Canonical actions and events
The system SHALL represent wheel-velocity and body-velocity commands without
binding them to a physics implementation, and SHALL expose version-stable event
bits.

#### Scenario: Validate body velocity
- **WHEN** a body-velocity command contains finite linear and angular velocities
- **THEN** canonical action validation succeeds

### Requirement: Complete match configuration
The system SHALL represent geometry, physical properties, timing, actuator
limits, reset rules, randomization ranges, seed, and backend settings.

#### Scenario: Reject incompatible timing
- **WHEN** control period is shorter than the fixed physics timestep
- **THEN** configuration validation fails at the control-period field
