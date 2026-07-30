# Policy observability

## ADDED Requirements

### Requirement: Strategic intent is distinct from actuation

Replay ticks SHALL preserve categorical policy intent separately from requested
and physically applied wheel motion.

#### Scenario: primitive replay

- **WHEN** an M24 policy selects a directed strike
- **THEN** the replay SHALL record its action index, skill, direction,
  confidence, alternatives, execution phase, and target
- **AND** the existing actuator telemetry SHALL remain available

### Requirement: Selected actor inspection

The viewer SHALL expand one selected actor and synchronize its policy,
primitive, actuator, and physical state.

#### Scenario: developer selects an actor

- **WHEN** an actor card is selected
- **THEN** the field SHALL overlay that actor's target and exit direction
- **AND** the card SHALL expose confidence and top alternatives

### Requirement: Navigable behavior timeline

The replay transport SHALL expose primitive segments and event markers.

#### Scenario: event marker is activated

- **WHEN** a developer activates a goal, contact, pass, interception, or reset
  marker
- **THEN** playback SHALL pause and seek to the corresponding frame

### Requirement: Categorical exploration

Primitive runs SHALL report exploration using categorical metrics.

#### Scenario: primitive metrics are rendered

- **WHEN** the action parser is primitive
- **THEN** the dashboard SHALL show normalized entropy and action-family usage
- **AND** it SHALL NOT label a fixed compatibility buffer as wheel log standard
  deviation
