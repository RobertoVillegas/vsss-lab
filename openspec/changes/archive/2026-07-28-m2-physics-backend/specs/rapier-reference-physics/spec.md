## ADDED Requirements

### Requirement: Fixed-step differential drive
The reference backend SHALL advance six robots and one ball at the configured
timestep using saturated wheel-speed commands.

#### Scenario: Straight command
- **WHEN** equal positive wheel speeds are applied
- **THEN** the robot moves forward without commanded rotation

### Requirement: Field collisions and goals
The reference backend SHALL collide bodies with field boundaries while leaving
goal mouths open and emitting the scoring-team goal event when the ball crosses
a goal line inside the mouth.

#### Scenario: Detect blue goal
- **WHEN** the ball crosses the positive-x goal line inside the goal mouth
- **THEN** the score-blue value increments and GOAL_BLUE is emitted

### Requirement: Replay determinism
Identical config, initial state, and action sequence SHALL yield identical
canonical states and checksums on the same platform and locked version.

#### Scenario: Repeat action sequence
- **WHEN** the same 100-tick sequence runs twice
- **THEN** every final scalar and checksum is identical
