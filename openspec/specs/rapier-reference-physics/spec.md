# rapier-reference-physics Specification

## Purpose
TBD - created by archiving change m2-physics-backend. Update Purpose after archive.
## Requirements
### Requirement: Fixed-step differential drive
The reference backend SHALL advance six robots and one ball at the configured
timestep using saturated wheel-speed targets and force-derived bounded actuator
response.

#### Scenario: Straight command
- **WHEN** equal positive wheel-speed targets are applied
- **THEN** the robot accelerates forward without commanded rotation

#### Scenario: Abrupt wheel target
- **WHEN** a wheel target changes by more than one-step actuator capacity
- **THEN** applied wheel speed approaches it without exceeding the derived delta

### Requirement: Field collisions and goals
The reference backend SHALL collide robots with field and goal-box boundaries
while emitting the scoring-team goal event when the ball crosses a goal line
inside the mouth.

#### Scenario: Detect blue goal
- **WHEN** the ball crosses the positive-x goal line inside the goal mouth
- **THEN** the score-blue value increments and GOAL_BLUE is emitted

#### Scenario: Robot enters a goal mouth
- **WHEN** a robot drives through a goal mouth toward its side or back boundary
- **THEN** collision geometry keeps it inside the modeled field and goal box

### Requirement: Replay determinism
Identical config, initial state, and action sequence SHALL yield identical
canonical states and checksums on the same platform and locked version.

#### Scenario: Repeat action sequence
- **WHEN** the same 100-tick sequence runs twice
- **THEN** every final scalar and checksum is identical

### Requirement: Sustained robot contact separation
The reference backend SHALL prevent commanded robots from materially
interpenetrating during sustained contact.

#### Scenario: Two robots drive head-on
- **WHEN** two 75 mm robots continuously command forward motion into each other
  for 1,000 fixed steps
- **THEN** their axis-aligned center separation remains at least 73.9 mm

#### Scenario: Robot drives into ball
- **WHEN** a robot continuously commands forward motion into a stationary ball
  for 1,000 fixed steps
- **THEN** the ball center remains outside the robot collider within the
  committed 1.1 mm contact tolerance

### Requirement: Full-ball edge-triggered goal
The reference backend SHALL score exactly once when the complete ball crosses a
goal line while inside the goal mouth.

#### Scenario: Ball center alone crosses
- **WHEN** the ball center crosses the goal line but its trailing edge does not
- **THEN** no goal is scored

#### Scenario: Complete ball crosses
- **WHEN** the trailing edge crosses the goal line inside the mouth
- **THEN** exactly one goal event and score increment are produced

### Requirement: Rule-aware field collisions

The reference backend SHALL contain robots and the ball using the configured
walls and goals plus the calibrated 70 mm VSSS corner chamfers.

#### Scenario: Ball reaches a field corner

- **WHEN** a ball travels diagonally toward a playing-field corner
- **THEN** the chamfer contact deflects it before it reaches the square corner
