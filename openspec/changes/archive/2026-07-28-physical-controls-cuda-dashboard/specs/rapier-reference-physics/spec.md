## MODIFIED Requirements

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
