## ADDED Requirements

### Requirement: Full-ball edge-triggered goal
The reference backend SHALL score exactly once when the complete ball crosses a
goal line while inside the goal mouth.

#### Scenario: Ball center alone crosses
- **WHEN** the ball center crosses the goal line but its trailing edge does not
- **THEN** no goal is scored

#### Scenario: Complete ball crosses
- **WHEN** the trailing edge crosses the goal line inside the mouth
- **THEN** exactly one goal event and score increment are produced
