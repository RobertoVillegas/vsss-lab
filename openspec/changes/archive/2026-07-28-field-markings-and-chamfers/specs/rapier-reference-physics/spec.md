## ADDED Requirements

### Requirement: Rule-aware field collisions

The reference backend SHALL contain robots and the ball using the configured
walls and goals plus the calibrated 70 mm VSSS corner chamfers.

#### Scenario: Ball reaches a field corner

- **WHEN** a ball travels diagonally toward a playing-field corner
- **THEN** the chamfer contact deflects it before it reaches the square corner
