## ADDED Requirements

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
