## ADDED Requirements

### Requirement: the strike arrives aligned

When the straight path from the robot to the behind-ball acquisition point would enter the
ball's contact radius, the strike primitive SHALL first target a clearing waypoint on the exit
line beyond the acquisition point (`ball − (0.10 + clearing_distance) · exit`), so that the
robot stays outside the contact radius while it turns onto the exit direction. The alignment
gate (0.11 m, 0.60 rad) and the drive-through geometry SHALL NOT change.

#### Scenario: angled approach clears the ball

- **GIVEN** a ball within scoring range and a striker placed 60 degrees off the shooting line,
  such that the straight segment to the behind-ball acquisition point passes within the ball's
  contact radius
- **WHEN** the strike primitive selects its target
- **THEN** the selected target SHALL keep the robot's path outside the ball's contact radius
  until the robot is within the alignment gate

#### Scenario: on-line approach is unchanged

- **GIVEN** a ball on the shooting line with the striker directly behind it
- **WHEN** the strike primitive selects its target
- **THEN** the behind-ball acquisition point SHALL be selected exactly as before, with no
  clearing waypoint inserted

#### Scenario: alignment gate is not widened

- **GIVEN** any strike execution
- **WHEN** the drive-through decision is made
- **THEN** the drive-through SHALL engage only within the pre-existing acquisition envelope
  (0.11 m) and alignment tolerance (0.60 rad), and a chance the geometry forbids SHALL remain
  unconverted

### Requirement: the approach phase is configurable and reversible

The ball-clearing approach SHALL be controlled by a configuration flag that defaults to the new
behaviour and that restores the straight-line approach exactly when disabled. Checkpoints written
before the flag existed SHALL load when the flag is at its neutral default.

#### Scenario: disabled flag reproduces the straight-line approach

- **GIVEN** a configuration with the clearing approach disabled
- **WHEN** the strike primitive executes from an angled start
- **THEN** the selected targets SHALL equal the straight-line approach of the prior behaviour

#### Scenario: legacy checkpoint loads with the flag off

- **GIVEN** a checkpoint whose stored configuration lacks the clearing flag
- **WHEN** it is loaded with the flag at its neutral default
- **THEN** loading SHALL succeed

#### Scenario: non-default flag rejects a legacy checkpoint

- **GIVEN** a checkpoint whose stored configuration lacks the clearing flag
- **WHEN** it is loaded with the clearing approach enabled
- **THEN** loading SHALL fail with a fingerprint mismatch

### Requirement: dribbling and navigation are untouched

The change SHALL live entirely inside the strike path; `navigate` and `stop` SHALL produce
identical wheel commands before and after.

#### Scenario: navigation is byte-identical

- **WHEN** a navigate intent is executed with the clearing approach enabled
- **THEN** the wheel commands SHALL equal those produced with the approach disabled
