## ADDED Requirements

### Requirement: Impasse restarts play rather than ending it

An impasse away from both goal areas SHALL be resolved by placing the ball on the free-ball
mark of the quadrant where it occurred and continuing the episode, at the interval the rules
specify. An episode SHALL NOT end because the ball stopped moving.

#### Scenario: Ball stalls in open play

- **GIVEN** a ball that has not moved for the configured impasse interval away from both
  goal areas
- **WHEN** the interval elapses
- **THEN** the ball is placed on the quadrant's free-ball mark at rest
- **AND** the episode continues with its clock and score intact

#### Scenario: Robot inside the clearance

- **WHEN** a robot stands within the clearance of the free-ball mark
- **THEN** it is moved to its own half before play resumes

#### Scenario: Impasse inside a goal area

- **WHEN** the impasse occurs inside a goal area
- **THEN** the ball is not repositioned, because that case is a goal kick and is not modelled

#### Scenario: Impasse remains observable

- **WHEN** free balls occur during a rollout
- **THEN** their count is reported, so removing the terminal does not hide the impasse rate
