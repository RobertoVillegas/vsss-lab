## ADDED Requirements

### Requirement: An angled chance is convertible

The action set SHALL contain a way to finish a chance the ball does not already share a line
with the goal for. A primitive that must first move away from the ball to line up does not
satisfy this, because the chance is gone by the time it has.

#### Scenario: A chance from the side

- **GIVEN** the ball within scoring range and the robot well off the line between the ball and
  the goal
- **WHEN** the policy asks to finish
- **THEN** the primitive approaches from the side the shot needs without first retreating, and
  the chance is convertible at a rate that is not zero

#### Scenario: Measured against dribbling

- **WHEN** the finishing primitive is compared to driving at the goal on the same drills
- **THEN** it is not worse on the shooting line, which is the case it exists for

### Requirement: Dribbling remains a choice

The finishing primitive SHALL NOT be introduced by removing or degrading the ability to drive
the ball forward. Driving at the goal measured better than striking in several cases, and a set
that can only finish has fewer options than one that can also carry.

#### Scenario: Both remain available

- **WHEN** the action space is enumerated after the change
- **THEN** driving toward a requested heading is still expressible, and the policy chooses
  between it and finishing rather than being left one route

### Requirement: The primitive is measured before it is ported

The finishing primitive SHALL NOT be ported to the native crate until its behaviour is settled
against the drills, because porting a primitive that is still being shaped costs two changes for
one and freezes an equivalence test against a moving reference.

#### Scenario: Porting early

- **WHEN** a port is proposed while the primitive's geometry is still under ablation
- **THEN** it is deferred until the ablation has chosen
