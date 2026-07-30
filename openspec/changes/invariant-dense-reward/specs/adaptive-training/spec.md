## ADDED Requirements

### Requirement: Terminal-invariant potential shaping

Potential shaping SHALL treat the potential of a terminal state as zero, so that
adding or removing the shaping term cannot change which policy is optimal.

#### Scenario: Episode ends in a favorable geometry

- **WHEN** an episode reaches a terminal state with a high potential
- **THEN** the shaping term of that transition pays nothing for the final geometry

#### Scenario: Shaping cannot reorder outcomes

- **GIVEN** two episodes whose terminal outcomes differ
- **WHEN** the shaping coefficient changes
- **THEN** their ordering by return does not change

### Requirement: Signed contact impulse

Useful-contact reward SHALL remain an impulse taken on the contact edge, so it
cannot become a rate reward for ball advancement, and SHALL carry the sign of the
ball's velocity change with respect to the attacking direction.

#### Scenario: Contact envelope oscillation

- **WHEN** a robot repeatedly leaves and re-enters the contact envelope on velocity
  noise, without advancing the ball
- **THEN** those impulses sum to zero rather than accumulating

#### Scenario: Ball moved the wrong way

- **WHEN** contact moves the ball against the attacking direction
- **THEN** the cost equals what the same magnitude toward the goal would earn

#### Scenario: Sustained contact is not a rate

- **WHEN** contact persists across consecutive decisions
- **THEN** no further impulse is paid until contact is re-established
