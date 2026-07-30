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

### Requirement: Contact reward attributable and signed

Useful-contact reward SHALL be signed with respect to the attacking direction and
SHALL be attributed to the contribution of the controlled robot, rather than
triggered by a team-level contact edge.

#### Scenario: Contact envelope oscillation

- **WHEN** a robot repeatedly leaves and re-enters the contact envelope without
  advancing the ball
- **THEN** it accumulates no positive return

#### Scenario: Ball moved the wrong way

- **WHEN** a controlled robot moves the ball against its attacking direction
- **THEN** the cost equals what the same magnitude toward the goal would earn

### Requirement: Coherent terminal pressure

Episode time cost, the draw terminal, and the stagnation terminal SHALL be
expressed on one scale, so no stalemate is the cheapest available outcome.

#### Scenario: Stalemate versus contest

- **WHEN** an episode avoids the stagnation terminal without attempting a goal
- **THEN** its return does not exceed that of a comparable contested episode
