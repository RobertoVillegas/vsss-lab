## ADDED Requirements

### Requirement: A carry is worth a stated fraction of a goal

The carry gradient's coefficient and the goal coefficient SHALL be set as one ratio and recorded
as such, because what a chance is worth against what a goal is worth is the design and not two
independent knobs.

#### Scenario: Reading the balance

- **WHEN** the configuration is inspected
- **THEN** the maximum a carry can pay across an episode, and what a goal pays, are both legible
  without running anything

#### Scenario: A carry that rivals a goal

- **WHEN** the carry's bound approaches the goal coefficient
- **THEN** the configuration is rejected: carrying the ball to a good position is not scoring,
  and a reward that says otherwise invites the team to hold position instead of shooting

### Requirement: The ratio is chosen by ablation

The ratio SHALL be selected from at least three measured settings, reported against goals scored
per minute and against how often the ball reaches a position the primitives can convert from —
not against total reward, which the shaping term inflates by construction.

#### Scenario: Selecting on total reward

- **WHEN** a setting is proposed because it earned more reward
- **THEN** it is rejected as evidence, because raising the carry coefficient raises reward
  without necessarily scoring
