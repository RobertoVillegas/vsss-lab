## ADDED Requirements

### Requirement: Difficulty modulates the scored demand

Each difficulty axis SHALL move the quantity its family is scored on, so that lowering
difficulty makes a failing family learnable. A generator change that alters scenario
geometry SHALL bump the generator revision, and holdouts of different revisions SHALL NOT
be compared.

#### Scenario: Easy end of a clearance drill

- **GIVEN** the lowest difficulty of the clearance family
- **WHEN** its scenario is compiled
- **THEN** the ball starts materially closer to the threshold it must cross than at the
  highest difficulty
- **AND** it still starts inside the defensive third

#### Scenario: Family failing at every band

- **WHEN** a family scores zero at its lowest difficulty band
- **THEN** the curriculum has no lower rung to offer and the family is unlearnable, so the
  generator is at fault rather than the policy

#### Scenario: Comparing across revisions

- **WHEN** an evaluation is read against one from a different generator revision
- **THEN** the two are reported as incomparable rather than merged
