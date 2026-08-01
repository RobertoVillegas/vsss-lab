## ADDED Requirements

### Requirement: Scenarios carry a match situation

Generated scenarios SHALL vary the starting score difference and the time remaining, so that the
context channels carrying them are not constant across the training distribution.

#### Scenario: The score channel carries variance

- **WHEN** observations are sampled across the scenario distribution
- **THEN** the score-difference channel takes more than one value, and leads as well as deficits
  appear

#### Scenario: The clock channel carries variance

- **WHEN** observations are sampled across the scenario distribution
- **THEN** the time-remaining channel spans the match rather than its first few per cent

### Requirement: The situation is an audited difficulty axis

The match situation SHALL be declared as a difficulty axis and audited by the same tool as every
other axis, so that an axis that does not order difficulty is reported rather than assumed.

#### Scenario: The axis does not order difficulty

- **WHEN** the audit finds no gradient, an inverted gradient, or invalid generation along the
  situation axis
- **THEN** the axis is reported with that verdict and is not used as a ladder until it is fixed

### Requirement: Holdouts are regenerated when the distribution changes

Immutable holdouts SHALL be regenerated under a new generator revision when the scenario
distribution changes, and results across the revision boundary SHALL NOT be compared as though
they came from one distribution.

#### Scenario: Comparing across a distribution change

- **WHEN** a score before the situation axis is compared to a score after it
- **THEN** the comparison is reported as spanning two generator revisions, not as a single trend
