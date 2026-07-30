## ADDED Requirements

### Requirement: Isotropic heading precision

The angular precision of a requested heading SHALL NOT depend on the direction
requested. Heading sampling, log-probability, and entropy SHALL be evaluated as
circular quantities, and concentration SHALL be produced per state rather than
held as one value for the whole policy.

#### Scenario: Axis-aligned and diagonal request at one concentration

- **GIVEN** two states whose requested headings differ only in direction, one
  aligned with the goal line and one diagonal
- **WHEN** both are sampled at the same concentration
- **THEN** the circular deviation of the executed heading is equal within
  tolerance

#### Scenario: Angular exploration is measured

- **WHEN** the policy becomes more concentrated in direction
- **THEN** the reported entropy decreases

#### Scenario: Wrap remains continuous

- **WHEN** a sampled heading crosses from +π to −π
- **THEN** its log-probability and its executed direction remain continuous

### Requirement: Intensity is reachable and respected

A requested drive intensity SHALL be attainable across its declared interval and
SHALL inform the reachability model that selects a strike intercept.

#### Scenario: Teacher target inside the interval

- **WHEN** a teacher demonstrates full authority
- **THEN** its distillation target lies inside the interval the policy can express

#### Scenario: Reduced authority selects a reachable intercept

- **GIVEN** a moving ball and a request below full authority
- **WHEN** the executor selects a strike intercept
- **THEN** the selected point is reachable at the requested authority

### Requirement: Heading contract is recorded

A checkpoint SHALL record the heading parameterization it was trained under, and
SHALL NOT load under a different one.

#### Scenario: Loading across parameterizations

- **WHEN** a policy trained on the bounded Cartesian heading is loaded by a run
  configured for the circular heading
- **THEN** the load is rejected rather than reinterpreted
