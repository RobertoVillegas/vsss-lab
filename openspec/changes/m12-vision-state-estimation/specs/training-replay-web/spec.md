## ADDED Requirements

### Requirement: Estimated and predicted state inspection

The replay viewer SHALL allow independent display of truth, measurements,
estimated state, predicted trajectory, uncertainty, estimate age, association
confidence, and post-hoc error when those layers are present.

#### Scenario: Inspect a goalkeeper decision

- **WHEN** the selected frame contains a trajectory interception query
- **THEN** the viewer aligns the predicted path, labeled future offsets,
  interception point, time, uncertainty, and actor action to the same decision
  tick

### Requirement: Honest prediction labeling

The viewer SHALL identify the source model and SHALL NOT describe a physical
projection as a neural-network prediction.

#### Scenario: Display an analytic projection

- **WHEN** the trajectory came from the calibrated analytic model
- **THEN** the overlay labels it as model projection and distinguishes it from
  policy value or learned outputs
