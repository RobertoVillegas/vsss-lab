## ADDED Requirements

### Requirement: Timestamped causal measurements

The perception boundary SHALL identify capture time, arrival time, source
sequence, calibration profile, detections, visibility, and association
confidence for every camera measurement.

#### Scenario: Delayed camera frame arrives

- **WHEN** a frame is processed after a newer decision deadline
- **THEN** its original capture time remains available and it is not represented
  as a current measurement

### Requirement: Explicit estimated state

The estimator SHALL produce ball position, velocity, and acceleration plus robot
pose, linear velocity, and angular velocity with covariance, age, visibility,
association confidence, and source measurement range.

#### Scenario: Policy receives a physical observation

- **WHEN** a decision is requested from camera-derived state
- **THEN** the policy receives the estimate aligned to that deadline rather than
  canonical truth or raw unaligned detections

### Requirement: Kalman and EKF reference estimators

The CPU reference SHALL estimate the ball using a constant-acceleration Kalman
filter and robots using a differential-drive EKF with normalized angular
innovation.

#### Scenario: Robot heading crosses pi

- **WHEN** consecutive measurements cross the `-pi`/`pi` boundary
- **THEN** the EKF applies the shortest angular innovation without a discontinuity

### Requirement: Outlier and dropout handling

The estimator SHALL gate implausible innovations, record rejection reasons,
predict through bounded dropouts, and mark estimates unavailable after the
configured maximum age.

#### Scenario: Ball detection is an outlier

- **WHEN** measurement innovation exceeds the configured statistical gate
- **THEN** the update is rejected, the causal prediction is retained, and the
  rejection is observable

#### Scenario: Ball remains occluded

- **WHEN** no accepted measurement arrives beyond the maximum prediction age
- **THEN** the ball estimate becomes stale/unavailable instead of extrapolating
  indefinitely

### Requirement: Ground-truth isolation

Canonical simulator truth SHALL remain separate from measurements and estimates
and SHALL be available only to simulation adjudication and evaluation paths.

#### Scenario: Evaluate a noisy estimate

- **WHEN** camera perturbations alter measurements and estimated state
- **THEN** scoring uses unchanged canonical truth while evaluation can compare
  the two representations
