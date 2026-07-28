## ADDED Requirements

### Requirement: Seeded camera perturbations

Simulation SHALL generate reproducible camera latency, measurement noise,
occlusion, false detections, and marker-association ambiguity while preserving
canonical truth.

#### Scenario: Repeat a camera-derived episode

- **WHEN** seed, truth trajectory, calibration, and perturbation profile are
  unchanged
- **THEN** measurements, accepted/rejected updates, estimated states, and
  policy-visible predictions are identical
