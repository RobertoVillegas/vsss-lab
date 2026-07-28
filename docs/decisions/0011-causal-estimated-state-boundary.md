# ADR 0011: Causal estimated-state boundary

- Status: accepted
- Date: 2026-07-28

## Decision

VSSS Lab represents camera measurements, causal estimates, physics truth, and
trajectory predictions as different contract types. Every camera-derived value
includes capture and arrival timing; every estimate records its source sequence,
effective time, covariance, acceptance status, and rejection reason.

Policies may consume only measurements and estimates available by their decision
deadline. Physics truth remains authoritative for simulation and evaluation but
is not interchangeable with an estimate. Prediction begins from one estimate
and never reads later simulator frames. Post-hoc prediction error is analysis
data and cannot enter the policy observation API.

The reference ball filter is a CPU constant-acceleration Kalman filter. The
reference robot filter is a CPU differential-drive EKF with wrapped heading
innovations. Calibration, innovation gates, and maximum prediction age are
versioned inputs rather than policy constants.

## Consequences

Simulation can reproduce camera latency, noise, outliers, and occlusion without
corrupting canonical truth. Hardware and simulation share the estimation
boundary. CUDA acceleration is permitted only behind equivalent contracts after
profiling identifies a vision bottleneck.
