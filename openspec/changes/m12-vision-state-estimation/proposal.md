## Why

M12 must convert delayed, noisy, and occasionally missing overhead-camera
detections into the same structured state used by policies. Julio De La Torre's
thesis demonstrates a useful separation: Kalman estimation for ball motion, EKF
estimation for differential-drive robots, and an explicit ball trajectory
prediction used for interception. VSSS Lab currently exposes exact simulator
state and velocity, so policies can learn anticipation implicitly, but it cannot
yet reproduce what a physical controller would actually observe.

## What Changes

- Introduce a timestamped perception-to-estimated-state boundary with confidence,
  visibility, association, and covariance.
- Add a deterministic CPU reference estimator: constant-acceleration Kalman
  filter for the ball and differential-drive EKF for robots.
- Add innovation gating for outlier rejection and bounded prediction through
  short camera dropouts.
- Add a ball trajectory predictor driven only by the current observed/estimated
  state and calibrated field physics.
- Record the exact estimated and predicted values visible to a policy.
- Add optional viewer overlays for predicted path, future-time markers,
  goalkeeper interception, uncertainty, and post-hoc prediction error.
- Add seeded camera latency, measurement noise, occlusion, and misassociation
  evaluation using simulator ground truth.
- Keep future simulator truth inaccessible to policy observations.

## Capabilities

### New Capabilities

- `vision-state-estimation`: timestamped detections, association, Kalman/EKF
  state estimation, uncertainty, and dropout handling.
- `predictive-ball-trajectory`: present-state-only ball projection and
  interception queries.

### Modified Capabilities

- `simulation-observer-stream`: carry optional perception, estimate, prediction,
  and post-hoc error diagnostics.
- `training-replay-web`: display the exact estimated state and prediction that
  were available at each recorded decision.
- `seeded-domain-randomization`: support camera-specific perturbations without
  corrupting canonical truth.

## PRD Milestone

M12 — Visión y hardware.

## Non-Goals

- Sending commands to physical robots.
- Replacing Rapier or using a second training physics backend.
- Feeding actual future simulator frames to a policy.
- Requiring OpenCV-CUDA before a CPU profile proves vision is the bottleneck.
- Making predictive features mandatory before a controlled ablation.
