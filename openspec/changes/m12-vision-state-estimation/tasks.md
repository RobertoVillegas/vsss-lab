## 1. Contracts and calibration

- [x] 1.1 Record the estimated-state/prediction boundary in an ADR
- [x] 1.2 Define versioned measurement, association, estimate, covariance, and
  prediction schemas
- [x] 1.3 Define camera/filter calibration profiles and stable validation errors
- [ ] 1.4 Add contract tests proving canonical truth and estimated state cannot be
  confused

## 2. Deterministic state estimation

- [x] 2.1 Implement a CPU ball constant-acceleration Kalman filter
- [x] 2.2 Implement a CPU differential-drive robot EKF with wrapped angles
- [ ] 2.3 Implement innovation gating, rejection diagnostics, and bounded
  prediction-only dropout handling
- [x] 2.4 Implement confidence-aware marker association input without binding
  marker identity to policy role
- [ ] 2.5 Add golden transition, covariance, outlier, occlusion, and seed tests

## 3. Predictive trajectory

- [x] 3.1 Implement analytic present-state ball projection
- [x] 3.2 Implement collision-aware projection against canonical walls, goals,
  damping, restitution, and chamfers
- [x] 3.3 Implement goalkeeper-line and general segment interception queries
- [ ] 3.4 Add uncertainty propagation and stale-estimate limits
- [ ] 3.5 Add a future-truth mutation test that blocks information leakage

## 4. Simulation and policy evaluation

- [x] 4.1 Generate timestamped synthetic camera measurements from canonical truth
- [x] 4.2 Add seeded latency, noise, occlusion, false detection, and
  misassociation profiles
- [ ] 4.3 Record exact estimates used at each policy decision
- [ ] 4.4 Add a versioned optional predictive observation adapter
- [ ] 4.5 Run identical-budget MAPPO ablation with and without predictive features
- [ ] 4.6 Report estimation and interception metrics against hidden truth

## 5. Replay viewer

- [ ] 5.1 Render selectable truth, measured, estimated, and predicted layers
- [ ] 5.2 Render path samples at labeled future offsets and an uncertainty band
- [ ] 5.3 Render goalkeeper interception point and time
- [ ] 5.4 Render accepted/rejected measurements, estimate age, visibility, and
  association confidence
- [ ] 5.5 Store post-hoc prediction error separately from policy-visible replay
  data

## 6. Physical camera and acceleration

- [ ] 6.1 Replay a recorded overhead-camera fixture through the CPU pipeline
- [ ] 6.2 Integrate the estimator with the ROS camera bridge
- [ ] 6.3 Profile decode, segmentation, association, filter, and transfer stages
- [ ] 6.4 Add CUDA vision only if CPU image processing is a measured bottleneck
- [ ] 6.5 Verify CPU/CUDA estimate agreement within calibrated tolerances

## 7. Gates and delivery

- [ ] 7.1 Define M12 accuracy, latency, staleness, and safety thresholds from
  recorded-camera evidence
- [ ] 7.2 Run doctor, build, test, lint, OpenSpec, and relevant container gates
- [ ] 7.3 Record benchmarks, artifacts, known limitations, and rollback evidence
- [ ] 7.4 Archive the completed OpenSpec and create small signed commits
