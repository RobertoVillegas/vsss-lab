## Context

Simulation has authoritative state at the current physics tick. A physical VSSS
system instead receives camera frames captured in the past, extracts imperfect
detections, associates visual tags, and estimates latent velocity and
orientation. These representations must coexist without confusing canonical
truth, what the controller observed, and later analytical hindsight.

The thesis reference uses a constant-acceleration Kalman model for the ball and a
nonlinear EKF for robot position, orientation, linear velocity, and angular
velocity. Its traditional goalkeeper projects the ball path to an interception
point. The same concepts are useful here, but the estimator and predictor must
remain independent from rewards, policies, rendering, and authoritative scoring.

## Decisions

### 1. Three explicitly different state classes

- `CanonicalTruth`: authoritative simulator state or unavailable on hardware.
- `EstimatedState`: causal estimate from measurements available by decision time.
- `Prediction`: causal projection from one estimated state and a pinned model.

Post-hoc ground-truth comparisons are evaluation artifacts and SHALL NOT be
readable through the policy observation API.

### 2. Time is part of every perception contract

Measurements carry monotonic capture time, arrival time, source sequence, and
calibration identity. Estimates carry their effective time, update time, age,
and source measurement range. The policy consumes an estimate aligned to its
decision deadline rather than whichever camera frame arrived last.

### 3. CPU reference before acceleration

The first implementation is deterministic and testable on CPU. A CUDA vision
front end may be added only after profiling shows image processing is the
bottleneck. Filter state transition and update semantics must remain equivalent
within documented tolerances across CPU/CUDA variants.

### 4. Causal filters

- Ball: state `[x, vx, ax, y, vy, ay]`, constant-acceleration transition, measured
  `[x, y]`.
- Robot: state `[x, y, theta, v, omega]`, differential-drive nonlinear
  transition, measured pose, angle-normalized innovations.
- Association: marker identity, confidence, visibility, and ambiguity remain
  explicit rather than silently rewriting logical robot identity.
- Outliers: innovation/Mahalanobis gating with recorded rejection reason.
- Dropouts: prediction-only updates are bounded by maximum age; stale entities
  become unavailable rather than indefinitely extrapolated.

Exact matrices, covariance initialization, and thresholds belong to versioned
calibration profiles, not hard-coded policy logic.

### 5. Two prediction fidelities

The trajectory API exposes:

1. an analytic short-horizon projection for low-cost policy features;
2. an optional collision-aware projection using pinned field geometry, damping,
   restitution, goals, and corner chamfers for viewer and interception analysis.

Both consume only the state available at the selected decision tick. Neither may
peek at future replay or simulator frames.

### 6. Prediction is optional policy input

The baseline MAPPO observation continues using current relative position and
velocity. A versioned experiment may add projected ball positions or
time-to-intercept. Promotion requires an ablation against the baseline with
identical seeds, budgets, and evaluation fixtures.

### 7. Observability without semantic ambiguity

Replay frames may contain:

- camera detections;
- filter estimate and covariance;
- accepted/rejected measurement indicators;
- predicted path sampled at configured future offsets;
- interception candidates;
- prediction error computed only after matching truth becomes available.

The viewer labels these layers `TRUTH`, `MEASURED`, `ESTIMATED`, and `PREDICTED`.
Recorded frames preserve what the policy saw; later error enrichment is stored
as a separate analysis artifact.

## Validation Strategy

- Golden numerical Kalman/EKF transitions and angle-wrap cases.
- Seeded noisy-camera replay produces byte-stable estimates on the reference
  implementation.
- Occlusion and outlier fixtures prove bounded extrapolation and rejection.
- Prediction tests cover damping, walls, goals, and 70 mm corner chamfers.
- A no-leakage contract test mutates future truth while holding current
  measurements fixed and requires identical policy observations/predictions.
- Ground-truth comparison reports position, velocity, heading, covariance
  calibration, rejection rate, estimate age, and interception error.
- Viewer tests seek exact ticks and verify layer/time alignment.

## Compatibility and Migration

Perception and prediction fields are optional observer extensions. Existing
canonical state, replay readers, headless training, and checkpoints remain
valid. Policy observation schema changes require a new explicit version and do
not silently alter existing models.

## Risks and Mitigations

- Model mismatch during impacts → reset/inflate covariance on detected contact
  and compare analytic versus collision-aware forecasts.
- Confident wrong association → preserve association confidence and ambiguity;
  never bind marker permanently to policy role.
- Training benefits from unavailable truth → enforce causal API and no-leakage
  tests.
- Viewer path mistaken for network reasoning → label model source, horizon, and
  uncertainty; do not call it a neural-network prediction unless it is one.
- Excessive CPU cost → batch filters, profile first, then accelerate only the
  demonstrated bottleneck.

## Rollback

Disable the optional perception observer and predictive observation adapter.
Canonical physics, current-state MAPPO observations, and ordinary replay remain
unchanged.
