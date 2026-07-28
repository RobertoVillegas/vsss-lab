## Why

The fast backend is deterministic but its fidelity claims need traceable
reference parameters, phenomenon-level scenarios, and explicit tolerances.

## What Changes

- Record extracted Julio simulator/thesis geometry and dynamics provenance.
- Add machine-readable straight, turn, and free-ball golden scenarios.
- Measure Rapier trajectories against reference equations and publish errors.
- Treat unavailable legacy runtime behavior as an explicit evidence gap.

## Capabilities

### New Capabilities

- `reference-physics-calibration`: versioned source parameters, scenarios,
  metrics, tolerances, and reproducible reports.

## Impact

Adds calibration fixtures, a dependency-free report command, tests, evidence,
and a gate without introducing ROS 1 or Gazebo Classic into the main workspace.
