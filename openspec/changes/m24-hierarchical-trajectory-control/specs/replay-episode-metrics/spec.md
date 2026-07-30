# Replay episode metrics

## ADDED Requirements

### Requirement: Episode identity

Every replay tick SHALL identify its episode within the captured replay.

#### Scenario: stagnation reset

- **WHEN** a captured rollout resets after stagnation
- **THEN** subsequent ticks SHALL carry a new episode number
- **AND** prediction errors SHALL NOT compare samples across the reset

### Requirement: Trajectory diagnostics

Offline analysis SHALL distinguish state-estimation error from trajectory,
contact, and reacquisition performance.

#### Scenario: low-motion valley

- **WHEN** the ball is stationary and all active robots remain outside contact
  range
- **THEN** analysis SHALL report the duration and nearest-ball distance
- **AND** it SHALL not infer that zero movement means zero policy action
