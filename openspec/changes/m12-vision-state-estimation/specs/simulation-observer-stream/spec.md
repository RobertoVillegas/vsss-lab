## ADDED Requirements

### Requirement: Perception and prediction diagnostics

The observer schema SHALL support optional timestamped detections, causal
estimates, predictions, uncertainty, and rejection diagnostics without changing
canonical state semantics or blocking simulation.

#### Scenario: Record what a policy observed

- **WHEN** a policy acts from an estimated state and projected trajectory
- **THEN** the replay records those exact decision-time values independently from
  canonical truth and later analytical error
