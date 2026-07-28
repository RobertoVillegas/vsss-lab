# seeded-domain-randomization Specification

## Purpose
TBD - created by archiving change m11-domain-randomization. Update Purpose after archive.
## Requirements
### Requirement: Seeded multi-domain perturbations
The environment SHALL sample friction, restitution, six motor multipliers,
latency, drops, and observation noise from versioned ranges at reset.

#### Scenario: Repeat a randomized episode
- **WHEN** seed, configuration, actions, and build are unchanged
- **THEN** realized parameters and canonical trajectory are identical

### Requirement: Canonical truth separation
Observation noise SHALL affect policy input but SHALL NOT corrupt canonical
ground truth, replay, scoring, or event adjudication.

#### Scenario: Observe a noisy state
- **WHEN** observation noise is nonzero
- **THEN** the evaluator can still score the unmodified canonical state

