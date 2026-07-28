# reference-physics-calibration Specification

## Requirements

### Requirement: Traceable reference parameters
Every adopted reference parameter SHALL record value, SI unit, source, and
whether it was measured, extracted, or inferred.

#### Scenario: Audit a calibration
- **WHEN** a report names a reference value
- **THEN** its provenance and uncertainty are discoverable in the committed manifest

### Requirement: Phenomenon-level golden scenarios
The system SHALL evaluate deterministic isolated trajectories with explicit
position, velocity, heading, rebound, or stop-time tolerances as applicable.

#### Scenario: Detect fidelity drift
- **WHEN** a backend measurement exceeds a scenario tolerance
- **THEN** the calibration gate fails and reports the metric and deviation

### Requirement: Honest evidence gaps
Unavailable legacy runtimes or unlicensed assets SHALL be reported as gaps and
SHALL NOT be represented as measured evidence.

#### Scenario: Reference runtime is unavailable
- **WHEN** no reproducible legacy trace can be executed
- **THEN** the report identifies the analytic or extracted substitute
