# rl-skill-curriculum Specification

## Purpose
TBD - created by archiving change m5-rl-skills. Update Purpose after archive.
## Requirements
### Requirement: Native go-to-target skill task
The system SHALL expose a bounded single-robot task backed by the native
simulator, with identity-free robot-centric observations, normalized wheel
actions, distance-progress reward, success radius, and time-limit truncation.

#### Scenario: Reach a sampled target
- **WHEN** the robot enters the configured success radius
- **THEN** the episode terminates successfully and reports success

#### Scenario: Exceed the horizon
- **WHEN** the robot has not reached the target at the configured step limit
- **THEN** the episode truncates without success

### Requirement: Curriculum C0 through C5
The system SHALL define ordered stages C0–C5 that monotonically broaden target
distance, bearing, initial heading, and starting-position variation.

#### Scenario: Promote a stage
- **WHEN** deterministic evaluation meets a stage's configured threshold
- **THEN** subsequent episodes use the next curriculum stage

#### Scenario: Hold a stage
- **WHEN** deterministic evaluation misses its configured threshold
- **THEN** training remains at the current stage

### Requirement: Deterministic skill gate
The evaluator SHALL use explicit fixed seeds and deterministic policy actions,
and SHALL pass C5 only at a success rate of at least 95%.

#### Scenario: Evaluate final skill
- **WHEN** the policy succeeds on at least 95% of the configured C5 seeds
- **THEN** the executable M5 skill gate exits successfully and records evidence

#### Scenario: Reject a weak policy
- **WHEN** the policy succeeds on fewer than 95% of the configured C5 seeds
- **THEN** the executable M5 skill gate exits unsuccessfully

