# self-play-run-capture Specification

## Purpose
TBD - created by archiving change m7-league-self-play. Update Purpose after archive.
## Requirements
### Requirement: Real self-play training iteration

Each training iteration SHALL collect a native trajectory against a frozen
registered opponent, optimize IPPO or MAPPO, increment policy version, permit
resume from the latest durable compatible checkpoint, and report progress and
estimated completion time.

#### Scenario: Complete one iteration

- **WHEN** a current policy trains against a selected historical opponent
- **THEN** the new checkpoint has the next policy version and references that matchup

#### Scenario: Resume interrupted sustained training

- **WHEN** an operator resumes a run with an existing league registry
- **THEN** training loads the latest registered checkpoint and continues with the next iteration number

#### Scenario: Request a graceful stop

- **WHEN** the operator sends SIGINT or SIGTERM during training
- **THEN** the current iteration completes, its latest policy is durably registered, and the process exits

### Requirement: Iteration capture

The run SHALL retain metrics for every iteration while writing checkpoints and
evaluation replays at independently configurable intervals without placing
viewer work in the rollout hot loop.

#### Scenario: Capture a selected iteration

- **WHEN** an iteration matches the capture interval
- **THEN** its immutable checkpoint is evaluated into a viewer-compatible replay

#### Scenario: Bound artifact growth

- **WHEN** checkpoint and capture intervals are greater than one
- **THEN** intermediate optimization metrics are retained without writing a checkpoint or replay every iteration

### Requirement: Learned-policy replay compatibility
Evaluation replay files SHALL embed simulator configuration, policy identities
and versions, exact snapshots/actions/events, and valid checksums consumable by
the existing replay viewer.

#### Scenario: View an intermediate policy
- **WHEN** a captured iteration replay is passed to the native viewer
- **THEN** field, robots, ball, actions, score, and events can be reproduced

### Requirement: Fixed simulated-time capture

Each evaluation capture SHALL represent the configured amount of simulated
time using the canonical control period, independently of wall-clock execution
speed.

#### Scenario: Capture one simulated minute

- **WHEN** capture duration is 60 seconds and control period is 20 milliseconds
- **THEN** the replay contains 3,000 decision frames and ends at 60 seconds simulated time

