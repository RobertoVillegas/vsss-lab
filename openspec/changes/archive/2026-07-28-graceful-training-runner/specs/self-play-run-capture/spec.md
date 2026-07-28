## MODIFIED Requirements

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
