# self-play-run-capture Specification

## Purpose
TBD - created by archiving change m7-league-self-play. Update Purpose after archive.
## Requirements
### Requirement: Real self-play training iteration
Each training iteration SHALL collect a native trajectory against a frozen
registered opponent, optimize IPPO or MAPPO, increment policy version, and write
an immutable checkpoint with the realized matchup.

#### Scenario: Complete one iteration
- **WHEN** a current policy trains against a selected historical opponent
- **THEN** the new checkpoint has the next policy version and references that matchup

### Requirement: Iteration capture
The run SHALL capture checkpoints, metrics, evaluation reports, and replays at a
configurable interval without placing viewer work in the rollout hot loop.

#### Scenario: Capture a selected iteration
- **WHEN** an iteration matches the capture interval
- **THEN** its immutable checkpoint is evaluated into a viewer-compatible replay

### Requirement: Learned-policy replay compatibility
Evaluation replay files SHALL embed simulator configuration, policy identities
and versions, exact snapshots/actions/events, and valid checksums consumable by
the existing replay viewer.

#### Scenario: View an intermediate policy
- **WHEN** a captured iteration replay is passed to the native viewer
- **THEN** field, robots, ball, actions, score, and events can be reproduced

