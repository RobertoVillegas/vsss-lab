## MODIFIED Requirements

### Requirement: Vector-world network batching
The runner SHALL own configurable independent worlds in one native batch and
batch shared actor, critic, and PPO tensor operations across them.

#### Scenario: Collect sixteen worlds
- **WHEN** a rollout explicitly requests sixteen vector worlds
- **THEN** its trajectory retains time, world, and three-agent batch dimensions

#### Scenario: Collect sixty-four worlds
- **WHEN** a rollout uses the default 64 vector worlds
- **THEN** its trajectory retains time, world, and three-agent batch dimensions
  while native physics steps the worlds without per-world Python calls

## ADDED Requirements

### Requirement: Persistent match-target training
The runner SHALL keep 30-second matches alive across shorter PPO rollouts and
SHALL support stopping after a requested number of completed matches.

#### Scenario: Train for one hundred thousand matches
- **WHEN** a run targets 100,000 matches
- **THEN** progress reports completed matches and matches/s and the final policy
  is checkpointed after reaching or exceeding the target
