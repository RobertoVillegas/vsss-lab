## ADDED Requirements

### Requirement: Environment-step training budget
The runner SHALL accept a positive environment-step target mutually exclusive
with a completed-match target and SHALL count one control decision in one world
as one step.

#### Scenario: Train for twenty million steps
- **WHEN** a CUDA run targets 20,000,000 environment steps
- **THEN** progress and ETA use completed steps and the runner checkpoints after
  the first complete PPO rollout that reaches or exceeds the target
