## ADDED Requirements

### Requirement: Evidence-gated promotion
The system SHALL promote a training configuration or checkpoint only from
terminal match objectives evaluated over paired independent seeds and holdout
scenarios, never from shaped training return or checkpoint recency alone.

#### Scenario: Candidate has higher shaped return but worse terminal outcomes
- **WHEN** a candidate improves training return but its paired terminal match
  confidence bound regresses against the promoted baseline
- **THEN** promotion is rejected and the objective vector is recorded

### Requirement: Valid adaptive scenario curriculum
The system SHALL allocate training across routine, frontier, prioritized
failure, and immutable holdout scenarios while validating every generated
canonical state before simulation.

#### Scenario: Teacher proposes an overlapping initial state
- **WHEN** a mutated scenario places a robot and ball in invalid contact
- **THEN** the scenario is rejected before entering a rollout

#### Scenario: Learner masters a frontier bucket
- **WHEN** a bucket's success and learning-progress thresholds indicate mastery
- **THEN** training allocation moves toward another frontier bucket while
  retaining a bounded rehearsal allocation

### Requirement: Reproducible multi-fidelity search
The system SHALL search only typed bounded parameters through resumable smoke,
screen, and confirmation stages that record code, configuration, seeds,
lineage, objectives, pruning, and compute.

#### Scenario: Resume an interrupted study
- **WHEN** a study restarts from its persistent identifier
- **THEN** completed trials are not repeated and pending trials preserve their
  declared seed sets and parent lineage

### Requirement: Isolated optional policy memory
The system SHALL isolate recurrent state by world and agent, reset it at episode
boundaries, and version it in checkpoints without changing canonical physics.

#### Scenario: One vectorized world terminates
- **WHEN** one world resets while other worlds continue
- **THEN** only the terminated world's agent memories are cleared

### Requirement: Verified teacher demonstrations
The system SHALL accept planner demonstrations only when they replay through the
authoritative simulator and satisfy a deterministic skill success predicate.

#### Scenario: Planner exploits invalid physics
- **WHEN** a high-scoring planned trajectory violates a physical invariant
- **THEN** the demonstration is rejected and cannot enter imitation training

### Requirement: Evidence-gated accelerator adoption
The system SHALL retain Rapier as the authoritative simulator unless a
device-resident prototype demonstrates physical trace parity and material
end-to-end training throughput improvement.

#### Scenario: GPU prototype is fast but changes contact outcomes
- **WHEN** an accelerator prototype improves frames per second but disagrees
  with the golden contact or goal corpus
- **THEN** it is rejected as a production backend and the mismatch is recorded
