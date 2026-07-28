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

### Requirement: Reward-independent replay analytics
The system SHALL derive versioned per-match, per-team, and per-robot possession,
pressure, positioning, movement, coordination, and event metrics from canonical
artifacts without changing training reward or replay state.

#### Scenario: Inspect a possession interval
- **WHEN** one team touches the ball and retains the last valid touch until an
  opponent contact
- **THEN** the interval is attributed with its start, end, duration,
  territorial progress, and terminating event

#### Scenario: Diagnostic metric improves without terminal improvement
- **WHEN** a candidate increases possession or touches but does not improve
  paired terminal outcomes
- **THEN** the metric is reported diagnostically and cannot independently
  promote the candidate

### Requirement: Explicit learner configuration boundaries
The system SHALL version scenario mutation, observation building, action
parsing, reward calculation, termination, and rendering independently while
preserving authoritative physics ownership.

#### Scenario: Compare an action abstraction
- **WHEN** a discrete or chunked action parser is evaluated against continuous
  wheel control
- **THEN** it starts a distinct lineage and is compared at the same physical
  control frequency and evaluation suite

### Requirement: Current prior-art evidence
Every major M14 implementation block SHALL include a dated review of maintained
external systems and recent primary research with explicit adopt, adapt, defer,
or reject decisions.

#### Scenario: Begin a policy-architecture ablation
- **WHEN** implementation of memory or entity attention starts
- **THEN** its evidence note identifies current comparable implementations,
  local differences, expected benefit, and a falsifiable acceptance gate
