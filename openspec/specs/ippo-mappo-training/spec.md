# ippo-mappo-training Specification

## Purpose
TBD - created by archiving change m6-marl-baselines. Update Purpose after archive.
## Requirements
### Requirement: IPPO shared-parameter baseline
The learner SHALL implement clipped PPO and GAE over three agents with a shared
decentralized actor and shared local critic.

#### Scenario: Optimize an IPPO batch
- **WHEN** a valid synchronous team trajectory is optimized
- **THEN** actor and local-critic parameters update with finite losses

### Requirement: MAPPO centralized-training baseline
The learner SHALL implement clipped PPO and GAE with the shared decentralized
actor and a permutation-equivariant centralized team critic.

#### Scenario: Optimize a MAPPO batch
- **WHEN** a valid synchronous team trajectory is optimized
- **THEN** actor and centralized-critic parameters update with finite losses

### Requirement: Versioned synchronous trajectories
Every multi-agent rollout SHALL identify its policy ID and monotonically
increasing policy version and SHALL retain the canonical M6 trajectory fields.

#### Scenario: Reject stale synchronous data
- **WHEN** a trajectory policy version differs from the learner's current version
- **THEN** optimization fails before parameters change

### Requirement: MARL checkpoint compatibility
Checkpoints SHALL identify algorithm, shared actor, critic, optimizer, policy
version, curriculum stage, configuration fingerprint, and RNG state.

#### Scenario: Load with another algorithm
- **WHEN** an IPPO checkpoint is loaded as MAPPO or vice versa
- **THEN** loading fails before model state is applied

### Requirement: Explicit training compute device
The MARL runner SHALL accept auto, CPU, and CUDA device selection; auto SHALL
select CUDA when available and visibly warn when falling back to CPU.

#### Scenario: CUDA is unavailable in auto mode
- **WHEN** a run starts with auto selection and PyTorch cannot use CUDA
- **THEN** training continues on CPU and emits a visible fallback warning

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

### Requirement: Long-run terminal observability
An interactive run SHALL keep a progress bar below current and rolling training
metrics and SHALL provide aligned line output when no interactive terminal is
available.

#### Scenario: Run in an interactive terminal
- **WHEN** an iteration completes
- **THEN** return, progress, losses, throughput, device, worlds, and checkpoint
  state update above the bottom progress bar

### Requirement: Action-change regularization
The team reward SHALL optionally penalize squared changes in normalized policy
actions independently from physical actuator dynamics.

#### Scenario: Abrupt consecutive actions
- **WHEN** action-delta regularization is positive and commands change abruptly
- **THEN** the reward includes the configured negative action-delta term

### Requirement: Persistent match-target training
The runner SHALL keep 30-second matches alive across shorter PPO rollouts and
SHALL support stopping after a requested number of completed matches.

#### Scenario: Train for one hundred thousand matches
- **WHEN** a run targets 100,000 matches
- **THEN** progress reports completed matches and matches/s and the final policy
  is checkpointed after reaching or exceeding the target

### Requirement: Environment-step training budget
The runner SHALL accept a positive environment-step target mutually exclusive
with a completed-match target and SHALL count one control decision in one world
as one step.

#### Scenario: Train for twenty million steps
- **WHEN** a CUDA run targets 20,000,000 environment steps
- **THEN** progress and ETA use completed steps and the runner checkpoints after
  the first complete PPO rollout that reaches or exceeds the target

### Requirement: Post-goal closure
The match environment SHALL continue for the configured one-second goal pause
after a goal event before reporting match termination.

#### Scenario: Goal at 50 Hz control
- **WHEN** a goal event occurs with a one-second configured pause
- **THEN** the event is rewarded once and termination occurs after 50 control
  frames

