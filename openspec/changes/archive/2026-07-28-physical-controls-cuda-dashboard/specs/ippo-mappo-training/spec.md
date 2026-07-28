## ADDED Requirements

### Requirement: Explicit training compute device
The MARL runner SHALL accept auto, CPU, and CUDA device selection; auto SHALL
select CUDA when available and visibly warn when falling back to CPU.

#### Scenario: CUDA is unavailable in auto mode
- **WHEN** a run starts with auto selection and PyTorch cannot use CUDA
- **THEN** training continues on CPU and emits a visible fallback warning

### Requirement: Vector-world network batching
The runner SHALL batch shared actor, critic, and PPO tensor operations across a
configurable positive number of independent worlds.

#### Scenario: Collect sixteen worlds
- **WHEN** a rollout uses sixteen vector worlds
- **THEN** its trajectory retains time, world, and three-agent batch dimensions

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
