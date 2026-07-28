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

