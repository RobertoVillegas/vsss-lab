## ADDED Requirements

### Requirement: Reproducible PPO training
The system SHALL train a continuous-action single-agent policy using clipped PPO,
GAE, locked PyTorch/TorchRL dependencies, and an explicit seed.

#### Scenario: Repeat a seeded smoke run
- **WHEN** two CPU smoke runs use the same configuration and seed
- **THEN** they produce the same final model checksum and metric values

### Requirement: Versioned checkpoint lifecycle
The trainer SHALL save and resume versioned checkpoints containing policy,
critic, optimizer, progress, curriculum, configuration fingerprint, and random
generator state.

#### Scenario: Resume at an update boundary
- **WHEN** training resumes from a compatible trusted checkpoint
- **THEN** the next update and metrics match uninterrupted execution

#### Scenario: Reject an incompatible checkpoint
- **WHEN** checkpoint schema or configuration fingerprint is incompatible
- **THEN** the trainer fails before collecting new rollout frames

### Requirement: Machine-readable metrics
The trainer SHALL append schema-versioned JSONL metrics with run identity, seed,
stage, update, frames, losses, return, success rate, and elapsed time.

#### Scenario: Inspect a completed run
- **WHEN** a smoke training run completes
- **THEN** every metrics line is valid JSON and required fields are present
