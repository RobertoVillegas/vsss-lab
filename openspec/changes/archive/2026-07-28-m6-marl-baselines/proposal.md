## Why

M5 proves the single-agent learning lifecycle, but it cannot represent three
independent decisions with shared weights or centralized training. M6 adds the
minimal auditable MARL baselines and makes identity leakage a blocking failure.

## What Changes

- Add agent-centric permutation-invariant team observations without robot IDs.
- Add a shared decentralized actor for three independent team decisions.
- Add IPPO with shared local critic and MAPPO with centralized team critic.
- Add versioned synchronous multi-agent trajectory metadata.
- Add C7 3v0 and C8 3v3-against-heuristic curriculum tasks.
- Add blocking identity-equivariance and better-than-random evaluation gates.
- Do not add self-play, league/history, async collection, Ray, recurrent actors,
  learned role labels, match protocols, or distributed training.

## Capabilities

### New Capabilities

- `permutation-safe-team-policy`: Identity-free observations and equivariant shared-actor execution.
- `ippo-mappo-training`: Synchronous IPPO/MAPPO losses, critics, trajectory metadata, and checkpoints.
- `marl-3v3-evaluation`: C7/C8 curriculum and deterministic better-than-random gates.

### Modified Capabilities

None.

## Impact

Extends `python/vsss_train`, experiment configs, tests, and local commands. It
uses the locked M5 PyTorch/TorchRL stack and does not alter the Rust hot loop.
