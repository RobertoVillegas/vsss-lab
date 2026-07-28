# Adaptive scenario evidence — 2026-07-28

## Maintained-system comparison

RLGym v2 keeps `StateMutator` separate from transition physics and permits
ordered mutation composition. Its maintained examples also distinguish natural
goal termination from timeout/no-touch truncation. VSSS adopts those boundaries
without importing Rocket League's engine or state model.

## Decisions

- **Adopt:** typed scenario roles and soccer-skill buckets.
- **Adapt:** deterministic bounded mutation followed by exact geometric
  validity checks before a state can reach Rapier.
- **Adopt:** a stable 20% routine, 50% frontier, 20% deduplicated failure, 10%
  holdout compute contract.
- **Adapt:** frontier rank uses absolute measured learning progress, but excludes
  already mastered and currently impossible buckets when evidence exists.
- **Adopt:** content hashes for suite deduplication and atomic suite writes.
- **Reject:** coordinate-only GAN generation and unchecked mutation.
- **Reject:** mutation of immutable holdouts.

## Falsifiable gate

Overlap, non-finite state, out-of-field geometry, duplicate canonical states,
and holdout mutation must fail before rollout. Allocation totals must equal the
requested batch and remain deterministic for identical progress evidence.
