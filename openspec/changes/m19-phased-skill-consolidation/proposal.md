# M19: Phased skill consolidation

## Why

The M18 confirmation run kept PPO diagnostics stable but exchanged semantic
skills: early global success regressed while a tiny rotation gain caused a later
checkpoint to be selected. Simultaneous practice across seven objectives and
six roster sizes creates avoidable gradient interference.

## What changes

- Focus training through foundation, defense, cooperation, rotation, and
  integration phases.
- Require consecutive immutable holdout passes before phase promotion.
- Rehearse mastered phases without reopening the complete task mixture.
- Keep all semantic families in evaluation at every phase.
- Rank checkpoints by phase evidence, passed gates, global success, and
  unresolved outcomes before minimum-family tie breakers.
- Reduce continuous shaping and add the causal useful-touch impulse signal.

## Non-goals

- No physics or MAPPO algorithm change.
- No removal of semantic rules or telemetry.
- No warm start from the conflicted M18 policy.
