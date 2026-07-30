# ADR 0013: Hierarchical primitive action contract

- Status: accepted
- Date: 2026-07-30

## Context

Direct 50 Hz wheel control made the policy learn navigation, ball acquisition,
contact timing, and team strategy simultaneously. Long M23 runs produced
non-zero actions but spent most captured time away from a stationary ball. The
robot-soccer reference architecture in Su et al. separates a slower strategic
policy from reusable locomotion and ball skills.

## Decision

M24 introduces a versioned categorical action set with 17 choices: stop, eight
field-relative navigation directions, and eight directed strike primitives.
The learned actor selects at the existing control rate; deterministic,
causal controllers convert the token into bounded differential wheel commands.
Yellow-team directions are reflected into canonical attacking coordinates.

The exact action table is frozen in
`tests/golden/m24_primitive_actions.json`. Checkpoints record
`action_parser = "primitive"` and cannot silently load under a continuous or
lattice parser.

Direct wheel control remains available through the earlier configurations as
the rollback and ablation baseline.

## Consequences

- Strategy gradients no longer need to rediscover basic pursuit and contact.
- Primitive behavior is inspectable and can be benchmarked independently of
  reward or learning.
- The first implementation is deterministic rather than a learned low-level
  policy, so sim-to-real calibration remains future work.
- MAPPO and IPPO can be compared with the same observations, primitives,
  rewards, seeds, and network width.
