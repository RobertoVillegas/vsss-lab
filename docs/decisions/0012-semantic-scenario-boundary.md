# ADR-0012: Semantic scenario compilation and termination boundary

## Status

Accepted for M15 on 2026-07-28.

## Context

M14 can restore arbitrary validated canonical states, but its scenario labels do
not define what constitutes a successful interception, save, clearance, shot,
or pass. Encoding those meanings in Rapier would couple curriculum policy to
authoritative physics. Encoding them only in reward would make outcome
attribution dependent on tunable shaping and invite event farming.

## Decision

M15 separates three responsibilities:

1. A typed, versioned scenario parameter value is compiled deterministically
   into a canonical `MatchState`.
2. Rapier advances that state and remains the sole authority for contacts,
   motion, boundaries, and goals.
3. A Python semantic evaluator observes the initial drill context and
   authoritative transitions and returns `running`, `success`, `failure`, or
   `unresolved` with a reason code.

Compilation must validate finite values, field bounds, body separation, and
nonterminal initial state before rollout. Parameters and canonical state receive
separate hashes. Team reflection changes positions, velocities, headings,
logical teams, target goal, and predicate orientation without selecting a
permanent policy or physical robot role.

Atomic drills terminate after their semantic predicate resolves and any bounded
confirmation window. Full matches preserve their existing goal grace and
termination behavior. Semantic outcomes and rewards are recorded separately
from base MAPPO reward.

## Consequences

- Scenario generation and skill meaning can evolve without changing Rapier or
  the canonical schema.
- A touch alone cannot be relabelled as a save or pass by reward tuning.
- Legacy M14 static scenarios remain readable but have no M15 predicate until
  explicitly migrated.
- Replay and checkpoint compatibility is preserved because actor tensor shapes
  do not change.
- Resume must checkpoint semantic curriculum state in addition to learner
  state.

## Validation

Property tests cover deterministic compilation, reflection, bounds, separation,
and seed diversity. Golden authoritative traces cover success, failure,
unresolved, and near-miss predicates. Vector tests prove termination isolation.

## Rollback

Select the immutable M14 configuration and legacy scenario suite. The semantic
compiler and evaluator are additive and do not modify canonical replays or
existing checkpoints.
