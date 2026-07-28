## Context

M3 exposes six simultaneous agents, M4 supplies a dynamic identity-free teacher,
and M5 establishes PPO artifacts. M6 must add centralized training with
decentralized execution without turning physical IDs or array slots into roles.

## Goals / Non-Goals

**Goals:** one actor applied independently to all three teammates, IPPO and
MAPPO critics, permutation-safe observations, synchronous policy-versioned
trajectories, C7/C8 tasks, and deterministic identity/random-baseline gates.

**Non-Goals:** league/self-play, asynchronous or distributed collectors,
recurrent state, explicit/latent roles, opponent pools, protocol servers, and
production-scale convergence claims.

## Decisions

1. Team coordinates are canonicalized so every controlled team attacks +x.
2. Each agent observation has self/ball/goals/context features plus Deep Sets
   pools for two teammates and three opponents. Entity records contain only
   relative geometry and motion; IDs and source slots are excluded.
3. A single feed-forward Gaussian actor is called on shape
   `[world, agent, feature]`; it never receives another agent's private slot or
   a one-hot identity.
4. IPPO uses a shared local value network. MAPPO uses a centralized critic that
   pools all three agent embeddings then combines the pool with each local
   embedding, producing one equivariant value per agent.
5. Both algorithms share the M5 clipped-PPO/GAE semantics. Multi-agent advantage
   normalization is global over samples but never separately keyed by identity.
6. Synchronous trajectory batches record run, episode, world, tick, team,
   policy ID/version, observations, actions, log probabilities, rewards,
   done/truncated, values, and global-state reference.
7. C7 is coordinated 3v0 and C8 is 3v3 against the M4 heuristic. The M6 local
   competence gate compares fixed-seed team progress against random actions;
   score-based league evaluation belongs to M7.
8. The M4 controller may initialize the shared actor by behavior distillation.
   Subsequent IPPO/MAPPO optimization is explicit and independently testable.

## Risks / Trade-offs

- **Sorted slots leak ordering at ties** → Deep Sets aggregation is the normative
  path and permutation tests include exact slot permutations.
- **Dense progress can differ from winning** → M6 only gates against random;
  M7 owns score, promotion, and historical robustness.
- **Central critic may leak execution state** → actor signature accepts only one
  agent observation and tests run it independently.
- **Tiny smoke rollouts do not establish algorithm superiority** → report this
  limitation and keep thresholds deterministic and narrow.

## Migration Plan

Add isolated Python modules and configs over existing canonical state rows.
Rollback removes M6 artifacts without changing M1–M5 APIs or checkpoints.

## Open Questions

Deep Sets versus attention and feed-forward versus GRU remain later controlled
experiments.
