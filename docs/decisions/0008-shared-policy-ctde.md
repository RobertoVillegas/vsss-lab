# ADR-0008: Shared policy and centralized training

- Status: accepted
- Date: 2026-07-28
- Owners: Roberto Villegas

## Context

M6 introduces three independent decisions per team. Physical identity, array
position, or centralized execution must not become a permanent tactical role.

## Decision

Canonicalize attack direction, exclude IDs, aggregate teammate and opponent
records with shared Deep Sets encoders, and apply one actor independently to
each agent observation. IPPO uses a shared local critic. MAPPO uses a
permutation-equivariant centralized critic that pools all three observations
but is discarded for execution.

Synchronous rollouts carry policy ID/version and are rejected when stale. C7/C8
use the native simulator; identity equivariance and improvement over fixed-seed
random actions are blocking M6 gates.

## Consequences

Actor parameter count is independent of team size and roles cannot be encoded
by an ID feature. Deep Sets loses some pairwise structure, while the centralized
critic sees team information unavailable during execution as intended by CTDE.

## Alternatives considered

One network emitting all three actions violates decentralized execution.
Separate networks per robot invite identity specialization. Sorted identity
slots are simpler but not robustly permutation invariant.

## Validation and rollback

Observation, actor, IPPO/MAPPO loss, stale-policy, checkpoint-algorithm,
permutation, C7/C8, and random-baseline tests gate M6. Rollback removes the MARL
package while leaving the M5 single-agent trainer intact.
