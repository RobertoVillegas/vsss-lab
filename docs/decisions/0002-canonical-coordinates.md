# ADR-0002: Canonical coordinates and team reflection

- Status: accepted
- Date: 2026-07-27
- Owners: Roberto Villegas

## Context

Shared policies must not learn a permanent tactical role from physical identity
or attack direction.

## Decision

The field origin is its center; positive x points toward the yellow goal,
positive y is blue's left, heading zero points positive x, and positive rotation
is counter-clockwise. Yellow perspective is canonicalized by a π rotation,
blue/yellow relabeling, score exchange, and team-event exchange. Robot IDs remain
physical identities.

## Consequences

Both teams can consume the same positive-x convention. Reflected headings are
normalized to `[-π, π)`, so callers must also keep source headings normalized
when testing exact involution.

## Alternatives considered

Mirroring x alone reverses handedness. Rotating without relabeling leaves
team-relative semantics inconsistent. Reassigning robot IDs would leak roles
into identity mapping.

## Validation and rollback

Symmetry tests require reflection twice to recover valid normalized states and
verify score/event exchange. Revert this ADR and its transform before consumers
persist canonical observations.
