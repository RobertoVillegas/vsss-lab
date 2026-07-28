# ADR-0003: Rapier2D reference physics backend

- Status: accepted
- Date: 2026-07-27
- Owners: Roberto Villegas

## Context

VSSS Lab needs deterministic headless physics before optimized batch execution.

## Decision

Use locked Rapier2D with enhanced determinism, zero gravity, fixed timestep,
dynamic cuboid robots, a CCD ball, and static field boundaries. The public API
uses only `vsss-spec`. Snapshots store canonical state and rebuild Rapier.

## Consequences

The same platform and dependency version are deterministic. Snapshot restoration
is transparent but slower than serializing engine internals.

## Alternatives considered

A custom solver is premature. Box2D adds C/C++ integration. Opaque Rapier
snapshots make schema compatibility harder to control.

## Validation and rollback

Correctness, snapshot replay, and checksum tests gate the backend. Revert the M2
crates and this ADR before downstream bindings if the backend changes.
