# ADR-0001: Canonical SI unit types

- Status: accepted
- Date: 2026-07-27
- Owners: Roberto Villegas

## Context

Physics, policies, protocols, and robot adapters must exchange physical values
without relying on implicit unit conventions.

## Decision

`vsss-spec` represents each physical dimension with a transparent `f32` newtype.
The scalar is always SI: metres, seconds, radians, kilograms, newtons, or
newton-metres. Serialized values remain JSON numbers while Rust APIs retain the
dimension.

## Consequences

Call sites must construct the intended unit explicitly. The representation stays
compact and backend-neutral, but does not provide compile-time unit algebra.

## Alternatives considered

Bare `f32` values permit accidental mixing. A dimensional-analysis dependency
adds generic types and conversions that M1 does not yet need.

## Validation and rollback

Contract tests verify scalar serialization and reject non-finite physical state.
Rollback restores bare scalars before downstream contracts are released.
