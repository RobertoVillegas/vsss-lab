# ADR-0005: Scripted baselines and M4 replay

- Status: accepted
- Date: 2026-07-27
- Owners: Roberto Villegas

## Context

Learning milestones require reproducible scripted opponents and inspectable
regression artifacts without assigning permanent tactical roles to robot IDs.

## Decision

Use normalized differential-drive skills and geometric minimum-cost role
assignment recalculated from state. IDs are metadata, never assignment inputs.
Record M4 matches as deterministic JSONL containing a header and one canonical
tick record with actions, snapshot, event flags, and checksum per step. Provide
a headless validation/summary CLI.

## Consequences

Baselines are simple and auditable. JSONL is intentionally verbose and dynamic
assignment may chatter; later milestones can add hysteresis and compact replay.

## Alternatives considered

Fixed robot roles violate the PRD. A learned assignment belongs after scripted
baselines. A binary replay schema is premature before protocol work.

## Validation and rollback

Unit, permutation, and byte-identical replay tests gate M4. Rollback removes the
Python packages and viewer without changing native contracts.
