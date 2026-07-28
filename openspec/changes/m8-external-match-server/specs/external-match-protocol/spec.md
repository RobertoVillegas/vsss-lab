# external-match-protocol Specification

## Requirements

### Requirement: Versioned heterogeneous wire contract
The system SHALL exchange verified, size-bounded FlatBuffers envelopes whose
protocol version, match, slot, sequence, tick, deadline, and payload kind are
validated before use.

#### Scenario: Reject incompatible input
- **WHEN** a controller sends an incompatible, malformed, stale, future, or
  wrong-slot envelope
- **THEN** the server rejects and records it without mutating match state

### Requirement: Additive protocol evolution
The protocol SHALL preserve stable field identifiers and gate schema changes
with conformity checks and committed cross-language fixtures.

#### Scenario: Regenerate bindings
- **WHEN** bindings are regenerated from the committed schema
- **THEN** Rust and Python decode the same golden envelopes
