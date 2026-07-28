## ADDED Requirements

### Requirement: Versioned typed envelope
The protocol SHALL encode every controller message in a verified FlatBuffers
envelope containing protocol version, match ID, controller slot, sequence,
server tick, timestamp, deadline, and one typed payload.

#### Scenario: Decode a supported controller action
- **WHEN** the server receives a valid action envelope from a negotiated protocol version
- **THEN** it exposes the typed bounded action and complete ordering metadata

### Requirement: Compatible schema evolution
Protocol schemas SHALL evolve by additive fields with stable field IDs and
defaults, and SHALL remain conformant with the previous accepted schema.

#### Scenario: Validate an additive schema revision
- **WHEN** a new optional field is appended with a compatible default
- **THEN** previous golden buffers remain readable and FlatBuffers conformity passes

### Requirement: Strict message validation
The server SHALL reject malformed, oversized, duplicate, stale, future,
wrong-slot, and unsupported-version messages with an auditable reason.

#### Scenario: Receive a stale sequence
- **WHEN** a controller sends a sequence not greater than its last accepted sequence
- **THEN** the server rejects it without changing the pending action
