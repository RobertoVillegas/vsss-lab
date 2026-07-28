# controller-sdk Specification

## Requirements

### Requirement: Typed Rust and Python lifecycle
The system SHALL provide Rust and Python controller SDKs for handshake,
capabilities, reset, observation/action, heartbeat, event, and result handling.

#### Scenario: Implement a controller
- **WHEN** a developer supplies a typed action callback
- **THEN** the SDK handles framing, validation, assignment, and transport

### Requirement: Logical identity is transport-stable
Controller identity SHALL remain independent from temporary side, tactical role,
and physical visual marker.

#### Scenario: Switch sides
- **WHEN** a controller is reassigned from blue to yellow
- **THEN** its transport and policy identity remain unchanged
