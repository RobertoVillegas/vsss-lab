## ADDED Requirements

### Requirement: Typed controller lifecycle
Rust and Python SDKs SHALL expose typed reset, observation/action, event, result,
heartbeat, and error behavior while hiding wire framing.

#### Scenario: Implement a Python controller
- **WHEN** a developer supplies the lifecycle callbacks
- **THEN** the SDK completes negotiation and exchanges typed messages with the Rust server

### Requirement: Cross-language semantic parity
Rust and Python SDKs SHALL produce and consume the same protocol golden fixtures
and enforce the same action bounds, sequence, and deadline semantics.

#### Scenario: Round-trip a golden observation
- **WHEN** either SDK decodes and re-encodes the accepted observation fixture
- **THEN** both expose equivalent canonical values and compatible bytes
