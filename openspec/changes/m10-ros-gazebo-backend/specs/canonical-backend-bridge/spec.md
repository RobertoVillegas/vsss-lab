# canonical-backend-bridge Specification

## Requirements

### Requirement: Policy-independent backend selection
A policy SHALL consume the same canonical observation and produce the same six
wheel-action pairs for native and bridged backends.

#### Scenario: Change backend
- **WHEN** backend configuration changes from native to bridge
- **THEN** policy code and policy API remain unchanged

### Requirement: Validated process bridge
The bridge SHALL validate sequence, state width, action shape, finite values,
and child lifecycle for every reset and step.

#### Scenario: Sidecar returns invalid state
- **WHEN** a response has wrong sequence, shape, or non-finite values
- **THEN** the adapter raises an infrastructure error
